//! [`SyncPipeline`] ZST -- associated functions implementing the
//! main-thread sync pipeline. Each method takes disjoint borrows of the
//! engine's sub-structs (`Scene`, `EntityAnnotations`, `VisoOptions`,
//! `GpuPipeline`, `AnimationState`) so the pipeline is expressible
//! without routing through `&mut self` on [`VisoEngine`].
//!
//! [`VisoEngine`]: super::super::VisoEngine

use std::collections::HashMap;
use std::sync::Arc;

use glam::Vec3;
use molex::entity::molecule::id::EntityId;
use molex::{Assembly, MoleculeType, SSType};
use rustc_hash::FxHashMap;

use super::super::annotations::EntityAnnotations;
use super::super::entity_view::EntityView;
use super::super::scene::Scene;
use super::super::scene_state::SceneRenderState;
use super::super::trajectory::TrajectoryFrame;
use crate::animation::transition::Transition;
use crate::animation::AnimationState;
use crate::options::{
    DisplayOptions, DrawingMode, GeometryOptions, VisoOptions,
};
use crate::renderer::gpu_pipeline::SceneChainData;
use crate::renderer::pipeline::prepared::{
    FullRebuildBody, FullRebuildEntity, PreparedRebuild,
};
use crate::renderer::pipeline::SceneRequest;
use crate::renderer::GpuPipeline;

/// Main-thread sync pipeline.
///
/// ZST -- all methods are associated functions taking disjoint borrows of
/// the engine's sub-structs. Per-sync working state (ribbon cache, flat
/// color buffers, etc.) is rebuilt inside each call rather than cached
/// on the pipeline.
pub(crate) struct SyncPipeline;

impl SyncPipeline {
    /// Rederive viso-side state from an `Assembly` snapshot.
    ///
    /// Called when the triple buffer yields a snapshot whose
    /// generation differs from `scene.last_seen_generation`.
    pub(crate) fn sync_from_assembly(
        scene: &mut Scene,
        annotations: &mut EntityAnnotations,
        options: &VisoOptions,
        assembly: &Assembly,
    ) {
        scene.render_state = SceneRenderState::from_assembly(assembly);

        let mut seen: std::collections::HashSet<EntityId> =
            std::collections::HashSet::default();
        for entity in assembly.entities() {
            let id = entity.id();
            let _ = seen.insert(id);
            let ss = assembly.ss_types(id);
            let ss_override = annotations.ss_overrides.get(&id).cloned();
            // Topology is composition + SS; it does not depend on coords. On a
            // coord-only republish (every wiggle/shake frame) the cached one
            // still describes the entity, so keep it rather than rebuild the
            // bond list and sidechain lookup map for every entity, every frame.
            let reused = scene
                .entity_state
                .get(&id)
                .filter(|state| {
                    crate::engine::entity_view::topology_reusable(
                        &state.topology,
                        entity,
                        ss,
                    )
                })
                .map(|state| Arc::clone(&state.topology));
            let topology = reused.unwrap_or_else(|| {
                Arc::new(crate::engine::entity_view::derive_topology(
                    entity, ss,
                ))
            });
            let drawing_mode = annotations.resolved_drawing_mode(
                options,
                id,
                topology.molecule_type,
            );
            let fresh_version = scene.bump_mesh_version();
            match scene.entity_state.entry(id) {
                std::collections::hash_map::Entry::Occupied(mut slot) => {
                    let state = slot.get_mut();
                    // sheet_offsets are entity-local-residue-indexed. They
                    // stay index-valid iff the residue layout (per-residue
                    // atom ranges) is unchanged. Clear only on a genuine
                    // re-layout, where a stale slice would index into a
                    // different residue space; the next prepared mesh
                    // refills it. On a coord-only republish (same layout,
                    // e.g. an interactive pull) retain the existing slice so
                    // sidechain-anchored geometry does not snap to the
                    // un-flattened position between async rebuilds.
                    let layout_unchanged = state.topology.residue_atom_ranges
                        == topology.residue_atom_ranges;
                    state.topology = topology;
                    state.ss_override = ss_override;
                    state.drawing_mode = drawing_mode;
                    state.mesh_version = fresh_version;
                    state.per_residue_colors = None;
                    // ribbon_anchors are entity-local-residue-indexed like
                    // sheet_offsets and stay index-valid only while the
                    // residue layout is unchanged; clear both on a genuine
                    // re-layout so a stale slice can't index a different
                    // residue space. The next prepared mesh refills them.
                    if !layout_unchanged {
                        state.sheet_offsets.clear();
                        state.ribbon_anchors.clear();
                        state.displayed_positions.clear();
                    }
                }
                std::collections::hash_map::Entry::Vacant(slot) => {
                    let _ = slot.insert(EntityView {
                        drawing_mode,
                        ss_override,
                        topology,
                        per_residue_colors: None,
                        sheet_offsets: Vec::new(),
                        ribbon_anchors: Vec::new(),
                        displayed_positions: Vec::new(),
                        mesh_version: fresh_version,
                    });
                }
            }
            scene
                .positions
                .insert_from_reference(id, entity.positions());

            // New entity? Seed visibility from ambient-type defaults.
            if let std::collections::hash_map::Entry::Vacant(slot) =
                annotations.visibility.entry(id)
            {
                let visible = match entity.molecule_type() {
                    MoleculeType::Water => options.display.show_waters,
                    MoleculeType::Ion => options.display.show_ions,
                    MoleculeType::Solvent => options.display.show_solvent,
                    _ => true,
                };
                let _ = slot.insert(visible);
            }
        }

        scene.entity_state.retain(|id, _| seen.contains(id));
        scene.positions.retain(|id| seen.contains(&id));
        annotations.retain_entities(|id| seen.contains(&id));
    }

    /// Drain any pending assembly snapshot, then submit a full-rebuild
    /// request to the background mesh processor using the current
    /// pending transitions.
    pub(crate) fn sync_now(
        scene: &mut Scene,
        annotations: &mut EntityAnnotations,
        options: &VisoOptions,
        gpu: &mut GpuPipeline,
        animation: &mut AnimationState,
    ) {
        Self::poll_assembly_force(scene, annotations, options);
        let transitions = animation.take_pending_transitions();
        Self::submit_full_rebuild(
            scene,
            annotations,
            options,
            gpu,
            animation,
            transitions,
        );
    }

    /// Drain any pending [`Assembly`] snapshot and
    /// immediately apply it. Used by [`Self::sync_now`] when the host
    /// has just pushed a new snapshot via
    /// [`crate::VisoEngine::set_assembly`] that must be reflected
    /// before the next render.
    fn poll_assembly_force(
        scene: &mut Scene,
        annotations: &mut EntityAnnotations,
        options: &VisoOptions,
    ) {
        let Some(assembly) = scene.pending.take() else {
            return;
        };
        if assembly.generation() == scene.last_seen_generation {
            return;
        }
        Self::sync_from_assembly(scene, annotations, options, &assembly);
        scene.current = assembly;
        scene.last_seen_generation = scene.current.generation();
    }

    /// Submit a `FullRebuild` request to the background mesh processor
    /// with per-entity transitions.
    ///
    /// Entities in the map animate with their transition; entities
    /// not in the map snap. Pass an empty map for a non-animated sync.
    pub(crate) fn submit_full_rebuild(
        scene: &mut Scene,
        annotations: &EntityAnnotations,
        options: &VisoOptions,
        gpu: &mut GpuPipeline,
        animation: &mut AnimationState,
        entity_transitions: HashMap<u32, Transition>,
    ) {
        let request_entities =
            Self::build_full_rebuild_entities(scene, annotations, options);

        let entity_options = Self::resolve_entity_options(annotations, options);
        animation.merge_pending_transitions(entity_transitions);

        let generation = gpu.scene_processor.next_generation();
        let entity_ids: rustc_hash::FxHashSet<EntityId> =
            request_entities.iter().map(|e| e.id).collect();
        let topology_generation =
            gpu.scene_processor.advance_topology_generation(entity_ids);
        log::debug!(
            "submit_full_rebuild: submitting FullRebuild gen={generation}, \
             topology_gen={topology_generation}, entity_count={}",
            request_entities.len(),
        );
        gpu.scene_processor
            .submit(SceneRequest::FullRebuild(Box::new(FullRebuildBody {
                entities: request_entities,
                display: options.display.clone(),
                colors: options.colors.clone(),
                geometry: options.resolved_geometry(),
                entity_options,
                generation,
                topology_generation,
            })));
    }

    fn resolve_entity_options(
        annotations: &EntityAnnotations,
        options: &VisoOptions,
    ) -> FxHashMap<u32, (DisplayOptions, GeometryOptions)> {
        let resolved_geometry = options.resolved_geometry();
        annotations
            .appearance
            .iter()
            .map(|(&id, ovr)| {
                (
                    id.raw(),
                    (
                        ovr.to_display_options(&options.display),
                        ovr.to_geometry_options(&resolved_geometry),
                    ),
                )
            })
            .collect()
    }

    /// Build the per-sync FullRebuild payload: for each visible entity,
    /// compute per-residue colors from current display options and
    /// positions. Caches colors onto `EntityView` so main-thread color
    /// uploads can concatenate without recomputing.
    fn build_full_rebuild_entities(
        scene: &mut Scene,
        annotations: &EntityAnnotations,
        options: &VisoOptions,
    ) -> Vec<FullRebuildEntity> {
        let assembly = Arc::clone(&scene.current);
        let b_range = assembly_b_range(scene);
        let mut result = Vec::new();

        for (entity_index, entity) in assembly.entities().iter().enumerate() {
            let eid = entity.id();
            if !annotations.is_visible(eid) {
                continue;
            }
            let Some(positions) = scene.positions.get(eid) else {
                continue;
            };
            let positions = positions.to_vec();
            let Some(state) = scene.entity_state.get_mut(&eid) else {
                continue;
            };
            let display = annotations.appearance.get(&eid).map_or_else(
                || options.display.clone(),
                |ovr| ovr.to_display_options(&options.display),
            );
            let ss_types: Vec<SSType> = state
                .ss_override
                .clone()
                .unwrap_or_else(|| state.topology.ss_types.clone());
            let backbone_chains =
                state.topology.protein_backbone_chains(&positions);
            let per_residue_colors = if state.topology.is_protein() {
                let b_factors = matches!(
                    display.backbone_color_scheme(),
                    crate::options::ColorScheme::BFactor
                )
                .then(|| residue_max_backbone_b(&state.topology));
                per_entity_colors(
                    entity_index,
                    &backbone_chains,
                    &ss_types,
                    annotations.scores.get(&eid).map(Vec::as_slice),
                    b_factors.as_deref(),
                    b_range,
                    &display,
                )
            } else {
                None
            };
            state.per_residue_colors.clone_from(&per_residue_colors);

            result.push(FullRebuildEntity {
                id: eid,
                mesh_version: state.mesh_version,
                drawing_mode: state.drawing_mode,
                topology: Arc::clone(&state.topology),
                positions,
                ss_override: state.ss_override.clone(),
                per_residue_colors,
            });
        }
        result
    }

    /// Upload prepared scene geometry to GPU renderers. Rebuilds the
    /// flat [`SceneChainData`] from entity_state + positions for the
    /// renderers that still consume it internally (backbone metadata
    /// cache used by frustum culling + LOD tier comparison).
    fn upload_prepared_to_gpu(
        scene: &Scene,
        annotations: &EntityAnnotations,
        gpu: &mut GpuPipeline,
        prepared: &PreparedRebuild,
        animating: bool,
    ) {
        let (backbone_chains, na_chains) =
            Self::flat_scene_chains(scene, annotations);
        let chains = SceneChainData {
            backbone_chains: &backbone_chains,
            na_chains: &na_chains,
        };
        gpu.upload_prepared(prepared, animating, &chains);
    }

    /// Flatten per-entity backbone / NA chains in assembly order. Only
    /// Cartoon-mode protein entities contribute to the flat backbone.
    fn flat_scene_chains(
        scene: &Scene,
        annotations: &EntityAnnotations,
    ) -> (
        Vec<crate::renderer::entity_topology::ProteinBackboneChain>,
        Vec<crate::renderer::entity_topology::NaBackboneChain>,
    ) {
        let mut backbone = Vec::new();
        let mut na = Vec::new();
        for (_, eid, state) in scene.visible_entities(annotations) {
            let Some(positions) = scene.positions.get(eid) else {
                continue;
            };
            if state.topology.is_protein()
                && state.drawing_mode == DrawingMode::Cartoon
            {
                backbone
                    .extend(state.topology.protein_backbone_chains(positions));
            } else if state.topology.is_nucleic_acid() {
                na.extend(
                    state.topology.na_backbone_chain_positions(positions),
                );
            }
        }
        (backbone, na)
    }

    /// Apply any pending scene data from the background `SceneProcessor`.
    ///
    /// Returns `true` when a prepared mesh was consumed (positions and
    /// sheet offsets just became final), so the caller re-resolves and
    /// re-uploads the bond capsules against the new positions.
    pub(crate) fn apply_pending_scene(
        scene: &mut Scene,
        annotations: &EntityAnnotations,
        options: &VisoOptions,
        gpu: &mut GpuPipeline,
        animation: &mut AnimationState,
    ) -> bool {
        let Some(prepared) = gpu.scene_processor.try_recv_rebuild() else {
            return false;
        };

        // Lift the per-entity sheet-flattening offsets and ribbon anchors
        // the mesh build just produced onto each EntityView, so
        // structural-bond endpoint resolution (disulfides, hbonds) re-anchors
        // sidechain atoms onto the same flattened sticks and attaches
        // backbone endpoints to the same drawn ribbon the mesh draws.
        Self::store_mesh_anchors(scene, &prepared);

        let entity_transitions = animation.take_pending_transitions();
        let animating = !entity_transitions.is_empty();

        if animating {
            Self::start_per_entity_animations(
                scene,
                animation,
                &entity_transitions,
            );
            Self::ensure_gpu_capacity_and_colors(scene, annotations, gpu);
            Self::submit_animation_frame(scene, options, gpu, animation);
        } else {
            Self::snap_from_prepared(scene, annotations, gpu);
        }

        Self::upload_prepared_to_gpu(
            scene,
            annotations,
            gpu,
            &prepared,
            animating,
        );
        true
    }

    /// Partition the prepared mesh's whole-assembly sheet-flattening
    /// offsets and ribbon anchors back onto each [`EntityView`], re-based to
    /// entity-local residue indices.
    ///
    /// `prepared.backbone.sheet_offsets` and `prepared.backbone.ribbon_anchors`
    /// are both global-residue-indexed (mesh concatenation rebases each
    /// entity's base-0 indices by its global residue base);
    /// `prepared.entity_residue_offsets` records each entity's base in
    /// ascending assembly-visible order. Both lists are ascending by residue
    /// index, so one linear pass per list buckets each item into its owning
    /// entity and subtracts the base. No geometry math runs here: these are
    /// the exact deltas and anchors the mesh produced.
    fn store_mesh_anchors(scene: &mut Scene, prepared: &PreparedRebuild) {
        Self::partition_anchors_onto_views(
            scene,
            &prepared.backbone.sheet_offsets,
            &prepared.backbone.ribbon_anchors,
            &prepared.entity_residue_offsets,
            &prepared.displayed_positions,
        );
    }

    /// Lift an animation frame's fresh sheet offsets + ribbon anchors onto
    /// each [`EntityView`] using the same partition the full-rebuild path
    /// uses. Routed through here (not the GPU layer) because this layer holds
    /// the [`Scene`] borrow `entity_residue_offsets` partitioning needs.
    pub(crate) fn store_animation_anchors(
        scene: &mut Scene,
        anchors: &crate::renderer::gpu_pipeline::AnimationAnchors,
    ) {
        Self::partition_anchors_onto_views(
            scene,
            &anchors.sheet_offsets,
            &anchors.ribbon_anchors,
            &anchors.entity_residue_offsets,
            &anchors.displayed_positions,
        );
    }

    /// Shared partition: bucket whole-assembly sheet offsets + ribbon anchors
    /// into their owning entities and rebase to entity-local residue indices.
    /// Both lists are ascending by global residue index and
    /// `entity_residue_offsets` records each entity's base in ascending
    /// assembly order, so one linear pass per list suffices. No geometry math
    /// runs here.
    fn partition_anchors_onto_views(
        scene: &mut Scene,
        sheet_offsets: &[crate::renderer::geometry::backbone::SheetOffset],
        ribbon_anchors: &[crate::renderer::geometry::backbone::RibbonAnchor],
        bases: &[(EntityId, u32)],
        displayed_positions: &[(EntityId, Vec<Vec3>)],
    ) {
        // Clear first so an entity whose strands/anchors disappeared this
        // rebuild ends up with empty slices rather than stale ones.
        for state in scene.entity_state.values_mut() {
            state.sheet_offsets.clear();
            state.ribbon_anchors.clear();
            state.displayed_positions.clear();
        }

        // Lift each entity's displayed-frame positions onto its view in
        // lockstep with the anchors below, so an overlay resolver pairs a
        // raw atom read with the ribbon/sheet transform from the same frame.
        for (eid, positions) in displayed_positions {
            if let Some(state) = scene.entity_state.get_mut(eid) {
                state.displayed_positions.clone_from(positions);
            }
        }

        for (i, &(eid, base)) in bases.iter().enumerate() {
            // This entity owns global residues `[base, next_base)`.
            let next_base = bases.get(i + 1).map_or(u32::MAX, |&(_, b)| b);
            let Some(state) = scene.entity_state.get_mut(&eid) else {
                continue;
            };
            state.sheet_offsets.extend(
                sheet_offsets
                    .iter()
                    .filter(|so| {
                        so.residue_idx >= base && so.residue_idx < next_base
                    })
                    .map(|so| {
                        crate::renderer::geometry::backbone::SheetOffset {
                            residue_idx: so.residue_idx - base,
                            offset: so.offset,
                        }
                    }),
            );
            state.ribbon_anchors.extend(
                ribbon_anchors
                    .iter()
                    .filter(|ra| {
                        ra.residue_idx >= base && ra.residue_idx < next_base
                    })
                    .map(|ra| {
                        crate::renderer::geometry::backbone::RibbonAnchor {
                            residue_idx: ra.residue_idx - base,
                            ..*ra
                        }
                    }),
            );
        }
    }

    /// Kick off per-entity animation runners using the current
    /// positions as `start` and each entity's reference positions as
    /// `target`.
    fn start_per_entity_animations(
        scene: &mut Scene,
        animation: &mut AnimationState,
        entity_transitions: &HashMap<u32, Transition>,
    ) {
        let targets: Vec<(EntityId, Vec<Vec3>)> = scene
            .current
            .entities()
            .iter()
            .map(|entity| (entity.id(), entity.positions().to_vec()))
            .collect();
        for (eid, target) in targets {
            let raw = eid.raw();
            let Some(transition) = entity_transitions.get(&raw) else {
                continue;
            };
            let current = scene
                .positions
                .get(eid)
                .map(<[Vec3]>::to_vec)
                .unwrap_or_default();
            if current.len() != target.len() {
                // Atom layout changed (e.g. mutation rebuilds
                // sidechains). Snap positions to target so the renderer
                // reflects the new layout immediately -- interpolation
                // is meaningless across mismatched shapes.
                scene.positions.set(eid, target.clone());
                // Mirror the snap into the displayed frame so overlays do
                // not read a stale (old-layout) snapshot for the frame
                // before the next prepared mesh refills it.
                if let Some(state) = scene.entity_state.get_mut(&eid) {
                    state.displayed_positions.clone_from(&target);
                }
                if !transition.allows_size_change {
                    continue;
                }
                // For size-change-aware transitions (collapse_ease_expand),
                // still install a runner so the phase timeline (which
                // controls sidechain visibility) plays through.
            }
            animation
                .animator
                .animate_entity(eid, current, target, transition);
        }
    }

    fn snap_from_prepared(
        scene: &mut Scene,
        annotations: &EntityAnnotations,
        gpu: &mut GpuPipeline,
    ) {
        // Snap mode: no animation is queued for this sync, so the
        // visual positions buffer must be overwritten with the new
        // assembly's atom positions. Without this, scene.positions
        // would stay frozen at whatever the previous sync produced,
        // and the mesh would never reflect a coord update from the
        // host (the bug surfaces as a stationary protein during
        // wiggle/shake when no transition is queued).
        for entity in scene.current.entities() {
            scene
                .positions
                .set(entity.id(), entity.positions().to_vec());
        }
        Self::ensure_gpu_capacity_and_colors(scene, annotations, gpu);
        let flat_colors = scene.flat_cartoon_colors(annotations);
        if !flat_colors.is_empty() {
            gpu.set_colors_immediate(&flat_colors);
        }
    }

    fn ensure_gpu_capacity_and_colors(
        scene: &Scene,
        annotations: &EntityAnnotations,
        gpu: &mut GpuPipeline,
    ) {
        let (backbone_chains, _na) =
            Self::flat_scene_chains(scene, annotations);
        let total_residues =
            crate::renderer::geometry::sheet_adjust::backbone_residue_count(
                &backbone_chains,
            );
        gpu.ensure_residue_capacity(total_residues);
        let flat_colors = scene.flat_cartoon_colors(annotations);
        if !flat_colors.is_empty() {
            gpu.set_target_colors(&flat_colors);
        }
    }

    /// Submit an animation frame to the background thread using the
    /// engine's current [`super::super::positions::EntityPositions`].
    pub(crate) fn submit_animation_frame(
        scene: &Scene,
        options: &VisoOptions,
        gpu: &GpuPipeline,
        animation: &AnimationState,
    ) {
        let include_sidechains = animation.animator.should_include_sidechains();
        gpu.submit_animation_frame(
            &scene.positions,
            &options.geometry,
            include_sidechains,
        );
    }

    /// Apply a trajectory frame's atom-index updates to
    /// [`super::super::positions::EntityPositions`].
    pub(crate) fn apply_trajectory_frame(
        scene: &mut Scene,
        frame: &TrajectoryFrame,
    ) {
        let Some(slot) = scene.positions.get_mut(frame.entity) else {
            return;
        };
        for (i, &idx) in frame.atom_indices.iter().enumerate() {
            let Some(pos) = frame.positions.get(i).copied() else {
                continue;
            };
            if let Some(target) = slot.get_mut(idx as usize) {
                *target = pos;
            }
        }
    }

    /// Concatenated SS across all Cartoon protein entities, in
    /// assembly order. Used by the `SelectSegment` command path.
    pub(crate) fn concatenated_cartoon_ss(
        scene: &Scene,
        annotations: &EntityAnnotations,
    ) -> Vec<SSType> {
        let mut ss = Vec::new();
        for (_, _, state) in scene.visible_entities(annotations) {
            if state.topology.is_protein()
                && state.drawing_mode == DrawingMode::Cartoon
            {
                let ss_slice = state
                    .ss_override
                    .as_deref()
                    .unwrap_or(&state.topology.ss_types);
                ss.extend_from_slice(ss_slice);
            }
        }
        ss
    }

    /// Recompute per-chain backbone colors and upload them immediately.
    /// Used by display-option changes that affect backbone tint but
    /// don't invalidate mesh geometry.
    pub(crate) fn recompute_backbone_colors(
        scene: &mut Scene,
        annotations: &EntityAnnotations,
        options: &VisoOptions,
        gpu: &mut GpuPipeline,
    ) {
        let assembly = Arc::clone(&scene.current);
        let b_range = assembly_b_range(scene);
        for (entity_index, entity) in assembly.entities().iter().enumerate() {
            let eid = entity.id();
            let Some(positions) = scene.positions.get(eid) else {
                continue;
            };
            let positions = positions.to_vec();
            let Some(state) = scene.entity_state.get_mut(&eid) else {
                continue;
            };
            if !state.topology.is_protein() {
                continue;
            }
            let display = annotations.appearance.get(&eid).map_or_else(
                || options.display.clone(),
                |ovr| ovr.to_display_options(&options.display),
            );
            let ss_types: Vec<SSType> = state
                .ss_override
                .clone()
                .unwrap_or_else(|| state.topology.ss_types.clone());
            let backbone_chains =
                state.topology.protein_backbone_chains(&positions);
            let b_factors = matches!(
                display.backbone_color_scheme(),
                crate::options::ColorScheme::BFactor
            )
            .then(|| residue_max_backbone_b(&state.topology));
            state.per_residue_colors = per_entity_colors(
                entity_index,
                &backbone_chains,
                &ss_types,
                annotations.scores.get(&eid).map(Vec::as_slice),
                b_factors.as_deref(),
                b_range,
                &display,
            );
        }
        let flat = scene.flat_cartoon_colors(annotations);
        if !flat.is_empty() {
            gpu.set_colors_immediate(&flat);
        }
    }
}

// -- Helpers --

/// Per-residue B-factor: the max over the residue's backbone atoms. The
/// backbone atom indices come from `protein_backbone_layout`, the single
/// owner of that derivation; segments are walked in order so the result is
/// residue-indexed in ascending residue order. An index past
/// `atom_b_factors` is skipped; a residue whose four indices are all out of
/// range contributes `0.0`.
fn residue_max_backbone_b(
    topology: &crate::renderer::entity_topology::EntityTopology,
) -> Vec<f32> {
    let b = &topology.atom_b_factors;
    let mut out = Vec::with_capacity(topology.residue_atom_ranges.len());
    for seg in &topology.protein_backbone_layout {
        for i in 0..seg.ca.len() {
            let max = [seg.n[i], seg.ca[i], seg.c[i], seg.o[i]]
                .into_iter()
                .filter_map(|idx| b.get(idx).copied())
                .reduce(f32::max)
                .unwrap_or(0.0);
            out.push(max);
        }
    }
    out
}

/// Assembly-global (min, max) B-factor across every entity's atoms.
/// Returns `(0.0, 0.0)` when no atom carries one.
fn assembly_b_range(scene: &Scene) -> (f32, f32) {
    let mut lo = f32::INFINITY;
    let mut hi = f32::NEG_INFINITY;
    for state in scene.entity_state.values() {
        for &b in &state.topology.atom_b_factors {
            if b.is_nan() {
                continue;
            }
            lo = lo.min(b);
            hi = hi.max(b);
        }
    }
    if lo.is_finite() && hi.is_finite() {
        (lo, hi)
    } else {
        (0.0, 0.0)
    }
}

fn per_entity_colors(
    entity_index: usize,
    backbone_chains: &[crate::renderer::entity_topology::ProteinBackboneChain],
    ss_types: &[SSType],
    scores: Option<&[f64]>,
    b_factors: Option<&[f32]>,
    b_range: (f32, f32),
    display: &DisplayOptions,
) -> Option<Vec<[f32; 3]>> {
    if backbone_chains.is_empty() {
        return None;
    }
    let scores_slice = [scores];
    let inputs = crate::options::score_color::SchemeInputs {
        scores: &scores_slice,
        b_factors,
        b_range,
    };
    let colors = crate::options::score_color::compute_per_residue_colors_styled(
        backbone_chains,
        ss_types,
        &inputs,
        &display.backbone_color_scheme(),
        &display.backbone_palette(),
        entity_index,
        display.overrides.provisional.unwrap_or(false),
    );
    if colors.is_empty() {
        None
    } else {
        Some(colors)
    }
}

#[cfg(test)]
mod tests {
    use molex::MoleculeType;

    use super::residue_max_backbone_b;
    use crate::renderer::entity_topology::{
        EntityTopology, ProteinBackboneIndices, SidechainLayout,
    };

    fn protein_topology(
        segments: Vec<ProteinBackboneIndices>,
        atom_b_factors: Vec<f32>,
    ) -> EntityTopology {
        let residue_count: usize = segments.iter().map(|s| s.ca.len()).sum();
        EntityTopology {
            molecule_type: MoleculeType::Protein,
            protein_backbone_layout: segments,
            na_backbone_chain_layout: Vec::new(),
            sidechain_layout: SidechainLayout::empty(),
            ring_topology: Vec::new(),
            na_residue_base_colors: Vec::new(),
            na_guide_atom_indices: Vec::new(),
            ss_types: Vec::new(),
            atom_elements: Vec::new(),
            atom_b_factors,
            atom_residue_index: Vec::new(),
            residue_names: Vec::new(),
            residue_atom_ranges: vec![0..0; residue_count],
            bonds: Vec::new(),
        }
    }

    #[test]
    fn max_over_backbone_atoms_in_segment_order() {
        // Two segments, four atoms per residue: B rises with atom index so
        // the O atom (highest index) is the per-residue max.
        let b: Vec<f32> = (0..12).map(|i| i as f32).collect();
        let seg0 = ProteinBackboneIndices {
            n: vec![0, 4],
            ca: vec![1, 5],
            c: vec![2, 6],
            o: vec![3, 7],
        };
        let seg1 = ProteinBackboneIndices {
            n: vec![8],
            ca: vec![9],
            c: vec![10],
            o: vec![11],
        };
        let topo = protein_topology(vec![seg0, seg1], b);
        assert_eq!(residue_max_backbone_b(&topo), vec![3.0, 7.0, 11.0]);
    }

    #[test]
    fn out_of_range_indices_yield_zero_without_panic() {
        // One residue whose four backbone indices all exceed
        // `atom_b_factors.len()`: contributes 0.0 rather than panicking.
        let seg = ProteinBackboneIndices {
            n: vec![50],
            ca: vec![51],
            c: vec![52],
            o: vec![53],
        };
        let topo = protein_topology(vec![seg], vec![1.0, 2.0, 3.0, 4.0]);
        assert_eq!(residue_max_backbone_b(&topo), vec![0.0]);
    }

    #[test]
    fn partially_out_of_range_takes_max_of_in_range() {
        // n/ca in range, c/o past the end: max over the resolvable pair.
        let seg = ProteinBackboneIndices {
            n: vec![0],
            ca: vec![1],
            c: vec![99],
            o: vec![100],
        };
        let topo = protein_topology(vec![seg], vec![5.0, 8.0]);
        assert_eq!(residue_max_backbone_b(&topo), vec![8.0]);
    }
}
