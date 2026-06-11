//! Background scene processor for non-blocking geometry generation.
//!
//! Moves all CPU-heavy mesh/instance generation off the main thread.
//! The main thread only does GPU uploads (<1ms) and render passes.
//!
//! Supports **per-entity mesh caching**: when an entity's `mesh_version`
//! hasn't changed between frames, its cached mesh is reused instead of
//! being regenerated. Global settings changes (view mode, display,
//! colors) clear the entire cache.

use std::sync::mpsc;

use molex::entity::molecule::id::EntityId;
use rustc_hash::{FxHashMap, FxHashSet};

use super::prepared::{
    AnimationFrameBody, CachedEntityMesh, FullRebuildBody, FullRebuildEntity,
    PreparedRebuild, SceneRequest,
};
use crate::engine::positions::EntityPositions;
use crate::options::{
    ChainLod, ColorOptions, DisplayOptions, DrawingMode, GeometryOptions,
};

// ---------------------------------------------------------------------------
// Platform-abstracted background thread spawn
// ---------------------------------------------------------------------------

/// Handle to a background worker. On native this is a joinable OS thread;
/// on WASM it is a no-op because the worker runs on a rayon pool thread
/// (backed by web workers via `wasm-bindgen-rayon` + `SharedArrayBuffer`)
/// and exits when the channel disconnects.
#[cfg(not(target_arch = "wasm32"))]
type WorkerHandle = Option<std::thread::JoinHandle<()>>;
#[cfg(target_arch = "wasm32")]
type WorkerHandle = ();

/// Spawn a long-lived closure on a background thread.
///
/// - **Native:** dedicated OS thread via `std::thread::Builder`.
/// - **WASM:** `rayon::spawn` onto the `wasm-bindgen-rayon` pool.
fn spawn_background(
    f: impl FnOnce() + Send + 'static,
) -> Result<WorkerHandle, std::io::Error> {
    #[cfg(not(target_arch = "wasm32"))]
    {
        std::thread::Builder::new()
            .name("scene-processor".into())
            .spawn(f)
            .map(Some)
    }
    #[cfg(target_arch = "wasm32")]
    {
        rayon::spawn(f);
        Ok(())
    }
}

/// Join a background worker, blocking until it finishes.
fn join_background(handle: &mut WorkerHandle) {
    #[cfg(not(target_arch = "wasm32"))]
    if let Some(h) = handle.take() {
        let _ = h.join();
    }
    #[cfg(target_arch = "wasm32")]
    let _ = handle;
}

// ---------------------------------------------------------------------------
// SceneProcessor
// ---------------------------------------------------------------------------

/// Background thread that generates CPU-side geometry from scene data.
pub(crate) struct SceneProcessor {
    request_tx: mpsc::Sender<SceneRequest>,
    rebuild_result: triple_buffer::Output<Option<PreparedRebuild>>,
    anim_result: triple_buffer::Output<Option<PreparedRebuild>>,
    worker: WorkerHandle,
    /// Monotonically increasing rebuild generation counter. Bumped
    /// each time a `FullRebuild` is submitted. Animation frame results
    /// with a lower generation are discarded as stale.
    rebuild_generation: u64,
    /// Topology generation. Advances only when the visible entity-id set
    /// changes (entity added or removed), NOT on every submit. A rebuild
    /// built for the current topology generation is applied even if
    /// same-topology submits have since bumped `rebuild_generation`.
    topology_generation: u64,
    /// Visible entity-id set of the most recent submit, used to detect
    /// topology changes that advance `topology_generation`.
    last_entity_ids: FxHashSet<EntityId>,
    /// True between `FullRebuild` submission and `PreparedRebuild`
    /// consumption. While set, the backbone renderer's cached chains are
    /// stale — LOD must not read them.
    rebuild_pending: bool,
}

impl SceneProcessor {
    /// Spawn the background scene processing thread.
    ///
    /// # Errors
    ///
    /// Returns [`std::io::Error`] if the background thread fails to spawn.
    pub(crate) fn new() -> Result<Self, std::io::Error> {
        let (request_tx, request_rx) = mpsc::channel::<SceneRequest>();
        let (rebuild_input, rebuild_output) =
            triple_buffer::triple_buffer(&None);
        let (anim_input, anim_output) = triple_buffer::triple_buffer(&None);

        let worker = spawn_background(move || {
            Self::thread_loop(request_rx, rebuild_input, anim_input);
        })?;

        Ok(Self {
            request_tx,
            rebuild_result: rebuild_output,
            anim_result: anim_output,
            worker,
            rebuild_generation: 0,
            topology_generation: 0,
            last_entity_ids: FxHashSet::default(),
            rebuild_pending: false,
        })
    }

    /// Increment and return the next rebuild generation counter.
    ///
    /// Also sets `rebuild_pending`, which prevents LOD from reading the
    /// backbone renderer's stale cached chains until the corresponding
    /// `PreparedRebuild` is consumed.
    pub(crate) fn next_generation(&mut self) -> u64 {
        self.rebuild_generation += 1;
        self.rebuild_pending = true;
        self.rebuild_generation
    }

    /// Current rebuild generation counter.
    pub(crate) fn generation(&self) -> u64 {
        self.rebuild_generation
    }

    /// Current topology generation counter. Advances only when the
    /// visible entity-id set changes; used to stamp animation frames so a
    /// same-topology coordinate stream does not discard them.
    pub(crate) fn topology_generation(&self) -> u64 {
        self.topology_generation
    }

    /// Return the topology generation for a submit covering `entity_ids`,
    /// advancing it first if the set differs from the previous submit's.
    ///
    /// A coordinate-only resubmit (same entity-id set, new positions)
    /// keeps the topology generation steady; an entity add/remove bumps
    /// it. The returned value is stamped onto the submitted body so the
    /// consumer can tell whether a built rebuild still matches the live
    /// scene.
    pub(crate) fn advance_topology_generation(
        &mut self,
        entity_ids: FxHashSet<EntityId>,
    ) -> u64 {
        if entity_ids != self.last_entity_ids {
            self.topology_generation += 1;
            self.last_entity_ids = entity_ids;
        }
        self.topology_generation
    }

    /// Submit a scene request (non-blocking send).
    pub(crate) fn submit(&self, request: SceneRequest) {
        let _ = self.request_tx.send(request);
    }

    /// Non-blocking check for a completed full rebuild.
    ///
    /// Discards a result only when its topology generation is behind the
    /// current one, i.e. the visible entity-id set changed since it was
    /// built (a stale structure after `replace_scene()`). A rebuild built
    /// for the current topology is applied even when newer same-topology
    /// submits have since bumped the per-submit `rebuild_generation`;
    /// those resubmits supersede it on their own arrival.
    ///
    /// Clears `rebuild_pending` on successful consumption so that LOD
    /// submission (gated by [`Self::is_rebuild_pending`]) resumes with
    /// the now-correct backbone renderer cache.
    pub(crate) fn try_recv_rebuild(&mut self) -> Option<PreparedRebuild> {
        let _ = self.rebuild_result.update();
        let prepared = self.rebuild_result.output_buffer_mut().take()?;
        if prepared.topology_generation < self.topology_generation {
            log::debug!(
                "try_recv_rebuild: DISCARDING stale rebuild (topology gen {} \
                 < current {})",
                prepared.topology_generation,
                self.topology_generation,
            );
            return None;
        }
        log::debug!(
            "try_recv_rebuild: ACCEPTED rebuild topology_gen={} (current={})",
            prepared.topology_generation,
            self.topology_generation,
        );
        self.rebuild_pending = false;
        Some(prepared)
    }

    /// Whether a `FullRebuild` has been submitted but its
    /// `PreparedRebuild` has not yet been consumed.
    ///
    /// While true, the backbone renderer's cached chains are stale —
    /// callers that read the cache to build `AnimationFrame` requests
    /// (notably LOD) must skip submission.
    pub(crate) fn is_rebuild_pending(&self) -> bool {
        self.rebuild_pending
    }

    /// Non-blocking check for completed animation frame.
    ///
    /// Discards a frame only when its topology generation is behind the
    /// current one, i.e. the visible entity-id set changed since it was
    /// built. A frame built for the current topology survives even when
    /// newer same-topology coordinate rebuilds have bumped the per-submit
    /// `rebuild_generation`; the latest-wins triple buffer already heads
    /// the newest frame toward the newest target.
    pub(crate) fn try_recv_animation(&mut self) -> Option<PreparedRebuild> {
        let _ = self.anim_result.update();
        let prepared = self.anim_result.output_buffer_mut().take()?;
        if prepared.topology_generation < self.topology_generation {
            log::debug!(
                "Discarding stale animation frame (topology gen {} < current \
                 {})",
                prepared.topology_generation,
                self.topology_generation,
            );
            return None;
        }
        Some(prepared)
    }

    /// Shut down the background thread and wait for it to finish.
    pub(crate) fn shutdown(&mut self) {
        let _ = self.request_tx.send(SceneRequest::Shutdown);
        join_background(&mut self.worker);
    }

    /// Background thread main loop with per-entity mesh caching.
    #[allow(clippy::needless_pass_by_value)]
    fn thread_loop(
        request_rx: mpsc::Receiver<SceneRequest>,
        mut rebuild_input: triple_buffer::Input<Option<PreparedRebuild>>,
        mut anim_input: triple_buffer::Input<Option<PreparedRebuild>>,
    ) {
        let mut cache = MeshCache::new();
        // Topology generation of the last FullRebuild processed on this
        // thread. Animation frames stamped with an older topology
        // generation are stale (the entity-id set changed under them).
        let mut last_topology_generation: u64 = 0;

        while let Ok(request) = request_rx.recv() {
            let latest = drain_latest(request, &request_rx);

            match latest {
                SceneRequest::Shutdown => break,
                SceneRequest::FullRebuild(body) => {
                    let FullRebuildBody {
                        entities,
                        display,
                        colors,
                        geometry,
                        entity_options,
                        generation,
                        topology_generation,
                    } = *body;
                    last_topology_generation = topology_generation;
                    cache.cache_stable_data(&entities, &entity_options);
                    let entity_meshes = cache.update(
                        &entities,
                        &display,
                        &colors,
                        &geometry,
                        &entity_options,
                    );
                    let mut prepared =
                        super::mesh_concat::concatenate_meshes(&entity_meshes);
                    prepared.generation = generation;
                    prepared.topology_generation = topology_generation;
                    rebuild_input.write(Some(prepared));
                }
                SceneRequest::AnimationFrame(body) => {
                    let AnimationFrameBody {
                        positions,
                        geometry,
                        per_chain_lod,
                        include_sidechains,
                        generation,
                        topology_generation,
                    } = *body;
                    if topology_generation < last_topology_generation {
                        continue;
                    }
                    let mut prepared = cache.regenerate_for_animation(
                        &positions,
                        &geometry,
                        per_chain_lod.as_deref(),
                        include_sidechains,
                    );
                    prepared.generation = generation;
                    prepared.topology_generation = topology_generation;
                    anim_input.write(Some(prepared));
                }
            }
        }
    }
}

impl Drop for SceneProcessor {
    fn drop(&mut self) {
        self.shutdown();
    }
}

/// Per-entity mesh cache with settings-based invalidation.
///
/// Caches per-entity geometry keyed on [`EntityId`], plus the last
/// rebuild's per-entity inputs ([`MeshCache::last_entities`]) so
/// `AnimationFrame` requests can regenerate every entity's mesh -- through
/// every drawing mode -- from interpolated positions alone.
struct MeshCache {
    meshes: FxHashMap<EntityId, (u64, CachedEntityMesh)>,
    last_display: Option<DisplayOptions>,
    last_colors: Option<ColorOptions>,
    last_geometry: Option<GeometryOptions>,
    /// Per-entity rebuild inputs from the last `FullRebuild`, retained so
    /// an `AnimationFrame` can regenerate every entity's mesh -- through
    /// every drawing mode -- from interpolated positions, using the same
    /// per-entity derivation the rebuild path uses. Positions on these
    /// snapshots are stale (the live ones arrive per frame); everything
    /// else (topology, drawing mode, colors, SS) is reused as-is.
    last_entities: Vec<FullRebuildEntity>,
    /// Per-entity display+geometry overrides captured at the last
    /// `FullRebuild`, parallel to [`Self::last_entities`].
    last_entity_options: FxHashMap<u32, (DisplayOptions, GeometryOptions)>,
}

impl MeshCache {
    fn new() -> Self {
        Self {
            meshes: FxHashMap::default(),
            last_display: None,
            last_colors: None,
            last_geometry: None,
            last_entities: Vec::new(),
            last_entity_options: FxHashMap::default(),
        }
    }

    /// Retain the per-entity rebuild inputs so subsequent
    /// `AnimationFrame`s can regenerate meshes from interpolated positions
    /// through the same per-entity derivation as the rebuild. No drawing
    /// mode is privileged here -- the retained snapshot carries every
    /// entity's topology, drawing mode, colors, and SS as-is.
    fn cache_stable_data(
        &mut self,
        entities: &[FullRebuildEntity],
        entity_options: &FxHashMap<u32, (DisplayOptions, GeometryOptions)>,
    ) {
        self.last_entities = entities.to_vec();
        self.last_entity_options.clone_from(entity_options);
    }

    /// Regenerate every retained entity's mesh from interpolated positions
    /// and concatenate into a `PreparedRebuild`.
    ///
    /// An animation frame is just lerped molecular data, so per-drawing-mode
    /// geometry is derived the same way the `FullRebuild` derives it -- via
    /// [`super::mesh_gen::generate_entity_mesh`] per entity. Cartoon,
    /// ball-and-stick, and nucleic-acid geometry therefore all animate.
    ///
    /// `per_chain_lod`, when `Some`, is the global per-chain LOD tier list
    /// from the camera-distance remesh; it is sliced per entity by each
    /// entity's cartoon chain count, in the retained entity order (the same
    /// order [`super::mesh_concat::concatenate_meshes`] stitched the chains
    /// in). Position-animation frames pass `None`.
    fn regenerate_for_animation(
        &self,
        positions: &EntityPositions,
        geometry: &GeometryOptions,
        per_chain_lod: Option<&[ChainLod]>,
        include_sidechains: bool,
    ) -> PreparedRebuild {
        let (Some(display), Some(colors)) =
            (self.last_display.as_ref(), self.last_colors.as_ref())
        else {
            // No rebuild has populated the cache yet; nothing to animate.
            return super::mesh_concat::concatenate_meshes(&[]);
        };

        let total_residues: usize = self
            .last_entities
            .iter()
            .map(|e| {
                let protein = e
                    .topology
                    .protein_backbone_layout
                    .iter()
                    .map(|s| s.ca.len())
                    .sum::<usize>();
                let na = e
                    .topology
                    .na_backbone_chain_layout
                    .iter()
                    .map(Vec::len)
                    .sum::<usize>();
                protein + na
            })
            .sum();
        let geometry = geometry.clamped_for_residues(total_residues);

        let mut meshes: Vec<CachedEntityMesh> =
            Vec::with_capacity(self.last_entities.len());
        let mut lod_offset = 0usize;
        for e in &self.last_entities {
            // Only Cartoon entities contribute backbone chains (and thus
            // per-chain LOD slots), in entity order; advance the offset by
            // this entity's chain count so the slice stays aligned with the
            // concatenated chain stream the remesh built `per_chain_lod` from.
            let chain_count = if e.drawing_mode == DrawingMode::Cartoon {
                e.topology.protein_backbone_layout.len()
                    + e.topology.na_backbone_chain_layout.len()
            } else {
                0
            };
            let entity_lod = per_chain_lod
                .and_then(|all| all.get(lod_offset..lod_offset + chain_count));
            lod_offset += chain_count;

            let (e_display, e_geometry) =
                self.last_entity_options.get(&e.id.raw()).map_or_else(
                    || (display, geometry.clone()),
                    |(d, g)| (d, g.clamped_for_residues(total_residues)),
                );

            let mut entity = e.clone();
            if let Some(p) = positions.get(e.id) {
                entity.positions = p.to_vec();
            }
            let mut mesh = super::mesh_gen::generate_entity_mesh(
                &entity,
                e_display,
                colors,
                &e_geometry,
                entity_lod,
            );
            if !include_sidechains {
                mesh.sidechain_instances.clear();
                mesh.sidechain_instance_count = 0;
            }
            meshes.push(mesh);
        }
        let refs: Vec<&CachedEntityMesh> = meshes.iter().collect();
        let mut prepared = super::mesh_concat::concatenate_meshes(&refs);
        // Backbone-only frames leave the previously uploaded sidechains
        // untouched on apply; their positions are unchanged by level-of-detail.
        prepared.sidechains_omitted = !include_sidechains;
        prepared
    }

    /// Update cached meshes and return entity-ordered references for
    /// concatenation.
    fn update(
        &mut self,
        entities: &[FullRebuildEntity],
        display: &DisplayOptions,
        colors: &ColorOptions,
        geometry: &GeometryOptions,
        entity_options: &FxHashMap<u32, (DisplayOptions, GeometryOptions)>,
    ) -> Vec<&CachedEntityMesh> {
        // Clamp geometry detail so the concatenated vertex buffer stays
        // under the wgpu 256 MB max.
        let total_residues: usize = entities
            .iter()
            .map(|e| {
                let protein = e
                    .topology
                    .protein_backbone_layout
                    .iter()
                    .map(|s| s.ca.len())
                    .sum::<usize>();
                let na = e
                    .topology
                    .na_backbone_chain_layout
                    .iter()
                    .map(Vec::len)
                    .sum::<usize>();
                protein + na
            })
            .sum();
        let geometry = geometry.clamped_for_residues(total_residues);

        // Any settings change (geometry, display, or colors) clears the
        // entire cache because backbone colors are baked into vertex data.
        let settings_changed = self.last_geometry.as_ref() != Some(&geometry)
            || self.last_display.as_ref() != Some(display)
            || self.last_colors.as_ref() != Some(colors);

        if settings_changed {
            self.meshes.clear();
        }
        self.last_display = Some(display.clone());
        self.last_colors = Some(colors.clone());
        self.last_geometry = Some(geometry.clone());

        // Generate or reuse per-entity meshes.
        for e in entities {
            let entity_u32 = *e.id;
            let needs_regen = self
                .meshes
                .get(&e.id)
                .is_none_or(|(v, _)| *v != e.mesh_version);
            if needs_regen {
                let (e_display, e_geometry) =
                    if let Some((d, g)) = entity_options.get(&entity_u32) {
                        (d, g.clamped_for_residues(total_residues))
                    } else {
                        (display, geometry.clone())
                    };
                let mesh = super::mesh_gen::generate_entity_mesh(
                    e,
                    e_display,
                    colors,
                    &e_geometry,
                    None,
                );
                drop(self.meshes.insert(e.id, (e.mesh_version, mesh)));
            }
        }

        // Evict removed entities.
        let active_ids: FxHashSet<EntityId> =
            entities.iter().map(|e| e.id).collect();
        self.meshes.retain(|id, _| active_ids.contains(id));

        // Collect references in entity order.
        entities
            .iter()
            .filter_map(|e| self.meshes.get(&e.id).map(|(_, mesh)| mesh))
            .collect()
    }
}

/// Drain queued requests, keeping only the latest.
///
/// Special case: a queued `AnimationFrame` does NOT replace a pending
/// `FullRebuild` — the rebuild must still run so the mesh cache is
/// populated before animation frames can reference it.
fn drain_latest(
    initial: SceneRequest,
    rx: &mpsc::Receiver<SceneRequest>,
) -> SceneRequest {
    let mut latest = initial;
    while let Ok(newer) = rx.try_recv() {
        match (&latest, &newer) {
            (SceneRequest::FullRebuild(_), SceneRequest::AnimationFrame(_)) => {
            }
            _ => {
                latest = newer;
            }
        }
    }
    latest
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A cache with no retained entities but populated scene state, so
    /// `regenerate_for_animation` runs to its main return without needing
    /// real topology or a GPU device. Backbone-only frames must mark the
    /// result so the apply side leaves the retained sidechains untouched.
    fn cache_with_empty_scene() -> MeshCache {
        let mut cache = MeshCache::new();
        cache.last_display = Some(DisplayOptions::default());
        cache.last_colors = Some(ColorOptions::default());
        cache.last_geometry = Some(GeometryOptions::default());
        cache
    }

    #[test]
    fn animation_frame_marks_omitted_when_sidechains_excluded() {
        let cache = cache_with_empty_scene();
        let prepared = cache.regenerate_for_animation(
            &EntityPositions::new(),
            &GeometryOptions::default(),
            None,
            false,
        );
        assert!(prepared.sidechains_omitted);
    }

    #[test]
    fn animation_frame_unmarked_when_sidechains_included() {
        let cache = cache_with_empty_scene();
        let prepared = cache.regenerate_for_animation(
            &EntityPositions::new(),
            &GeometryOptions::default(),
            None,
            true,
        );
        assert!(!prepared.sidechains_omitted);
    }
}
