//! Constraint resolution: resolve band/pull specs to world-space
//! positions by walking
//! [`crate::renderer::entity_topology::EntityTopology`] +
//! [`super::positions::EntityPositions`].
//!
//! Each frame's resolution pass builds a [`ConstraintContext`] once
//! (O(visible cartoon proteins)) and then resolves every band + the
//! pull against it. Atom lookups inside the context are O(log n) to
//! find the owning entity (binary search on the cartoon-residue range
//! table) plus O(1) for the sidechain name-to-atom-index lookup
//! ([`SidechainLayout::atom_index`](crate::renderer::entity_topology::SidechainLayout::atom_index)).

use glam::{UVec2, Vec2, Vec3};
use molex::entity::molecule::id::EntityId;
use rustc_hash::FxHashMap;

use super::annotations::EntityAnnotations;
use super::command::{
    AtomRef, BandInfo, BandTarget, ClashEndpoint, ClashInfo,
    ExposedHydrophobicInfo, PullInfo, ResolvedBand, ResolvedClash,
    ResolvedExposedHydro, ResolvedPull,
};
use super::entity_view::{EntityView, RibbonBackbone};
use super::scene::Scene;
use super::scene_state::rendered_atom_position;
use super::{ConstraintSpecs, VisoEngine};
use crate::camera::controller::CameraController;
use crate::options::{DrawingMode, VisoOptions};
use crate::renderer::GpuPipeline;

/// Pre-computed per-frame cache for constraint resolution.
///
/// Built once at the top of [`VisoEngine::resolve_and_render_constraints`]
/// and reused for every band + the pull resolution in that frame.
/// Without this cache, each [`AtomRef`] resolution did a linear walk
/// over `engine.scene.current.entities()` — O(bands × entities × log
/// residues) per frame.
pub(super) struct ConstraintContext<'a> {
    scene: &'a Scene,
    /// Cartoon-mode protein entities in assembly order, with their
    /// flat residue ranges. `AtomRef.residue` is a flat index across
    /// these entities (matching
    /// [`VisoEngine::concatenated_cartoon_ss`]); binary-search this
    /// table to locate the owning entity.
    cartoon_ranges: Vec<CartoonRange>,
}

/// One cartoon-mode protein entity's flat residue range in assembly
/// order.
struct CartoonRange {
    /// Flat residue index where this entity's residues start.
    start: u32,
    /// Flat residue index where this entity's residues end (exclusive).
    end: u32,
    /// Owning entity id.
    entity: EntityId,
}

impl<'a> ConstraintContext<'a> {
    pub(super) fn new(
        scene: &'a Scene,
        annotations: &'a EntityAnnotations,
    ) -> Self {
        let mut cartoon_ranges = Vec::new();
        let mut cursor: u32 = 0;
        for (_, eid, state) in scene.visible_entities(annotations) {
            if state.topology.is_protein()
                && state.drawing_mode == DrawingMode::Cartoon
            {
                let count = state.topology.residue_atom_ranges.len() as u32;
                cartoon_ranges.push(CartoonRange {
                    start: cursor,
                    end: cursor + count,
                    entity: eid,
                });
                cursor += count;
            }
        }
        Self {
            scene,
            cartoon_ranges,
        }
    }

    /// Resolve an [`AtomRef`] to world-space. Binary-searches the
    /// cartoon-residue range table for the owning entity, then looks
    /// up the atom by name (O(1) via
    /// [`SidechainLayout::atom_index`](crate::renderer::entity_topology::SidechainLayout::atom_index)
    /// for sidechain atoms, O(1) range-indexed for backbone N/CA/C).
    fn resolve_atom_ref(&self, atom: &AtomRef) -> Option<Vec3> {
        let range = self
            .cartoon_ranges
            .binary_search_by(|r| {
                use std::cmp::Ordering;
                if atom.residue < r.start {
                    Ordering::Greater
                } else if atom.residue >= r.end {
                    Ordering::Less
                } else {
                    Ordering::Equal
                }
            })
            .ok()
            .map(|i| &self.cartoon_ranges[i])?;
        let state = self.scene.entity_state.get(&range.entity)?;
        let positions = displayed_positions(state, self.scene, range.entity)?;
        let local_residue = atom.residue - range.start;
        resolve_atom_in_entity(state, positions, local_residue, &atom.atom_name)
    }
}

/// The positions an overlay resolver should read for `entity`: the
/// displayed-frame snapshot lifted off the last applied mesh, so a raw atom
/// read pairs with the ribbon/sheet transform from the same frame. Falls
/// back to the live positions before the first mesh has landed (the snapshot
/// is empty until then), where no transform is applied either.
pub(super) fn displayed_positions<'a>(
    state: &'a EntityView,
    scene: &'a Scene,
    entity: EntityId,
) -> Option<&'a [Vec3]> {
    if state.displayed_positions.is_empty() {
        scene.positions.get(entity)
    } else {
        Some(&state.displayed_positions)
    }
}

/// Resolve a single band spec to world-space endpoint positions.
fn resolve_band(
    ctx: &ConstraintContext<'_>,
    band: &BandInfo,
) -> Option<ResolvedBand> {
    let endpoint_a = ctx.resolve_atom_ref(&band.anchor_a)?;
    let endpoint_b = match &band.anchor_b {
        BandTarget::Atom(atom) => ctx.resolve_atom_ref(atom)?,
        BandTarget::Position(pos) => *pos,
    };
    let is_space_pull = matches!(band.anchor_b, BandTarget::Position(_));

    Some(ResolvedBand {
        endpoint_a,
        endpoint_b,
        is_disabled: band.is_disabled,
        strength: band.strength,
        target_length: band.target_length,
        residue_idx: band.anchor_a.residue,
        is_space_pull,
        band_type: band.band_type,
        from_script: band.from_script,
    })
}

/// Resolve a single clash spec to world-space endpoint positions.
///
/// Each endpoint names its owning entity and an entity-local residue, so
/// resolution is direct: look the entity up in the Scene and resolve the
/// atom in place, with no flat-index indirection. Each endpoint then routes
/// through [`resolve_hbond_endpoint`] → [`rendered_atom_position`] (the same
/// wrapper the hbond/disulfide paths use) so a clash on a Cartoon-mode
/// residue lands on the DRAWN atom: a backbone endpoint projects onto the
/// ribbon spline, a sidechain endpoint picks up the sheet-flattening offset.
/// The clash is skipped (returns `None`) if either endpoint's entity is
/// absent or its atom fails to resolve, exactly as bands skip.
fn resolve_clash(
    ctx: &ConstraintContext<'_>,
    ribbons: &FxHashMap<EntityId, RibbonBackbone<'_>>,
    clash: &ClashInfo,
) -> Option<ResolvedClash> {
    let endpoint_a = resolve_hbond_endpoint(ctx.scene, ribbons, &clash.a)?;
    let endpoint_b = resolve_hbond_endpoint(ctx.scene, ribbons, &clash.b)?;
    Some(ResolvedClash {
        endpoint_a,
        endpoint_b,
        severity: clash.severity,
        seed: clash_seed(&clash.a, &clash.b),
    })
}

/// Hash a clash's atom pair into a stable per-clash seed in `[0, 1)`.
///
/// Deterministic per atom-pair so the procedural lightning bolt's jag and
/// flicker are consistent across frames for a given clash but decorrelated
/// between distinct clashes. A plain FNV-1a walk over the endpoint fields
/// keeps it independent of world-space positions (which animate).
fn clash_seed(a: &ClashEndpoint, b: &ClashEndpoint) -> f32 {
    let mut h: u32 = 0x811c_9dc5;
    let mut mix = |bytes: &[u8]| {
        for &byte in bytes {
            h ^= u32::from(byte);
            h = h.wrapping_mul(0x0100_0193);
        }
    };
    mix(&a.entity.raw().to_le_bytes());
    mix(&a.residue.to_le_bytes());
    mix(a.atom_name.as_bytes());
    mix(&b.entity.raw().to_le_bytes());
    mix(&b.residue.to_le_bytes());
    mix(b.atom_name.as_bytes());
    // Map the 32-bit hash into [0, 1).
    (h as f32) / (u32::MAX as f32)
}

/// Resolve one per-entity clash endpoint to world-space. Returns `None`
/// if the entity is not present in the Scene or the atom does not resolve.
fn resolve_clash_endpoint(scene: &Scene, ep: &ClashEndpoint) -> Option<Vec3> {
    let state = scene.entity_state.get(&ep.entity)?;
    let positions = displayed_positions(state, scene, ep.entity)?;
    resolve_atom_in_entity(state, positions, ep.residue, &ep.atom_name)
}

/// Resolve one hbond endpoint to its rendered world-space position via
/// the shared [`rendered_atom_position`] resolver.
///
/// The resolver applies both Cartoon-mode render transforms: backbone N/O/C
/// project onto the ribbon spline, and a sidechain endpoint on a flattened
/// beta-strand picks up the sheet offset (so a sidechain↔backbone hbond's
/// sidechain end lands on the drawn flattened stick, not the raw atom).
/// `ep.residue` is the entity-local residue index the resolver expects.
fn resolve_hbond_endpoint(
    scene: &Scene,
    ribbons: &FxHashMap<EntityId, RibbonBackbone<'_>>,
    ep: &ClashEndpoint,
) -> Option<Vec3> {
    let raw = resolve_clash_endpoint(scene, ep)?;
    let Some(state) = scene.entity_state.get(&ep.entity) else {
        return Some(raw);
    };
    Some(rendered_atom_position(
        raw,
        state.drawing_mode,
        state.topology.is_protein(),
        ribbons.get(&ep.entity),
        &state.sheet_offsets,
        ep.residue,
        &ep.atom_name,
    ))
}

/// View each Cartoon-mode protein entity's stored ribbon anchors (the ones
/// the cartoon mesh emitted), keyed by entity id. Shared each frame by the
/// clash endpoints and the hbond/disulfide block so backbone endpoints
/// attach to the drawn ribbon. Entities not in Cartoon mode, non-proteins,
/// or without anchors yet are absent -- their backbone endpoints fall back
/// to raw.
pub(super) fn build_hbond_ribbons(
    scene: &Scene,
) -> FxHashMap<EntityId, RibbonBackbone<'_>> {
    scene
        .entity_state
        .iter()
        .filter(|(_, state)| {
            state.drawing_mode == DrawingMode::Cartoon
                && state.topology.is_protein()
        })
        .filter_map(|(&id, state)| {
            let ribbon = RibbonBackbone::from_anchors(&state.ribbon_anchors)?;
            Some((id, ribbon))
        })
        .collect()
}

/// Resolve a single exposed-hydrophobic spec to a world-space sidechain
/// anchor. Prefers the CB atom; falls back to the sidechain heavy-atom
/// centroid, then to CA. The raw anchor is then routed through
/// [`rendered_atom_position`] as a sidechain atom so a bead on a flattened
/// beta-strand picks up the sheet-flattening offset and sits on the drawn
/// sidechain rather than the raw atom. Returns `None` if the entity is
/// absent or no anchor atom resolves (skipped exactly as clashes skip).
fn resolve_exposed_hydro(
    scene: &Scene,
    bead: &ExposedHydrophobicInfo,
) -> Option<ResolvedExposedHydro> {
    let state = scene.entity_state.get(&bead.entity)?;
    let positions = displayed_positions(state, scene, bead.entity)?;
    let raw = resolve_sidechain_anchor(state, positions, bead.residue)?;
    // The anchor is a sidechain point (CB / centroid / CA fallback); the
    // sheet offset is per-residue and uniform across the sidechain, so route
    // it through the resolver as a sidechain atom name ("CB"). The ribbon is
    // irrelevant here (sidechain branch never touches it), so pass none.
    let center = rendered_atom_position(
        raw,
        state.drawing_mode,
        state.topology.is_protein(),
        None,
        &state.sheet_offsets,
        bead.residue,
        "CB",
    );
    Some(ResolvedExposedHydro {
        center,
        seed: exposed_hydro_seed(bead),
    })
}

/// World-space sidechain anchor for a residue: CB if present, else the
/// centroid of the residue's sidechain heavy atoms, else CA.
fn resolve_sidechain_anchor(
    state: &EntityView,
    positions: &[Vec3],
    local_residue: u32,
) -> Option<Vec3> {
    if let Some(cb_idx) = state
        .topology
        .sidechain_layout
        .atom_index(local_residue, "CB")
    {
        if let Some(pos) = positions.get(cb_idx as usize) {
            return Some(*pos);
        }
    }

    if let Some(atom_map) = state
        .topology
        .sidechain_layout
        .atom_lookup
        .get(&local_residue)
    {
        let mut sum = Vec3::ZERO;
        let mut count = 0u32;
        for &atom_idx in atom_map.values() {
            if let Some(pos) = positions.get(atom_idx as usize) {
                sum += *pos;
                count += 1;
            }
        }
        if count > 0 {
            return Some(sum / count as f32);
        }
    }

    // Fall back to CA (backbone offset 1).
    let range = state
        .topology
        .residue_atom_ranges
        .get(local_residue as usize)?;
    positions.get(range.start as usize + 1).copied()
}

/// Hash an exposed-hydrophobic residue ref into a stable per-bead seed in
/// `[0, 1)`. Deterministic per (entity, residue) so the procedural "boil"
/// is consistent across frames for a given bead but decorrelated between
/// distinct beads (mirrors [`clash_seed`]).
fn exposed_hydro_seed(bead: &ExposedHydrophobicInfo) -> f32 {
    let mut h: u32 = 0x811c_9dc5;
    let mut mix = |bytes: &[u8]| {
        for &byte in bytes {
            h ^= u32::from(byte);
            h = h.wrapping_mul(0x0100_0193);
        }
    };
    mix(&bead.entity.raw().to_le_bytes());
    mix(&bead.residue.to_le_bytes());
    (h as f32) / (u32::MAX as f32)
}

/// Resolve a pull spec to world-space atom and target positions.
fn resolve_pull(
    ctx: &ConstraintContext<'_>,
    camera: &CameraController,
    viewport: (u32, u32),
    pull: &PullInfo,
) -> Option<ResolvedPull> {
    let atom_pos = ctx.resolve_atom_ref(&pull.atom)?;
    let target_pos = camera.screen_to_world_at_depth(
        Vec2::new(pull.screen_target.0, pull.screen_target.1),
        UVec2::new(viewport.0, viewport.1),
        atom_pos,
    );

    Some(ResolvedPull {
        atom_pos,
        target_pos,
        residue_idx: pull.atom.residue,
    })
}

/// Public entry point used by [`VisoEngine::resolve_atom_position`].
///
/// Builds a single-shot [`ConstraintContext`] — cheap (O(visible
/// cartoon proteins)), but callers with multiple resolutions per
/// frame should build a shared context via
/// [`ConstraintContext::new`].
pub(super) fn resolve_atom_ref_pub(
    scene: &Scene,
    annotations: &EntityAnnotations,
    atom: &AtomRef,
) -> Option<Vec3> {
    ConstraintContext::new(scene, annotations).resolve_atom_ref(atom)
}

/// Full breakdown of which atom a screen-space pick resolves to inside
/// a cartoon-mode residue.
///
/// Used by drag dispatchers that need the owning entity id and the
/// entity-local residue index in addition to the atom name (e.g.
/// classifying backbone vs sidechain for the pull op router).
pub struct PickedResidueAtom {
    /// Owning entity (molex `EntityId.raw()`).
    pub entity_id: u32,
    /// Entity-local residue index (0-based).
    pub local_residue: u32,
    /// PDB atom name of the atom that projects closest to `screen_pos`.
    pub atom_name: String,
}

/// Same as [`closest_atom_in_residue`] but also returns the owning
/// entity id and the entity-local residue index. Lets the host
/// classify the pick (backbone vs sidechain, protein vs non-protein)
/// and build the param map for op dispatch in one pass.
pub(super) fn picked_residue_atom(
    scene: &Scene,
    annotations: &EntityAnnotations,
    camera: &CameraController,
    viewport: UVec2,
    residue: u32,
    screen_pos: Vec2,
) -> Option<PickedResidueAtom> {
    let ctx = ConstraintContext::new(scene, annotations);
    let range = ctx
        .cartoon_ranges
        .iter()
        .find(|r| residue >= r.start && residue < r.end)?;
    let entity_id = range.entity.raw();
    let local_residue = residue - range.start;
    let atom_name = closest_atom_in_residue(
        scene,
        annotations,
        camera,
        viewport,
        residue,
        screen_pos,
    )?;
    Some(PickedResidueAtom {
        entity_id,
        local_residue,
        atom_name,
    })
}

/// Find the heavy-atom in `residue` whose world position projects
/// closest to `screen_pos` (in pixels). Used by the host to anchor
/// pull/drag actions to the atom under the cursor instead of always
/// defaulting to CA.
pub(super) fn closest_atom_in_residue(
    scene: &Scene,
    annotations: &EntityAnnotations,
    camera: &CameraController,
    viewport: UVec2,
    residue: u32,
    screen_pos: Vec2,
) -> Option<String> {
    let ctx = ConstraintContext::new(scene, annotations);
    let range = ctx
        .cartoon_ranges
        .iter()
        .find(|r| residue >= r.start && residue < r.end)?;
    let state = scene.entity_state.get(&range.entity)?;
    let positions = scene.positions.get(range.entity)?;
    let local_residue = residue - range.start;

    let mut best: Option<(f32, String)> = None;
    let mut consider = |name: &str, pos: Vec3| {
        if let Some(screen) = camera.world_to_screen(pos, viewport) {
            let dx = screen.x - screen_pos.x;
            let dy = screen.y - screen_pos.y;
            let d2 = dx * dx + dy * dy;
            if best.as_ref().is_none_or(|(b, _)| d2 < *b) {
                best = Some((d2, name.to_owned()));
            }
        }
    };

    if let Some(bb_range) = state
        .topology
        .residue_atom_ranges
        .get(local_residue as usize)
    {
        for (offset, name) in [(0_usize, "N"), (1, "CA"), (2, "C")] {
            if let Some(pos) = positions.get(bb_range.start as usize + offset) {
                consider(name, *pos);
            }
        }
    }

    if let Some(atom_map) = state
        .topology
        .sidechain_layout
        .atom_lookup
        .get(&local_residue)
    {
        for (name, &atom_idx) in atom_map {
            if let Some(pos) = positions.get(atom_idx as usize) {
                consider(name.as_ref(), *pos);
            }
        }
    }

    best.map(|(_, name)| name)
}

fn resolve_atom_in_entity(
    state: &EntityView,
    positions: &[Vec3],
    local_residue: u32,
    atom_name: &str,
) -> Option<Vec3> {
    match atom_name {
        "N" | "CA" | "C" => {
            let range = state
                .topology
                .residue_atom_ranges
                .get(local_residue as usize)?;
            let offset = match atom_name {
                "N" => 0,
                "CA" => 1,
                "C" => 2,
                _ => return None,
            };
            let idx = range.start as usize + offset;
            positions.get(idx).copied()
        }
        other => {
            let atom_idx = state
                .topology
                .sidechain_layout
                .atom_index(local_residue, other)?;
            positions.get(atom_idx as usize).copied()
        }
    }
}

// ConstraintSpecs: per-frame resolution

impl ConstraintSpecs {
    /// Resolve stored band/pull specs to world-space and update the
    /// band + pull GPU renderers.
    pub(crate) fn resolve_and_render(
        &self,
        scene: &Scene,
        annotations: &EntityAnnotations,
        options: &VisoOptions,
        camera: &CameraController,
        gpu: &mut GpuPipeline,
    ) {
        let viewport = (gpu.context.config.width, gpu.context.config.height);
        // Views over each Cartoon-mode protein entity's stored ribbon anchors,
        // shared by the clash endpoints (which may be backbone atoms reading the
        // drawn-ribbon anchor). Sidechain endpoints ignore it (they take the
        // sheet offset), and entities with no anchors yet are absent so those
        // endpoints fall back to raw. Only built when something consumes it
        // (clashes); empty otherwise.
        let ribbons = if self.clash_specs.is_empty() {
            FxHashMap::default()
        } else {
            build_hbond_ribbons(scene)
        };
        // Resolve bands, pull, clashes, and exposed-hydrophobic beads
        // against one shared context, then drop it before taking `&mut
        // gpu` for the upload.
        let (bands, pull, clashes, beads) = {
            let ctx = ConstraintContext::new(scene, annotations);
            let bands: Vec<_> = self
                .band_specs
                .iter()
                .filter_map(|b| resolve_band(&ctx, b))
                .collect();
            let pull = self
                .pull_spec
                .as_ref()
                .and_then(|p| resolve_pull(&ctx, camera, viewport, p));
            let clashes: Vec<_> = self
                .clash_specs
                .iter()
                .filter_map(|c| resolve_clash(&ctx, &ribbons, c))
                .collect();
            let beads: Vec<_> = self
                .exposed_hydro_specs
                .iter()
                .filter_map(|b| resolve_exposed_hydro(ctx.scene, b))
                .collect();
            (bands, pull, clashes, beads)
        };

        gpu.renderers.band.update(
            &gpu.context.device,
            &gpu.context.queue,
            &bands,
            Some(&options.colors),
        );
        gpu.renderers.pull.update(
            &gpu.context.device,
            &gpu.context.queue,
            pull.as_ref(),
        );
        gpu.renderers.clash.update(
            &gpu.context.device,
            &gpu.context.queue,
            &clashes,
        );
        gpu.renderers.grease.update(
            &gpu.context.device,
            &gpu.context.queue,
            &beads,
        );
    }
}

// Engine-side dispatchers

impl VisoEngine {
    /// Resolve stored band/pull specs to world-space and update
    /// renderers.
    pub(super) fn resolve_and_render_constraints(&mut self) {
        self.constraints.resolve_and_render(
            &self.scene,
            &self.annotations,
            &self.options,
            &self.camera_controller,
            &mut self.gpu,
        );
    }

    /// Resolve the hydrogen-bond and disulfide connection links to capsules
    /// and re-upload the bond buffer. Fired at every moment positions
    /// become final (sync, consumed mesh, animation/trajectory tick) so the
    /// capsules never strand mid-motion.
    pub(super) fn resolve_and_upload_bond_connections(&mut self) {
        super::bond_connections::resolve_and_upload_bond_connections(
            &self.scene,
            &self.annotations,
            &self.options,
            &mut self.gpu,
        );
    }

    /// Replace the current set of constraint bands.
    pub fn update_bands(&mut self, bands: Vec<BandInfo>) {
        self.constraints.band_specs = bands;
        self.resolve_and_render_constraints();
    }

    /// Set or clear the active pull constraint.
    pub fn update_pull(&mut self, pull: Option<PullInfo>) {
        self.constraints.pull_spec = pull;
        self.resolve_and_render_constraints();
    }

    /// Replace the current set of steric clash arcs. An empty vec clears
    /// them.
    pub fn update_clashes(&mut self, clashes: Vec<ClashInfo>) {
        self.constraints.clash_specs = clashes;
        self.resolve_and_render_constraints();
    }

    /// Replace the current set of exposed-hydrophobic "grease bead"
    /// markers. An empty vec clears them. Each frame the stored refs
    /// re-resolve to a world-space sidechain anchor so the beads track the
    /// residues live (mirrors [`Self::update_clashes`]).
    pub fn update_exposed_hydrophobics(
        &mut self,
        beads: Vec<ExposedHydrophobicInfo>,
    ) {
        self.constraints.exposed_hydro_specs = beads;
        self.resolve_and_render_constraints();
    }

    /// Replace the per-residue non-designable set. `non_designable` is the
    /// per-entity authoritative set of residues that may NOT be designed
    /// (mutated) in the current puzzle; the geometry shaders desaturate
    /// these residues toward white, composing on top of score color and the
    /// selection highlight. An empty map clears every whiteout (the
    /// non-design-puzzle case).
    ///
    /// Like [`Self::set_selection`], viso stores the per-entity set as the
    /// source of truth and re-derives the flat GPU bitset from its own
    /// always-current per-entity residue offsets, both here and on every
    /// mesh rebuild, so the overlay can never go stale relative to a
    /// shifting residue space.
    pub fn set_non_designable(
        &mut self,
        non_designable: &std::collections::BTreeMap<
            EntityId,
            std::collections::BTreeSet<u32>,
        >,
    ) {
        self.gpu.pick.set_non_designable(non_designable.clone());
        self.gpu
            .pick
            .update_non_designable_buffer(&self.gpu.context.queue);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn endpoint(entity: u32, residue: u32, atom: &str) -> ClashEndpoint {
        ClashEndpoint {
            entity: EntityId::from_raw(entity),
            residue,
            atom_name: atom.to_owned(),
        }
    }

    #[test]
    fn clash_seed_is_deterministic_per_pair() {
        let a = endpoint(0, 7, "CA");
        let b = endpoint(1, 12, "CB");
        assert_eq!(clash_seed(&a, &b), clash_seed(&a, &b));
    }

    #[test]
    fn clash_seed_decorrelates_distinct_pairs() {
        let a = endpoint(0, 7, "CA");
        let b = endpoint(1, 12, "CB");
        let c = endpoint(0, 8, "CA");
        assert_ne!(clash_seed(&a, &b), clash_seed(&a, &c));
    }

    #[test]
    fn clash_seed_in_unit_range() {
        let s = clash_seed(&endpoint(3, 5, "N"), &endpoint(3, 9, "O"));
        assert!((0.0..1.0).contains(&s), "seed {s} out of [0, 1)");
    }
}
