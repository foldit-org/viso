//! Assembly consumption + derived per-entity state.
//!
//! [`Scene`] bundles the one coherent data flow that makes the engine
//! a consumer of the host's [`Assembly`]: the pending-snapshot slot
//! written by [`crate::VisoEngine::set_assembly`], the latest applied
//! snapshot, the generation tracker, and the render-ready derived
//! state ([`SceneRenderState`] + per-entity [`EntityView`] +
//! [`EntityPositions`]).
//!
//! `sync_from_assembly` reads the three ingest fields and writes the
//! four derived fields, so they're kept together. The `next_mesh_version`
//! dispenser also lives here — it's the counter that stamps freshness
//! onto `EntityView.mesh_version`, and must survive `reset_scene_local_state`
//! so a Vacant insert never collides with a stale worker `MeshCache`
//! entry for the same `EntityId`.
//!
//! Every overlay / command entry point that arrives with a raw `u32`
//! entity ID translates once via [`Scene::entity_id`] and then passes
//! the opaque [`EntityId`] down. Internal code never walks the
//! assembly looking up u32s.

use std::sync::Arc;

use molex::entity::molecule::id::EntityId;
use molex::{Assembly, MoleculeEntity};
use rustc_hash::FxHashMap;

use super::annotations::EntityAnnotations;
use super::entity_view::EntityView;
use super::positions::EntityPositions;
use super::scene_state::SceneRenderState;
use crate::options::DrawingMode;

/// Assembly consumption + derived per-entity state.
pub(crate) struct Scene {
    // Ingest
    /// Latest assembly snapshot pushed by the host via
    /// [`crate::VisoEngine::set_assembly`] but not yet applied. Drained
    /// by `poll_assembly` / `poll_assembly_force` on the next sync tick.
    pub(crate) pending: Option<Arc<Assembly>>,
    /// Last `Assembly` snapshot applied by `sync_from_assembly`. Held
    /// for read-only queries (entity metadata for the UI panel, atom
    /// positions for the picking pipeline). Always up to date with
    /// `entity_state` / `positions`.
    pub(crate) current: Arc<Assembly>,
    /// Generation of the most recently consumed snapshot. Initialized
    /// to `u64::MAX` so the first snapshot (generation 0) always
    /// triggers a sync.
    pub(crate) last_seen_generation: u64,

    // Derived
    /// Cross-entity rendering data (disulfide + H-bond endpoints +
    /// per-sync resolved structural bonds). Rederived on every sync;
    /// lives on the main thread only.
    pub(crate) render_state: SceneRenderState,
    /// Per-entity rendering state (topology + drawing mode + SS
    /// override + mesh version) rederived on every sync.
    pub(crate) entity_state: FxHashMap<EntityId, EntityView>,
    /// Per-entity animator write surface; renderer reads back each
    /// frame. Seeded from the assembly's reference positions when
    /// new entities appear, animated locally thereafter.
    pub(crate) positions: EntityPositions,

    /// Monotonic mesh-version dispenser. Survives
    /// `reset_scene_local_state` so a Vacant insert never collides
    /// with a stale worker `MeshCache` entry for the same `EntityId`
    /// (fresh per-file allocators reuse low ids across
    /// `replace_scene`).
    pub(crate) next_mesh_version: u64,
}

/// Iterator item for [`Scene::visible_entities`].
pub(crate) type VisibleEntity<'a> =
    (&'a MoleculeEntity, EntityId, &'a EntityView);

impl Scene {
    /// Empty scene with a valid (but empty) assembly snapshot. The
    /// host pushes its first snapshot via
    /// [`crate::VisoEngine::set_assembly`].
    pub(crate) fn new() -> Self {
        Self {
            pending: None,
            current: Arc::new(Assembly::new(Vec::new())),
            last_seen_generation: u64::MAX,
            render_state: SceneRenderState::new(),
            entity_state: FxHashMap::default(),
            positions: EntityPositions::new(),
            next_mesh_version: 1,
        }
    }

    /// Pull a fresh, never-reused `mesh_version` from the dispenser.
    /// All writes to `EntityView.mesh_version` must route through
    /// this so a worker `MeshCache` entry for an `EntityId` is never
    /// invalidated by a colliding version.
    pub(crate) fn bump_mesh_version(&mut self) -> u64 {
        let v = self.next_mesh_version;
        self.next_mesh_version = self.next_mesh_version.wrapping_add(1);
        v
    }

    /// Look up the opaque [`EntityId`] for a raw `u32` id. Returns
    /// `None` if no entity with that raw id exists. Callers that hold
    /// a raw id should translate here *once* and then pass `EntityId`
    /// down.
    ///
    /// Also checks `pending` so callers can stage annotation overrides
    /// for newly-published entities before the next sync tick consumes
    /// the snapshot — without this, a `set_*` call sandwiched between
    /// `set_assembly` and the next `update` is a silent no-op.
    #[must_use]
    pub(crate) fn entity_id(&self, raw: u32) -> Option<EntityId> {
        let in_current = self
            .current
            .entities()
            .iter()
            .map(|e| e.id())
            .find(|eid| eid.raw() == raw);
        if in_current.is_some() {
            return in_current;
        }
        self.pending.as_ref().and_then(|asm| {
            asm.entities()
                .iter()
                .map(|e| e.id())
                .find(|eid| eid.raw() == raw)
        })
    }

    /// Iterate every visible entity that has an `entity_state` slot,
    /// yielding `(entity, entity_id, view)`. Skips entities toggled off
    /// in `annotations` and entities not yet reconciled into
    /// `entity_state`. Callers that also need positions look them up via
    /// `self.positions.get`.
    pub(crate) fn visible_entities<'a>(
        &'a self,
        annotations: &'a EntityAnnotations,
    ) -> impl Iterator<Item = VisibleEntity<'a>> + 'a {
        self.current.entities().iter().filter_map(|entity| {
            let entity: &MoleculeEntity = entity;
            let eid = entity.id();
            if !annotations.is_visible(eid) {
                return None;
            }
            let state = self.entity_state.get(&eid)?;
            Some((entity, eid, state))
        })
    }

    /// Concatenated per-residue colors across every visible Cartoon-mode
    /// protein entity, in assembly order. Entities without cached colors
    /// contribute a default gray block sized to their residue count.
    pub(crate) fn flat_cartoon_colors(
        &self,
        annotations: &EntityAnnotations,
    ) -> Vec<[f32; 3]> {
        let mut out = Vec::new();
        for (_, _, state) in self.visible_entities(annotations) {
            if !state.topology.is_protein()
                || state.drawing_mode != DrawingMode::Cartoon
            {
                continue;
            }
            if let Some(colors) = &state.per_residue_colors {
                out.extend_from_slice(colors);
            } else {
                let residue_count = state.topology.residue_atom_ranges.len();
                out.extend(std::iter::repeat_n(
                    [0.5_f32, 0.5, 0.5],
                    residue_count,
                ));
            }
        }
        out
    }

    /// Reset all scene-local derived state (positions, entity_state).
    /// Keeps `next_mesh_version` (the dispenser must outlive
    /// `replace_scene` so fresh per-file entity IDs don't collide with
    /// stale worker mesh cache entries) and any pending snapshot. Also
    /// resets `last_seen_generation` to `u64::MAX` so the next snapshot
    /// triggers a sync unconditionally.
    pub(crate) fn reset_local_state(&mut self) {
        self.entity_state.clear();
        self.positions = EntityPositions::new();
        self.render_state = SceneRenderState::new();
        self.current = Arc::new(Assembly::new(Vec::new()));
        self.last_seen_generation = u64::MAX;
    }
}
