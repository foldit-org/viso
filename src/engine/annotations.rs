//! Per-entity annotations: user-authored state that rides alongside
//! the [`Assembly`](molex::Assembly) snapshot.
//!
//! Everything in [`EntityAnnotations`] is state the caller (UI, IPC,
//! command path) has expressed *about* entities — not state derived
//! from the Assembly itself. The Assembly-derived half of the engine
//! (topology, positions, render state) is rebuilt on every sync;
//! annotations persist across syncs and only get reconciled when an
//! entity leaves the assembly entirely.
//!
//! All maps are keyed on the opaque [`EntityId`] so lookups are O(1).
//! Callers arriving with a raw `u32` (UI commands, IPC) translate
//! *once* at the API boundary via [`VisoEngine::entity_id`] and then
//! pass the [`EntityId`] down. Engine-side mutation methods live at
//! the bottom of this file as impls on [`VisoEngine`].
//!
//! Reconciliation on [`Assembly`](molex::Assembly) sync happens via
//! [`EntityAnnotations::retain_entities`], which drops entries for
//! IDs that have left the assembly.
//!
//! The `Focus` arm holds an [`EntityId`] for the same reason the maps
//! do; the UI-side `u32` is recovered via `EntityId::raw()` when
//! needed by external consumers (the bridge, the webview).
//!
//! [`AnnotationsScene`] is the disjoint-borrow write view: most
//! annotation mutations also bump per-entity `mesh_version`, so the
//! write path owns coordinated access to `annotations` plus the two
//! scene fields it stamps (`entity_state`, `next_mesh_version`).
//! `VisoEngine` exposes it through `annotations_mut()`; all annotation
//! mutators on `VisoEngine` become one-line dispatchers.

use std::collections::HashMap;

use molex::entity::molecule::id::EntityId;
use molex::{MoleculeType, SSType};
use rustc_hash::FxHashMap;

use super::density_store::DensityStore;
use super::focus::Focus;
use super::scene::Scene;
use super::surface::{EntitySurface, SurfaceKind};
use super::surface_regen::{regenerate_surfaces, SurfaceRegen};
use super::{ExternalVoidField, VisoEngine};
use crate::animation::transition::Transition;
use crate::options::overrides::RenderInvalidation;
use crate::options::{DisplayOverrides, DrawingMode, VisoOptions};

/// Per-entity user-authored state that isn't derived from the
/// [`Assembly`](molex::Assembly).
#[derive(Default)]
pub(crate) struct EntityAnnotations {
    /// Focus state (session-wide or a specific entity).
    pub(crate) focus: Focus,
    /// Per-entity visibility (`true` = drawn). Missing = visible.
    pub(crate) visibility: FxHashMap<EntityId, bool>,
    /// Per-entity animation behavior overrides.
    pub(crate) behaviors: FxHashMap<EntityId, Transition>,
    /// Per-entity appearance overrides (None-fields inherit global).
    pub(crate) appearance: FxHashMap<EntityId, DisplayOverrides>,
    /// Per-entity scores (for color-by-score visualization).
    pub(crate) scores: FxHashMap<EntityId, Vec<f64>>,
    /// Per-entity SS overrides (from puzzle annotations).
    pub(crate) ss_overrides: FxHashMap<EntityId, Vec<SSType>>,
    /// Per-entity molecular surfaces.
    pub(crate) surfaces: FxHashMap<EntityId, EntitySurface>,
}

impl EntityAnnotations {
    /// Whether `id` is currently visible. Entities with no explicit
    /// entry default to visible.
    #[must_use]
    pub(crate) fn is_visible(&self, id: EntityId) -> bool {
        self.visibility.get(&id).copied().unwrap_or(true)
    }

    /// Per-entity animation behavior override, if set.
    #[must_use]
    pub(crate) fn behavior(&self, id: EntityId) -> Option<&Transition> {
        self.behaviors.get(&id)
    }

    /// Record a per-entity animation behavior override.
    pub(crate) fn set_behavior(
        &mut self,
        id: EntityId,
        transition: Transition,
    ) {
        let _ = self.behaviors.insert(id, transition);
    }

    /// Drop a per-entity animation behavior override.
    pub(crate) fn clear_behavior(&mut self, id: EntityId) {
        let _ = self.behaviors.remove(&id);
    }

    /// Per-entity appearance override, if set.
    #[must_use]
    pub(crate) fn appearance(&self, id: EntityId) -> Option<&DisplayOverrides> {
        self.appearance.get(&id)
    }

    /// Drop entries for entities no longer present in the assembly.
    pub(crate) fn retain_entities(&mut self, keep: impl Fn(EntityId) -> bool) {
        self.visibility.retain(|&id, _| keep(id));
        self.behaviors.retain(|&id, _| keep(id));
        self.appearance.retain(|&id, _| keep(id));
        self.scores.retain(|&id, _| keep(id));
        self.ss_overrides.retain(|&id, _| keep(id));
        self.surfaces.retain(|&id, _| keep(id));
    }

    /// Advance `focus` to the next entity in `focusable`, wrapping
    /// back to `Session` after the last. Returns the new focus.
    ///
    /// `focusable` is the engine-filtered list of entities eligible
    /// for focus (typically: visible, focusable molecule type). The
    /// annotation owns focus state but not the filter, so callers
    /// build the list themselves.
    pub(crate) fn cycle_focus(&mut self, focusable: &[EntityId]) -> Focus {
        self.focus = match self.focus {
            Focus::All => focusable
                .first()
                .map_or(Focus::All, |&id| Focus::Entity(id)),
            Focus::Entity(current) => {
                let idx = focusable.iter().position(|&id| id == current);
                match idx {
                    Some(i) if i + 1 < focusable.len() => {
                        Focus::Entity(focusable[i + 1])
                    }
                    _ => Focus::All,
                }
            }
        };
        self.focus
    }

    /// Resolve the drawing mode for an entity: per-entity override wins,
    /// else global (with Cartoon falling back to type-default).
    #[must_use]
    pub(crate) fn resolved_drawing_mode(
        &self,
        options: &VisoOptions,
        eid: EntityId,
        mol_type: MoleculeType,
    ) -> DrawingMode {
        resolve_drawing_mode(self.appearance(eid), options, mol_type)
    }
}

/// Resolve drawing mode against explicit overrides rather than the
/// annotations map. Used by `set_entity_appearance` /
/// `clear_entity_appearance`, which need to resolve against the NEW
/// overrides before the map has been updated — reading through the map
/// would return stale values and `state.drawing_mode` would lag one
/// mutation behind.
#[must_use]
pub(crate) fn resolve_drawing_mode(
    overrides: Option<&DisplayOverrides>,
    options: &VisoOptions,
    mol_type: MoleculeType,
) -> DrawingMode {
    overrides
        .and_then(|ovr| ovr.drawing_mode)
        .unwrap_or_else(|| {
            if options.display.drawing_mode() == DrawingMode::Cartoon {
                DrawingMode::default_for(mol_type)
            } else {
                options.display.drawing_mode()
            }
        })
}

impl EntityAnnotations {
    /// Clear annotation state back to the default (focus session-wide,
    /// every map emptied) EXCEPT per-entity scores and appearance
    /// overrides.
    ///
    /// Scores and appearance deliberately survive `reset`. A per-publish
    /// topology swap (`replace_assembly`) routes through here;
    /// blanket-clearing either map would blank the surviving committed
    /// entities for the one frame between the reset and the recolor (a
    /// gray flash), and would drop user-authored per-entity appearance on
    /// every republish. Membership is instead reconciled by
    /// [`Self::retain_entities`] on the next sync, which drops only the
    /// entities that actually left the assembly. Genuine teardown, where
    /// the next topology may reuse entity ids, must drop scores explicitly
    /// via [`Self::clear_scores`] and appearance via
    /// [`VisoEngine::clear_all_appearance`].
    pub(crate) fn reset(&mut self) {
        self.focus = Focus::default();
        self.visibility.clear();
        self.behaviors.clear();
        self.ss_overrides.clear();
        self.surfaces.clear();
    }

    /// Drop every per-entity score entry. Scores survive [`Self::reset`]
    /// (reconciled by id on the next sync), so a genuine topology
    /// teardown / reload that may reuse entity ids must clear them here or
    /// the next molecule inherits the previous one's colors.
    pub(crate) fn clear_scores(&mut self) {
        self.scores.clear();
    }
}

/// Disjoint-borrow write view over the annotation maps plus the scene
/// fields every mutation has to stamp.
///
/// Annotation writes almost always bump an entity's `mesh_version`, so
/// this struct bundles `annotations` together with `&mut Scene` (whose
/// `entity_state` and `next_mesh_version` get stamped) — two disjoint
/// `&mut`s — so the write path is expressible without `&mut self`
/// methods fighting over `VisoEngine`. `VisoEngine::annotations_mut`
/// is the constructor.
///
/// Surface mutations also live here (rather than in a parallel handle
/// that would conflict on `&mut EntityAnnotations`); they need the
/// extra read borrows of [`DensityStore`], [`VisoOptions`], and
/// [`SurfaceRegen`] (plus a reborrow of `&Scene` from the held
/// `&mut Scene`) to call [`regenerate_surfaces`].
pub(crate) struct AnnotationsScene<'a> {
    annotations: &'a mut EntityAnnotations,
    scene: &'a mut Scene,
    density: &'a DensityStore,
    external_void_field: Option<&'a ExternalVoidField>,
    options: &'a VisoOptions,
    regen: &'a SurfaceRegen,
}

impl<'a> AnnotationsScene<'a> {
    /// Construct the view from the two disjoint `&mut` fields plus the
    /// three immutable borrows surface regeneration needs.
    pub(crate) fn new(
        annotations: &'a mut EntityAnnotations,
        scene: &'a mut Scene,
        density: &'a DensityStore,
        external_void_field: Option<&'a ExternalVoidField>,
        options: &'a VisoOptions,
        regen: &'a SurfaceRegen,
    ) -> Self {
        Self {
            annotations,
            scene,
            density,
            external_void_field,
            options,
            regen,
        }
    }

    /// Pull a fresh `mesh_version` from the dispenser.
    fn bump(&mut self) -> u64 {
        let v = self.scene.next_mesh_version;
        self.scene.next_mesh_version =
            self.scene.next_mesh_version.wrapping_add(1);
        v
    }

    /// Bump the dispenser and stamp the new version onto `eid`'s
    /// `EntityView` if one exists.
    fn bump_for(&mut self, eid: EntityId) {
        let v = self.bump();
        if let Some(state) = self.scene.entity_state.get_mut(&eid) {
            state.mesh_version = v;
        }
    }

    /// Record visibility for `eid` and invalidate its mesh.
    pub(crate) fn set_visible(&mut self, eid: EntityId, visible: bool) {
        let _ = self.annotations.visibility.insert(eid, visible);
        self.bump_for(eid);
    }

    /// Record (or clear, with `None`) per-residue scores for `eid`.
    pub(crate) fn set_per_residue_scores(
        &mut self,
        eid: EntityId,
        scores: Option<Vec<f64>>,
    ) {
        match scores {
            Some(s) => {
                let _ = self.annotations.scores.insert(eid, s);
            }
            None => {
                let _ = self.annotations.scores.remove(&eid);
            }
        }
        self.bump_for(eid);
    }

    /// Record an SS override for `eid`.
    pub(crate) fn set_ss_override(&mut self, eid: EntityId, ss: Vec<SSType>) {
        let _ = self.annotations.ss_overrides.insert(eid, ss.clone());
        // Also stamp entity_state directly: the rebuild path reads
        // state.ss_override (sync_from_assembly is the only other writer,
        // so an override applied between syncs would otherwise be ignored
        // on the next rebuild).
        if let Some(state) = self.scene.entity_state.get_mut(&eid) {
            state.ss_override = Some(ss);
        }
        self.bump_for(eid);
    }

    /// Record an appearance override for `eid` and restamp both
    /// `mesh_version` and the resolved `drawing_mode`.
    ///
    /// The resolved drawing mode is passed in because resolution reads
    /// `VisoOptions`, which lives outside the fields this view owns;
    /// the caller resolves it before taking the view.
    pub(crate) fn set_appearance(
        &mut self,
        eid: EntityId,
        overrides: DisplayOverrides,
        drawing_mode: DrawingMode,
    ) {
        let _ = self.annotations.appearance.insert(eid, overrides);
        let v = self.bump();
        if let Some(state) = self.scene.entity_state.get_mut(&eid) {
            state.mesh_version = v;
            state.drawing_mode = drawing_mode;
        }
    }

    /// Drop an appearance override for `eid` and restamp `mesh_version`
    /// plus the resolved `drawing_mode` (see [`Self::set_appearance`]
    /// for why the caller resolves it).
    pub(crate) fn clear_appearance(
        &mut self,
        eid: EntityId,
        drawing_mode: DrawingMode,
    ) {
        let _ = self.annotations.appearance.remove(&eid);
        let v = self.bump();
        if let Some(state) = self.scene.entity_state.get_mut(&eid) {
            state.mesh_version = v;
            state.drawing_mode = drawing_mode;
        }
    }

    /// Trigger a regeneration of all isosurface meshes (density,
    /// entity surfaces, cavities) on the background thread.
    fn regenerate_surfaces(&self) {
        regenerate_surfaces(
            &*self.scene,
            &*self.annotations,
            self.density,
            self.external_void_field,
            self.options,
            self.regen,
        );
    }

    /// Add a Gaussian surface for an entity.
    pub(crate) fn add_gaussian_surface(
        &mut self,
        entity_id: EntityId,
        color: [f32; 4],
    ) {
        self.annotations.set_entity_surface(
            entity_id,
            EntitySurface {
                kind: SurfaceKind::Gaussian,
                color,
                ..Default::default()
            },
        );
        self.regenerate_surfaces();
    }

    /// Add a solvent-excluded (Connolly) surface for an entity.
    pub(crate) fn add_ses_surface(
        &mut self,
        entity_id: EntityId,
        color: [f32; 4],
    ) {
        self.annotations.set_entity_surface(
            entity_id,
            EntitySurface {
                kind: SurfaceKind::Ses,
                color,
                ..Default::default()
            },
        );
        self.regenerate_surfaces();
    }

    /// Update a single color channel or opacity on an entity's surface.
    pub(crate) fn set_surface_color_channel(
        &mut self,
        entity_id: EntityId,
        channel: usize,
        value: f32,
    ) {
        if self
            .annotations
            .set_surface_color_channel(entity_id, channel, value)
        {
            self.regenerate_surfaces();
        }
    }

    /// Remove the molecular surface for an entity.
    ///
    /// When a global surface is active, this stores an invisible sentinel
    /// so the entity explicitly opts out instead of falling back to the
    /// global default.
    pub(crate) fn remove_surface(&mut self, entity_id: EntityId) {
        let global_kind = self.options.display.surface_kind();
        if self
            .annotations
            .remove_entity_surface(entity_id, global_kind)
        {
            log::info!("removed surface for entity {}", entity_id.raw());
            self.regenerate_surfaces();
        }
    }

    /// Advance focus to the next visible, focusable entity. Wraps to
    /// session after the last. Returns the new focus.
    ///
    /// The eligibility filter (visible + focusable molecule type) is
    /// computed from the assembly held on `scene`, keeping the policy
    /// next to the focus state that answers to it.
    pub(crate) fn cycle_focus(&mut self) -> Focus {
        let focusable: Vec<EntityId> = self
            .scene
            .current
            .entities()
            .iter()
            .filter(|e| self.annotations.is_visible(e.id()) && e.is_focusable())
            .map(|e| e.id())
            .collect();
        self.annotations.cycle_focus(&focusable)
    }
}

// ── Engine-side annotation dispatchers ─────────────────────────────
//
// Each entry point either (a) takes a raw `u32`, translates to
// `EntityId` once, and forwards to [`AnnotationsScene`] /
// [`EntityAnnotations`], or (b) takes an `EntityId` and forwards
// directly. Any follow-up work that needs more than the three
// annotation-scene fields (e.g. `sync_scene_to_renderers`, reading
// `VisoOptions`) stays in the dispatcher.

impl VisoEngine {
    /// Disjoint-borrow write view over the annotations + the scene
    /// fields every annotation mutation stamps.
    pub(crate) fn annotations_mut(&mut self) -> AnnotationsScene<'_> {
        AnnotationsScene::new(
            &mut self.annotations,
            &mut self.scene,
            &self.density,
            self.external_void_field.as_ref(),
            &self.options,
            &self.surface_regen,
        )
    }

    /// Whether an entity is currently visible. Absent entries default
    /// to visible.
    #[must_use]
    pub fn is_entity_visible(&self, id: u32) -> bool {
        self.entity_id(id)
            .is_none_or(|eid| self.annotations.is_visible(eid))
    }

    /// Record the visibility of a single entity. Visibility is
    /// engine-side state — mutating it does not require re-publishing
    /// the [`Assembly`](molex::Assembly).
    pub fn set_entity_visible(&mut self, id: u32, visible: bool) {
        if let Some(eid) = self.entity_id(id) {
            self.annotations_mut().set_visible(eid, visible);
        }
    }

    /// Set visibility for every entity of a given molecule type. This
    /// is the low-level helper; the public toggle that also syncs the
    /// matching `options.display.show_*` flag and re-syncs the scene
    /// is [`Self::set_type_visibility`].
    pub(crate) fn apply_type_visibility(
        &mut self,
        mol_type: MoleculeType,
        visible: bool,
    ) {
        let ids: Vec<EntityId> = self
            .scene
            .current
            .entities()
            .iter()
            .filter(|e| e.molecule_type() == mol_type)
            .map(|e| e.id())
            .collect();
        let mut view = self.annotations_mut();
        for eid in ids {
            view.set_visible(eid, visible);
        }
    }

    /// Set per-entity scores (`None` clears). Per-residue scores are
    /// engine-side state used by the score-based color modes;
    /// mutating them does not require re-publishing the
    /// [`Assembly`](molex::Assembly).
    pub fn set_per_residue_scores(
        &mut self,
        id: u32,
        scores: Option<Vec<f64>>,
    ) {
        if let Some(eid) = self.entity_id(id) {
            self.annotations_mut().set_per_residue_scores(eid, scores);
            // Scores feed the per-residue color buffer, not geometry, so
            // recolor only. The recolor path uploads the new colors to
            // the GPU on its own without a mesh rebuild.
            self.apply_entity_invalidation(RenderInvalidation::RE_COLOR);
        }
    }

    /// Drop all per-entity score annotations and recolor.
    ///
    /// Scores survive a per-publish `replace_assembly` (reconciled by id
    /// on the next sync) so surviving entities don't flash neutral for a
    /// frame. A genuine teardown / reload is different: the next topology
    /// may reuse entity ids, so the host must clear scores explicitly here
    /// or a reused id inherits the previous molecule's colors.
    pub fn clear_scores(&mut self) {
        self.annotations.clear_scores();
        self.apply_entity_invalidation(RenderInvalidation::RE_COLOR);
    }

    /// Drop every per-entity appearance override and repaint.
    ///
    /// Appearance overrides survive a per-publish `replace_assembly`
    /// (reconciled by id on the next sync) so user-authored per-entity
    /// settings persist across republishes. A genuine teardown / reload is
    /// different: the next topology may reuse entity ids, so the host must
    /// clear appearance explicitly here or a reused id inherits the
    /// previous molecule's overrides. The symmetric counterpart to
    /// [`Self::clear_scores`]. Distinct from the per-entity
    /// `clear_appearance`, which drops a single id.
    pub fn clear_all_appearance(&mut self) {
        self.annotations.appearance.clear();
        self.apply_entity_invalidation(
            RenderInvalidation::RE_COLOR | RenderInvalidation::RE_MESH,
        );
    }

    /// Set an SS override for an entity. The override is engine-side
    /// state and replaces DSSP-derived secondary structure for
    /// rendering only — it does not mutate the
    /// [`Assembly`](molex::Assembly).
    pub fn set_ss_override(&mut self, id: u32, ss: Vec<SSType>) {
        if let Some(eid) = self.entity_id(id) {
            self.annotations_mut().set_ss_override(eid, ss);
            // SS feeds cartoon ribbon geometry (helix/sheet/coil shape);
            // remesh so the override shows immediately.
            self.apply_entity_invalidation(RenderInvalidation::RE_MESH);
        }
    }

    /// Read access to the per-entity behavior override for `id`.
    pub(crate) fn entity_behavior(&self, id: u32) -> Option<&Transition> {
        self.annotations.behavior(self.entity_id(id)?)
    }

    /// Set a persistent animation-behavior override for an entity.
    /// Survives across syncs until [`Self::clear_entity_behavior`] is
    /// called. The override is consulted internally by
    /// `queue_entity_transition` when deciding what transition to
    /// actually stage.
    pub fn set_entity_behavior(
        &mut self,
        entity_id: EntityId,
        transition: Transition,
    ) {
        self.annotations.set_behavior(entity_id, transition);
    }

    /// Clear a per-entity behavior override.
    pub fn clear_entity_behavior(&mut self, entity_id: EntityId) {
        self.annotations.clear_behavior(entity_id);
    }

    /// Stage a per-entity animation transition for the **next** sync.
    /// One-shot: the staged transition is consumed when the next
    /// `update`/`sync_now` applies a new Assembly generation, and
    /// entities with no staged transition snap.
    ///
    /// If a persistent behavior override is set on this entity (via
    /// [`Self::set_entity_behavior`]) it wins — `transition` is the
    /// fallback used only when no override exists. Hosts that don't
    /// use persistent overrides can ignore the precedence and treat
    /// this as a plain "queue this transition" call.
    ///
    /// Hosts typically pair this with `set_assembly`: stage the
    /// transition first, then push the new snapshot.
    pub fn queue_entity_transition(&mut self, id: u32, transition: Transition) {
        let effective = self.entity_behavior(id).cloned().unwrap_or(transition);
        self.animation.queue_transition(id, effective);
    }

    /// Bulk form of [`Self::queue_entity_transition`]: stage a set of
    /// per-entity transitions for the next sync, with persistent
    /// behavior overrides honored per entity. Replaces any previously
    /// staged transitions.
    pub fn queue_entity_transitions(
        &mut self,
        transitions: HashMap<u32, Transition>,
    ) {
        let mut resolved = HashMap::with_capacity(transitions.len());
        for (id, fallback) in transitions {
            let effective =
                self.entity_behavior(id).cloned().unwrap_or(fallback);
            let _ = resolved.insert(id, effective);
        }
        self.animation.set_pending_transitions(resolved);
    }

    /// Drop all transitions staged via [`Self::queue_entity_transition`]
    /// or [`Self::queue_entity_transitions`]. The next sync snaps for
    /// every entity.
    pub fn clear_pending_transitions(&mut self) {
        self.animation.clear_pending_transitions();
    }

    /// Set per-entity appearance overrides.
    ///
    /// Diffs against the entity's previous overrides and dispatches only
    /// the invalidations that actually matter — a `surface_kind` change
    /// fires `RE_SURFACE`, a `color_scheme` change fires `RE_COLOR`, and
    /// so on. Previously this blindly called `sync_scene_to_renderers`
    /// regardless of which field changed (and never triggered surface
    /// regen for per-entity `surface_kind` changes — now fixed).
    pub fn set_entity_appearance(
        &mut self,
        entity_id: EntityId,
        overrides: DisplayOverrides,
    ) {
        let previous = self
            .annotations
            .appearance
            .get(&entity_id)
            .cloned()
            .unwrap_or_default();
        let inv = previous.diff(&overrides);

        let mol_type = self
            .scene
            .entity_state
            .get(&entity_id)
            .map(|s| s.topology.molecule_type);
        if let Some(mol_type) = mol_type {
            // Resolve against the NEW overrides, not the stale map. The
            // map still holds `previous` until `set_appearance` inserts
            // — reading via `resolved_drawing_mode` here would write the
            // OLD resolved value into `state.drawing_mode` and the
            // change wouldn't show until the next mutation.
            let drawing_mode =
                resolve_drawing_mode(Some(&overrides), &self.options, mol_type);
            self.annotations_mut().set_appearance(
                entity_id,
                overrides,
                drawing_mode,
            );
        } else {
            let _ = self.annotations.appearance.insert(entity_id, overrides);
        }
        self.apply_entity_invalidation(inv);
    }

    /// Clear a per-entity appearance override.
    pub fn clear_entity_appearance(&mut self, entity_id: EntityId) {
        let previous = self
            .annotations
            .appearance
            .get(&entity_id)
            .cloned()
            .unwrap_or_default();
        let inv = previous.diff(&DisplayOverrides::default());

        let mol_type = self
            .scene
            .entity_state
            .get(&entity_id)
            .map(|s| s.topology.molecule_type);
        if let Some(mol_type) = mol_type {
            // After clear, the entity has no overrides — resolve falls
            // through to global. Same staleness fix as `set_entity_appearance`.
            let drawing_mode =
                resolve_drawing_mode(None, &self.options, mol_type);
            self.annotations_mut()
                .clear_appearance(entity_id, drawing_mode);
        } else {
            let _ = self.annotations.appearance.remove(&entity_id);
        }
        self.apply_entity_invalidation(inv);
    }

    /// Apply a single per-entity appearance override field, keyed by raw id.
    ///
    /// Resolves the raw id, merges `field`/`value` into the entity's current
    /// overrides via [`DisplayOverrides::apply_json_field`], then either
    /// clears the entry (if the merge left it empty) or stores it back. An
    /// unknown field is logged and skipped. Shared by the bridge dispatch and
    /// the host command path so both edit appearance through one code path.
    pub fn apply_entity_appearance_field(
        &mut self,
        entity_id: u32,
        field: &str,
        value: &serde_json::Value,
    ) {
        let Some(eid) = self.entity_id(entity_id) else {
            return;
        };
        let mut ovr = self.entity_appearance(eid).cloned().unwrap_or_default();
        if let Err(unknown) = ovr.apply_json_field(field, value) {
            log::warn!("Unknown entity appearance field: {unknown}");
            return;
        }
        if ovr.is_empty() {
            self.clear_entity_appearance(eid);
        } else {
            self.set_entity_appearance(eid, ovr);
        }
    }

    /// Look up a per-entity appearance override.
    #[must_use]
    pub fn entity_appearance(
        &self,
        entity_id: EntityId,
    ) -> Option<&DisplayOverrides> {
        self.annotations.appearance(entity_id)
    }

    /// Resolve the drawing mode for an entity: per-entity override wins,
    /// else global (with Cartoon falling back to type-default).
    pub(crate) fn resolved_drawing_mode(
        &self,
        eid: EntityId,
        mol_type: MoleculeType,
    ) -> DrawingMode {
        self.annotations
            .resolved_drawing_mode(&self.options, eid, mol_type)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::options::ColorScheme;

    #[test]
    fn appearance_survives_reset_and_reconciles_by_retain() {
        let mut ann = EntityAnnotations::default();
        let keep_id = EntityId::from_raw(0);
        let drop_id = EntityId::from_raw(1);
        let kept = DisplayOverrides {
            color_scheme: Some(ColorScheme::BFactor),
            ..Default::default()
        };
        let _ = ann.appearance.insert(keep_id, kept.clone());
        let _ = ann.appearance.insert(drop_id, DisplayOverrides::default());

        // A per-publish topology swap routes through reset; appearance
        // must survive it (only membership reconciliation drops entries).
        ann.reset();
        assert_eq!(ann.appearance.get(&keep_id), Some(&kept));
        assert!(ann.appearance.contains_key(&drop_id));

        // retain_entities reconciles by id: the departed entity's override
        // is dropped, the surviving one's is kept.
        ann.retain_entities(|id| id == keep_id);
        assert_eq!(ann.appearance.get(&keep_id), Some(&kept));
        assert!(!ann.appearance.contains_key(&drop_id));
    }
}
