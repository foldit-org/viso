//! Engine-side methods that propagate option / preset changes through
//! the GPU, scene, and overlay sub-structs.
//!
//! Every method here takes `&mut self` on [`VisoEngine`] and touches
//! multiple sibling sub-structs (`gpu`, `scene`, `annotations`,
//! `camera_controller`) — the code can't live in `options/` because
//! that would invert the dependency direction (options is a leaf
//! module, engine depends on it).
//!
//! ## Dispatch model
//!
//! `set_options` computes two diff products:
//!
//! - [`GlobalsChange`] — bools for concerns that only exist at global scope
//!   (lighting, camera, present mode, type-level visibility, etc.)
//! - [`crate::options::overrides::RenderInvalidation`] — per-field projection
//!   from a [`crate::options::DisplayOverrides::diff`] onto classes of GPU work
//!   (mesh/color/surface/LOD/drawing-mode-resolve)
//!
//! The dispatcher fires each `RenderInvalidation` flag at most once per
//! call — dedup is a structural property of the type, not a runtime
//! discipline. Fixes the historical triple-sync bug (same invalidation
//! reached from three different `OptionsChange` bools at once).

use std::collections::HashMap;
#[cfg(not(target_arch = "wasm32"))]
use std::path::Path;

use molex::entity::molecule::id::EntityId;
use molex::MoleculeType;

use super::VisoEngine;
use crate::options::overrides::RenderInvalidation;
use crate::options::VisoOptions;

/// Global-only diff. Covers every section of [`VisoOptions`] whose
/// changes cannot be driven by per-entity overrides.
///
/// The overridable subset of `display` is diffed separately via
/// [`crate::options::DisplayOverrides::diff`] into a
/// [`RenderInvalidation`] flag set.
#[derive(Debug, Default, Clone, Copy)]
#[allow(clippy::struct_excessive_bools)]
pub(crate) struct GlobalsChange {
    /// Lighting uniform (`options.lighting`) changed.
    pub(crate) lighting: bool,
    /// Post-processing stack (`options.post_processing`) changed.
    pub(crate) post_processing: bool,
    /// Camera options (`options.camera`) changed.
    pub(crate) camera: bool,
    /// Debug overlay options (`options.debug`) changed.
    pub(crate) debug: bool,
    /// Surface present mode changed (requires swapchain reconfigure).
    pub(crate) present_mode: bool,
    /// Geometry tuning section changed.
    pub(crate) geometry: bool,
    /// Colors section changed.
    pub(crate) colors: bool,
    /// Water visibility toggled.
    pub(crate) waters: bool,
    /// Ion visibility toggled.
    pub(crate) ions: bool,
    /// Solvent visibility toggled.
    pub(crate) solvent: bool,
    /// Structural bond options (`options.display.bonds`) changed.
    pub(crate) bonds: bool,
}

impl GlobalsChange {
    /// Compute the global-only diff between two options.
    pub(crate) fn diff(old: &VisoOptions, new: &VisoOptions) -> Self {
        Self {
            lighting: old.lighting != new.lighting,
            post_processing: old.post_processing != new.post_processing,
            camera: old.camera != new.camera,
            debug: old.debug != new.debug,
            present_mode: old.display.present_mode != new.display.present_mode,
            geometry: old.geometry != new.geometry,
            colors: old.colors != new.colors,
            waters: old.display.show_waters != new.display.show_waters,
            ions: old.display.show_ions != new.display.show_ions,
            solvent: old.display.show_solvent != new.display.show_solvent,
            bonds: old.display.bonds != new.display.bonds,
        }
    }

    /// True if any global concern changed.
    pub(crate) fn any(&self) -> bool {
        // Exhaustive destructuring — adding a field forces a compile
        // error here until `any()` is updated to include it.
        let Self {
            lighting,
            post_processing,
            camera,
            debug,
            present_mode,
            geometry,
            colors,
            waters,
            ions,
            solvent,
            bonds,
        } = *self;
        lighting
            || post_processing
            || camera
            || debug
            || present_mode
            || geometry
            || colors
            || waters
            || ions
            || solvent
            || bonds
    }
}

/// Classes of mesh/render work invalidated by a `GeometryOptions` change.
///
/// Separate from the override-driven `RenderInvalidation` because
/// `GeometryOptions` is a global-only concern (LOD parameters, not
/// overridable per-entity) — its invalidation is always global scope.
fn geometry_invalidation() -> RenderInvalidation {
    RenderInvalidation::RE_MESH | RenderInvalidation::LOD_REMESH
}

/// Classes of mesh work invalidated by a `ColorOptions` change.
fn colors_invalidation() -> RenderInvalidation {
    RenderInvalidation::RE_MESH | RenderInvalidation::RE_COLOR
}

impl VisoEngine {
    /// Push lighting options to the GPU uniform.
    pub(super) fn apply_lighting(&mut self) {
        self.gpu.apply_lighting(&self.options.lighting);
    }

    /// Push post-processing options to the composite pass.
    pub(super) fn apply_post_processing(&mut self) {
        self.gpu.apply_post_processing(&self.options);
    }

    /// Push camera options to the controller.
    pub(super) fn apply_camera(&mut self) {
        self.camera_controller
            .apply_camera_options(&self.options.camera);
    }

    /// Push debug options to the camera uniform.
    pub(super) fn apply_debug(&mut self) {
        self.camera_controller
            .apply_debug_options(&self.options.debug, &self.gpu.context.queue);
    }

    /// Replace options and apply only the sections that changed.
    pub fn set_options(&mut self, new: VisoOptions) {
        let globals = GlobalsChange::diff(&self.options, &new);
        // Override-field diff: per-field mapping onto invalidation classes.
        let mut inv =
            self.options.display.overrides.diff(&new.display.overrides);
        if globals.geometry {
            inv |= geometry_invalidation();
        }
        if globals.colors {
            inv |= colors_invalidation();
        }
        if !globals.any() && inv.is_empty() {
            return;
        }
        self.options = new;
        self.apply_global_invalidation(globals, inv);
    }

    /// Dispatch one action per [`GlobalsChange`] flag and per
    /// [`RenderInvalidation`] bit. Each invalidation fires at most once
    /// — the bitflag union is the dedup mechanism.
    ///
    /// Ordering: `DRAWING_MODE_RESOLVE` must precede `RE_MESH` so the
    /// subsequent scene sync picks up the newly-resolved drawing mode.
    /// Enforced by arm order in this function body.
    fn apply_global_invalidation(
        &mut self,
        globals: GlobalsChange,
        inv: RenderInvalidation,
    ) {
        // --- Global-only concerns ---
        if globals.waters {
            self.apply_type_visibility(
                MoleculeType::Water,
                self.options.display.show_waters,
            );
        }
        if globals.ions {
            self.apply_type_visibility(
                MoleculeType::Ion,
                self.options.display.show_ions,
            );
        }
        if globals.solvent {
            self.apply_type_visibility(
                MoleculeType::Solvent,
                self.options.display.show_solvent,
            );
        }
        if globals.lighting {
            self.apply_lighting();
        }
        if globals.post_processing {
            self.apply_post_processing();
        }
        if globals.camera {
            self.apply_camera();
        }
        if globals.debug {
            self.apply_debug();
        }
        if globals.present_mode {
            self.gpu
                .context
                .set_present_mode(self.options.display.present_mode.to_wgpu());
        }

        // --- Override-driven invalidations. Each flag fires at most once. ---
        if inv.contains(RenderInvalidation::DRAWING_MODE_RESOLVE) {
            self.reresolve_drawing_modes();
        }
        if inv.contains(RenderInvalidation::RE_MESH) {
            self.invalidate_all_mesh_versions();
        }
        if inv.contains(RenderInvalidation::RE_COLOR) {
            self.recompute_backbone_colors();
        }
        if inv.contains(RenderInvalidation::LOD_REMESH) {
            let camera_eye = self.camera_controller.camera.eye;
            self.submit_per_chain_lod_remesh(camera_eye);
        }
        if inv.contains(RenderInvalidation::RE_SURFACE) {
            super::surface_regen::regenerate_surfaces(
                &self.scene,
                &self.annotations,
                &self.density,
                self.external_void_field.as_ref(),
                &self.options,
                &self.surface_regen,
            );
        }
        if inv.contains(RenderInvalidation::RE_SURFACE_OPACITY) {
            // Global surface opacity is a shader uniform, so a slider drag
            // is a cheap queue write with no mesh regeneration.
            self.gpu
                .set_surface_opacity(self.options.display.surface_opacity());
        }
        // Single final sync for geometry and structural-bond changes. A
        // color-only invalidation already uploaded its result in the
        // `RE_COLOR` arm above (`recompute_backbone_colors` writes the
        // per-residue color buffer directly), so it must not submit a
        // scene rebuild. A `display.bonds` change has no
        // `RenderInvalidation` flag of its own — the full rebuild here
        // re-resolves and re-uploads structural bonds with the new flags.
        // Folding it into this single condition keeps a combined
        // mesh+bonds change to one sync.
        if inv.contains(RenderInvalidation::RE_MESH) || globals.bonds {
            self.sync_scene_to_renderers(HashMap::new());
        }
    }

    /// Dispatch [`RenderInvalidation`] flags at per-entity scope. Called
    /// from `set_entity_appearance` / `clear_entity_appearance` after
    /// diffing the entity's old vs. new overrides.
    ///
    /// Shares the `RenderInvalidation` vocabulary with the global
    /// dispatcher but elides global-only concerns (lighting, camera,
    /// visibility toggles, etc.) — those are unreachable at per-entity
    /// scope.
    ///
    /// Fires `RE_SURFACE` for per-entity surface changes, which was
    /// previously a bug: a per-entity `surface_kind` change never
    /// triggered surface regeneration because the only path to
    /// `regenerate_surfaces` went through the global `change.surface`
    /// bool.
    pub(crate) fn apply_entity_invalidation(
        &mut self,
        inv: RenderInvalidation,
    ) {
        if inv.is_empty() {
            return;
        }
        // No DRAWING_MODE_RESOLVE arm — `set_appearance` /
        // `clear_appearance` already re-resolve the single entity's
        // `state.drawing_mode` before the mesh regen. That matches the
        // ordering invariant enforced globally (resolve-before-mesh).
        if inv.contains(RenderInvalidation::RE_COLOR) {
            self.recompute_backbone_colors();
        }
        if inv.contains(RenderInvalidation::LOD_REMESH) {
            let camera_eye = self.camera_controller.camera.eye;
            self.submit_per_chain_lod_remesh(camera_eye);
        }
        // Per-entity opacity is baked into vertex alpha, so a per-entity
        // opacity change re-meshes just like a kind/cavities change.
        if inv.contains(RenderInvalidation::RE_SURFACE)
            || inv.contains(RenderInvalidation::RE_SURFACE_OPACITY)
        {
            super::surface_regen::regenerate_surfaces(
                &self.scene,
                &self.annotations,
                &self.density,
                self.external_void_field.as_ref(),
                &self.options,
                &self.surface_regen,
            );
        }
        // Geometry changes need a scene resubmit; a color-only change
        // already uploaded its colors in the `RE_COLOR` arm above and
        // must not trigger a full rebuild.
        if inv.contains(RenderInvalidation::RE_MESH) {
            self.sync_scene_to_renderers(HashMap::new());
        }
    }

    /// Force-refresh all subsystems from current options (escape hatch).
    pub fn apply_options(&mut self) {
        self.gpu
            .context
            .set_present_mode(self.options.display.present_mode.to_wgpu());
        self.apply_lighting();
        self.apply_post_processing();
        self.apply_camera();
        self.apply_debug();
        self.refresh_ball_and_stick();

        let camera_eye = self.camera_controller.camera.eye;
        self.submit_per_chain_lod_remesh(camera_eye);
        self.recompute_backbone_colors();
    }

    /// Load a named view preset from the presets directory.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn load_preset(&mut self, name: &str, presets_dir: &Path) -> bool {
        let path = presets_dir.join(format!("{name}.toml"));
        match VisoOptions::load(&path) {
            Ok(opts) => {
                log::info!("Loaded view preset '{name}'");
                self.set_options(opts);
                self.active_preset = Some(name.to_owned());
                true
            }
            Err(e) => {
                log::error!("Failed to load view preset '{name}': {e}");
                false
            }
        }
    }

    /// Save the current options as a named view preset.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn save_preset(&mut self, name: &str, presets_dir: &Path) -> bool {
        let path = presets_dir.join(format!("{name}.toml"));
        match self.options.save(&path) {
            Ok(()) => {
                log::info!("Saved view preset '{name}'");
                self.active_preset = Some(name.to_owned());
                true
            }
            Err(e) => {
                log::error!("Failed to save view preset '{name}': {e}");
                false
            }
        }
    }

    /// Set the GPU render scale (supersampling factor).
    pub fn set_render_scale(&mut self, scale: u32) {
        self.gpu.context.render_scale = scale;
    }

    /// Update render scale from a DPI scale factor and resize if needed.
    pub fn set_surface_scale(&mut self, scale: f64) {
        if self.gpu.context.set_surface_scale(scale) {
            let w = self.gpu.context.config.width;
            let h = self.gpu.context.config.height;
            self.gpu.resize(w, h);
        }
    }

    /// Bump every entity's `mesh_version` so the next sync regenerates
    /// all meshes.
    fn invalidate_all_mesh_versions(&mut self) {
        let ids: Vec<EntityId> =
            self.scene.entity_state.keys().copied().collect();
        for id in ids {
            let v = self.scene.bump_mesh_version();
            if let Some(state) = self.scene.entity_state.get_mut(&id) {
                state.mesh_version = v;
            }
        }
    }

    /// Re-resolve every entity's `drawing_mode` against the current
    /// global option + per-entity appearance overrides. Needed when the
    /// global `display.drawing_mode` changes between Assembly syncs —
    /// `sync_from_assembly` only runs on generation bumps, so without
    /// this the per-entity `state.drawing_mode` stays stale.
    fn reresolve_drawing_modes(&mut self) {
        let pairs: Vec<(EntityId, MoleculeType)> = self
            .scene
            .entity_state
            .iter()
            .map(|(id, state)| (*id, state.topology.molecule_type))
            .collect();
        for (id, mol_type) in pairs {
            let resolved = self.resolved_drawing_mode(id, mol_type);
            if let Some(state) = self.scene.entity_state.get_mut(&id) {
                state.drawing_mode = resolved;
            }
        }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::options::PresentMode;

    /// Direct regression guard: a `display.bonds` toggle must register as
    /// a global change. Before bonds were added to `GlobalsChange`, this
    /// diff reported `any() == false` and `set_options` silently dropped
    /// the new value.
    #[test]
    fn globals_change_detects_bonds() {
        let a = VisoOptions::default();
        let mut b = a.clone();
        b.display.bonds.hydrogen_bonds.visible = true;
        assert!(GlobalsChange::diff(&a, &b).any());
    }

    /// Coverage guard against future "silently dropped control"
    /// regressions: every `GlobalsChange`-owned category must surface a
    /// single-field mutation as a change. Override-driven `display`
    /// fields are diffed separately via `overrides.diff`, so they are not
    /// asserted here.
    #[test]
    fn globals_change_detects_each_category() {
        let base = VisoOptions::default();

        let mut lighting = base.clone();
        lighting.lighting.light1_intensity += 1.0;
        assert!(
            GlobalsChange::diff(&base, &lighting).any(),
            "lighting change not detected"
        );

        let mut post = base.clone();
        post.post_processing.outline_thickness += 1.0;
        assert!(
            GlobalsChange::diff(&base, &post).any(),
            "post_processing change not detected"
        );

        let mut camera = base.clone();
        camera.camera.fovy += 1.0;
        assert!(
            GlobalsChange::diff(&base, &camera).any(),
            "camera change not detected"
        );

        let mut debug = base.clone();
        debug.debug.show_normals = !debug.debug.show_normals;
        assert!(
            GlobalsChange::diff(&base, &debug).any(),
            "debug change not detected"
        );

        let mut geometry = base.clone();
        geometry.geometry.sheet_arrows = !geometry.geometry.sheet_arrows;
        assert!(
            GlobalsChange::diff(&base, &geometry).any(),
            "geometry change not detected"
        );

        let mut colors = base.clone();
        colors.colors.lipid_carbon_tint[0] += 0.5;
        assert!(
            GlobalsChange::diff(&base, &colors).any(),
            "colors change not detected"
        );

        let mut present_mode = base.clone();
        present_mode.display.present_mode = PresentMode::Immediate;
        assert!(
            GlobalsChange::diff(&base, &present_mode).any(),
            "present_mode change not detected"
        );

        let mut waters = base.clone();
        waters.display.show_waters = !waters.display.show_waters;
        assert!(
            GlobalsChange::diff(&base, &waters).any(),
            "show_waters change not detected"
        );

        let mut ions = base.clone();
        ions.display.show_ions = !ions.display.show_ions;
        assert!(
            GlobalsChange::diff(&base, &ions).any(),
            "show_ions change not detected"
        );

        let mut solvent = base.clone();
        solvent.display.show_solvent = !solvent.display.show_solvent;
        assert!(
            GlobalsChange::diff(&base, &solvent).any(),
            "show_solvent change not detected"
        );

        let mut bonds = base.clone();
        bonds.display.bonds.hydrogen_bonds.visible = true;
        assert!(
            GlobalsChange::diff(&base, &bonds).any(),
            "bonds change not detected"
        );
    }
}
