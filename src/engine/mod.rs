pub(crate) mod annotations;
mod bootstrap;
/// Structural reference types for atom-anchored constraints.
pub(crate) mod command;
pub(crate) mod constraint;
mod culling;
mod density;
pub(crate) mod density_store;
pub(crate) mod entity_view;
/// Focus state for tab cycling.
pub(crate) mod focus;
/// Pointer / scroll / modifier intake and click-expansion helpers.
mod intake;
mod options_apply;
pub(crate) mod positions;
pub(crate) mod scene;
/// Scene operations callable directly on `VisoEngine`.
mod scene_ops;
pub(crate) mod scene_state;
pub(crate) mod surface;
pub(crate) mod surface_regen;
mod sync;
pub(crate) mod trajectory;

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use annotations::EntityAnnotations;
pub(crate) use bootstrap::FrameTiming;
use density_store::DensityStore;
use focus::Focus;
use molex::entity::molecule::id::EntityId;
use molex::{Assembly, MoleculeEntity};
use scene::Scene;
use web_time::Instant;

use crate::animation::{
    build_animation, Advance, AnimationPlayer, AnimationState,
};
use crate::camera;
use crate::camera::controller::CameraController;
use crate::options::{SurfaceKindOption, VisoOptions};
use crate::renderer::GpuPipeline;

/// Quiet window the publish stream must stay silent for before the
/// molecular surface is regenerated: long enough that a continuous edit
/// (wiggle, drag) never crosses it mid-motion, short enough to re-mesh
/// promptly once the conformation comes to rest.
const SURFACE_SETTLE_WINDOW: std::time::Duration =
    std::time::Duration::from_millis(180);

/// A host-supplied void distance field stored on the engine, meshed as a
/// smooth blob on every surface regen.
///
/// `phi` is a flat row-major scalar grid that is HIGH at void centers and
/// ~0 at atom walls / exterior; the worker meshes its isosurface at
/// `threshold` directly. Pushed via
/// [`VisoEngine::set_external_void_field`].
pub(crate) struct ExternalVoidField {
    /// Grid dimensions `[nx, ny, nz]`.
    pub(crate) dims: [usize; 3],
    /// World-space origin of grid cell `(0,0,0)` (Angstroms).
    pub(crate) origin: [f32; 3],
    /// Per-axis grid spacing (Angstroms).
    pub(crate) spacing: [f32; 3],
    /// Flat row-major distance field.
    pub(crate) phi: Vec<f32>,
    /// Positive iso-level: the void-surface level to wrap.
    pub(crate) threshold: f32,
}

/// Stored constraint specifications (bands + pull + clashes), resolved to
/// world-space each frame.
pub(crate) struct ConstraintSpecs {
    /// Band constraint specs.
    pub(crate) band_specs: Vec<command::BandInfo>,
    /// Pull constraint spec.
    pub(crate) pull_spec: Option<command::PullInfo>,
    /// Steric clash arc specs.
    pub(crate) clash_specs: Vec<command::ClashInfo>,
}

/// The core rendering engine for protein visualization.
///
/// `VisoEngine` is a read-only consumer of structural state: the host
/// application owns an [`molex::Assembly`] and pushes the latest
/// snapshot via [`VisoEngine::set_assembly`]. In standalone deployments
/// ([`crate::VisoApp`]) the app plays the host role.
///
/// The engine is never mutated directly for structural changes — the
/// host mutates its own `Assembly` and re-publishes via
/// [`VisoEngine::set_assembly`]. Viso-side annotations (appearance
/// overrides, behavior overrides, camera state) are mutated through
/// engine methods directly.
pub struct VisoEngine {
    // ── GPU + camera ──────────────────────────────────────────────
    /// All GPU infrastructure (device, renderers, picking, post-process,
    /// lighting, cursor, culling state).
    pub(crate) gpu: GpuPipeline,
    /// Orbital camera controller.
    pub(crate) camera_controller: CameraController,

    // ── Runtime state ─────────────────────────────────────────────
    /// Stored band/pull constraint specs.
    pub(crate) constraints: ConstraintSpecs,
    /// Structural animation, trajectory, and pending transitions.
    pub(crate) animation: AnimationState,
    /// Runtime display, lighting, color, and geometry options.
    pub(crate) options: VisoOptions,
    /// Currently applied options preset name, if any.
    pub(crate) active_preset: Option<String>,
    /// Per-frame timing and FPS tracking.
    pub(crate) frame_timing: FrameTiming,
    /// Loaded electron density maps.
    pub(crate) density: DensityStore,
    /// Host-supplied void distance field pushed via
    /// [`VisoEngine::set_external_void_field`]; meshed as a smooth blob
    /// into the cavity stream. `None` when no field is set.
    pub(crate) external_void_field: Option<ExternalVoidField>,

    // ── Assembly ingest + derived per-entity state ────────────────
    /// Pending snapshot pushed by the host, latest applied snapshot,
    /// generation tracker, plus the per-entity render-ready derived
    /// state rebuilt on every sync (`SceneRenderState`, `EntityView`s,
    /// positions). Also owns the monotonic `mesh_version` dispenser.
    /// See [`Scene`].
    pub(crate) scene: Scene,

    // ── User-authored per-entity annotations ──────────────────────
    /// Per-entity opinions that ride alongside the Assembly: focus,
    /// visibility, behaviors, appearance overrides, scores, SS
    /// overrides, surfaces. All maps keyed on [`EntityId`] so lookups
    /// are O(1). See [`EntityAnnotations`].
    pub(crate) annotations: EntityAnnotations,

    // ── Background isosurface-mesh regeneration ───────────────────
    /// Holder for the surface-regen submit channel used by
    /// [`surface_regen::regenerate_surfaces`]. Requests run on the shared
    /// scene-processor worker; main-thread polling happens in
    /// [`GpuPipeline::apply_pending_surface`].
    pub(crate) surface_regen: surface_regen::SurfaceRegen,
    /// The `scene.last_seen_generation` the molecular surface was last
    /// regenerated against; when it lags the displayed generation the
    /// surface is stale. Inits to `u64::MAX` (matching
    /// `last_seen_generation`) so a never-built surface reads stale only
    /// once real geometry has been published.
    surface_built_for_generation: u64,
    /// Wall-clock instant of the most recent consumed publish, used to
    /// detect when the publish stream has gone quiet. A
    /// `web_time::Instant`, not a `dt` accumulator: the web frame loop
    /// feeds a fixed `dt`, so only a wall clock measures real elapsed time.
    last_publish_at: Instant,

    // ── Input state ───────────────────────────────────────────────
    /// Multi-click classifier + drag-detection state, fed by
    /// [`Self::feed_pointer_motion`] and [`Self::feed_pointer_button`].
    pub(crate) input_state: crate::input::click_state::InputState,
    /// Whether the primary mouse button is currently held.
    pub(crate) mouse_pressed: bool,
    /// Whether the shift modifier is currently held.
    pub(crate) shift_pressed: bool,
}

// ── Frame loop ──

impl VisoEngine {
    /// Per-frame updates: animation ticks, uniform uploads, frustum
    /// culling.
    fn pre_render(&mut self) {
        self.apply_pending_animation();
        self.tick_animation();

        self.camera_controller.uniform.hovered_residue =
            self.gpu.pick.hovered_target.as_residue_i32();
        self.camera_controller.uniform.time = self.frame_timing.elapsed_secs();
        self.camera_controller.update_gpu(&self.gpu.context.queue);

        let fog_start = self.camera_controller.distance();
        let fog_density =
            2.0 / self.camera_controller.bounding_radius().max(10.0);
        self.gpu.post_process.update_fog(
            &self.gpu.context.queue,
            fog_start,
            fog_density,
        );

        self.check_and_submit_lod();
        self.gpu
            .pick
            .update_selection_buffer(&self.gpu.context.queue);
        let _color_transitioning =
            self.gpu.pick.residue_colors.update(&self.gpu.context.queue);
        self.gpu.update_headlamp(&self.camera_controller.camera);
        self.update_frustum_culling();

        if !self.constraints.band_specs.is_empty()
            || self.constraints.pull_spec.is_some()
            || !self.constraints.clash_specs.is_empty()
        {
            self.resolve_and_render_constraints();
        }

        let _ = self.gpu.apply_pending_surface();
    }

    /// Tick animation (both trajectory and structural), submitting any
    /// interpolated frame to the background thread.
    fn tick_animation(&mut self) {
        let now = Instant::now();
        let trajectory_frame = self.animation.advance_trajectory(now);
        if let Some(frame) = trajectory_frame {
            self.apply_trajectory_frame(&frame);
            self.submit_animation_frame();
        }
        if self.animation.tick(now, &mut self.scene.positions) {
            self.submit_animation_frame();
        }
    }

    /// Core render — geometry, post-process, picking — targeting the
    /// given view. Returns the encoder so the caller can submit it.
    fn render_to_view(
        &mut self,
        view: &wgpu::TextureView,
    ) -> wgpu::CommandEncoder {
        self.gpu.render_to_view(view, &self.camera_controller)
    }

    /// Execute one frame: update animations, run the geometry pass,
    /// post-process, and present.
    ///
    /// # Errors
    ///
    /// Returns [`wgpu::SurfaceError`] if the swapchain frame cannot be
    /// acquired.
    pub fn render(&mut self) -> Result<(), wgpu::SurfaceError> {
        if !self.frame_timing.should_render() {
            return Ok(());
        }

        self.pre_render();

        let frame = self.gpu.context.get_next_frame()?;
        let view = frame
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        let encoder = self.render_to_view(&view);
        self.gpu.context.submit(encoder);

        self.gpu.pick.picking.start_readback();
        self.gpu.pick.poll_and_resolve(&self.gpu.context.device);

        frame.present();
        self.frame_timing.end_frame();

        Ok(())
    }

    /// Render the scene to the given texture view (for embedding in
    /// dioxus/etc). The caller owns the texture — no surface present
    /// happens.
    pub fn render_to_texture(&mut self, view: &wgpu::TextureView) {
        self.pre_render();
        let encoder = self.render_to_view(view);
        self.gpu.context.submit(encoder);
        self.frame_timing.end_frame();
    }

    /// Resize all GPU surfaces and the camera projection to match the
    /// new window size.
    pub fn resize(&mut self, width: u32, height: u32) {
        if width > 0 && height > 0 {
            self.gpu.resize(width, height);
            self.camera_controller.resize(width, height);
        }
    }
}

// ── Lifecycle + queries ──

impl VisoEngine {
    /// Advance camera animation and apply any pending scene from the
    /// background processor.
    pub fn update(&mut self, dt: f32) {
        let _ = self.camera_controller.update_animation(dt);
        let now = Instant::now();

        // A newer publish arriving mid-play coalesces to the latest target:
        // re-aim while still collapsing/easing (or on a plain ease) by
        // dropping the player so it rebuilds toward the latest below; but
        // let an in-flight expand finish before starting fresh.
        if self.has_new_pending()
            && self
                .animation
                .player
                .as_ref()
                .is_some_and(|p| !p.past_adopt())
        {
            self.animation.player = None;
        }

        // Begin a new sequence from a freshly published snapshot.
        if self.animation.player.is_none() {
            self.begin_pending_animation();
        }

        // Drive the in-flight sequence from the single call site.
        if let Some(mut player) = self.animation.player.take() {
            match player.advance(&mut self.animation.animator, &self.scene, now)
            {
                Advance::Running => self.animation.player = Some(player),
                Advance::Adopt(b) => {
                    // Adopt B via the normal rebuild path, then re-collapse
                    // the mutated sidechains so the expand grows them out.
                    self.scene.pending = Some(b);
                    let _ = self.poll_assembly();
                    player.seed_adopted(&mut self.scene);
                    let transitions = self.animation.take_pending_transitions();
                    self.sync_scene_to_renderers(transitions);
                    self.animation.player = Some(player);
                }
                Advance::Done => {}
            }
        }

        self.apply_pending_scene();
        self.maybe_settle_surface();
    }

    /// Whether a snapshot with an unseen generation is waiting.
    fn has_new_pending(&self) -> bool {
        self.scene
            .pending
            .as_ref()
            .is_some_and(|p| p.generation() != self.scene.last_seen_generation)
    }

    /// Build and start an animation from the pending snapshot, if one is
    /// waiting. A same-topology change adopts B up front (a same-length
    /// adopt keeps the current positions) so the player's ease animates the
    /// kept positions toward B's coords; a topology-changing mutation defers
    /// adoption to the `AdoptTarget` waypoint and leaves the scene on A until
    /// then. `None` from the builder (a residue insert/delete, or no
    /// movement) snaps via the normal adopt.
    fn begin_pending_animation(&mut self) {
        if !self.has_new_pending() {
            return;
        }
        let Some(pending) = self.scene.pending.clone() else {
            return;
        };
        match build_animation(self.scene.current.as_ref(), &pending) {
            None => {
                // Snap: adopt + rebuild via the normal path.
                if self.poll_assembly() {
                    let transitions = self.animation.take_pending_transitions();
                    self.sync_scene_to_renderers(transitions);
                }
            }
            Some(animation) => {
                let player = AnimationPlayer::new(animation);
                if player.has_adopt() {
                    // Defer adoption: the player holds B in its AdoptTarget
                    // step; consume the pending slot without adopting.
                    self.scene.pending = None;
                } else if self.poll_assembly() {
                    // Same topology: adopt up front (positions kept), then
                    // the Lerp animates the kept positions toward B.
                    let transitions = self.animation.take_pending_transitions();
                    self.sync_scene_to_renderers(transitions);
                }
                self.animation.player = Some(player);
            }
        }
    }

    /// Push a new [`Assembly`] snapshot from the host. The next
    /// `update` (or `sync_now`) tick will rederive viso-side state
    /// and submit mesh generation.
    pub fn set_assembly(&mut self, assembly: Arc<Assembly>) {
        self.scene.pending = Some(assembly);
    }

    /// Atomic topology swap: tear down scene-local state (animation,
    /// surfaces, derived per-entity views), stage the new snapshot,
    /// and force a sync so subsequent calls (`set_ss_override`,
    /// camera pose, etc.) operate against synced state. Use this for
    /// puzzle/file reloads — `set_assembly` alone leaves stale state
    /// from the previous topology around until the next `update`.
    pub fn replace_assembly(&mut self, assembly: Arc<Assembly>) {
        self.reset_scene_local_state();
        self.scene.pending = Some(assembly);
        self.sync_now();
        // `sync_now` advanced `last_seen_generation` to the new snapshot;
        // the reset already regenerated the surface against this scene, so
        // re-align the marker to the synced generation to stop a redundant
        // rebuild right after the swap.
        self.surface_built_for_generation = self.scene.last_seen_generation;
    }

    /// Combined centroid of every visible entity in the synced scene,
    /// weighted by atom count. `None` if the scene is empty. Use this
    /// to anchor a camera pose on the molecule rather than on a saved
    /// look-at target.
    pub fn focus_centroid(&self) -> Option<glam::Vec3> {
        let visible: Vec<&MoleculeEntity> = self
            .scene
            .current
            .entities()
            .iter()
            .filter(|e| self.is_entity_visible(e.id().raw()))
            .map(Arc::as_ref)
            .collect();
        camera::fit::combined_bounding_sphere(visible).map(|(c, _)| c)
    }

    /// Snap (non-animated) version of [`Self::fit_camera_to_focus`].
    /// Sets `focus_point`, orbit `distance`, and `bounding_radius`
    /// instantly to the molecule's bounding sphere — needed when a
    /// caller follows up with a manual `set_camera_pose` and would
    /// otherwise leave `bounding_radius` (the fog driver) tied to the
    /// previous topology.
    pub fn snap_camera_to_focus(&mut self) {
        let visible: Vec<&MoleculeEntity> = self
            .scene
            .current
            .entities()
            .iter()
            .filter(|e| self.is_entity_visible(e.id().raw()))
            .map(Arc::as_ref)
            .collect();
        if let Some((centroid, radius)) =
            camera::fit::combined_bounding_sphere(visible)
        {
            self.camera_controller.fit_to_sphere(centroid, radius);
        }
    }

    /// Drain any pending Assembly snapshot and, if its generation
    /// differs from the last applied one, rederive viso-side state.
    /// Returns `true` if a new generation was consumed (caller should
    /// follow up with mesh-rebuild work); `false` if there was nothing
    /// to apply.
    fn poll_assembly(&mut self) -> bool {
        let Some(assembly) = self.scene.pending.take() else {
            return false;
        };
        if assembly.generation() == self.scene.last_seen_generation {
            return false;
        }
        self.sync_from_assembly(&assembly);
        self.scene.current = assembly;
        self.scene.last_seen_generation = self.scene.current.generation();
        // Restart the rest-detection clock: each consumed publish pushes
        // the quiet window out, so a continuous stream never crosses it and
        // the surface only re-meshes once motion stops.
        self.last_publish_at = Instant::now();
        true
    }

    /// Whether any molecular surface could currently be shown. Errs toward
    /// `true`: a false positive runs one harmless gather that yields no
    /// jobs, while a false negative would leave a real surface frozen.
    fn surface_active(&self) -> bool {
        let display = &self.options.display;
        if display.surface_kind() != SurfaceKindOption::None
            || display.show_cavities()
            || !self.annotations.surfaces.is_empty()
        {
            return true;
        }
        // A per-entity appearance override can turn a surface on even when
        // the global display has none.
        self.annotations.appearance.values().any(|ovr| {
            matches!(
                ovr.surface_kind,
                Some(SurfaceKindOption::Gaussian | SurfaceKindOption::Ses)
            ) || ovr.show_cavities == Some(true)
        })
    }

    /// Regenerate the molecular surface once the conformation has come to
    /// rest. Called once per `update` tick after the latest publish has
    /// been consumed; the expensive marching-cubes re-mesh runs at most
    /// once per rest, never per wiggle/drag frame (each publish restarts
    /// the quiet window).
    fn maybe_settle_surface(&mut self) {
        // Decide before borrowing fields so the `&self` read and the later
        // `&`-field reads plus marker write don't overlap.
        let settle = surface_regen::should_settle_surface(
            self.surface_active(),
            self.scene.last_seen_generation,
            self.surface_built_for_generation,
            self.last_publish_at.elapsed(),
            SURFACE_SETTLE_WINDOW,
        );
        if settle {
            // At rest the scene's reference coords (`se.positions()`, read
            // by `regenerate_surfaces`) equal the displayed coords, so the
            // re-meshed surface matches what is on screen.
            surface_regen::regenerate_surfaces(
                &self.scene,
                &self.annotations,
                &self.density,
                self.external_void_field.as_ref(),
                &self.options,
                &self.surface_regen,
            );
            self.surface_built_for_generation = self.scene.last_seen_generation;
        }
    }

    /// Stop the background scene processor thread.
    pub fn shutdown(&mut self) {
        self.gpu.shutdown();
    }

    /// Load a DCD trajectory file and begin playback against the first
    /// visible protein entity.
    pub fn load_trajectory(&mut self, path: &std::path::Path) {
        self.animation.load_trajectory_from_path(
            path,
            &self.scene,
            &self.annotations,
        );
    }

    /// Position the camera explicitly from world-space center / eye / up.
    /// Used by puzzle loaders to apply a saved viewpoint.
    pub fn set_camera_pose(
        &mut self,
        center: glam::Vec3,
        eye: glam::Vec3,
        up: glam::Vec3,
    ) {
        self.camera_controller.set_pose(center, eye, up);
    }

    /// Fit the camera to the currently focused element (all-entities
    /// bounding sphere, or the focused entity's bounding sphere).
    pub fn fit_camera_to_focus(&mut self) {
        match self.annotations.focus {
            Focus::All => self.fit_all_camera(),
            Focus::Entity(eid) => {
                if let Some(entity) =
                    self.scene.current.entities().iter().find(|e| e.id() == eid)
                {
                    camera::fit::fit_to_entity(
                        &mut self.camera_controller,
                        entity.as_ref(),
                    );
                }
            }
        }
    }

    /// Fit the camera to the combined bounding sphere of every visible
    /// entity.
    pub(crate) fn fit_all_camera(&mut self) {
        let visible: Vec<&MoleculeEntity> = self
            .scene
            .current
            .entities()
            .iter()
            .filter(|e| self.is_entity_visible(e.id().raw()))
            .map(Arc::as_ref)
            .collect();
        camera::fit::fit_to_entities(&mut self.camera_controller, visible);
    }

    /// Reset all scene-local state (animation, scene ingest, derived
    /// per-entity views, annotations). Called when replacing or
    /// clearing the scene.
    ///
    /// Also resets `last_seen_generation` to `u64::MAX` so that the
    /// next Assembly snapshot triggers a sync unconditionally — the
    /// app-side replace path rebuilds the assembly in one shot via
    /// `Assembly::new(...)`, which starts at generation 0 and would
    /// otherwise collide with a previously-observed
    /// `last_seen_generation` of 0.
    pub(crate) fn reset_scene_local_state(&mut self) {
        self.animation = AnimationState::new();
        self.scene.reset_local_state();
        self.annotations.reset();
        surface_regen::regenerate_surfaces(
            &self.scene,
            &self.annotations,
            &self.density,
            self.external_void_field.as_ref(),
            &self.options,
            &self.surface_regen,
        );
        // Surface just regenerated against the freshly-reset scene; align
        // the marker with the reset generation so the gate stays quiet
        // until the next real publish advances the displayed generation.
        self.surface_built_for_generation = self.scene.last_seen_generation;
    }

    /// Look up the opaque [`EntityId`] for a raw `u32` id. Returns
    /// `None` if no entity with that raw id exists. The boundary
    /// translator: callers arriving from a wire format (IPC, TOML,
    /// CLI) translate *once* here and then pass [`EntityId`] to the
    /// per-entity engine methods.
    #[must_use]
    pub fn entity_id(&self, raw: u32) -> Option<EntityId> {
        self.scene.entity_id(raw)
    }

    /// The pick target currently under the cursor (resolved from the
    /// previous frame's GPU picking pass).
    pub fn hovered_target(&self) -> crate::renderer::picking::PickTarget {
        self.gpu.pick.hovered_target
    }

    /// The currently focused entity ID, or `None` when focus is `Focus::All`.
    #[must_use]
    pub fn focused_entity(&self) -> Option<EntityId> {
        match self.annotations.focus {
            Focus::Entity(id) => Some(id),
            Focus::All => None,
        }
    }

    /// Current smoothed frames-per-second.
    #[must_use]
    pub fn fps(&self) -> f32 {
        self.frame_timing.fps()
    }

    /// Read-only access to the current options.
    #[must_use]
    pub fn options(&self) -> &VisoOptions {
        &self.options
    }

    /// Name of the currently active preset, if any.
    #[must_use]
    pub fn active_preset(&self) -> Option<&str> {
        self.active_preset.as_deref()
    }

    /// Whether a trajectory is loaded.
    #[must_use]
    pub fn has_trajectory(&self) -> bool {
        self.animation.trajectory_player.is_some()
    }

    /// Current focus state.
    #[must_use]
    pub fn focus(&self) -> Focus {
        self.annotations.focus
    }

    /// Resolve an atom position from structural references, using
    /// interpolated visual positions during animation.
    #[must_use]
    pub fn resolve_atom_position(
        &self,
        residue: u32,
        atom_name: &str,
    ) -> Option<glam::Vec3> {
        constraint::resolve_atom_ref_pub(
            &self.scene,
            &self.annotations,
            &command::AtomRef {
                residue,
                atom_name: atom_name.to_owned(),
            },
        )
    }

    /// Find the heavy-atom in `residue` whose current world position
    /// projects closest to `screen_pos` (pixels, origin top-left).
    /// Returns the PDB atom name, or `None` if the residue can't be
    /// resolved or has no heavy atoms.
    #[must_use]
    pub fn closest_atom_in_residue(
        &self,
        residue: u32,
        screen_pos: (f32, f32),
    ) -> Option<String> {
        constraint::closest_atom_in_residue(
            &self.scene,
            &self.annotations,
            &self.camera_controller,
            self.viewport_size(),
            residue,
            glam::Vec2::new(screen_pos.0, screen_pos.1),
        )
    }

    /// Full breakdown of a cartoon-residue pick: owning entity id +
    /// entity-local residue index + PDB atom name of the heavy atom
    /// projecting closest to `screen_pos`. The host uses this to
    /// classify the pick (backbone vs sidechain, protein vs other)
    /// and build a pull-op dispatch in one pass.
    #[must_use]
    pub fn picked_residue_atom(
        &self,
        flat_residue: u32,
        screen_pos: (f32, f32),
    ) -> Option<constraint::PickedResidueAtom> {
        constraint::picked_residue_atom(
            &self.scene,
            &self.annotations,
            &self.camera_controller,
            self.viewport_size(),
            flat_residue,
            glam::Vec2::new(screen_pos.0, screen_pos.1),
        )
    }

    /// Number of entities currently in the scene.
    #[must_use]
    pub fn entity_count(&self) -> usize {
        self.scene.current.entities().len()
    }

    /// Read-only access to the last `Assembly` snapshot applied to
    /// viso-side state.
    #[must_use]
    pub fn assembly(&self) -> &Assembly {
        self.scene.current.as_ref()
    }

    /// Current viewport dimensions in physical pixels.
    #[must_use]
    pub fn viewport_size(&self) -> glam::UVec2 {
        glam::UVec2::new(
            self.gpu.context.config.width,
            self.gpu.context.config.height,
        )
    }

    /// Project screen coordinates onto a plane parallel to the camera
    /// at the depth of `world_point`. Useful for drag-anchor math
    /// (e.g. translating cursor motion into world-space delta on the
    /// camera plane through a clicked atom).
    #[must_use]
    pub fn screen_to_world_at_depth(
        &self,
        screen_pos: glam::Vec2,
        world_point: glam::Vec3,
    ) -> glam::Vec3 {
        self.camera_controller.screen_to_world_at_depth(
            screen_pos,
            self.viewport_size(),
            world_point,
        )
    }

    /// Update the cursor position for GPU picking.
    pub fn set_cursor_pos(&mut self, x: f32, y: f32) {
        self.gpu.cursor_pos = (x, y);
    }

    /// Replace the residue selection. `selection` is the per-entity
    /// authoritative selection (the same shape foldit-core's
    /// `App.selection` holds). viso stores it as the source of truth and
    /// re-derives the flat GPU bitset from its own always-current
    /// per-entity residue offsets, both here and on every mesh rebuild, so
    /// the highlight can never go stale relative to a shifting residue
    /// space.
    pub fn set_selection(
        &mut self,
        selection: &BTreeMap<EntityId, BTreeSet<u32>>,
    ) {
        self.gpu.pick.set_selection(selection.clone());
        self.gpu
            .pick
            .update_selection_buffer(&self.gpu.context.queue);
    }

    /// Project a world-space point to screen coordinates (pixels,
    /// origin top-left). Returns `None` if the point is at or behind
    /// the camera.
    #[must_use]
    pub fn world_to_screen(&self, world: glam::Vec3) -> Option<glam::Vec2> {
        self.camera_controller
            .world_to_screen(world, self.viewport_size())
    }
}
