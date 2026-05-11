pub(crate) mod annotations;
mod bootstrap;
/// The engine's complete interactive vocabulary.
pub(crate) mod command;
mod constraint;
mod culling;
mod density;
pub(crate) mod density_store;
pub(crate) mod entity_view;
/// Focus state for tab cycling.
pub(crate) mod focus;
mod options_apply;
pub(crate) mod positions;
pub(crate) mod scene;
pub(crate) mod scene_state;
pub(crate) mod surface;
pub(crate) mod surface_regen;
mod sync;
pub(crate) mod trajectory;

use std::collections::HashMap;
use std::sync::Arc;

use annotations::EntityAnnotations;
pub(crate) use bootstrap::FrameTiming;
use density_store::DensityStore;
use focus::Focus;
use molex::entity::molecule::id::EntityId;
use molex::{Assembly, MoleculeEntity, MoleculeType};
use scene::Scene;
use web_time::Instant;

use crate::animation::AnimationState;
use crate::camera;
use crate::camera::controller::CameraController;
use crate::options::VisoOptions;
use crate::renderer::GpuPipeline;
use crate::VisoError;

/// Stored constraint specifications (bands + pull), resolved to world-space
/// each frame.
pub(crate) struct ConstraintSpecs {
    /// Band constraint specs.
    pub(crate) band_specs: Vec<command::BandInfo>,
    /// Pull constraint spec.
    pub(crate) pull_spec: Option<command::PullInfo>,
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
    /// Owner of the worker→main channel sender used by
    /// [`surface_regen::regenerate_surfaces`]. The matching receiver
    /// lives on [`GpuPipeline`]; main-thread polling happens in
    /// [`GpuPipeline::apply_pending_density_mesh`].
    pub(crate) surface_regen: surface_regen::SurfaceRegen,
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
        {
            self.resolve_and_render_constraints();
        }

        let _ = self.gpu.apply_pending_density_mesh();
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
        self.gpu.render_to_view(
            view,
            &self.camera_controller,
            self.options.display.show_sidechains(),
        )
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

    /// Upload or clear an Adobe ASCII `.cube` 3D LUT on the GPU (`Rgba16Float`).
    ///
    /// Rendering is unchanged until post-process shaders sample the texture (PR3).
    ///
    /// # Errors
    ///
    /// Returns [`VisoError::GpuResource`] if `lut.size` exceeds
    /// [`wgpu::Limits::max_texture_dimension_3d`].
    pub fn set_adobe_cube_lut(
        &mut self,
        lut: Option<crate::LutRgbCube3d>,
    ) -> Result<(), VisoError> {
        self.gpu.set_adobe_cube_lut(lut)
    }
}

// ── Command dispatch ──

impl VisoEngine {
    /// Execute a single [`command::VisoCommand`].
    ///
    /// Returns a [`command::CommandOutcome`] describing what (if
    /// anything) observably changed. Callers that only care about a
    /// specific outcome can `matches!` on the relevant variant.
    pub fn execute(
        &mut self,
        cmd: command::VisoCommand,
    ) -> command::CommandOutcome {
        use command::{CommandOutcome, VisoCommand};
        match cmd {
            // Camera
            VisoCommand::RecenterCamera => {
                self.fit_camera_to_focus();
                CommandOutcome::NoEffect
            }
            VisoCommand::ToggleAutoRotate => {
                let _ = self.camera_controller.toggle_auto_rotate();
                CommandOutcome::NoEffect
            }
            VisoCommand::RotateCamera { delta } => {
                self.camera_controller.rotate(delta);
                CommandOutcome::NoEffect
            }
            VisoCommand::PanCamera { delta } => {
                self.camera_controller.pan(delta);
                CommandOutcome::NoEffect
            }
            VisoCommand::Zoom { delta } => {
                self.camera_controller.zoom(delta);
                CommandOutcome::NoEffect
            }
            // Focus
            VisoCommand::CycleFocus => {
                let _ = self.annotations_mut().cycle_focus();
                self.fit_camera_to_focus();
                CommandOutcome::FocusChanged
            }
            VisoCommand::ResetFocus => {
                self.annotations.focus = Focus::Session;
                self.fit_camera_to_focus();
                CommandOutcome::FocusChanged
            }
            // Playback
            VisoCommand::ToggleTrajectory => {
                self.animation.toggle_trajectory();
                CommandOutcome::NoEffect
            }
            // Selection
            VisoCommand::ClearSelection => {
                selection_outcome(self.gpu.pick.clear_selection())
            }
            VisoCommand::SelectResidue { index, extend } => selection_outcome(
                self.gpu.pick.picking.handle_click(index, extend),
            ),
            VisoCommand::SelectSegment { index, extend } => {
                let ss = self.concatenated_cartoon_ss();
                selection_outcome(
                    self.gpu.pick.select_segment(index, &ss, extend),
                )
            }
            VisoCommand::SelectChain { index, extend } => {
                let chains = self.gpu.renderers.backbone.cached_chains();
                selection_outcome(
                    self.gpu.pick.select_chain(index, chains, extend),
                )
            }
            // Entity management
            VisoCommand::FocusEntity { id } => {
                if let Some(eid) = self.entity_id(id) {
                    self.annotations.focus = Focus::Entity(eid);
                    self.fit_camera_to_focus();
                    CommandOutcome::FocusChanged
                } else {
                    CommandOutcome::NoEffect
                }
            }
            VisoCommand::ToggleEntityVisibility { id } => {
                let currently_visible = self.is_entity_visible(id);
                self.set_entity_visible(id, !currently_visible);
                self.sync_scene_to_renderers(HashMap::new());
                CommandOutcome::VisibilityChanged
            }
            VisoCommand::RemoveEntity { .. } => {
                log::warn!(
                    "VisoCommand::RemoveEntity dispatched to \
                     VisoEngine::execute — must go through VisoApp; ignoring"
                );
                CommandOutcome::Unhandled
            }
            // Display toggles — flip per-type visibility for every
            // entity of that type and keep the display option in sync.
            VisoCommand::SetTypeVisibility { mol_type, visible } => {
                self.set_molecule_type_visibility(mol_type, visible);
                CommandOutcome::VisibilityChanged
            }
            VisoCommand::CycleLipidMode => {
                self.options.display.overrides.lipid_mode =
                    Some(if self.options.display.lipid_ball_and_stick() {
                        crate::options::LipidMode::Coarse
                    } else {
                        crate::options::LipidMode::BallAndStick
                    });
                self.refresh_ball_and_stick();
                CommandOutcome::NoEffect
            }
        }
    }

    /// Resolve a `VisoCommand::SetTypeVisibility`: update the matching
    /// `options.display.show_*` flag (toggling when `visible` is
    /// `None`), broadcast the new value to every entity of `mol_type`,
    /// and resync the scene. Unknown types no-op.
    fn set_molecule_type_visibility(
        &mut self,
        mol_type: MoleculeType,
        visible: Option<bool>,
    ) {
        let flag: Option<&mut bool> = match mol_type {
            MoleculeType::Ion => Some(&mut self.options.display.show_ions),
            MoleculeType::Water => Some(&mut self.options.display.show_waters),
            MoleculeType::Solvent => {
                Some(&mut self.options.display.show_solvent)
            }
            _ => None,
        };
        let Some(flag) = flag else {
            return;
        };
        let next = visible.unwrap_or(!*flag);
        *flag = next;
        self.set_type_visibility(mol_type, next);
        self.sync_scene_to_renderers(HashMap::new());
    }
}

/// Map a selection-helper bool ("did selection actually change?") to
/// the corresponding [`command::CommandOutcome`].
fn selection_outcome(changed: bool) -> command::CommandOutcome {
    if changed {
        command::CommandOutcome::SelectionChanged
    } else {
        command::CommandOutcome::NoEffect
    }
}

// ── Lifecycle + queries ──

impl VisoEngine {
    /// Advance camera animation and apply any pending scene from the
    /// background processor.
    pub fn update(&mut self, dt: f32) {
        let _ = self.camera_controller.update_animation(dt);
        if self.poll_assembly() {
            // Fresh assembly snapshot consumed: queue mesh generation
            // for the background processor. Without this, viso-side
            // derived state would advance but no meshes would be
            // produced for the new entities.
            let transitions = self.animation.take_pending_transitions();
            self.sync_scene_to_renderers(transitions);
        }
        self.apply_pending_scene();
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
        true
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

    /// Fit the camera to the currently focused element (session-wide
    /// bounding sphere, or the focused entity's bounding sphere).
    pub fn fit_camera_to_focus(&mut self) {
        match self.annotations.focus {
            Focus::Session => self.fit_session_camera(),
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
    pub(crate) fn fit_session_camera(&mut self) {
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
            &self.options,
            &self.surface_regen,
        );
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

    /// The currently focused entity ID, or `None` when focus is session-wide.
    #[must_use]
    pub fn focused_entity(&self) -> Option<EntityId> {
        match self.annotations.focus {
            Focus::Entity(id) => Some(id),
            Focus::Session => None,
        }
    }

    /// Current smoothed frames-per-second.
    #[must_use]
    pub fn fps(&self) -> f32 {
        self.frame_timing.fps()
    }

    /// Currently selected residue indices.
    #[must_use]
    pub fn selected_residues(&self) -> &[i32] {
        self.gpu.pick.selected_residues()
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
}
