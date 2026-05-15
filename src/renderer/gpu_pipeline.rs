//! All GPU infrastructure grouped together.

use std::sync::mpsc;

use glam::{Mat4, Vec3};

use crate::camera::controller::CameraController;
use crate::camera::core::Camera;
use crate::engine::positions::EntityPositions;
use crate::error::VisoError;
use crate::gpu::adobe_cube_lut::AdobeCubeLutTexture;
use crate::gpu::lighting::Lighting;
use crate::gpu::{RenderContext, ShaderComposer};
use crate::options::{GeometryOptions, LightingOptions, VisoOptions};
use crate::renderer::draw_context::DrawBindGroups;
use crate::renderer::geometry::isosurface::IsosurfaceVertex;
use crate::renderer::geometry::{
    PreparedBackboneData, PreparedBallAndStickData, SidechainView,
};
use crate::renderer::picking::PickingSystem;
use crate::renderer::pipeline::prepared::{
    AnimationFrameBody, PreparedRebuild,
};
use crate::renderer::pipeline::{SceneProcessor, SceneRequest};
use crate::renderer::postprocess::post_process::PostProcessCamera;
use crate::renderer::postprocess::PostProcessStack;
use crate::renderer::{GeometryPassInput, Renderers};
use crate::util::lut_adobe_cube::LutRgbCube3d;

/// Borrowed scene chain data needed by [`GpuPipeline::upload_prepared`].
pub(crate) struct SceneChainData<'a> {
    /// Backbone chains (interpolated or at-rest).
    pub(crate) backbone_chains: &'a [Vec<Vec3>],
    /// Nucleic-acid chains.
    pub(crate) na_chains: &'a [Vec<Vec3>],
}

/// All GPU infrastructure grouped together: device/queue, renderers,
/// picking, background mesh processor, post-processing, lighting, and
/// per-frame cursor/culling state.
pub(crate) struct GpuPipeline {
    /// Core wgpu device, queue, and surface.
    pub(crate) context: RenderContext,
    /// All geometry renderers (backbone, sidechain, band, pull,
    /// ball-and-stick, nucleic acid).
    pub(crate) renderers: Renderers,
    /// GPU picking, selection, and per-residue color buffers.
    pub(crate) pick: PickingSystem,
    /// Background thread for off-main-thread mesh generation.
    pub(crate) scene_processor: SceneProcessor,
    /// Post-processing pass stack (SSAO, bloom, composite, FXAA).
    pub(crate) post_process: PostProcessStack,
    /// GPU lighting uniform and bind group.
    pub(crate) lighting: Lighting,
    /// Current cursor position in physical pixels (set by the viewer /
    /// input processor each frame for GPU picking).
    pub(crate) cursor_pos: (f32, f32),
    /// Camera eye position at the last frustum-culling update.
    pub(crate) last_cull_camera_eye: Vec3,
    /// Retained so compiled shader modules stay alive for the engine lifetime.
    #[allow(dead_code)]
    pub(crate) shader_composer: ShaderComposer,
    /// Receiver for background-extracted isosurface meshes (density
    /// maps, entity surfaces, cavities). The matching sender lives on
    /// [`crate::engine::surface_regen::SurfaceRegen`].
    pub(crate) density_rx: mpsc::Receiver<(Vec<IsosurfaceVertex>, Vec<u32>)>,
    /// Optional Adobe `.cube` color LUT uploaded as `Rgba16Float`; composite
    /// binds and samples it when set via [`GpuPipeline::set_adobe_cube_lut`].
    pub(crate) adobe_cube_lut: Option<AdobeCubeLutTexture>,
}

impl GpuPipeline {
    /// Core render — geometry, post-process, picking — targeting the given
    /// view. Returns the encoder so the caller can submit it.
    pub(crate) fn render_to_view(
        &mut self,
        view: &wgpu::TextureView,
        camera: &CameraController,
        show_sidechains: bool,
    ) -> wgpu::CommandEncoder {
        let mut encoder = self.context.create_encoder();

        // Geometry pass
        let input = GeometryPassInput {
            color: self.post_process.color_view(),
            normal: &self.post_process.normal_view,
            depth: &self.post_process.depth_view,
            show_sidechains,
        };
        let bind_groups = DrawBindGroups {
            camera: &camera.bind_group,
            lighting: &self.lighting.bind_group,
            selection: &self.pick.selection.bind_group,
            color: Some(&self.pick.residue_colors.bind_group),
        };
        let frustum = camera.frustum();
        self.renderers.encode_isosurface_backface_pass(
            &mut encoder,
            &self.post_process.backface_depth_view,
            &camera.bind_group,
        );
        self.renderers.encode_geometry_pass(
            &mut encoder,
            &input,
            &bind_groups,
            &frustum,
        );

        // Post-processing: SSAO → bloom → composite → FXAA
        let cam = &camera.camera;
        self.post_process.render(
            &mut encoder,
            &self.context.queue,
            &PostProcessCamera {
                proj: cam.build_projection(),
                view_matrix: Mat4::look_at_rh(cam.eye, cam.target, cam.up),
                znear: cam.znear,
                zfar: cam.zfar,
            },
            view.clone(),
        );

        // GPU Picking pass
        let picking_geometry =
            self.pick.build_geometry(&self.renderers, show_sidechains);
        self.pick.picking.render(
            &mut encoder,
            &camera.bind_group,
            &picking_geometry,
            (self.cursor_pos.0 as u32, self.cursor_pos.1 as u32),
        );

        encoder
    }

    /// Resize all GPU surfaces to match the new window size.
    pub(crate) fn resize(&mut self, width: u32, height: u32) {
        self.context.resize(width, height);
        self.post_process.resize(&self.context);
        self.sync_composite_adobe_cube_lut();
        self.renderers.isosurface.set_back_face_depth_view(
            &self.context.device,
            &self.post_process.backface_depth_view,
        );
        self.pick
            .picking
            .resize(&self.context.device, width, height);
    }

    fn sync_composite_adobe_cube_lut(&mut self) {
        self.post_process.sync_adobe_cube_lut(
            &self.context,
            self.adobe_cube_lut.as_ref(),
        );
    }

    /// Upload prepared scene geometry to GPU renderers.
    pub(crate) fn upload_prepared(
        &mut self,
        prepared: &PreparedRebuild,
        animating: bool,
        suppress_sidechains: bool,
        scene: &SceneChainData<'_>,
    ) {
        if animating {
            self.renderers
                .backbone
                .update_metadata(scene.backbone_chains, scene.na_chains);
        } else {
            self.renderers.backbone.apply_prepared(
                &self.context.device,
                &self.context.queue,
                PreparedBackboneData {
                    vertices: &prepared.backbone.vertices,
                    tube_indices: &prepared.backbone.tube_indices,
                    ribbon_indices: &prepared.backbone.ribbon_indices,
                    tube_index_count: prepared.backbone.tube_index_count,
                    ribbon_index_count: prepared.backbone.ribbon_index_count,
                    sheet_offsets: prepared.backbone.sheet_offsets.clone(),
                    chain_ranges: prepared.backbone.chain_ranges.clone(),
                    cached_chains: scene.backbone_chains,
                    cached_na_chains: scene.na_chains,
                },
            );
            if !suppress_sidechains {
                let _ = self.renderers.sidechain.apply_prepared(
                    &self.context.device,
                    &self.context.queue,
                    &prepared.sidechain_instances,
                    prepared.sidechain_instance_count,
                );
            }
        }
        self.upload_non_backbone(prepared);
    }

    /// Upload BnS, NA, and pick data (shared by animating and non-animating).
    fn upload_non_backbone(&mut self, prepared: &PreparedRebuild) {
        self.renderers.ball_and_stick.apply_prepared(
            &self.context.device,
            &self.context.queue,
            &PreparedBallAndStickData {
                sphere_bytes: &prepared.bns.sphere_instances,
                sphere_count: prepared.bns.sphere_count,
                capsule_bytes: &prepared.bns.capsule_instances,
                capsule_count: prepared.bns.capsule_count,
            },
        );
        self.renderers.nucleic_acid.apply_prepared(
            &self.context.device,
            &self.context.queue,
            &prepared.na,
        );
        self.pick.pick_map = Some(prepared.pick_map.clone());
        self.pick.groups.rebuild_all(
            &self.pick.picking,
            &self.context.device,
            &self.renderers.sidechain,
            &self.renderers.ball_and_stick,
        );
    }

    /// Apply any pending animation frame from the background thread.
    ///
    /// Returns `true` if a frame was applied, `false` otherwise.
    pub(crate) fn apply_pending_animation(&mut self) -> bool {
        let Some(prepared) = self.scene_processor.try_recv_animation() else {
            return false;
        };

        self.renderers.backbone.apply_mesh(
            &self.context.device,
            &self.context.queue,
            prepared.backbone,
        );

        if let Some(ref instances) = prepared.sidechain_instances {
            let reallocated = self.renderers.sidechain.apply_prepared(
                &self.context.device,
                &self.context.queue,
                instances,
                prepared.sidechain_instance_count,
            );
            if reallocated {
                self.pick.groups.rebuild_capsule(
                    &self.pick.picking,
                    &self.context.device,
                    &self.renderers.sidechain,
                );
            }
        }

        true
    }

    /// Submit an animation frame to the background thread using the
    /// engine's current interpolated positions. The worker reconstructs
    /// per-entity backbone / sidechain mesh from cached topology.
    pub(crate) fn submit_animation_frame(
        &self,
        positions: &EntityPositions,
        geometry: &GeometryOptions,
        include_sidechains: bool,
    ) {
        self.scene_processor
            .submit(SceneRequest::AnimationFrame(Box::new(
                AnimationFrameBody {
                    positions: positions.clone(),
                    geometry: geometry.clone(),
                    per_chain_lod: None,
                    include_sidechains,
                    generation: self.scene_processor.generation(),
                },
            )));
    }

    /// Submit a backbone-only remesh with per-chain LOD to the background
    /// thread. Each chain gets its own `(spr, csv)` based on its distance
    /// from the camera. No sidechains — they don't change with LOD.
    ///
    /// The base geometry is first clamped via
    /// `GeometryOptions::clamped_for_residues` to stay within the 256 MB
    /// buffer limit, then each chain is further scaled by its distance tier.
    /// For very large structures (>50 K residues) this per-chain scaling is
    /// critical — without it the vertex buffer can exceed GPU limits.
    pub(crate) fn submit_lod_remesh(
        &self,
        camera_eye: Vec3,
        geometry: &GeometryOptions,
        positions: &EntityPositions,
    ) {
        // While a FullRebuild is in flight, the backbone renderer's
        // cached chains are stale. Submitting a LOD remesh now would
        // produce an AnimationFrame with old backbone geometry that
        // could overwrite the correct PreparedRebuild upload.
        if self.scene_processor.is_rebuild_pending() {
            return;
        }
        use crate::options::{lod_scaled, select_chain_lod_tier};

        // Use clamped geometry as the base for LOD scaling
        let total_residues =
            crate::renderer::geometry::sheet_adjust::backbone_residue_count(
                self.renderers.backbone.cached_chains(),
            ) + self
                .renderers
                .backbone
                .cached_na_chains()
                .iter()
                .map(Vec::len)
                .sum::<usize>();
        let base_geo = geometry.clamped_for_residues(total_residues);
        let max_spr = base_geo.segments_per_residue;
        let max_csv = base_geo.cross_section_verts;

        let per_chain_lod: Vec<(usize, usize)> = self
            .renderers
            .backbone
            .chain_ranges()
            .iter()
            .map(|r| {
                let tier = select_chain_lod_tier(r.bounding_center, camera_eye);
                lod_scaled(max_spr, max_csv, tier)
            })
            .collect();

        self.scene_processor
            .submit(SceneRequest::AnimationFrame(Box::new(
                AnimationFrameBody {
                    positions: positions.clone(),
                    geometry: base_geo,
                    per_chain_lod: Some(per_chain_lod),
                    include_sidechains: false,
                    generation: self.scene_processor.generation(),
                },
            )));
    }

    /// Check per-chain LOD tiers and submit a background remesh if any
    /// chain's tier has changed. Skipped while a `FullRebuild` is
    /// pending — the backbone renderer's cached chains are stale.
    pub(crate) fn check_and_submit_lod(
        &mut self,
        camera_eye: Vec3,
        geometry: &GeometryOptions,
        positions: &EntityPositions,
    ) {
        if self.scene_processor.is_rebuild_pending() {
            return;
        }
        let per_chain_tiers: Vec<u8> = self
            .renderers
            .backbone
            .chain_ranges()
            .iter()
            .map(|r| {
                crate::options::select_chain_lod_tier(
                    r.bounding_center,
                    camera_eye,
                )
            })
            .collect();
        if per_chain_tiers != self.renderers.backbone.cached_lod_tiers() {
            self.renderers
                .backbone
                .set_cached_lod_tiers(per_chain_tiers);
            self.submit_lod_remesh(camera_eye, geometry, positions);
        }
    }

    /// Push lighting options to the GPU uniform.
    pub(crate) fn apply_lighting(&mut self, lo: &LightingOptions) {
        self.lighting.apply_options(lo, &self.context.queue);
    }

    /// Push post-processing options to the composite pass.
    pub(crate) fn apply_post_processing(&mut self, options: &VisoOptions) {
        self.post_process
            .apply_options(options, &self.context.queue);
    }

    /// Update headlamp direction from the camera and push to the GPU.
    pub(crate) fn update_headlamp(&mut self, camera: &Camera) {
        self.lighting.update_headlamp_from_camera(camera);
        self.lighting.update_gpu(&self.context.queue);
    }

    /// Poll for pending density mesh data and upload to GPU.
    ///
    /// Drains all queued results and only applies the latest one,
    /// so rapid slider changes don't queue up stale meshes.
    pub(crate) fn apply_pending_density_mesh(&mut self) -> bool {
        let mut latest = None;
        while let Ok(data) = self.density_rx.try_recv() {
            latest = Some(data);
        }
        let Some((vertices, indices)) = latest else {
            return false;
        };
        log::info!(
            "applying density mesh: {} verts, {} indices",
            vertices.len(),
            indices.len()
        );
        self.renderers.isosurface.apply_prepared(
            &self.context.device,
            &self.context.queue,
            &vertices,
            &indices,
        );
        true
    }

    /// Stop the background scene processor thread.
    pub(crate) fn shutdown(&mut self) {
        self.scene_processor.shutdown();
    }

    /// Ensure the selection and residue-color buffers have enough capacity.
    pub(crate) fn ensure_residue_capacity(&mut self, total_residues: usize) {
        self.pick
            .selection
            .ensure_capacity(&self.context.device, total_residues);
        self.pick
            .residue_colors
            .ensure_capacity(&self.context.device, total_residues);
    }

    /// Immediately upload per-residue colors (no transition).
    pub(crate) fn set_colors_immediate(&mut self, colors: &[[f32; 3]]) {
        self.pick
            .residue_colors
            .set_colors_immediate(&self.context.queue, colors);
    }

    /// Set target per-residue colors (animated transition).
    pub(crate) fn set_target_colors(&mut self, colors: &[[f32; 3]]) {
        self.pick.residue_colors.set_target_colors(colors);
    }

    /// Upload a sheet-adjusted, frustum-culled sidechain view and
    /// rebuild the capsule pick bind group. Exposed as a primitive so
    /// the engine can compose it against whichever state feeds the
    /// sidechain topology.
    pub(crate) fn upload_frustum_culled_sidechains(
        &mut self,
        view: &SidechainView,
        frustum: &crate::camera::frustum::Frustum,
        per_residue_colors: Option<&[[f32; 3]]>,
    ) {
        self.renderers.sidechain.update_with_frustum(
            &self.context.device,
            &self.context.queue,
            view,
            Some(frustum),
            per_residue_colors,
        );
        self.pick.groups.rebuild_capsule(
            &self.pick.picking,
            &self.context.device,
            &self.renderers.sidechain,
        );
    }

    /// Record the camera eye at which frustum culling last ran. Used
    /// by the engine to gate re-culling on camera motion.
    pub(crate) fn set_last_cull_camera_eye(&mut self, eye: Vec3) {
        self.last_cull_camera_eye = eye;
    }

    /// Read-only access to the backbone renderer's current sheet-offset
    /// list for main-thread sidechain adjustment.
    pub(crate) fn backbone_sheet_offsets(&self) -> &[(u32, Vec3)] {
        self.renderers.backbone.sheet_offsets()
    }

    /// Load or unload an Adobe LUT as a GPU 3D `Rgba16Float` texture and sync
    /// the composite pass bind group and grid uniform.
    ///
    /// # Errors
    ///
    /// Returns [`VisoError::GpuResource`] when [`LutRgbCube3d::size`]
    /// exceeds the adapter's [`wgpu::Limits::max_texture_dimension_3d`].
    pub(crate) fn set_adobe_cube_lut(
        &mut self,
        lut: Option<LutRgbCube3d>,
    ) -> Result<(), VisoError> {
        if let Some(parsed) = lut {
            let n = parsed.size;
            let gpu_lut = AdobeCubeLutTexture::try_new(&self.context, &parsed)?;
            self.adobe_cube_lut = Some(gpu_lut);
            log::info!("Adobe cube LUT active (LUT_3D_SIZE {n})");
        } else {
            self.adobe_cube_lut = None;
            log::info!("Adobe cube LUT cleared");
        }
        self.sync_composite_adobe_cube_lut();
        Ok(())
    }
}
