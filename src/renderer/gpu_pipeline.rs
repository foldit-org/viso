//! All GPU infrastructure grouped together.

use glam::{Mat4, Vec3};

use crate::camera::controller::CameraController;
use crate::camera::core::Camera;
use crate::engine::positions::EntityPositions;
use crate::gpu::lighting::Lighting;
use crate::gpu::{RenderContext, ShaderComposer};
use crate::options::{GeometryOptions, LightingOptions, VisoOptions};
use crate::renderer::draw_context::DrawBindGroups;
use crate::renderer::geometry::PreparedBallAndStickData;
use crate::renderer::impostor::CapsuleInstance;
use crate::renderer::picking::PickingSystem;
use crate::renderer::pipeline::prepared::{
    AnimationFrameBody, PreparedRebuild,
};
use crate::renderer::pipeline::{SceneProcessor, SceneRequest};
use crate::renderer::postprocess::post_process::PostProcessCamera;
use crate::renderer::postprocess::PostProcessStack;
use crate::renderer::{GeometryPassInput, Renderers};

/// Borrowed scene chain data needed by [`GpuPipeline::upload_prepared`].
pub(crate) struct SceneChainData<'a> {
    /// Protein backbone chains (interpolated or at-rest), SoA form.
    pub(crate) backbone_chains:
        &'a [crate::renderer::entity_topology::ProteinBackboneChain],
    /// Nucleic-acid chains.
    pub(crate) na_chains:
        &'a [crate::renderer::entity_topology::NaBackboneChain],
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
    /// The full, already-colored sidechain capsule set from the most
    /// recent rebuild or animation frame, retained on the main thread so
    /// per-camera frustum culling can re-filter it without regenerating.
    /// Pick ids are global (the rebuild path patches them in `mesh_concat`).
    pub(crate) retained_sidechains: Vec<CapsuleInstance>,
    /// Retained so compiled shader modules stay alive for the engine lifetime.
    #[allow(dead_code)]
    pub(crate) shader_composer: ShaderComposer,
}

impl GpuPipeline {
    /// Core render -- geometry, post-process, picking -- targeting the given
    /// view. Returns the encoder so the caller can submit it.
    pub(crate) fn render_to_view(
        &mut self,
        view: &wgpu::TextureView,
        camera: &CameraController,
    ) -> wgpu::CommandEncoder {
        let mut encoder = self.context.create_encoder();

        // Geometry pass
        let input = GeometryPassInput {
            color: self.post_process.color_view(),
            normal: &self.post_process.normal_view,
            depth: &self.post_process.depth_view,
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

        // Post-processing: SSAO -> bloom -> composite -> FXAA
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
        let picking_geometry = self.pick.build_geometry(&self.renderers);
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
        self.renderers.isosurface.set_back_face_depth_view(
            &self.context.device,
            &self.post_process.backface_depth_view,
        );
        self.pick
            .picking
            .resize(&self.context.device, width, height);
    }

    /// Upload prepared scene geometry to GPU renderers.
    pub(crate) fn upload_prepared(
        &mut self,
        prepared: &PreparedRebuild,
        animating: bool,
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
                &prepared.backbone.vertices,
                &prepared.backbone.tube_indices,
                &prepared.backbone.ribbon_indices,
                prepared.backbone.tube_index_count,
                prepared.backbone.ribbon_index_count,
                prepared.backbone.sheet_offsets.clone(),
                prepared.backbone.chain_ranges.clone(),
                scene.backbone_chains,
                scene.na_chains,
            );
            let _ = self.renderers.sidechain.apply_prepared(
                &self.context.device,
                &self.context.queue,
                &prepared.sidechain_instances,
                prepared.sidechain_instance_count,
            );
            self.retain_sidechains(&prepared.sidechain_instances);
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
        self.pick.entity_residue_offsets =
            prepared.entity_residue_offsets.iter().copied().collect();
        // The flat selection bitset is keyed by global residue index, which
        // shifts whenever the offsets table is rebuilt. Re-derive it from the
        // per-entity selection against the fresh offsets so the highlight
        // stays pinned to the selected residues across rebuilds (the
        // per-frame `update_selection_buffer` uploads the refreshed cache).
        self.pick.rederive_selection();
        // The non-designable bitset is keyed by the same global residue index
        // and shifts with the offsets table, so re-derive it on the same gate
        // (the per-frame `update_non_designable_buffer` uploads the cache).
        self.pick.rederive_non_designable();
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

        // A backbone-only frame carries no sidechains because the producer
        // omitted them, not because the scene has none. Sidechain positions
        // are unchanged by level-of-detail, so leave both the GPU buffer and
        // the retained set holding the prior (correct) sidechains. Touching
        // them here would replace the retained set with empty and the
        // frustum cull would then filter an empty set forever.
        if !prepared.sidechains_omitted {
            let reallocated = self.renderers.sidechain.apply_prepared(
                &self.context.device,
                &self.context.queue,
                &prepared.sidechain_instances,
                prepared.sidechain_instance_count,
            );
            self.retain_sidechains(&prepared.sidechain_instances);
            if reallocated {
                self.pick.groups.rebuild_capsule(
                    &self.pick.picking,
                    &self.context.device,
                    &self.renderers.sidechain,
                );
            }
        }

        // Ball-and-stick + nucleic-acid geometry is derived from the same
        // interpolated positions, so non-cartoon entities animate too. The
        // pick map is not rebuilt per frame: instance counts are stable
        // within an animation (a mutation's atom-count change rides a real
        // rebuild/adoption, not a frame), so the existing map stays valid.
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
                    topology_generation: self
                        .scene_processor
                        .topology_generation(),
                },
            )));
    }

    /// Submit a backbone-only remesh with per-chain LOD to the background
    /// thread. Each chain gets its own `(spr, csv)` based on its distance
    /// from the camera. No sidechains -- they don't change with LOD.
    ///
    /// The base geometry is first clamped via
    /// `GeometryOptions::clamped_for_residues` to stay within the 256 MB
    /// buffer limit, then each chain is further scaled by its distance tier.
    /// For very large structures (>50 K residues) this per-chain scaling is
    /// critical -- without it the vertex buffer can exceed GPU limits.
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
                .map(|c| c.p().len())
                .sum::<usize>();
        let base_geo = geometry.clamped_for_residues(total_residues);
        let max_spr = base_geo.segments_per_residue;
        let max_csv = base_geo.cross_section_verts;

        let per_chain_lod: Vec<crate::options::ChainLod> = self
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
                    topology_generation: self
                        .scene_processor
                        .topology_generation(),
                },
            )));
    }

    /// Check per-chain LOD tiers and submit a background remesh if any
    /// chain's tier has changed. Skipped while a `FullRebuild` is
    /// pending -- the backbone renderer's cached chains are stale.
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

    /// Set the global molecular-surface opacity uniform and write it to
    /// the GPU immediately. Cheap per-tick write for the global surface
    /// opacity slider — no mesh regeneration.
    pub(crate) fn set_surface_opacity(&mut self, opacity: f32) {
        self.lighting.uniform.surface_opacity = opacity;
        self.lighting.update_gpu(&self.context.queue);
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

    /// Poll the scene processor for a completed isosurface mesh (density
    /// maps, entity surfaces, cavities) and upload it to the GPU.
    ///
    /// The generation-gated `try_recv_surface` already discards a result
    /// superseded by a newer regen, so rapid option changes never apply a
    /// stale mesh.
    pub(crate) fn apply_pending_surface(&mut self) -> bool {
        let Some(prepared) = self.scene_processor.try_recv_surface() else {
            return false;
        };
        log::info!(
            "applying surface mesh: {} verts, {} indices",
            prepared.vertices.len(),
            prepared.indices.len()
        );
        self.renderers.isosurface.apply_prepared(
            &self.context.device,
            &self.context.queue,
            &prepared.vertices,
            &prepared.indices,
        );
        true
    }

    /// Stop the background scene processor thread.
    pub(crate) fn shutdown(&mut self) {
        self.scene_processor.shutdown();
    }

    /// Ensure the per-residue overlay (selection + non-designable) and
    /// residue-color buffers have enough capacity for `total_residues`
    /// residues. The overlay buffer grows both bitsets together.
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

    /// Retain a copy of the prepared sidechain capsules so per-camera
    /// frustum culling can re-filter them. The bytes are the globalized,
    /// already-colored instances that were just uploaded, so the retained
    /// pick ids stay global.
    fn retain_sidechains(&mut self, instances: &[u8]) {
        self.retained_sidechains = capsule_instances_from_bytes(instances);
    }

    /// Re-upload only the retained sidechain capsules whose bounding sphere
    /// intersects the camera frustum, then rebuild the capsule pick bind
    /// group. This is a pure filter over the prepared set: color, sheet
    /// adjustment, and global pick ids are inherited from the retained
    /// instances unchanged.
    pub(crate) fn upload_frustum_culled_sidechains(
        &mut self,
        frustum: &crate::camera::frustum::Frustum,
    ) {
        self.renderers.sidechain.update_with_frustum(
            &self.context.device,
            &self.context.queue,
            &self.retained_sidechains,
            frustum,
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
}

/// Decode concatenated capsule-instance bytes into a typed Vec.
///
/// The source bytes come from a `Vec<u8>` and so carry only 1-byte
/// alignment, while `CapsuleInstance` needs 16-byte alignment. A direct
/// `cast_slice` would panic on that mismatch, so each stride-sized chunk
/// is read with `pod_read_unaligned`, which copies the bytes without an
/// alignment precondition. The copy is the once-per-apply retain cost.
fn capsule_instances_from_bytes(bytes: &[u8]) -> Vec<CapsuleInstance> {
    let stride = size_of::<CapsuleInstance>();
    bytes
        .chunks_exact(stride)
        .map(bytemuck::pod_read_unaligned::<CapsuleInstance>)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_instances() -> Vec<CapsuleInstance> {
        vec![
            CapsuleInstance {
                endpoint_a: [1.0, 2.0, 3.0, 0.5],
                endpoint_b: [4.0, 5.0, 6.0, 7.0],
                color_a: [0.1, 0.2, 0.3, 0.0],
                color_b: [0.4, 0.5, 0.6, 0.0],
            },
            CapsuleInstance {
                endpoint_a: [-1.0, -2.0, -3.0, 1.5],
                endpoint_b: [-4.0, -5.0, -6.0, 8.0],
                color_a: [0.9, 0.8, 0.7, 0.0],
                color_b: [0.6, 0.5, 0.4, 0.0],
            },
            CapsuleInstance {
                endpoint_a: [10.0, 20.0, 30.0, 2.5],
                endpoint_b: [40.0, 50.0, 60.0, 9.0],
                color_a: [0.05, 0.15, 0.25, 0.0],
                color_b: [0.35, 0.45, 0.55, 0.0],
            },
        ]
    }

    fn assert_same(decoded: &[CapsuleInstance], expected: &[CapsuleInstance]) {
        assert_eq!(decoded.len(), expected.len());
        for (got, want) in decoded.iter().zip(expected.iter()) {
            assert_eq!(got.endpoint_a, want.endpoint_a);
            assert_eq!(got.endpoint_b, want.endpoint_b);
            assert_eq!(got.color_a, want.color_a);
            assert_eq!(got.color_b, want.color_b);
        }
    }

    /// Decoding bytes that are only 1-byte aligned (the real retain path
    /// feeds a `Vec<u8>`) must round-trip without panicking. The previous
    /// `cast_slice` implementation panicked here on alignment.
    #[test]
    fn from_bytes_handles_byte_aligned_source() {
        let expected = sample_instances();
        // A Vec<u8> copy carries only 1-byte alignment, matching the live
        // input that crashed the old cast.
        let bytes: Vec<u8> = bytemuck::cast_slice(&expected).to_vec();
        let decoded = capsule_instances_from_bytes(&bytes);
        assert_same(&decoded, &expected);
    }

    /// Decoding from a deliberately misaligned offset must also succeed,
    /// proving the path does not rely on the source happening to land on a
    /// 16-byte boundary.
    #[test]
    fn from_bytes_handles_misaligned_slice() {
        let expected = sample_instances();
        let body: Vec<u8> = bytemuck::cast_slice(&expected).to_vec();
        // Prepend one byte and slice past it so the input start is offset
        // by one from whatever alignment the allocation had.
        let mut padded = Vec::with_capacity(body.len() + 1);
        padded.push(0u8);
        padded.extend_from_slice(&body);
        let decoded = capsule_instances_from_bytes(&padded[1..]);
        assert_same(&decoded, &expected);
    }
}
