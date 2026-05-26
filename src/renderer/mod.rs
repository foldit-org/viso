//! Rendering subsystems for molecular visualization.
//!
//! Contains molecular renderers (tubes, ribbons, sidechains, ball-and-stick,
//! nucleic acids) and post-processing effects (SSAO, bloom, FXAA).

/// Bind groups shared across all molecular draw calls.
pub(crate) mod draw_context;
/// Render-ready per-entity contract -- [`EntityTopology`] plus the
/// [`SidechainLayout`] / [`NucleotideRingLayout`] consumed across the
/// renderer.
///
/// [`EntityTopology`]: entity_topology::EntityTopology
/// [`SidechainLayout`]: entity_topology::SidechainLayout
/// [`NucleotideRingLayout`]: entity_topology::NucleotideRingLayout
pub(crate) mod entity_topology;
/// All GPU infrastructure grouped together (device, renderers, picking, etc.).
pub(crate) mod gpu_pipeline;
pub(crate) use gpu_pipeline::GpuPipeline;
/// Molecular geometry assemblers (backbone, sidechain, ball-and-stick, etc.).
pub(crate) mod geometry;
/// Reusable impostor-pass primitives and instance types.
pub(crate) mod impostor;
/// Shared indexed-mesh draw-pass abstraction.
pub(crate) mod mesh;
/// GPU-based object picking and selection management.
pub(crate) mod picking;
/// Background mesh generation pipeline (scene -> GPU-ready buffers).
pub(crate) mod pipeline;
pub(crate) mod pipeline_util;
/// Post-processing effects (SSAO, bloom, FXAA).
pub(crate) mod postprocess;

use self::draw_context::DrawBindGroups;
use self::geometry::isosurface::IsosurfaceRenderer;
use self::geometry::{
    BackboneRenderer, BallAndStickRenderer, BandRenderer, BondRenderer,
    NucleicAcidRenderer, PullRenderer, SidechainRenderer, SidechainView,
};
use crate::camera::frustum::Frustum;
use crate::gpu::{RenderContext, ShaderComposer};

/// Bind group layouts shared by all molecular geometry pipelines.
///
/// Every molecular renderer (backbone, sidechain, ball-and-stick, band, pull,
/// nucleic acid) binds the same camera, lighting, and selection groups.
pub(crate) struct PipelineLayouts {
    pub(crate) camera: wgpu::BindGroupLayout,
    pub(crate) lighting: wgpu::BindGroupLayout,
    pub(crate) selection: wgpu::BindGroupLayout,
    pub(crate) color: wgpu::BindGroupLayout,
}

// ---------------------------------------------------------------------------
// GeometryPassInput
// ---------------------------------------------------------------------------

/// Input for the main geometry render pass.
pub(crate) struct GeometryPassInput<'a> {
    /// Color attachment (post-process input).
    pub(crate) color: &'a wgpu::TextureView,
    /// Normal attachment (post-process input).
    pub(crate) normal: &'a wgpu::TextureView,
    /// Depth attachment.
    pub(crate) depth: &'a wgpu::TextureView,
    /// Whether sidechain capsules should be drawn.
    pub(crate) show_sidechains: bool,
}

// ---------------------------------------------------------------------------
// Renderers
// ---------------------------------------------------------------------------

/// All geometry renderers grouped together.
pub(crate) struct Renderers {
    pub(crate) backbone: BackboneRenderer,
    pub(crate) sidechain: SidechainRenderer,
    pub(crate) bond: BondRenderer,
    pub(crate) band: BandRenderer,
    pub(crate) pull: PullRenderer,
    pub(crate) ball_and_stick: BallAndStickRenderer,
    pub(crate) nucleic_acid: NucleicAcidRenderer,
    pub(crate) isosurface: IsosurfaceRenderer,
}

impl Renderers {
    /// Create all geometry renderers with empty initial GPU buffers.
    ///
    /// First-frame geometry arrives via the background scene processor's
    /// `PreparedRebuild`, not from renderer construction.
    pub(crate) fn new(
        context: &RenderContext,
        layouts: &PipelineLayouts,
        shader_composer: &mut ShaderComposer,
        backface_depth_view: &wgpu::TextureView,
    ) -> Result<Self, crate::error::VisoError> {
        let backbone =
            BackboneRenderer::new(context, layouts, &[], &[], shader_composer)?;
        let sidechain = SidechainRenderer::new(
            context,
            layouts,
            &SidechainView {
                positions: &[],
                bonds: &[],
                backbone_bonds: &[],
                hydrophobicity: &[],
                residue_indices: &[],
            },
            shader_composer,
        )?;
        let bond = BondRenderer::new(context, layouts, shader_composer)?;
        let band = BandRenderer::new(context, layouts, shader_composer)?;
        let pull = PullRenderer::new(context, layouts, shader_composer)?;
        let ball_and_stick =
            BallAndStickRenderer::new(context, layouts, shader_composer)?;
        let nucleic_acid =
            NucleicAcidRenderer::new(context, layouts, shader_composer)?;
        let isosurface = IsosurfaceRenderer::new(
            context,
            layouts,
            shader_composer,
            backface_depth_view,
        )?;
        Ok(Self {
            backbone,
            sidechain,
            bond,
            band,
            pull,
            ball_and_stick,
            nucleic_acid,
            isosurface,
        })
    }

    /// Encode the isosurface back-face depth pre-pass.
    ///
    /// Renders all isosurface back-faces (front-face culling) into the
    /// R32Float `backface_depth_view`, writing linear view-space z. The
    /// main isosurface fragment shader samples this texture to compute
    /// thickness for Beer-Lambert absorption.
    pub(crate) fn encode_isosurface_backface_pass(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        backface_depth_view: &wgpu::TextureView,
        camera_bind_group: &wgpu::BindGroup,
    ) {
        let mut rp = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("isosurface backface depth pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: backface_depth_view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color {
                        r: 0.0,
                        g: 0.0,
                        b: 0.0,
                        a: 0.0,
                    }),
                    store: wgpu::StoreOp::Store,
                },
                depth_slice: None,
            })],
            depth_stencil_attachment: None,
            ..Default::default()
        });
        self.isosurface
            .draw_back_face_pass(&mut rp, camera_bind_group);
    }

    /// Encode the main geometry render pass.
    pub(crate) fn encode_geometry_pass(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        input: &GeometryPassInput<'_>,
        bind_groups: &DrawBindGroups<'_>,
        frustum: &Frustum,
    ) {
        let mut rp = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("main render pass"),
            color_attachments: &[
                Some(wgpu::RenderPassColorAttachment {
                    view: input.color,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                    depth_slice: None,
                }),
                Some(wgpu::RenderPassColorAttachment {
                    view: input.normal,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                        store: wgpu::StoreOp::Store,
                    },
                    depth_slice: None,
                }),
            ],
            depth_stencil_attachment: Some(
                wgpu::RenderPassDepthStencilAttachment {
                    view: input.depth,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                },
            ),
            ..Default::default()
        });

        self.backbone.draw_culled(&mut rp, bind_groups, frustum);

        if input.show_sidechains {
            self.sidechain.draw(&mut rp, bind_groups);
        }

        self.ball_and_stick.draw(&mut rp, bind_groups);
        self.nucleic_acid.draw(&mut rp, bind_groups);
        self.bond.draw(&mut rp, bind_groups);
        self.band.draw(&mut rp, bind_groups);
        self.pull.draw(&mut rp, bind_groups);
        self.isosurface.draw(&mut rp, bind_groups);
    }

    /// GPU buffer sizes across all renderers.
    ///
    /// Each entry is `(label, used_bytes, allocated_bytes)`.
    pub(crate) fn buffer_info(&self) -> Vec<(&str, usize, usize)> {
        let mut stats = Vec::new();
        stats.extend(self.backbone.buffer_info());
        stats.extend(self.sidechain.buffer_info());
        stats.extend(self.ball_and_stick.buffer_info());
        stats.extend(self.bond.buffer_info());
        stats.extend(self.band.buffer_info());
        stats.extend(self.pull.buffer_info());
        stats.extend(self.nucleic_acid.buffer_info());
        stats.extend(self.isosurface.buffer_info());
        stats
    }
}
