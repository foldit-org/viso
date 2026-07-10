use glam::Mat4;

use super::{
    BloomPass, CompositeInputs, CompositePass, FxaaPass, ScreenPass,
    SsaoRenderer,
    overlay::OverlayPass,
};
use crate::error::VisoError;
use crate::gpu::pipeline_helpers::create_render_texture;
use crate::gpu::{RenderContext, ShaderComposer};
use crate::options::VisoOptions;

/// Camera parameters needed for post-processing passes.
pub(crate) struct PostProcessCamera {
    pub(crate) proj: Mat4,
    pub(crate) view_matrix: Mat4,
    pub(crate) znear: f32,
    pub(crate) zfar: f32,
}

/// Owns the full post-processing pipeline: depth/normal G-buffers,
/// SSAO, bloom, composite, and FXAA passes.
pub(crate) struct PostProcessStack {
    pub(crate) depth_texture: wgpu::Texture,
    pub(crate) depth_view: wgpu::TextureView,
    pub(crate) normal_texture: wgpu::Texture,
    pub(crate) normal_view: wgpu::TextureView,
    pub(crate) backface_depth_texture: wgpu::Texture,
    pub(crate) backface_depth_view: wgpu::TextureView,
    pub(crate) ssao_renderer: SsaoRenderer,
    pub(crate) bloom_pass: BloomPass,
    pub(crate) composite_pass: CompositePass,
    pub(crate) fxaa_pass: FxaaPass,
    /// Host-supplied overlay, blended onto the swapchain after FXAA.
    pub(crate) overlay_pass: OverlayPass,
}

impl PostProcessStack {
    /// Build the full post-processing stack (depth/normal textures + all
    /// passes).
    pub(crate) fn new(
        context: &RenderContext,
        shader_composer: &mut ShaderComposer,
    ) -> Result<Self, VisoError> {
        let (depth_texture, depth_view) = Self::create_depth_texture(context);
        let (normal_texture, normal_view) =
            Self::create_normal_texture(context);
        let (backface_depth_texture, backface_depth_view) =
            Self::create_backface_depth_texture(context);

        let ssao_renderer = SsaoRenderer::new(
            context,
            &depth_view,
            &normal_view,
            shader_composer,
        )?;

        let mut bloom_pass =
            BloomPass::new(context, &normal_view, shader_composer)?;

        let mut composite_pass = CompositePass::new(
            context,
            &CompositeInputs {
                ssao: ssao_renderer.get_ssao_view(),
                depth: &depth_view,
                normal: &normal_view,
                bloom: bloom_pass.get_output_view(),
            },
            shader_composer,
        )?;

        bloom_pass.rebind_input(context, composite_pass.get_color_view());

        // If sRGB, hardware does gamma correction → gamma = 1.0
        // If linear, apply gamma = 1/2.2 in shader
        composite_pass.params.gamma = if context.config.format.is_srgb() {
            1.0
        } else {
            1.0 / 2.2
        };

        let fxaa_pass = FxaaPass::new(context, shader_composer)?;
        let overlay_pass = OverlayPass::new(context, shader_composer)?;
        composite_pass.set_output_view(fxaa_pass.get_input_view().clone());

        Ok(Self {
            depth_texture,
            depth_view,
            normal_texture,
            normal_view,
            backface_depth_texture,
            backface_depth_view,
            ssao_renderer,
            bloom_pass,
            composite_pass,
            fxaa_pass,
            overlay_pass,
        })
    }

    /// Recreate all resolution-dependent resources.
    pub(crate) fn resize(&mut self, context: &RenderContext) {
        let (depth_texture, depth_view) = Self::create_depth_texture(context);
        self.depth_texture = depth_texture;
        self.depth_view = depth_view;
        let (normal_texture, normal_view) =
            Self::create_normal_texture(context);
        self.normal_texture = normal_texture;
        self.normal_view = normal_view;
        let (backface_depth_texture, backface_depth_view) =
            Self::create_backface_depth_texture(context);
        self.backface_depth_texture = backface_depth_texture;
        self.backface_depth_view = backface_depth_view;

        self.ssao_renderer.set_geometry_views(
            self.depth_view.clone(),
            self.normal_view.clone(),
        );
        self.ssao_renderer.resize(context);

        self.bloom_pass.resize(context);

        self.composite_pass.set_external_views(
            self.ssao_renderer.get_ssao_view().clone(),
            self.depth_view.clone(),
            self.normal_view.clone(),
            self.bloom_pass.get_output_view().clone(),
        );
        self.composite_pass.resize(context);

        self.bloom_pass
            .rebind_input(context, self.composite_pass.get_color_view());
        self.fxaa_pass.resize(context);
        self.composite_pass
            .set_output_view(self.fxaa_pass.get_input_view().clone());
    }

    /// Run the SSAO → bloom → composite → FXAA sequence.
    pub(crate) fn render(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        queue: &wgpu::Queue,
        camera: &PostProcessCamera,
        final_view: wgpu::TextureView,
    ) {
        // SSAO pass
        self.ssao_renderer.update_matrices(queue, camera);
        self.ssao_renderer.render_ssao(encoder);

        // Bloom pass
        self.bloom_pass.render(encoder);

        // Composite pass → writes into FXAA input texture
        self.composite_pass.render(encoder);

        // FXAA pass → writes to swapchain
        self.fxaa_pass.set_output_view(final_view);
        self.fxaa_pass.render(encoder);
    }

    /// Update fog uniforms.
    pub(crate) fn update_fog(
        &mut self,
        queue: &wgpu::Queue,
        fog_start: f32,
        fog_density: f32,
    ) {
        self.composite_pass
            .update_fog(queue, fog_start, fog_density);
    }

    /// Push post-processing option values to GPU.
    pub(crate) fn apply_options(
        &mut self,
        options: &VisoOptions,
        queue: &wgpu::Queue,
    ) {
        let pp = &options.post_processing;
        self.composite_pass.params.outline_thickness = pp.outline_thickness;
        self.composite_pass.params.outline_strength = pp.outline_strength;
        self.composite_pass.params.ao_strength = pp.ao_strength;
        self.composite_pass.params.normal_outline_strength =
            pp.normal_outline_strength;
        self.composite_pass.params.exposure = pp.exposure;
        self.composite_pass.params.bloom_intensity = pp.bloom_intensity;
        self.composite_pass.flush_params(queue);

        self.bloom_pass.threshold = pp.bloom_threshold;
        self.bloom_pass.intensity = pp.bloom_intensity;
        self.bloom_pass.update_params(queue);

        self.ssao_renderer.radius = pp.ao_radius;
        self.ssao_renderer.bias = pp.ao_bias;
        self.ssao_renderer.power = pp.ao_power;
    }

    /// The color texture view used as render target for the geometry pass.
    pub(crate) fn color_view(&self) -> &wgpu::TextureView {
        &self.composite_pass.color_view
    }

    fn create_depth_texture(
        context: &RenderContext,
    ) -> (wgpu::Texture, wgpu::TextureView) {
        create_render_texture(
            &context.device,
            context.render_width(),
            context.render_height(),
            wgpu::TextureFormat::Depth32Float,
            "Depth Texture",
        )
    }

    fn create_normal_texture(
        context: &RenderContext,
    ) -> (wgpu::Texture, wgpu::TextureView) {
        create_render_texture(
            &context.device,
            context.render_width(),
            context.render_height(),
            wgpu::TextureFormat::Rgba16Float,
            "Normal G-Buffer",
        )
    }

    fn create_backface_depth_texture(
        context: &RenderContext,
    ) -> (wgpu::Texture, wgpu::TextureView) {
        create_render_texture(
            &context.device,
            context.render_width(),
            context.render_height(),
            wgpu::TextureFormat::R32Float,
            "Isosurface Backface Depth",
        )
    }
}
