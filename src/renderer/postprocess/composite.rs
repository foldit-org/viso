//! Composite pass - applies SSAO and outline effects to the rendered scene
//!
//! This pass takes the geometry color buffer, SSAO buffer, and depth buffer,
//! combining them to produce the final image with ambient occlusion and
//! silhouette outlines applied.

use half::f16;
use wgpu::util::DeviceExt;

use super::screen_pass::{run_screen_pass, ScreenPass, ScreenPassDesc};
use crate::error::VisoError;
use crate::gpu::adobe_cube_lut::AdobeCubeLutTexture;
use crate::gpu::pipeline_helpers::{
    create_render_texture, create_screen_space_pipeline, depth_texture_2d,
    filtering_sampler, linear_sampler, non_filtering_sampler, texture_2d,
    texture_3d_float, uniform_buffer, ScreenSpacePipelineDef,
};
use crate::gpu::{RenderContext, Shader, ShaderComposer};

/// External texture view inputs for creating a composite pass.
pub(crate) struct CompositeInputs<'a> {
    pub(crate) ssao: &'a wgpu::TextureView,
    pub(crate) depth: &'a wgpu::TextureView,
    pub(crate) normal: &'a wgpu::TextureView,
    pub(crate) bloom: &'a wgpu::TextureView,
}

/// Parameters for the composite pass effects (SSAO strength, outlines, etc.)
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct CompositeParams {
    /// Screen dimensions in pixels `[width, height]`.
    pub(crate) screen_size: [f32; 2],
    /// Outline thickness in texels.
    pub(crate) outline_thickness: f32,
    /// Outline darkness strength (0.0–1.0).
    pub(crate) outline_strength: f32,
    /// SSAO contribution strength.
    pub(crate) ao_strength: f32,
    /// Near clipping plane distance.
    pub(crate) near: f32,
    /// Far clipping plane distance.
    pub(crate) far: f32,
    /// Distance at which depth fog begins.
    pub(crate) fog_start: f32,
    /// Fog density factor.
    pub(crate) fog_density: f32,
    /// Normal-based outline strength.
    pub(crate) normal_outline_strength: f32,
    /// Exposure multiplier for tone mapping.
    pub(crate) exposure: f32,
    /// Gamma correction exponent.
    pub(crate) gamma: f32,
    /// Bloom blend intensity.
    pub(crate) bloom_intensity: f32,
    /// Padding for GPU alignment.
    pub(crate) _pad: f32,
    /// Padding for GPU alignment.
    pub(crate) _pad2: f32,
    /// Adobe `.cube` grid edge `N`; `0` skips LUT sampling (see composite WGSL).
    pub(crate) adobe_lut_grid_size: u32,
}

impl Default for CompositeParams {
    fn default() -> Self {
        Self {
            screen_size: [1920.0, 1080.0],
            outline_thickness: 1.0,
            outline_strength: 0.7,
            ao_strength: 0.85,
            near: 5.0,
            far: 2000.0,
            fog_start: 100.0,
            fog_density: 0.005,
            normal_outline_strength: 0.5,
            exposure: 1.0,
            gamma: 1.0,
            bloom_intensity: 0.0,
            _pad: 0.0,
            _pad2: 0.0,
            adobe_lut_grid_size: 0,
        }
    }
}

struct CompositeViews<'a> {
    pub color: &'a wgpu::TextureView,
    pub ssao: &'a wgpu::TextureView,
    pub depth: &'a wgpu::TextureView,
    pub normal: &'a wgpu::TextureView,
    pub bloom: &'a wgpu::TextureView,
    pub sampler: &'a wgpu::Sampler,
    pub depth_sampler: &'a wgpu::Sampler,
    pub params_buffer: &'a wgpu::Buffer,
    /// Placeholder 3D LUT (`1³`);
    pub adobe_lut_tex: &'a wgpu::TextureView,
    pub adobe_lut_sampler: &'a wgpu::Sampler,
}

/// Composite pass renderer
pub(crate) struct CompositePass {
    pipeline: wgpu::RenderPipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    bind_group: wgpu::BindGroup,
    sampler: wgpu::Sampler,
    depth_sampler: wgpu::Sampler,

    /// Intermediate color texture (geometry renders here instead of
    /// swapchain).
    pub(crate) color_texture: wgpu::Texture,
    /// View into the intermediate color texture.
    pub(crate) color_view: wgpu::TextureView,

    /// Output view (FXAA input texture), set before render.
    output_view: Option<wgpu::TextureView>,
    /// Stored SSAO view for bind group recreation on resize.
    ssao_view: wgpu::TextureView,
    /// Stored depth view for bind group recreation on resize.
    depth_view: wgpu::TextureView,
    /// Stored normal view for bind group recreation on resize.
    normal_view: wgpu::TextureView,
    /// Stored bloom view for bind group recreation on resize.
    bloom_view: wgpu::TextureView,

    /// `1³` `Rgba16Float` volume reserved for Adobe LUT bindings (B1 placeholder).
    #[allow(dead_code)]
    adobe_lut_dummy_texture: wgpu::Texture,
    adobe_lut_dummy_view: wgpu::TextureView,
    /// Linear + clamp sampler for binding 9 (shared by dummy and real LUT).
    adobe_lut_sampler: wgpu::Sampler,
    /// Dummy view or cloned view from [`AdobeCubeLutTexture`].
    lut_bind_view: wgpu::TextureView,

    /// Composite effect parameters (outline, AO, fog, tone-mapping).
    pub(crate) params: CompositeParams,
    params_buffer: wgpu::Buffer,

    width: u32,
    height: u32,
}

impl CompositePass {
    /// Create a new composite pass with all textures, samplers, and pipeline.
    pub(crate) fn new(
        context: &RenderContext,
        inputs: &CompositeInputs,
        shader_composer: &mut ShaderComposer,
    ) -> Result<Self, VisoError> {
        let width = context.render_width();
        let height = context.render_height();

        let (color_texture, color_view) =
            Self::create_color_texture(context, width, height);

        let (sampler, depth_sampler) = Self::create_samplers(context);

        let params = CompositeParams {
            screen_size: [width as f32, height as f32],
            ..Default::default()
        };
        let params_buffer = context.device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("Composite Params Buffer"),
                contents: bytemuck::cast_slice(&[params]),
                usage: wgpu::BufferUsages::UNIFORM
                    | wgpu::BufferUsages::COPY_DST,
            },
        );

        let (adobe_lut_dummy_texture, adobe_lut_dummy_view) =
            Self::create_adobe_lut_dummy_3d(context);
        let adobe_lut_sampler = Self::create_adobe_lut_sampler(context);
        let lut_bind_view = adobe_lut_dummy_view.clone();

        let bind_group_layout = Self::create_bind_group_layout(context);

        let bind_group = Self::create_bind_group(
            context,
            &bind_group_layout,
            &CompositeViews {
                color: &color_view,
                ssao: inputs.ssao,
                depth: inputs.depth,
                normal: inputs.normal,
                bloom: inputs.bloom,
                sampler: &sampler,
                depth_sampler: &depth_sampler,
                params_buffer: &params_buffer,
                adobe_lut_tex: &lut_bind_view,
                adobe_lut_sampler: &adobe_lut_sampler,
            },
        );

        let pipeline = Self::create_pipeline(
            context,
            shader_composer,
            &bind_group_layout,
        )?;

        Ok(Self {
            pipeline,
            bind_group_layout,
            bind_group,
            sampler,
            depth_sampler,
            color_texture,
            color_view,
            output_view: None,
            ssao_view: inputs.ssao.clone(),
            depth_view: inputs.depth.clone(),
            normal_view: inputs.normal.clone(),
            bloom_view: inputs.bloom.clone(),
            adobe_lut_dummy_texture,
            adobe_lut_dummy_view,
            adobe_lut_sampler,
            lut_bind_view,
            params,
            params_buffer,
            width,
            height,
        })
    }

    /// Upload a neutral `1³` `Rgba16Float` volume for Adobe LUT binding slot 8.
    fn create_adobe_lut_dummy_3d(
        context: &RenderContext,
    ) -> (wgpu::Texture, wgpu::TextureView) {
        let extent = wgpu::Extent3d {
            width: 1,
            height: 1,
            depth_or_array_layers: 1,
        };
        let texture = context.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Composite Adobe LUT dummy (1³)"),
            size: extent,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D3,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let view = texture.create_view(&wgpu::TextureViewDescriptor {
            label: Some("Composite Adobe LUT dummy view"),
            dimension: Some(wgpu::TextureViewDimension::D3),
            mip_level_count: Some(1),
            ..Default::default()
        });
        let one = f16::from_f32(1.0).to_le_bytes();
        let mut px = [0u8; 8];
        for ch in 0..4 {
            px[ch * 2..ch * 2 + 2].copy_from_slice(&one);
        }
        context.queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &px,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(8),
                rows_per_image: Some(1),
            },
            extent,
        );
        (texture, view)
    }

    fn create_adobe_lut_sampler(context: &RenderContext) -> wgpu::Sampler {
        context.device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Composite Adobe LUT sampler"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Nearest,
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            ..Default::default()
        })
    }

    fn create_samplers(
        context: &RenderContext,
    ) -> (wgpu::Sampler, wgpu::Sampler) {
        let sampler = linear_sampler(&context.device, "Composite Sampler");
        let depth_sampler =
            context.device.create_sampler(&wgpu::SamplerDescriptor {
                label: Some("Composite Depth Sampler"),
                address_mode_u: wgpu::AddressMode::ClampToEdge,
                address_mode_v: wgpu::AddressMode::ClampToEdge,
                mag_filter: wgpu::FilterMode::Nearest,
                min_filter: wgpu::FilterMode::Nearest,
                ..Default::default()
            });
        (sampler, depth_sampler)
    }

    fn create_bind_group_layout(
        context: &RenderContext,
    ) -> wgpu::BindGroupLayout {
        context.device.create_bind_group_layout(
            &wgpu::BindGroupLayoutDescriptor {
                label: Some("Composite Bind Group Layout"),
                entries: &[
                    texture_2d(0),
                    texture_2d(1),
                    depth_texture_2d(2),
                    filtering_sampler(3),
                    non_filtering_sampler(4),
                    uniform_buffer(5),
                    texture_2d(6),
                    texture_2d(7),
                    texture_3d_float(8),
                    filtering_sampler(9),
                ],
            },
        )
    }

    fn create_pipeline(
        context: &RenderContext,
        shader_composer: &mut ShaderComposer,
        bind_group_layout: &wgpu::BindGroupLayout,
    ) -> Result<wgpu::RenderPipeline, VisoError> {
        let shader =
            shader_composer.compose(&context.device, Shader::Composite)?;
        Ok(create_screen_space_pipeline(
            &context.device,
            &ScreenSpacePipelineDef {
                label: "Composite",
                shader: &shader,
                format: context.config.format,
                blend: None,
                bind_group_layouts: &[bind_group_layout],
            },
        ))
    }

    fn create_color_texture(
        context: &RenderContext,
        width: u32,
        height: u32,
    ) -> (wgpu::Texture, wgpu::TextureView) {
        create_render_texture(
            &context.device,
            width,
            height,
            wgpu::TextureFormat::Rgba16Float,
            "Intermediate Color Texture",
        )
    }

    fn create_bind_group(
        context: &RenderContext,
        layout: &wgpu::BindGroupLayout,
        views: &CompositeViews,
    ) -> wgpu::BindGroup {
        context
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Composite Bind Group"),
                layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(
                            views.color,
                        ),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(
                            views.ssao,
                        ),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::TextureView(
                            views.depth,
                        ),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: wgpu::BindingResource::Sampler(views.sampler),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: wgpu::BindingResource::Sampler(
                            views.depth_sampler,
                        ),
                    },
                    wgpu::BindGroupEntry {
                        binding: 5,
                        resource: views.params_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 6,
                        resource: wgpu::BindingResource::TextureView(
                            views.normal,
                        ),
                    },
                    wgpu::BindGroupEntry {
                        binding: 7,
                        resource: wgpu::BindingResource::TextureView(
                            views.bloom,
                        ),
                    },
                    wgpu::BindGroupEntry {
                        binding: 8,
                        resource: wgpu::BindingResource::TextureView(
                            views.adobe_lut_tex,
                        ),
                    },
                    wgpu::BindGroupEntry {
                        binding: 9,
                        resource: wgpu::BindingResource::Sampler(
                            views.adobe_lut_sampler,
                        ),
                    },
                ],
            })
    }

    /// Point composite binding 8 at `lut` (or dummy) and set
    /// [`CompositeParams::adobe_lut_grid_size`], then rebuild the bind group.
    pub(crate) fn sync_adobe_cube_lut(
        &mut self,
        context: &RenderContext,
        lut: Option<&AdobeCubeLutTexture>,
    ) {
        if let Some(l) = lut {
            self.params.adobe_lut_grid_size = l.grid_size();
            self.lut_bind_view = l.texture_view().clone();
        } else {
            self.params.adobe_lut_grid_size = 0;
            self.lut_bind_view = self.adobe_lut_dummy_view.clone();
        }
        context.queue.write_buffer(
            &self.params_buffer,
            0,
            bytemuck::cast_slice(&[self.params]),
        );
        self.bind_group = Self::create_bind_group(
            context,
            &self.bind_group_layout,
            &CompositeViews {
                color: &self.color_view,
                ssao: &self.ssao_view,
                depth: &self.depth_view,
                normal: &self.normal_view,
                bloom: &self.bloom_view,
                sampler: &self.sampler,
                depth_sampler: &self.depth_sampler,
                params_buffer: &self.params_buffer,
                adobe_lut_tex: &self.lut_bind_view,
                adobe_lut_sampler: &self.adobe_lut_sampler,
            },
        );
    }

    /// Set the output view (FXAA input texture) for this frame.
    pub(crate) fn set_output_view(&mut self, view: wgpu::TextureView) {
        self.output_view = Some(view);
    }

    /// Update the external texture views used in bind group recreation.
    pub(crate) fn set_external_views(
        &mut self,
        ssao: wgpu::TextureView,
        depth: wgpu::TextureView,
        normal: wgpu::TextureView,
        bloom: wgpu::TextureView,
    ) {
        self.ssao_view = ssao;
        self.depth_view = depth;
        self.normal_view = normal;
        self.bloom_view = bloom;
    }

    /// Get the color view for geometry rendering
    pub(crate) fn get_color_view(&self) -> &wgpu::TextureView {
        &self.color_view
    }

    /// Update fog parameters (called each frame from engine)
    pub(crate) fn update_fog(
        &mut self,
        queue: &wgpu::Queue,
        fog_start: f32,
        fog_density: f32,
    ) {
        self.params.fog_start = fog_start;
        self.params.fog_density = fog_density;
        queue.write_buffer(
            &self.params_buffer,
            0,
            bytemuck::cast_slice(&[self.params]),
        );
    }

    /// Flush the current params to the GPU buffer.
    pub(crate) fn flush_params(&self, queue: &wgpu::Queue) {
        queue.write_buffer(
            &self.params_buffer,
            0,
            bytemuck::cast_slice(&[self.params]),
        );
    }
}

impl ScreenPass for CompositePass {
    fn render(&self, encoder: &mut wgpu::CommandEncoder) {
        let Some(output_view) = &self.output_view else {
            return;
        };
        run_screen_pass(
            encoder,
            &ScreenPassDesc {
                label: "Composite Pass",
                view: output_view,
                pipeline: &self.pipeline,
                bind_group: &self.bind_group,
                clear_color: wgpu::Color::BLACK,
            },
        );
    }

    fn resize(&mut self, context: &RenderContext) {
        if context.render_width() == self.width
            && context.render_height() == self.height
        {
            return;
        }

        self.width = context.render_width();
        self.height = context.render_height();

        // Recreate color texture
        let (color_texture, color_view) =
            Self::create_color_texture(context, self.width, self.height);
        self.color_texture = color_texture;
        self.color_view = color_view;

        // Update screen_size in params
        self.params.screen_size = [self.width as f32, self.height as f32];
        context.queue.write_buffer(
            &self.params_buffer,
            0,
            bytemuck::cast_slice(&[self.params]),
        );

        // Recreate bind group with stored views
        self.bind_group = Self::create_bind_group(
            context,
            &self.bind_group_layout,
            &CompositeViews {
                color: &self.color_view,
                ssao: &self.ssao_view,
                depth: &self.depth_view,
                normal: &self.normal_view,
                bloom: &self.bloom_view,
                sampler: &self.sampler,
                depth_sampler: &self.depth_sampler,
                params_buffer: &self.params_buffer,
                adobe_lut_tex: &self.lut_bind_view,
                adobe_lut_sampler: &self.adobe_lut_sampler,
            },
        );
    }
}
