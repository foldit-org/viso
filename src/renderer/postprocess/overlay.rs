//! Overlay pass: blend a host-supplied texture over the presented frame.
//!
//! Runs last, after FXAA has written the swapchain view, and loads rather than
//! clears so the 3D scene survives underneath. The source is premultiplied
//! alpha and composites source-over.
//!
//! The overlay is deliberately not part of the render-scale (SSAA) chain: it is
//! sampled 1:1 against the swapchain, so a host compositing UI text through it
//! is never resampled.

use crate::error::VisoError;
use crate::gpu::pipeline_helpers::linear_sampler;
use crate::gpu::{RenderContext, Shader, ShaderComposer};

/// Source-over blending for a premultiplied-alpha source.
const PREMULTIPLIED_OVER: wgpu::BlendState = wgpu::BlendState {
    color: wgpu::BlendComponent {
        src_factor: wgpu::BlendFactor::One,
        dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
        operation: wgpu::BlendOperation::Add,
    },
    alpha: wgpu::BlendComponent {
        src_factor: wgpu::BlendFactor::One,
        dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
        operation: wgpu::BlendOperation::Add,
    },
};

pub(crate) struct OverlayPass {
    pipeline: wgpu::RenderPipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    sampler: wgpu::Sampler,
    /// `None` until a host installs a texture; the pass is then a no-op.
    bind_group: Option<wgpu::BindGroup>,
}

impl OverlayPass {
    pub(crate) fn new(
        context: &RenderContext,
        shader_composer: &mut ShaderComposer,
    ) -> Result<Self, VisoError> {
        let shader = shader_composer.compose(&context.device, Shader::Overlay)?;
        let bind_group_layout =
            context
                .device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("Overlay Bind Group Layout"),
                    entries: &[
                        wgpu::BindGroupLayoutEntry {
                            binding: 0,
                            visibility: wgpu::ShaderStages::FRAGMENT,
                            ty: wgpu::BindingType::Texture {
                                sample_type: wgpu::TextureSampleType::Float { filterable: true },
                                view_dimension: wgpu::TextureViewDimension::D2,
                                multisampled: false,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 1,
                            visibility: wgpu::ShaderStages::FRAGMENT,
                            ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                            count: None,
                        },
                    ],
                });

        // Not `create_screen_space_pipeline`: the fragment entry point depends
        // on the color space the blend unit will composite in.
        let format = context.config.format;
        let layout = context
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Overlay Pipeline Layout"),
                bind_group_layouts: &[Some(&bind_group_layout)],
                immediate_size: 0,
            });
        let pipeline = context
            .device
            .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("Overlay Pipeline"),
                layout: Some(&layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some("vs_main"),
                    buffers: &[],
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: Some(if format.is_srgb() {
                        "fs_linear_target"
                    } else {
                        "fs_main"
                    }),
                    targets: &[Some(wgpu::ColorTargetState {
                        format,
                        blend: Some(PREMULTIPLIED_OVER),
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                }),
                primitive: wgpu::PrimitiveState::default(),
                depth_stencil: None,
                multisample: wgpu::MultisampleState::default(),
                multiview_mask: None,
                cache: None,
            });

        Ok(Self {
            pipeline,
            bind_group_layout,
            sampler: linear_sampler(&context.device, "Overlay Sampler"),
            bind_group: None,
        })
    }

    /// Install (or clear) the overlay source. The bind group is rebuilt only
    /// here, so a host that keeps one texture alive pays nothing per frame.
    pub(crate) fn set_texture(
        &mut self,
        device: &wgpu::Device,
        view: Option<&wgpu::TextureView>,
    ) {
        self.bind_group = view.map(|view| {
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Overlay Bind Group"),
                layout: &self.bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Sampler(&self.sampler),
                    },
                ],
            })
        });
    }

    /// Blend the overlay onto `target`. Loads the existing contents; a
    /// `run_screen_pass` dispatch cannot be reused because it always clears.
    pub(crate) fn render(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        target: &wgpu::TextureView,
    ) {
        let Some(bind_group) = self.bind_group.as_ref() else {
            return;
        };
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("Overlay Pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: target,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Load,
                    store: wgpu::StoreOp::Store,
                },
                depth_slice: None,
            })],
            depth_stencil_attachment: None,
            ..Default::default()
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, bind_group, &[]);
        pass.draw(0..3, 0..1);
    }
}
