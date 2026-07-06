//! Isosurface mesh renderer for electron density maps and molecular
//! surfaces.
//!
//! Renders isosurfaces as triangle meshes via marching cubes on the
//! CPU, uploaded through a `MeshPass` + `DynamicBuffer` vertex buffer.
//! The surface alpha-composites its color over the opaque scene. It
//! writes depth like opaque geometry, so it self-occludes its own folds
//! and is occluded by nearer opaque surfaces, but it is masked out of
//! the normal render target, so it stays out of the normal-edge outline
//! and SSAO.

pub(crate) mod cavity;
pub(crate) mod cpu_marching_cubes;
pub(crate) mod density;
pub(crate) mod gaussian_surface;
pub(crate) mod mesh_smooth;
pub(crate) mod ses;
pub(crate) mod tables;

use crate::error::VisoError;
use crate::gpu::dynamic_buffer::DynamicBuffer;
use crate::gpu::pipeline_helpers::texture_2d_unfilterable;
use crate::gpu::{RenderContext, Shader, ShaderComposer};
use crate::renderer::draw_context::DrawBindGroups;
use crate::renderer::mesh::{create_mesh_pipeline, MeshPass, MeshPipelineDef};
use crate::renderer::PipelineLayouts;

/// Discriminator tagging which kind of isosurface a vertex belongs to.
///
/// Stored as a `u32` in [`IsosurfaceVertex::kind`] so the shared
/// isosurface shader can branch on the source kind without inspecting
/// vertex color or relying on separate draw calls. Cavity-specific
/// effects (pulsing rim, future depth absorption, etc.) gate on this.
///
/// Values are pinned because they're mirrored verbatim in the WGSL
/// shader (`isosurface.wgsl`).
pub(crate) mod isosurface_kind {
    /// Molecular surface mesh — Gaussian or SES. The global surface
    /// opacity uniform scales fragments of this kind; per-entity opacity
    /// is still baked into vertex alpha. Default value.
    pub(crate) const SURFACE: u32 = 0;
    /// Internal cavity mesh — gets the cavity-specific pulsing rim and
    /// (eventually) volumetric / depth-absorption shading.
    pub(crate) const CAVITY: u32 = 1;
    /// Electron-density map mesh — its opacity comes from the density
    /// entry's own baked alpha and is excluded from the surface-opacity
    /// uniform scale.
    pub(crate) const DENSITY: u32 = 2;
}

/// A vertex on the extracted isosurface.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct IsosurfaceVertex {
    /// World-space position.
    pub(crate) position: [f32; 3],
    /// Surface normal (central-difference gradient).
    pub(crate) normal: [f32; 3],
    /// Vertex color RGBA (alpha controls transparency).
    pub(crate) color: [f32; 4],
    /// Source kind discriminator — see [`isosurface_kind`].
    pub(crate) kind: u32,
    /// World-space centroid of the parent cavity, baked at mesh build
    /// time. Only meaningful when `kind == isosurface_kind::CAVITY` —
    /// other kinds set this to `[0.0; 3]` and the shader ignores it.
    /// Used by the vertex shader for radial-breath displacement.
    pub(crate) cavity_center: [f32; 3],
}

/// Isosurface vertex buffer layout for wgpu.
pub(crate) fn isosurface_vertex_layout() -> wgpu::VertexBufferLayout<'static> {
    wgpu::VertexBufferLayout {
        array_stride: size_of::<IsosurfaceVertex>() as wgpu::BufferAddress,
        step_mode: wgpu::VertexStepMode::Vertex,
        attributes: &[
            wgpu::VertexAttribute {
                format: wgpu::VertexFormat::Float32x3,
                offset: 0,
                shader_location: 0,
            },
            wgpu::VertexAttribute {
                format: wgpu::VertexFormat::Float32x3,
                offset: 12,
                shader_location: 1,
            },
            wgpu::VertexAttribute {
                format: wgpu::VertexFormat::Float32x4,
                offset: 24,
                shader_location: 2,
            },
            wgpu::VertexAttribute {
                format: wgpu::VertexFormat::Uint32,
                offset: 40,
                shader_location: 3,
            },
            wgpu::VertexAttribute {
                format: wgpu::VertexFormat::Float32x3,
                offset: 44,
                shader_location: 4,
            },
        ],
    }
}

/// Renderer for isosurface meshes (cavities, SES, Gaussian, density).
///
/// Owns the vertex buffer, the main `MeshPass` (index buffer + main
/// pipeline) and a separate back-face depth pipeline used for the
/// thickness pre-pass. The back-face pre-pass writes linear view-space
/// z to an external R32Float texture; the main pass samples that
/// texture (bound as group 2) to compute thickness for Beer-Lambert.
pub(crate) struct IsosurfaceRenderer {
    mesh_pass: MeshPass,
    vertex_buffer: DynamicBuffer,
    vertex_count: u32,
    back_face_pipeline: wgpu::RenderPipeline,
    back_face_bind_group_layout: wgpu::BindGroupLayout,
    back_face_bind_group: wgpu::BindGroup,
}

impl IsosurfaceRenderer {
    /// Create a new isosurface renderer.
    ///
    /// `back_face_depth_view` is the external R32Float texture view that
    /// the back-face pre-pass writes to and the main pass samples. Pass
    /// the same view that's resized in lockstep with the framebuffer
    /// (see `PostProcessStack::backface_depth_view`).
    pub(crate) fn new(
        context: &RenderContext,
        layouts: &PipelineLayouts,
        shader_composer: &mut ShaderComposer,
        back_face_depth_view: &wgpu::TextureView,
    ) -> Result<Self, VisoError> {
        let back_face_bind_group_layout = context
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Isosurface Backface Depth Layout"),
                entries: &[texture_2d_unfilterable(0)],
            });

        let pipeline = create_mesh_pipeline(
            context,
            &MeshPipelineDef {
                label: "Isosurface",
                shader: Shader::Isosurface,
                cull_mode: Some(wgpu::Face::Back),
                vertex_layout: isosurface_vertex_layout(),
                transparent: true,
            },
            &[
                &layouts.camera,
                &layouts.lighting,
                &back_face_bind_group_layout,
            ],
            shader_composer,
        )?;

        let back_face_pipeline = create_back_face_depth_pipeline(
            context,
            shader_composer,
            &layouts.camera,
        )?;

        let mesh_pass =
            MeshPass::new(&context.device, "Isosurface Indices", pipeline, &[]);

        let vertex_buffer = DynamicBuffer::new(
            &context.device,
            "Isosurface Vertices",
            size_of::<IsosurfaceVertex>() * 1000,
            wgpu::BufferUsages::VERTEX,
        );

        let back_face_bind_group = create_back_face_bind_group(
            &context.device,
            &back_face_bind_group_layout,
            back_face_depth_view,
        );

        Ok(Self {
            mesh_pass,
            vertex_buffer,
            vertex_count: 0,
            back_face_pipeline,
            back_face_bind_group_layout,
            back_face_bind_group,
        })
    }

    /// Rebuild the back-face depth bind group when the underlying
    /// texture is recreated (e.g. after a window resize).
    pub(crate) fn set_back_face_depth_view(
        &mut self,
        device: &wgpu::Device,
        view: &wgpu::TextureView,
    ) {
        self.back_face_bind_group = create_back_face_bind_group(
            device,
            &self.back_face_bind_group_layout,
            view,
        );
    }

    /// Upload new isosurface mesh data to the GPU.
    pub(crate) fn apply_prepared(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        vertices: &[IsosurfaceVertex],
        indices: &[u32],
    ) {
        if vertices.is_empty() || indices.is_empty() {
            self.vertex_count = 0;
            self.mesh_pass.write_indices(device, queue, &[]);
            return;
        }
        let _ = self.vertex_buffer.write(device, queue, vertices);
        self.vertex_count = vertices.len() as u32;
        self.mesh_pass.write_indices(device, queue, indices);
    }

    /// Draw the isosurface mesh into the given render pass.
    pub(crate) fn draw<'a>(
        &'a self,
        render_pass: &mut wgpu::RenderPass<'a>,
        bind_groups: &DrawBindGroups<'a>,
    ) {
        if self.vertex_count == 0 {
            return;
        }
        render_pass.set_bind_group(0, bind_groups.camera, &[]);
        render_pass.set_bind_group(1, bind_groups.lighting, &[]);
        render_pass.set_bind_group(2, &self.back_face_bind_group, &[]);
        self.mesh_pass
            .draw_indexed(render_pass, self.vertex_buffer.buffer());
    }

    /// Draw the back-face depth pre-pass.
    ///
    /// Caller is responsible for setting up a render pass with the
    /// R32Float color attachment and binding the camera bind group.
    /// Renders all isosurface back-faces (front-face culling) writing
    /// linear view-space z.
    pub(crate) fn draw_back_face_pass<'a>(
        &'a self,
        render_pass: &mut wgpu::RenderPass<'a>,
        camera_bind_group: &'a wgpu::BindGroup,
    ) {
        if self.vertex_count == 0 {
            return;
        }
        render_pass.set_pipeline(&self.back_face_pipeline);
        render_pass.set_bind_group(0, camera_bind_group, &[]);
        render_pass.set_vertex_buffer(0, self.vertex_buffer.buffer().slice(..));
        render_pass.set_index_buffer(
            self.mesh_pass.index_buffer().slice(..),
            wgpu::IndexFormat::Uint32,
        );
        render_pass.draw_indexed(0..self.mesh_pass.index_count, 0, 0..1);
    }

    /// GPU buffer sizes: `(label, used_bytes, allocated_bytes)`.
    pub(crate) fn buffer_info(&self) -> Vec<(&'static str, usize, usize)> {
        vec![
            (
                "Isosurface Vertices",
                self.vertex_buffer.len(),
                self.vertex_buffer.capacity(),
            ),
            (
                "Isosurface Indices",
                self.mesh_pass.index_buffer_len(),
                self.mesh_pass.index_buffer_capacity(),
            ),
        ]
    }
}

fn create_back_face_bind_group(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    view: &wgpu::TextureView,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Isosurface Backface Depth Bind Group"),
        layout,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: wgpu::BindingResource::TextureView(view),
        }],
    })
}

fn create_back_face_depth_pipeline(
    context: &RenderContext,
    shader_composer: &mut ShaderComposer,
    camera_layout: &wgpu::BindGroupLayout,
) -> Result<wgpu::RenderPipeline, VisoError> {
    let shader =
        shader_composer.compose(&context.device, Shader::BackfaceDepth)?;

    let pipeline_layout = context.device.create_pipeline_layout(
        &wgpu::PipelineLayoutDescriptor {
            label: Some("Isosurface Backface Depth Layout"),
            bind_group_layouts: &[Some(camera_layout)],
            immediate_size: 0,
        },
    );

    let vertex_layout = isosurface_vertex_layout();
    Ok(context
        .device
        .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Isosurface Backface Depth"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: std::slice::from_ref(&vertex_layout),
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: wgpu::TextureFormat::R32Float,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                cull_mode: Some(wgpu::Face::Front),
                ..Default::default()
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        }))
}
