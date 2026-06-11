//! Shared indexed-mesh draw-pass abstraction.
//!
//! `MeshPass` owns a render pipeline and an index buffer, providing a
//! reusable draw/write interface for indexed-mesh renderers. The vertex
//! buffer is external so multiple passes can share one (e.g. backbone
//! tube + ribbon passes reference the same vertices).
//!
//! Used by `BackboneRenderer` today; designed for future use by isosurface
//! renderers and any other indexed-mesh geometry.

use crate::error::VisoError;
use crate::gpu::dynamic_buffer::DynamicBuffer;
use crate::gpu::{RenderContext, Shader, ShaderComposer};
use crate::renderer::pipeline_util;

/// Description of an indexed-mesh render pipeline.
pub(crate) struct MeshPipelineDef {
    pub(crate) label: &'static str,
    pub(crate) shader: Shader,
    pub(crate) cull_mode: Option<wgpu::Face>,
    pub(crate) vertex_layout: wgpu::VertexBufferLayout<'static>,
    /// Exclude this mesh from the normal G-buffer. The mesh still writes
    /// depth like opaque geometry (so it self-occludes its own folds and
    /// is occluded by nearer opaque surfaces), but its normal target is
    /// fully masked so the normal-edge outline and SSAO never key off it.
    /// Used for alpha-composited surfaces that should depth-test and
    /// self-occlude correctly yet cast no normal-edge silhouette. Meshes
    /// that belong in the normal G-buffer set this `false`.
    pub(crate) transparent: bool,
}

/// Create a standard indexed-mesh render pipeline.
pub(crate) fn create_mesh_pipeline(
    context: &RenderContext,
    def: &MeshPipelineDef,
    bind_group_layouts: &[&wgpu::BindGroupLayout],
    shader_composer: &mut ShaderComposer,
) -> Result<wgpu::RenderPipeline, VisoError> {
    let label = def.label;
    let shader = shader_composer.compose(&context.device, def.shader)?;

    let pipeline_layout = context.device.create_pipeline_layout(
        &wgpu::PipelineLayoutDescriptor {
            label: Some(&format!("{label} Layout")),
            bind_group_layouts,
            push_constant_ranges: &[],
        },
    );

    // `transparent` meshes keep the opaque depth state (depth-write on,
    // depth-compare Less) so they self-occlude their own folds and are
    // occluded by nearer opaque geometry; only their normal target is
    // fully masked so the normal-edge outline and SSAO ignore them.
    // Opaque meshes use the shared defaults unchanged.
    let mut fragment_targets = pipeline_util::hdr_fragment_targets();
    let depth_stencil = pipeline_util::depth_stencil_state();
    if def.transparent {
        if let Some(normal_target) = &mut fragment_targets[1] {
            normal_target.write_mask = wgpu::ColorWrites::empty();
        }
    }

    Ok(context
        .device
        .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some(label),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: std::slice::from_ref(&def.vertex_layout),
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                targets: &fragment_targets,
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                cull_mode: def.cull_mode,
                ..Default::default()
            },
            depth_stencil: Some(depth_stencil),
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        }))
}

/// An indexed-mesh draw pass: pipeline + index buffer.
///
/// Vertex buffer is external so passes can share vertices.
pub(crate) struct MeshPass {
    pipeline: wgpu::RenderPipeline,
    index_buffer: DynamicBuffer,
    pub(crate) index_count: u32,
}

impl MeshPass {
    /// Create a pass with initial index data.
    pub(crate) fn new(
        device: &wgpu::Device,
        label: &str,
        pipeline: wgpu::RenderPipeline,
        indices: &[u32],
    ) -> Self {
        let index_buffer = if indices.is_empty() {
            DynamicBuffer::new(
                device,
                label,
                size_of::<u32>() * 3000,
                wgpu::BufferUsages::INDEX,
            )
        } else {
            DynamicBuffer::new_with_data(
                device,
                label,
                indices,
                wgpu::BufferUsages::INDEX,
            )
        };
        Self {
            pipeline,
            index_buffer,
            index_count: indices.len() as u32,
        }
    }

    /// Set pipeline, vertex buffer, index buffer, and draw.
    ///
    /// Caller must set bind groups before calling this.
    pub(crate) fn draw_indexed<'a>(
        &'a self,
        render_pass: &mut wgpu::RenderPass<'a>,
        vertex_buffer: &'a wgpu::Buffer,
    ) {
        if self.index_count == 0 {
            return;
        }
        render_pass.set_pipeline(&self.pipeline);
        render_pass.set_vertex_buffer(0, vertex_buffer.slice(..));
        render_pass.set_index_buffer(
            self.index_buffer.buffer().slice(..),
            wgpu::IndexFormat::Uint32,
        );
        render_pass.draw_indexed(0..self.index_count, 0, 0..1);
    }

    /// Draw a sub-range of the index buffer (for frustum culling).
    ///
    /// Caller must set bind groups before calling this.
    pub(crate) fn draw_indexed_range<'a>(
        &'a self,
        render_pass: &mut wgpu::RenderPass<'a>,
        vertex_buffer: &'a wgpu::Buffer,
        index_range: std::ops::Range<u32>,
    ) {
        if index_range.is_empty() {
            return;
        }
        render_pass.set_pipeline(&self.pipeline);
        render_pass.set_vertex_buffer(0, vertex_buffer.slice(..));
        render_pass.set_index_buffer(
            self.index_buffer.buffer().slice(..),
            wgpu::IndexFormat::Uint32,
        );
        render_pass.draw_indexed(index_range, 0, 0..1);
    }

    /// Write typed index data.
    pub(crate) fn write_indices(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        indices: &[u32],
    ) {
        if !indices.is_empty() {
            let _ = self.index_buffer.write(device, queue, indices);
        }
        self.index_count = indices.len() as u32;
    }

    /// Write raw index bytes (from scene processor).
    pub(crate) fn write_indices_bytes(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        data: &[u8],
        count: u32,
    ) {
        if !data.is_empty() {
            let _ = self.index_buffer.write_bytes(device, queue, data);
        }
        self.index_count = count;
    }

    /// Get the underlying index buffer (for picking).
    pub(crate) fn index_buffer(&self) -> &wgpu::Buffer {
        self.index_buffer.buffer()
    }

    /// Current index data size in bytes.
    pub(crate) fn index_buffer_len(&self) -> usize {
        self.index_buffer.len()
    }

    /// Allocated index buffer capacity in bytes.
    pub(crate) fn index_buffer_capacity(&self) -> usize {
        self.index_buffer.capacity()
    }
}
