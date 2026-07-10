//! Reusable impostor-pass primitives.
//!
//! Every impostor renderer (capsule, sphere, cone, polygon) follows the same
//! pattern: one storage buffer, one bind group, one pipeline, and a
//! `draw(0..N, 0..instance_count)` call.  `ImpostorPass<T>` extracts that
//! boilerplate so each renderer just composes from shared primitives.

pub(crate) mod capsule;
pub(crate) mod cone;
pub(crate) mod polygon;
pub(crate) mod select_sphere;
pub(crate) mod sphere;

use bytemuck::{Pod, Zeroable};
pub(crate) use capsule::CapsuleInstance;
pub(crate) use cone::ConeInstance;
pub(crate) use polygon::ExtrudedPolygonInstance;
pub(crate) use select_sphere::SelectSphereInstance;
pub(crate) use sphere::SphereInstance;

use crate::error::VisoError;
use crate::gpu::dynamic_buffer::TypedBuffer;
use crate::gpu::{RenderContext, Shader, ShaderComposer};
use crate::renderer::{pipeline_util, PipelineLayouts};

/// Shader identity for an impostor pass.
pub(crate) struct ShaderDef {
    pub(crate) label: &'static str,
    pub(crate) shader: Shader,
}

/// A single impostor draw pass: pipeline + typed storage buffer + bind group.
///
/// All impostor shaders use the same bind group layout convention:
/// - group(0): camera uniform
/// - group(1): lighting uniform + textures
/// - group(2): per-residue overlay bitsets (binding 0 = selection, binding 1 =
///   non-designable)
/// - group(3): storage buffer (instances)
pub(crate) struct ImpostorPass<T: Pod + Zeroable> {
    pipeline: wgpu::RenderPipeline,
    instance_buffer: TypedBuffer<T>,
    bind_group_layout: wgpu::BindGroupLayout,
    bind_group: wgpu::BindGroup,
    pub(crate) instance_count: u32,
    vertices_per_instance: u32,
}

impl<T: Pod + Zeroable> ImpostorPass<T> {
    /// Create a new impostor pass with the given shader.
    pub(crate) fn new(
        context: &RenderContext,
        shader_def: &ShaderDef,
        layouts: &PipelineLayouts,
        vertices_per_instance: u32,
        shader_composer: &mut ShaderComposer,
    ) -> Result<Self, VisoError> {
        let label = shader_def.label;
        let instance_buffer = TypedBuffer::new_with_data(
            &context.device,
            &format!("{label} Buffer"),
            &[T::zeroed()],
            wgpu::BufferUsages::STORAGE,
        );

        let bind_group_layout =
            Self::create_bind_group_layout(&context.device, label);
        let bind_group = Self::create_bind_group(
            &context.device,
            &bind_group_layout,
            &instance_buffer,
            label,
        );
        let pipeline = Self::create_pipeline(
            context,
            shader_def,
            &bind_group_layout,
            layouts,
            shader_composer,
        )?;

        Ok(Self {
            pipeline,
            instance_buffer,
            bind_group_layout,
            bind_group,
            instance_count: 0,
            vertices_per_instance,
        })
    }

    fn create_bind_group_layout(
        device: &wgpu::Device,
        label: &str,
    ) -> wgpu::BindGroupLayout {
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some(&format!("{label} Layout")),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX
                    | wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        })
    }

    fn create_bind_group(
        device: &wgpu::Device,
        layout: &wgpu::BindGroupLayout,
        buffer: &TypedBuffer<T>,
        label: &str,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: buffer.buffer().as_entire_binding(),
            }],
            label: Some(&format!("{label} Bind Group")),
        })
    }

    fn create_pipeline(
        context: &RenderContext,
        shader_def: &ShaderDef,
        bind_group_layout: &wgpu::BindGroupLayout,
        layouts: &PipelineLayouts,
        shader_composer: &mut ShaderComposer,
    ) -> Result<wgpu::RenderPipeline, VisoError> {
        let label = shader_def.label;
        let shader =
            shader_composer.compose(&context.device, shader_def.shader)?;

        let pipeline_layout = context.device.create_pipeline_layout(
            &wgpu::PipelineLayoutDescriptor {
                label: Some(&format!("{label} Pipeline Layout")),
                bind_group_layouts: &[
                    Some(&layouts.camera),
                    Some(&layouts.lighting),
                    Some(&layouts.selection),
                    Some(bind_group_layout),
                ],
                immediate_size: 0,
            },
        );

        Ok(context.device.create_render_pipeline(
            &wgpu::RenderPipelineDescriptor {
                label: Some(&format!("{label} Pipeline")),
                layout: Some(&pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some("vs_main"),
                    buffers: &[],
                    compilation_options: Default::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: Some("fs_main"),
                    targets: &pipeline_util::hdr_fragment_targets(),
                    compilation_options: Default::default(),
                }),
                primitive: wgpu::PrimitiveState::default(),
                depth_stencil: Some(pipeline_util::depth_stencil_state()),
                multisample: wgpu::MultisampleState::default(),
                multiview_mask: None,
                cache: None,
            },
        ))
    }

    /// Write typed instances to the GPU buffer. Recreates the bind group if
    /// the buffer was reallocated.
    ///
    /// Returns `true` if the underlying buffer was reallocated (callers that
    /// hold external bind groups — e.g. picking — need to recreate them).
    pub(crate) fn write_instances(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        instances: &[T],
    ) -> bool {
        let data = if instances.is_empty() {
            &[T::zeroed()]
        } else {
            instances
        };
        let reallocated = self.instance_buffer.write(device, queue, data);
        if reallocated {
            self.bind_group = Self::create_bind_group(
                device,
                &self.bind_group_layout,
                &self.instance_buffer,
                "reallocated",
            );
        }
        self.instance_count = instances.len() as u32;
        reallocated
    }

    /// Write raw instance bytes to the GPU buffer. Recreates the bind group if
    /// the buffer was reallocated.
    ///
    /// Returns `true` if the underlying buffer was reallocated.
    pub(crate) fn write_bytes(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        bytes: &[u8],
        count: u32,
    ) -> bool {
        let zeroed = T::zeroed();
        let data = if bytes.is_empty() {
            bytemuck::bytes_of(&zeroed)
        } else {
            bytes
        };
        let reallocated = self.instance_buffer.write_bytes(device, queue, data);
        if reallocated {
            self.bind_group = Self::create_bind_group(
                device,
                &self.bind_group_layout,
                &self.instance_buffer,
                "reallocated",
            );
        }
        self.instance_count = count;
        reallocated
    }

    /// Issue the draw call for this pass.
    ///
    /// Sets the pipeline and bind groups 0–3, then draws.
    pub(crate) fn draw<'a>(
        &'a self,
        render_pass: &mut wgpu::RenderPass<'a>,
        bind_groups: &super::draw_context::DrawBindGroups<'a>,
    ) {
        if self.instance_count == 0 {
            return;
        }
        render_pass.set_pipeline(&self.pipeline);
        render_pass.set_bind_group(0, bind_groups.camera, &[]);
        render_pass.set_bind_group(1, bind_groups.lighting, &[]);
        render_pass.set_bind_group(2, bind_groups.selection, &[]);
        render_pass.set_bind_group(3, &self.bind_group, &[]);
        render_pass.draw(0..self.vertices_per_instance, 0..self.instance_count);
    }

    /// The underlying `wgpu::Buffer` (for picking bind groups, etc.).
    pub(crate) fn buffer(&self) -> &wgpu::Buffer {
        self.instance_buffer.buffer()
    }

    /// `(label, used_bytes, allocated_bytes)` for debug overlay.
    pub(crate) fn buffer_info(
        &self,
        label: &'static str,
    ) -> (&'static str, usize, usize) {
        (
            label,
            self.instance_buffer.len_bytes(),
            self.instance_buffer.capacity_bytes(),
        )
    }
}
