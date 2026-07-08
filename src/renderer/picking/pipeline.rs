//! GPU-based picking using a dedicated picking render pass
//!
//! Renders residue indices to an offscreen buffer, then reads back
//! the pixel at the mouse position to determine which residue is under the
//! cursor. This is exact - it matches exactly what's rendered on screen.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use wgpu::util::DeviceExt;

use crate::error::VisoError;
use crate::gpu::pipeline_helpers::read_only_storage_buffer;
use crate::gpu::{RenderContext, Shader, ShaderComposer};
use crate::renderer::geometry::backbone::backbone_vertex_buffer_layout;

/// Per-residue overlay buffers for GPU shaders. Owns the three bit-array
/// storage buffers that share group 2 in every geometry shader: the selection
/// highlight (`@binding(0)`), the non-designable bitset (`@binding(1)`), and
/// the pulse bitset (`@binding(2)`). One bind group carries all three, so the
/// geometry passes need only a single `set_bind_group(2, ...)` to feed every
/// overlay. All buffers are sized to the same `capacity` (one tracked
/// word-size invariant covers them all).
pub(crate) struct SelectionBuffer {
    buffer: wgpu::Buffer,
    /// Non-designable bitset (1 = residue may not be mutated). Same sizing
    /// and upload semantics as `buffer`; lives at `@binding(1)` of the shared
    /// group-2 layout.
    non_designable_buffer: wgpu::Buffer,
    /// Pulse bitset (1 = residue is pulsing). Same sizing and upload
    /// semantics as `buffer`; lives at `@binding(2)` of the shared group-2
    /// layout.
    pulse_buffer: wgpu::Buffer,
    /// Bind group layout for the shared group-2 overlay buffers (binding 0 =
    /// selection, binding 1 = non-designable, binding 2 = pulse).
    pub(crate) layout: wgpu::BindGroupLayout,
    /// Bind group referencing all overlay storage buffers.
    pub(crate) bind_group: wgpu::BindGroup,
    /// Number of residues (for sizing). All buffers share this capacity.
    capacity: usize,
}

impl SelectionBuffer {
    /// Create the overlay buffers sized for up to `max_residues` residues.
    pub(crate) fn new(device: &wgpu::Device, max_residues: usize) -> Self {
        // Round up to multiple of 32 bits
        let num_words = max_residues.div_ceil(32);
        let data = vec![0u32; num_words.max(1)];

        let buffer =
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Selection Buffer"),
                contents: bytemuck::cast_slice(&data),
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_DST,
            });
        let non_designable_buffer =
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Designability Buffer"),
                contents: bytemuck::cast_slice(&data),
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_DST,
            });
        let pulse_buffer =
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Pulse Buffer"),
                contents: bytemuck::cast_slice(&data),
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_DST,
            });

        let layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Residue Overlay Bind Group Layout"),
                entries: &[
                    read_only_storage_buffer(0),
                    read_only_storage_buffer(1),
                    read_only_storage_buffer(2),
                ],
            });

        let bind_group = Self::create_bind_group(
            device,
            &layout,
            &buffer,
            &non_designable_buffer,
            &pulse_buffer,
        );

        Self {
            buffer,
            non_designable_buffer,
            pulse_buffer,
            layout,
            bind_group,
            capacity: max_residues,
        }
    }

    /// Build the shared group-2 bind group binding every overlay buffer.
    fn create_bind_group(
        device: &wgpu::Device,
        layout: &wgpu::BindGroupLayout,
        selection: &wgpu::Buffer,
        non_designable: &wgpu::Buffer,
        pulse: &wgpu::Buffer,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Residue Overlay Bind Group"),
            layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: selection.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: non_designable.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: pulse.as_entire_binding(),
                },
            ],
        })
    }

    /// Pack a list of global residue indices into a bit-array sized to the
    /// current capacity.
    fn pack_bits(&self, residues: &[i32]) -> Vec<u32> {
        let num_words = self.capacity.div_ceil(32);
        let mut data = vec![0u32; num_words.max(1)];

        for &idx in residues {
            if idx >= 0 && (idx as usize) < self.capacity {
                let word_idx = idx as usize / 32;
                let bit_idx = idx as usize % 32;
                data[word_idx] |= 1u32 << bit_idx;
            }
        }
        data
    }

    /// Update selection state from a list of selected residue indices
    pub(crate) fn update(
        &self,
        queue: &wgpu::Queue,
        selected_residues: &[i32],
    ) {
        let data = self.pack_bits(selected_residues);
        queue.write_buffer(&self.buffer, 0, bytemuck::cast_slice(&data));
    }

    /// Update the non-designable bitset from a list of global residue indices.
    /// Mirrors [`Self::update`] but writes the `@binding(1)` overlay buffer.
    pub(crate) fn update_non_designable(
        &self,
        queue: &wgpu::Queue,
        non_designable_residues: &[i32],
    ) {
        let data = self.pack_bits(non_designable_residues);
        queue.write_buffer(
            &self.non_designable_buffer,
            0,
            bytemuck::cast_slice(&data),
        );
    }

    /// Update the pulse bitset from a list of global residue indices.
    /// Mirrors [`Self::update`] but writes the `@binding(2)` overlay buffer.
    pub(crate) fn update_pulse(
        &self,
        queue: &wgpu::Queue,
        pulse_residues: &[i32],
    ) {
        let data = self.pack_bits(pulse_residues);
        queue.write_buffer(&self.pulse_buffer, 0, bytemuck::cast_slice(&data));
    }

    /// Ensure every overlay buffer has capacity for at least `required`
    /// residues. Recreates the buffers and the shared bind group if the
    /// current capacity is insufficient.
    pub(crate) fn ensure_capacity(
        &mut self,
        device: &wgpu::Device,
        required: usize,
    ) {
        if required <= self.capacity {
            return;
        }

        // Need to grow - recreate the buffers with new capacity
        let new_capacity = required;
        let num_words = new_capacity.div_ceil(32);
        let data = vec![0u32; num_words.max(1)];

        self.buffer =
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Selection Buffer"),
                contents: bytemuck::cast_slice(&data),
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_DST,
            });
        self.non_designable_buffer =
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Designability Buffer"),
                contents: bytemuck::cast_slice(&data),
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_DST,
            });
        self.pulse_buffer =
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Pulse Buffer"),
                contents: bytemuck::cast_slice(&data),
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_DST,
            });

        self.bind_group = Self::create_bind_group(
            device,
            &self.layout,
            &self.buffer,
            &self.non_designable_buffer,
            &self.pulse_buffer,
        );

        self.capacity = new_capacity;
    }

    /// GPU buffer sizes: `(label, used_bytes, allocated_bytes)`.
    pub(crate) fn buffer_info(&self) -> Vec<(&'static str, usize, usize)> {
        let bytes = self.capacity.div_ceil(32).max(1) * 4;
        vec![
            ("Selection", bytes, bytes),
            ("Designability", bytes, bytes),
            ("Pulse", bytes, bytes),
        ]
    }
}

/// Geometry buffers needed for the picking render pass.
pub(crate) struct PickingGeometry<'a> {
    /// Backbone vertex buffer (shared by tube and ribbon passes).
    pub(crate) backbone_vertex_buffer: &'a wgpu::Buffer,
    /// Backbone tube index buffer (back-face culled pass).
    pub(crate) backbone_tube_index_buffer: &'a wgpu::Buffer,
    /// Number of backbone tube indices to draw.
    pub(crate) backbone_tube_index_count: u32,
    /// Backbone ribbon index buffer (no-cull pass).
    pub(crate) backbone_ribbon_index_buffer: &'a wgpu::Buffer,
    /// Number of backbone ribbon indices to draw.
    pub(crate) backbone_ribbon_index_count: u32,
    /// Sidechain capsule bind group for picking.
    pub(crate) capsule_bind_group: Option<&'a wgpu::BindGroup>,
    /// Number of sidechain capsule instances.
    pub(crate) capsule_count: u32,
    /// Ball-and-stick capsule bind group for picking.
    pub(crate) bns_capsule_bind_group: Option<&'a wgpu::BindGroup>,
    /// Number of ball-and-stick capsule instances.
    pub(crate) bns_capsule_count: u32,
    /// Ball-and-stick sphere bind group for picking.
    pub(crate) bns_sphere_bind_group: Option<&'a wgpu::BindGroup>,
    /// Number of ball-and-stick sphere instances.
    pub(crate) bns_sphere_count: u32,
}

/// Manages GPU-based residue picking via an offscreen R32Uint render pass.
pub(crate) struct Picking {
    /// Picking texture (R32Uint format for residue indices)
    texture: wgpu::Texture,
    texture_view: wgpu::TextureView,
    /// Depth texture for the picking pass
    depth_texture: wgpu::Texture,
    depth_view: wgpu::TextureView,
    /// Staging buffer for reading back pixel data
    staging_buffer: wgpu::Buffer,
    /// Pipeline for rendering tubes to picking buffer
    tube_pipeline: wgpu::RenderPipeline,
    /// Pipeline for rendering capsules to picking buffer
    capsule_pipeline: wgpu::RenderPipeline,
    /// Bind group layout for capsule storage buffer
    capsule_bind_group_layout: wgpu::BindGroupLayout,
    /// Pipeline for rendering spheres to picking buffer
    sphere_pipeline: wgpu::RenderPipeline,
    /// Bind group layout for sphere storage buffer
    sphere_bind_group_layout: wgpu::BindGroupLayout,
    /// Current dimensions
    width: u32,
    height: u32,
    /// Currently selected residue indices
    pub(crate) selected_residues: Vec<i32>,
    /// Whether a readback is in flight (buffer mapping requested)
    readback_in_flight: bool,
    /// Flag set by callback when buffer mapping is complete
    map_complete: Arc<AtomicBool>,
}

impl Picking {
    /// Create a new picking system with pipelines and textures sized to the
    /// current context.
    pub(crate) fn new(
        context: &RenderContext,
        camera_bind_group_layout: &wgpu::BindGroupLayout,
        shader_composer: &mut ShaderComposer,
    ) -> Result<Self, VisoError> {
        let width = context.config.width;
        let height = context.config.height;

        let (texture, texture_view) =
            Self::create_picking_texture(&context.device, width, height);
        let (depth_texture, depth_view) =
            Self::create_depth_texture(&context.device, width, height);

        let staging_buffer =
            context.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Picking Staging Buffer"),
                size: 256,
                usage: wgpu::BufferUsages::COPY_DST
                    | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            });

        let tube_pipeline = Self::create_tube_pipeline(
            context,
            camera_bind_group_layout,
            shader_composer,
        )?;
        let (capsule_pipeline, capsule_bind_group_layout) =
            Self::create_capsule_pipeline(
                context,
                camera_bind_group_layout,
                shader_composer,
            )?;
        let (sphere_pipeline, sphere_bind_group_layout) =
            Self::create_sphere_pipeline(
                context,
                camera_bind_group_layout,
                shader_composer,
            )?;

        Ok(Self {
            texture,
            texture_view,
            depth_texture,
            depth_view,
            staging_buffer,
            tube_pipeline,
            capsule_pipeline,
            capsule_bind_group_layout,
            sphere_pipeline,
            sphere_bind_group_layout,
            width,
            height,
            selected_residues: Vec::new(),
            readback_in_flight: false,
            map_complete: Arc::new(AtomicBool::new(false)),
        })
    }

    fn create_tube_pipeline(
        context: &RenderContext,
        camera_bind_group_layout: &wgpu::BindGroupLayout,
        shader_composer: &mut ShaderComposer,
    ) -> Result<wgpu::RenderPipeline, VisoError> {
        let shader =
            shader_composer.compose(&context.device, Shader::PickingMesh)?;

        let layout = context.device.create_pipeline_layout(
            &wgpu::PipelineLayoutDescriptor {
                label: Some("Picking Tube Pipeline Layout"),
                bind_group_layouts: &[Some(camera_bind_group_layout)],
                immediate_size: 0,
            },
        );

        Ok(context.device.create_render_pipeline(
            &wgpu::RenderPipelineDescriptor {
                label: Some("Picking Tube Pipeline"),
                layout: Some(&layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some("vs_main"),
                    buffers: &[backbone_vertex_buffer_layout()],
                    compilation_options: Default::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: Some("fs_main"),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: wgpu::TextureFormat::R32Uint,
                        blend: None,
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                    compilation_options: Default::default(),
                }),
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleList,
                    cull_mode: Some(wgpu::Face::Back),
                    ..Default::default()
                },
                depth_stencil: Some(picking_depth_stencil()),
                multisample: wgpu::MultisampleState::default(),
                multiview_mask: None,
                cache: None,
            },
        ))
    }

    fn create_capsule_pipeline(
        context: &RenderContext,
        camera_bind_group_layout: &wgpu::BindGroupLayout,
        shader_composer: &mut ShaderComposer,
    ) -> Result<(wgpu::RenderPipeline, wgpu::BindGroupLayout), VisoError> {
        Self::create_impostor_picking_pipeline(
            context,
            camera_bind_group_layout,
            shader_composer,
            Shader::PickingCapsule,
            "Picking Capsule",
        )
    }

    fn create_sphere_pipeline(
        context: &RenderContext,
        camera_bind_group_layout: &wgpu::BindGroupLayout,
        shader_composer: &mut ShaderComposer,
    ) -> Result<(wgpu::RenderPipeline, wgpu::BindGroupLayout), VisoError> {
        Self::create_impostor_picking_pipeline(
            context,
            camera_bind_group_layout,
            shader_composer,
            Shader::PickingSphere,
            "Picking Sphere",
        )
    }

    /// Shared pipeline creation for impostor-based picking (capsule, sphere).
    fn create_impostor_picking_pipeline(
        context: &RenderContext,
        camera_bind_group_layout: &wgpu::BindGroupLayout,
        shader_composer: &mut ShaderComposer,
        shader_variant: Shader,
        label: &str,
    ) -> Result<(wgpu::RenderPipeline, wgpu::BindGroupLayout), VisoError> {
        let shader =
            shader_composer.compose(&context.device, shader_variant)?;

        let bind_group_layout = storage_bind_group_layout(
            &context.device,
            &format!("{label} Bind Group Layout"),
        );

        let layout = context.device.create_pipeline_layout(
            &wgpu::PipelineLayoutDescriptor {
                label: Some(&format!("{label} Pipeline Layout")),
                bind_group_layouts: &[
                    Some(camera_bind_group_layout),
                    Some(&bind_group_layout),
                ],
                immediate_size: 0,
            },
        );

        let pipeline = context.device.create_render_pipeline(
            &wgpu::RenderPipelineDescriptor {
                label: Some(&format!("{label} Pipeline")),
                layout: Some(&layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some("vs_main"),
                    buffers: &[],
                    compilation_options: Default::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: Some("fs_main"),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: wgpu::TextureFormat::R32Uint,
                        blend: None,
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                    compilation_options: Default::default(),
                }),
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleList,
                    cull_mode: None,
                    ..Default::default()
                },
                depth_stencil: Some(picking_depth_stencil()),
                multisample: wgpu::MultisampleState::default(),
                multiview_mask: None,
                cache: None,
            },
        );

        Ok((pipeline, bind_group_layout))
    }

    fn create_picking_texture(
        device: &wgpu::Device,
        width: u32,
        height: u32,
    ) -> (wgpu::Texture, wgpu::TextureView) {
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Picking Texture"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::R32Uint,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        (texture, view)
    }

    fn create_depth_texture(
        device: &wgpu::Device,
        width: u32,
        height: u32,
    ) -> (wgpu::Texture, wgpu::TextureView) {
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Picking Depth Texture"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth32Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        (texture, view)
    }

    /// Resize the picking and depth textures to match the new dimensions.
    pub(crate) fn resize(
        &mut self,
        device: &wgpu::Device,
        width: u32,
        height: u32,
    ) {
        if width == self.width && height == self.height {
            return;
        }
        self.width = width;
        self.height = height;
        let (texture, texture_view) =
            Self::create_picking_texture(device, width, height);
        self.texture = texture;
        self.texture_view = texture_view;
        let (depth_texture, depth_view) =
            Self::create_depth_texture(device, width, height);
        self.depth_texture = depth_texture;
        self.depth_view = depth_view;
    }

    /// Create a bind group for capsule storage buffer
    pub(crate) fn create_capsule_bind_group(
        &self,
        device: &wgpu::Device,
        capsule_buffer: &wgpu::Buffer,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Picking Capsule Bind Group"),
            layout: &self.capsule_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: capsule_buffer.as_entire_binding(),
            }],
        })
    }

    /// Create a bind group for sphere storage buffer
    pub(crate) fn create_sphere_bind_group(
        &self,
        device: &wgpu::Device,
        sphere_buffer: &wgpu::Buffer,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Picking Sphere Bind Group"),
            layout: &self.sphere_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: sphere_buffer.as_entire_binding(),
            }],
        })
    }

    /// Render the picking pass and request readback at mouse position.
    ///
    /// In Ribbon mode, pass ribbon buffers to render helices/sheets for
    /// picking. Tubes are still rendered (for coils in ribbon mode, or
    /// everything in tube mode).
    pub(crate) fn render(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        camera_bind_group: &wgpu::BindGroup,
        geometry: &PickingGeometry,
        mouse_pos: (u32, u32),
    ) {
        self.encode_picking_pass(encoder, camera_bind_group, geometry);
        self.copy_pixel_to_staging(encoder, mouse_pos);
    }

    fn encode_picking_pass(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        camera_bind_group: &wgpu::BindGroup,
        geometry: &PickingGeometry,
    ) {
        let mut render_pass =
            encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Picking Render Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.texture_view,
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
                depth_stencil_attachment: Some(
                    wgpu::RenderPassDepthStencilAttachment {
                        view: &self.depth_view,
                        depth_ops: Some(wgpu::Operations {
                            load: wgpu::LoadOp::Clear(1.0),
                            store: wgpu::StoreOp::Store,
                        }),
                        stencil_ops: None,
                    },
                ),
                ..Default::default()
            });

        draw_picking_geometry(
            &mut render_pass,
            self,
            camera_bind_group,
            geometry,
        );
    }

    /// Copy the pixel at `mouse_pos` to the staging buffer for readback.
    fn copy_pixel_to_staging(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        (mouse_x, mouse_y): (u32, u32),
    ) {
        if mouse_x >= self.width
            || mouse_y >= self.height
            || self.readback_in_flight
        {
            return;
        }
        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: &self.texture,
                mip_level: 0,
                origin: wgpu::Origin3d {
                    x: mouse_x,
                    y: mouse_y,
                    z: 0,
                },
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &self.staging_buffer,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(256),
                    rows_per_image: Some(1),
                },
            },
            wgpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
        );
    }

    /// Start async readback (call after queue.submit)
    /// This initiates the buffer mapping without blocking
    pub(crate) fn start_readback(&mut self) {
        if self.readback_in_flight {
            return;
        }
        self.readback_in_flight = true;
        self.map_complete.store(false, Ordering::SeqCst);
        let map_complete = self.map_complete.clone();
        let buffer_slice = self.staging_buffer.slice(..4);
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            if result.is_ok() {
                map_complete.store(true, Ordering::SeqCst);
            }
        });
    }

    /// Try to complete the readback without blocking.
    /// Returns the raw pick ID if data was read, `None` if still pending.
    /// Uses previous frame's hover result until new data is ready.
    pub(crate) fn complete_readback(
        &mut self,
        device: &wgpu::Device,
    ) -> Option<u32> {
        if !self.readback_in_flight {
            return None;
        }

        // Poll without waiting - process callbacks
        let _ = device.poll(wgpu::PollType::Poll);

        // Check if the callback has signaled completion
        if !self.map_complete.load(Ordering::SeqCst) {
            return None;
        }

        // Buffer is mapped - read the data
        let buffer_slice = self.staging_buffer.slice(..4);
        let data = buffer_slice.get_mapped_range();
        let raw_id = u32::from_le_bytes([data[0], data[1], data[2], data[3]]);
        drop(data);
        self.staging_buffer.unmap();
        self.readback_in_flight = false;

        Some(raw_id)
    }
}

fn storage_bind_group_layout(
    device: &wgpu::Device,
    label: &str,
) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some(label),
        entries: &[read_only_storage_buffer(0)],
    })
}

fn picking_depth_stencil() -> wgpu::DepthStencilState {
    wgpu::DepthStencilState {
        format: wgpu::TextureFormat::Depth32Float,
        depth_write_enabled: Some(true),
        depth_compare: Some(wgpu::CompareFunction::Less),
        stencil: wgpu::StencilState::default(),
        bias: wgpu::DepthBiasState::default(),
    }
}

fn draw_picking_geometry(
    render_pass: &mut wgpu::RenderPass<'_>,
    picking: &Picking,
    camera_bind_group: &wgpu::BindGroup,
    geometry: &PickingGeometry,
) {
    // Draw backbone geometry (tube + ribbon share the same vertex buffer)
    render_pass.set_pipeline(&picking.tube_pipeline);
    render_pass.set_bind_group(0, camera_bind_group, &[]);
    render_pass.set_vertex_buffer(0, geometry.backbone_vertex_buffer.slice(..));

    if geometry.backbone_tube_index_count > 0 {
        render_pass.set_index_buffer(
            geometry.backbone_tube_index_buffer.slice(..),
            wgpu::IndexFormat::Uint32,
        );
        render_pass.draw_indexed(
            0..geometry.backbone_tube_index_count,
            0,
            0..1,
        );
    }

    if geometry.backbone_ribbon_index_count > 0 {
        render_pass.set_index_buffer(
            geometry.backbone_ribbon_index_buffer.slice(..),
            wgpu::IndexFormat::Uint32,
        );
        render_pass.draw_indexed(
            0..geometry.backbone_ribbon_index_count,
            0,
            0..1,
        );
    }

    // Draw capsules (sidechains)
    if let Some(capsule_bg) = geometry.capsule_bind_group {
        if geometry.capsule_count > 0 {
            render_pass.set_pipeline(&picking.capsule_pipeline);
            render_pass.set_bind_group(0, camera_bind_group, &[]);
            render_pass.set_bind_group(1, capsule_bg, &[]);
            render_pass.draw(0..6, 0..geometry.capsule_count);
        }
    }

    // Draw ball-and-stick bond capsules for picking
    if let Some(bns_bg) = geometry.bns_capsule_bind_group {
        if geometry.bns_capsule_count > 0 {
            render_pass.set_pipeline(&picking.capsule_pipeline);
            render_pass.set_bind_group(0, camera_bind_group, &[]);
            render_pass.set_bind_group(1, bns_bg, &[]);
            render_pass.draw(0..6, 0..geometry.bns_capsule_count);
        }
    }

    // Draw ball-and-stick spheres for picking
    if let Some(bns_sphere_bg) = geometry.bns_sphere_bind_group {
        if geometry.bns_sphere_count > 0 {
            render_pass.set_pipeline(&picking.sphere_pipeline);
            render_pass.set_bind_group(0, camera_bind_group, &[]);
            render_pass.set_bind_group(1, bns_sphere_bg, &[]);
            render_pass.draw(0..6, 0..geometry.bns_sphere_count);
        }
    }
}
