//! Unified backbone renderer for protein and nucleic acid chains.
//!
//! Two GPU pipelines share one vertex buffer:
//! - **Tube pass** (back-face culling): round cross-sections
//! - **Ribbon pass** (no culling): flat cross-sections

pub(crate) mod arrows;
pub(crate) mod curve;
pub(crate) mod index;
pub(crate) mod mesh;
pub(crate) mod path;
pub(crate) mod profile;
pub(crate) mod sheet_trace;
pub(crate) mod spline;

use glam::Vec3;
pub(crate) use mesh::ChainRange;
use molex::SSType;
pub(crate) use path::SheetOffset;

/// Output of backbone mesh generation.
#[derive(Default)]
pub(crate) struct BackboneMeshOutput {
    pub(crate) vertices: Vec<BackboneVertex>,
    pub(crate) tube_indices: Vec<u32>,
    pub(crate) ribbon_indices: Vec<u32>,
    pub(crate) sheet_offsets: Vec<SheetOffset>,
    pub(crate) chain_ranges: Vec<ChainRange>,
}

impl BackboneMeshOutput {
    /// Merge a per-chain mesh, recording a [`ChainRange`].
    fn push_chain(
        &mut self,
        chain: Self,
        bounding_center: Vec3,
        bounding_radius: f32,
    ) {
        let tube_index_start = self.tube_indices.len() as u32;
        let ribbon_index_start = self.ribbon_indices.len() as u32;

        self.vertices.extend(chain.vertices);
        self.tube_indices.extend(chain.tube_indices);
        self.ribbon_indices.extend(chain.ribbon_indices);
        self.sheet_offsets.extend(chain.sheet_offsets);

        self.chain_ranges.push(ChainRange::new(
            tube_index_start..self.tube_indices.len() as u32,
            ribbon_index_start..self.ribbon_indices.len() as u32,
            bounding_center,
            bounding_radius,
        ));
    }
}

use std::hash::{Hash, Hasher};

use rustc_hash::FxHasher;

use crate::camera::frustum::Frustum;
use crate::error::VisoError;
use crate::gpu::dynamic_buffer::DynamicBuffer;
use crate::gpu::{RenderContext, Shader, ShaderComposer};
use crate::options::{ChainLod, GeometryOptions};
use crate::renderer::draw_context::DrawBindGroups;
use crate::renderer::entity_topology::{NaBackboneChain, ProteinBackboneChain};
use crate::renderer::mesh::{create_mesh_pipeline, MeshPass, MeshPipelineDef};
use crate::util::hash::hash_vec3_slice_summary;

// ==================== VERTEX FORMAT ====================

/// 52-byte backbone vertex, shared by tube and ribbon pipelines.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct BackboneVertex {
    pub(crate) position: [f32; 3],
    pub(crate) normal: [f32; 3],
    pub(crate) color: [f32; 3],
    pub(crate) residue_idx: u32,
    pub(crate) center_pos: [f32; 3],
}

pub(crate) fn backbone_vertex_buffer_layout(
) -> wgpu::VertexBufferLayout<'static> {
    wgpu::VertexBufferLayout {
        array_stride: size_of::<BackboneVertex>() as wgpu::BufferAddress,
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
                format: wgpu::VertexFormat::Float32x3,
                offset: 24,
                shader_location: 2,
            },
            wgpu::VertexAttribute {
                format: wgpu::VertexFormat::Uint32,
                offset: 36,
                shader_location: 3,
            },
            wgpu::VertexAttribute {
                format: wgpu::VertexFormat::Float32x3,
                offset: 40,
                shader_location: 4,
            },
        ],
    }
}

// ==================== RENDERER ====================

/// Unified backbone renderer with tube + ribbon passes.
pub(crate) struct BackboneRenderer {
    tube_pass: MeshPass,
    ribbon_pass: MeshPass,
    vertex_buffer: DynamicBuffer,
    last_hash: u64,
    cached_chains: Vec<ProteinBackboneChain>,
    cached_na_chains: Vec<NaBackboneChain>,
    sheet_offsets: Vec<SheetOffset>,
    chain_ranges: Vec<ChainRange>,
    cached_lod_tiers: Vec<u8>,
}

impl BackboneRenderer {
    pub(crate) fn new(
        context: &RenderContext,
        layouts: &crate::renderer::PipelineLayouts,
        protein: &[ProteinBackboneChain],
        na: &[NaBackboneChain],
        shader_composer: &mut ShaderComposer,
    ) -> Result<Self, VisoError> {
        // No mesh generation here -- the scene processor background thread
        // handles all mesh generation. We just create empty GPU buffers
        // and pipelines. The first frame will be blank until
        // apply_prepared() uploads the background thread's result.

        let device = &context.device;
        let vertex_buffer = new_buffer::<BackboneVertex>(
            device,
            "Backbone Vertex",
            &[],
            wgpu::BufferUsages::VERTEX,
        );

        let mesh_layouts = &[
            &layouts.camera,
            &layouts.lighting,
            &layouts.selection,
            &layouts.color,
        ];
        let vl = backbone_vertex_buffer_layout();

        let tube_pipeline = create_mesh_pipeline(
            context,
            &MeshPipelineDef {
                label: "Backbone Tube",
                shader: Shader::BackboneTube,
                cull_mode: Some(wgpu::Face::Back),
                vertex_layout: vl.clone(),
            },
            mesh_layouts,
            shader_composer,
        )?;
        let ribbon_pipeline = create_mesh_pipeline(
            context,
            &MeshPipelineDef {
                label: "Backbone Ribbon",
                shader: Shader::BackboneTube,
                cull_mode: None,
                vertex_layout: vl,
            },
            mesh_layouts,
            shader_composer,
        )?;

        let tube_pass =
            MeshPass::new(device, "Backbone Tube Index", tube_pipeline, &[]);
        let ribbon_pass = MeshPass::new(
            device,
            "Backbone Ribbon Index",
            ribbon_pipeline,
            &[],
        );

        Ok(Self {
            tube_pass,
            ribbon_pass,
            vertex_buffer,
            last_hash: combined_hash(protein, na),
            cached_chains: protein.to_vec(),
            cached_na_chains: na.to_vec(),
            sheet_offsets: Vec::new(),
            chain_ranges: Vec::new(),
            cached_lod_tiers: Vec::new(),
        })
    }

    // -- Draw --

    /// Frustum-culled draw: skip chains whose bounding sphere is off-screen.
    pub(crate) fn draw_culled<'a>(
        &'a self,
        render_pass: &mut wgpu::RenderPass<'a>,
        bind_groups: &DrawBindGroups<'a>,
        frustum: &Frustum,
    ) {
        let Some(color) = bind_groups.color else {
            return;
        };
        render_pass.set_bind_group(0, bind_groups.camera, &[]);
        render_pass.set_bind_group(1, bind_groups.lighting, &[]);
        render_pass.set_bind_group(2, bind_groups.selection, &[]);
        render_pass.set_bind_group(3, color, &[]);

        let vb = self.vertex_buffer.buffer();

        if self.chain_ranges.is_empty() {
            // No chain range data -- fall back to full draw
            self.tube_pass.draw_indexed(render_pass, vb);
            self.ribbon_pass.draw_indexed(render_pass, vb);
            return;
        }

        for range in &self.chain_ranges {
            if !frustum
                .intersects_sphere(range.bounding_center, range.bounding_radius)
            {
                continue;
            }

            self.tube_pass
                .draw_indexed_range(render_pass, vb, range.tube());
            self.ribbon_pass.draw_indexed_range(
                render_pass,
                vb,
                range.ribbon(),
            );
        }
    }

    // -- Scene-processor path --

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn apply_prepared(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        vertices: &[u8],
        tube_indices: &[u8],
        ribbon_indices: &[u8],
        tube_index_count: u32,
        ribbon_index_count: u32,
        sheet_offsets: Vec<SheetOffset>,
        chain_ranges: Vec<ChainRange>,
        cached_chains: &[ProteinBackboneChain],
        cached_na_chains: &[NaBackboneChain],
    ) {
        if !vertices.is_empty() {
            // `write_bytes` only errors if a buffer grow/realloc fails;
            // a dropped upload yields a stale or blank frame that the
            // next sync overwrites -- not a fatal condition, so the
            // Result is deliberately ignored here.
            let _ = self.vertex_buffer.write_bytes(device, queue, vertices);
        }
        self.tube_pass.write_indices_bytes(
            device,
            queue,
            tube_indices,
            tube_index_count,
        );
        self.ribbon_pass.write_indices_bytes(
            device,
            queue,
            ribbon_indices,
            ribbon_index_count,
        );
        self.sheet_offsets = sheet_offsets;
        self.chain_ranges = chain_ranges;
        self.cached_chains.clear();
        self.cached_chains.extend_from_slice(cached_chains);
        self.cached_na_chains.clear();
        self.cached_na_chains.extend_from_slice(cached_na_chains);
        self.last_hash =
            combined_hash(&self.cached_chains, &self.cached_na_chains);
    }

    pub(crate) fn update_metadata(
        &mut self,
        cached_chains: &[ProteinBackboneChain],
        cached_na_chains: &[NaBackboneChain],
    ) {
        self.cached_chains.clear();
        self.cached_chains.extend_from_slice(cached_chains);
        self.cached_na_chains.clear();
        self.cached_na_chains.extend_from_slice(cached_na_chains);
        self.last_hash =
            combined_hash(&self.cached_chains, &self.cached_na_chains);
    }

    pub(crate) fn apply_mesh(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        mesh: crate::renderer::pipeline::prepared::BackboneMeshData,
    ) {
        if !mesh.vertices.is_empty() {
            // See `apply_prepared`: a failed buffer write only costs one
            // stale/blank frame, recovered on the next sync, so the
            // Result is intentionally not propagated.
            let _ =
                self.vertex_buffer
                    .write_bytes(device, queue, &mesh.vertices);
            self.tube_pass.write_indices_bytes(
                device,
                queue,
                &mesh.tube_indices,
                mesh.tube_index_count,
            );
            self.ribbon_pass.write_indices_bytes(
                device,
                queue,
                &mesh.ribbon_indices,
                mesh.ribbon_index_count,
            );
        }
        self.sheet_offsets = mesh.sheet_offsets;
        self.chain_ranges = mesh.chain_ranges;
    }

    // -- Accessors --

    pub(crate) fn chain_ranges(&self) -> &[ChainRange] {
        &self.chain_ranges
    }
    pub(crate) fn cached_lod_tiers(&self) -> &[u8] {
        &self.cached_lod_tiers
    }
    pub(crate) fn set_cached_lod_tiers(&mut self, tiers: Vec<u8>) {
        self.cached_lod_tiers = tiers;
    }
    pub(crate) fn sheet_offsets(&self) -> &[SheetOffset] {
        &self.sheet_offsets
    }
    pub(crate) fn cached_chains(&self) -> &[ProteinBackboneChain] {
        &self.cached_chains
    }
    pub(crate) fn cached_na_chains(&self) -> &[NaBackboneChain] {
        &self.cached_na_chains
    }
    pub(crate) fn vertex_buffer(&self) -> &wgpu::Buffer {
        self.vertex_buffer.buffer()
    }
    pub(crate) fn tube_index_buffer(&self) -> &wgpu::Buffer {
        self.tube_pass.index_buffer()
    }
    pub(crate) fn ribbon_index_buffer(&self) -> &wgpu::Buffer {
        self.ribbon_pass.index_buffer()
    }
    pub(crate) fn tube_index_count(&self) -> u32 {
        self.tube_pass.index_count
    }
    pub(crate) fn ribbon_index_count(&self) -> u32 {
        self.ribbon_pass.index_count
    }

    /// GPU buffer sizes: `(label, used_bytes, allocated_bytes)`.
    pub(crate) fn buffer_info(&self) -> Vec<(&'static str, usize, usize)> {
        vec![
            (
                "Backbone Vertex",
                self.vertex_buffer.len(),
                self.vertex_buffer.capacity(),
            ),
            (
                "Backbone Tube Idx",
                self.tube_pass.index_buffer_len(),
                self.tube_pass.index_buffer_capacity(),
            ),
            (
                "Backbone Ribbon Idx",
                self.ribbon_pass.index_buffer_len(),
                self.ribbon_pass.index_buffer_capacity(),
            ),
        ]
    }

    // -- Static mesh generation --

    pub(crate) fn generate_mesh_colored(
        protein: &[ProteinBackboneChain],
        na: &[NaBackboneChain],
        ss_override: Option<&[SSType]>,
        per_residue_colors: Option<&[[f32; 3]]>,
        geo: &GeometryOptions,
        per_chain_lod: Option<&[ChainLod]>,
        na_residue_colors: Option<&[[f32; 3]]>,
        na_seeds: Option<&[Option<Vec3>]>,
        na_guide_dirs: Option<&[Vec3]>,
    ) -> BackboneMeshOutput {
        mesh::generate_mesh_colored(
            protein,
            na,
            ss_override,
            per_residue_colors,
            geo,
            per_chain_lod,
            na_residue_colors,
            na_seeds,
            na_guide_dirs,
        )
    }
}

// -- Helpers --

fn new_buffer<T: bytemuck::Pod>(
    device: &wgpu::Device,
    label: &str,
    data: &[T],
    usage: wgpu::BufferUsages,
) -> DynamicBuffer {
    if data.is_empty() {
        DynamicBuffer::new(device, label, size_of::<T>() * 1000, usage)
    } else {
        DynamicBuffer::new_with_data(device, label, data, usage)
    }
}

fn combined_hash(
    protein_chains: &[ProteinBackboneChain],
    na_chains: &[NaBackboneChain],
) -> u64 {
    let mut h = FxHasher::default();
    // Hash CA positions per protein chain -- sufficient to detect mesh
    // rebuilds; N/C/O move alongside CA so CA alone is a reliable proxy.
    protein_chains.len().hash(&mut h);
    for chain in protein_chains {
        hash_vec3_slice_summary(chain.ca(), &mut h);
    }
    na_chains.len().hash(&mut h);
    for chain in na_chains {
        hash_vec3_slice_summary(chain.p(), &mut h);
    }
    h.finish()
}
