//! Capsule sidechain renderer
//!
//! Renders sidechains as capsule chains (cylinders with hemispherical caps).
//! This replaces both AtomRenderer and CylinderImpostorRenderer:
//! - Bonds are rendered as capsules
//! - "Atoms" are simply the hemispherical caps at capsule endpoints
//! - No separate sphere pass needed, no junction artifacts
//!
//! Uses the same capsule_impostor.wgsl shader as the tube renderer.

use glam::Vec3;
use molex::Element;

use crate::camera::frustum::Frustum;
use crate::error::VisoError;
use crate::gpu::{RenderContext, Shader, ShaderComposer};
use crate::renderer::impostor::{CapsuleInstance, ImpostorPass, ShaderDef};

/// Radius used for frustum culling (capsule bounding sphere)
const CULL_RADIUS: f32 = 5.0;

const CAPSULE_RADIUS: f32 = 0.3;

/// Borrowed view of sidechain geometry data for GPU rendering.
///
/// This is the renderer's input type — a set of borrowed slices that travel
/// together.
pub(crate) struct SidechainView<'a> {
    /// World-space atom positions.
    pub(crate) positions: &'a [Vec3],
    /// Intra-sidechain bonds as `(atom_a, atom_b)` index pairs.
    pub(crate) bonds: &'a [(u32, u32)],
    /// CA-CB backbone bonds as `(ca_position, cb_sidechain_index)`.
    pub(crate) backbone_bonds: &'a [(Vec3, u32)],
    /// Per-atom hydrophobicity flags.
    pub(crate) hydrophobicity: &'a [bool],
    /// Residue index for each sidechain atom.
    pub(crate) residue_indices: &'a [u32],
}

/// Renders sidechains as capsule chains (cylinders with hemispherical caps).
pub(crate) struct SidechainRenderer {
    pass: ImpostorPass<CapsuleInstance>,
}

impl SidechainRenderer {
    /// `sidechain.backbone_bonds`: backbone-anchor -> sidechain bonds
    /// (CA-CB for every residue, N-CD for proline's ring closure) as
    /// (anchor_position, cb_sidechain_index) tuples.
    pub(crate) fn new(
        context: &RenderContext,
        layouts: &crate::renderer::PipelineLayouts,
        sidechain: &SidechainView,
        shader_composer: &mut ShaderComposer,
    ) -> Result<Self, VisoError> {
        let mut pass = ImpostorPass::new(
            context,
            &ShaderDef {
                label: "Capsule Sidechain",
                shader: Shader::Capsule,
            },
            layouts,
            6,
            shader_composer,
        )?;

        // No frustum culling on initial creation
        let instances =
            Self::generate_instances(sidechain, None, None, None, None);

        let _ =
            pass.write_instances(&context.device, &context.queue, &instances);

        Ok(Self { pass })
    }

    /// Generate capsule instances from sidechain data.
    /// - `frustum`: Optional frustum for culling (None = no culling)
    /// - `sidechain_colors`: Optional (hydrophobic, hydrophilic) color override
    /// - `per_residue_colors`: When set, each sidechain atom is colored by its
    ///   residue's backbone color (overrides `sidechain_colors`).
    /// - `cpk_elements`: When set, a layout-indexed element slice; non-carbon
    ///   atoms take their element (CPK) color and carbon falls through to the
    ///   color mode selected above. `None` disables CPK coloring.
    pub(crate) fn generate_instances(
        sidechain: &SidechainView,
        frustum: Option<&Frustum>,
        sidechain_colors: Option<([f32; 3], [f32; 3])>,
        per_residue_colors: Option<&[[f32; 3]]>,
        cpk_elements: Option<&[Element]>,
    ) -> Vec<CapsuleInstance> {
        let mut instances = Vec::with_capacity(
            sidechain.bonds.len() + sidechain.backbone_bonds.len(),
        );

        // Callers pass explicit colors; the neutral fallback only applies
        // when there is no sidechain geometry to color (empty initial view).
        let (hydrophobic_color, hydrophilic_color) =
            sidechain_colors.unwrap_or(([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]));

        // Helper to get color for an atom index
        let get_color = |idx: usize| -> [f32; 3] {
            if let Some(color) = cpk_elements
                .and_then(|elements| elements.get(idx))
                .filter(|&&elem| elem != Element::C)
                .map(Element::cpk_color)
            {
                return color;
            }
            // Backbone color mode: look up the residue's backbone color.
            if let Some(colors) = per_residue_colors {
                let res =
                    sidechain.residue_indices.get(idx).copied().unwrap_or(0)
                        as usize;
                return colors.get(res).copied().unwrap_or([0.5, 0.5, 0.5]);
            }
            if sidechain.hydrophobicity.get(idx).copied().unwrap_or(false) {
                hydrophobic_color
            } else {
                hydrophilic_color
            }
        };

        // Helper to get residue index for an atom
        let get_residue_idx = |idx: usize| -> f32 {
            sidechain.residue_indices.get(idx).copied().unwrap_or(0) as f32
        };

        // Helper to check if a capsule is visible (either endpoint in frustum)
        let is_visible = |pos_a: Vec3, pos_b: Vec3| -> bool {
            frustum.is_none_or(|f| {
                let center = (pos_a + pos_b) * 0.5;
                let half_len = (pos_a - pos_b).length() * 0.5;
                f.intersects_sphere(center, half_len + CULL_RADIUS)
            })
        };

        // Sidechain-sidechain bonds
        for &(a, b) in sidechain.bonds {
            let a_idx = a as usize;
            let b_idx = b as usize;
            if a_idx >= sidechain.positions.len()
                || b_idx >= sidechain.positions.len()
            {
                continue;
            }

            let pos_a = sidechain.positions[a_idx];
            let pos_b = sidechain.positions[b_idx];

            // Frustum cull
            if !is_visible(pos_a, pos_b) {
                continue;
            }

            let color_a = get_color(a_idx);
            let color_b = get_color(b_idx);
            // Use residue index from first atom (both should be same residue
            // for internal bonds)
            let res_idx = get_residue_idx(a_idx);

            instances.push(CapsuleInstance {
                endpoint_a: [pos_a.x, pos_a.y, pos_a.z, CAPSULE_RADIUS],
                endpoint_b: [pos_b.x, pos_b.y, pos_b.z, res_idx],
                color_a: [color_a[0], color_a[1], color_a[2], 0.0],
                color_b: [color_b[0], color_b[1], color_b[2], 0.0],
            });
        }

        Self::emit_backbone_bonds(
            sidechain,
            &get_color,
            &get_residue_idx,
            &is_visible,
            &mut instances,
        );

        instances
    }

    /// Emit backbone-sidechain bonds (CA to CB) as capsule instances.
    fn emit_backbone_bonds(
        sidechain: &SidechainView,
        get_color: &impl Fn(usize) -> [f32; 3],
        get_residue_idx: &impl Fn(usize) -> f32,
        is_visible: &impl Fn(Vec3, Vec3) -> bool,
        instances: &mut Vec<CapsuleInstance>,
    ) {
        for &(ca_pos, cb_idx) in sidechain.backbone_bonds {
            let cb_idx = cb_idx as usize;
            if cb_idx >= sidechain.positions.len() {
                continue;
            }

            let cb_pos = sidechain.positions[cb_idx];

            // Frustum cull
            if !is_visible(ca_pos, cb_pos) {
                continue;
            }

            let cb_color = get_color(cb_idx);
            let res_idx = get_residue_idx(cb_idx);

            // CA end uses same color as CB for visual continuity
            instances.push(CapsuleInstance {
                endpoint_a: [ca_pos.x, ca_pos.y, ca_pos.z, CAPSULE_RADIUS],
                endpoint_b: [cb_pos.x, cb_pos.y, cb_pos.z, res_idx],
                color_a: [cb_color[0], cb_color[1], cb_color[2], 0.0],
                color_b: [cb_color[0], cb_color[1], cb_color[2], 0.0],
            });
        }
    }

    /// Re-upload the subset of already-prepared capsules whose bounding
    /// sphere intersects the camera frustum. This is a pure filter: each
    /// survivor keeps its color and global pick id verbatim, so no color
    /// or visibility is re-derived here.
    pub(crate) fn update_with_frustum(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        prepared: &[CapsuleInstance],
        frustum: &Frustum,
    ) {
        let visible: Vec<CapsuleInstance> = prepared
            .iter()
            .copied()
            .filter(|c| {
                let pos_a = Vec3::new(
                    c.endpoint_a[0],
                    c.endpoint_a[1],
                    c.endpoint_a[2],
                );
                let pos_b = Vec3::new(
                    c.endpoint_b[0],
                    c.endpoint_b[1],
                    c.endpoint_b[2],
                );
                let center = (pos_a + pos_b) * 0.5;
                let half_len = (pos_a - pos_b).length() * 0.5;
                frustum.intersects_sphere(center, half_len + CULL_RADIUS)
            })
            .collect();

        let _ = self.pass.write_instances(device, queue, &visible);
    }

    /// Draw sidechain capsules into the given render pass.
    pub(crate) fn draw<'a>(
        &'a self,
        render_pass: &mut wgpu::RenderPass<'a>,
        bind_groups: &crate::renderer::draw_context::DrawBindGroups<'a>,
    ) {
        self.pass.draw(render_pass, bind_groups);
    }

    /// Apply pre-computed instance data (GPU upload only, no CPU generation).
    ///
    /// Returns `true` if the buffer was reallocated (picking bind groups need
    /// recreation).
    pub(crate) fn apply_prepared(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        instances: &[u8],
        instance_count: u32,
    ) -> bool {
        self.pass
            .write_bytes(device, queue, instances, instance_count)
    }

    /// Get the capsule instance buffer for picking
    pub(crate) fn capsule_buffer(&self) -> &wgpu::Buffer {
        self.pass.buffer()
    }

    /// Number of capsule instances currently drawn.
    pub(crate) fn instance_count(&self) -> u32 {
        self.pass.instance_count
    }

    /// GPU buffer sizes: `(label, used_bytes, allocated_bytes)`.
    pub(crate) fn buffer_info(&self) -> Vec<(&'static str, usize, usize)> {
        vec![self.pass.buffer_info("Sidechain Capsules")]
    }
}
