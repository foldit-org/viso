//! Pull renderer
//!
//! Renders the active pull constraint during drag operations.
//! A pull is a temporary constraint used to drag an atom toward the mouse.
//!
//! Visual style matches original Foldit:
//! - Cylinder from atom to near the mouse position
//! - Cone/arrow at the mouse end pointing toward the target
//! - Purple color
//!
//! Only one pull can be active at a time.

use crate::engine::command::ResolvedPull;
use crate::error::VisoError;
use crate::gpu::{RenderContext, Shader, ShaderComposer};
use crate::renderer::impostor::{
    CapsuleInstance, ConeInstance, ImpostorPass, ShaderDef,
};

// Pull visual constants - match band defaults
const PULL_COLOR: [f32; 3] = [0.5, 0.0, 0.5]; // Purple - same as BAND_COLOR
const PULL_CYLINDER_RADIUS: f32 = 0.25; // Same as BAND_MID_RADIUS (default strength)
const PULL_CONE_RADIUS: f32 = 0.6; // Larger than cylinder for visible arrow
const PULL_CONE_LENGTH: f32 = 2.0;

/// Renders the active pull constraint (capsule cylinder + cone arrow).
pub(crate) struct PullRenderer {
    capsule_pass: ImpostorPass<CapsuleInstance>,
    cone_pass: ImpostorPass<ConeInstance>,
}

impl PullRenderer {
    /// Create a new pull renderer with empty instance buffers.
    pub(crate) fn new(
        context: &RenderContext,
        layouts: &crate::renderer::PipelineLayouts,
        shader_composer: &mut ShaderComposer,
    ) -> Result<Self, VisoError> {
        let capsule_pass = ImpostorPass::new(
            context,
            &ShaderDef {
                label: "Pull Capsule",
                shader: Shader::Capsule,
            },
            layouts,
            6,
            shader_composer,
        )?;

        let cone_pass = ImpostorPass::new(
            context,
            &ShaderDef {
                label: "Pull Cone",
                shader: Shader::Cone,
            },
            layouts,
            6,
            shader_composer,
        )?;

        Ok(Self {
            capsule_pass,
            cone_pass,
        })
    }

    /// Update with the active pull, or clear if None
    pub(crate) fn update(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        pull: Option<&ResolvedPull>,
    ) {
        if let Some(p) = pull {
            let (capsules, cones) = Self::generate_instances(p);
            let _ = self.capsule_pass.write_instances(device, queue, &capsules);
            let _ = self.cone_pass.write_instances(device, queue, &cones);
        } else {
            self.capsule_pass.instance_count = 0;
            self.cone_pass.instance_count = 0;
        }
    }

    /// GPU buffer sizes: `(label, used_bytes, allocated_bytes)`.
    pub(crate) fn buffer_info(&self) -> Vec<(&'static str, usize, usize)> {
        vec![
            self.capsule_pass.buffer_info("Pull Capsules"),
            self.cone_pass.buffer_info("Pull Cones"),
        ]
    }

    fn generate_instances(
        pull: &ResolvedPull,
    ) -> (Vec<CapsuleInstance>, Vec<ConeInstance>) {
        let mut capsules = Vec::with_capacity(1);
        let mut cones = Vec::with_capacity(1);

        let atom_pos = pull.atom_pos;
        let target_pos = pull.target_pos;
        let to_target = target_pos - atom_pos;
        let distance = to_target.length();

        if distance < 0.001 {
            return (capsules, cones);
        }

        let direction = to_target / distance;

        // Cone base is PULL_CONE_LENGTH from the target, pointing toward target
        let cone_base = if distance > PULL_CONE_LENGTH {
            target_pos - direction * PULL_CONE_LENGTH
        } else {
            atom_pos
        };

        // Cylinder from atom to cone base
        if atom_pos.distance_squared(cone_base) > 0.001 {
            capsules.push(CapsuleInstance {
                endpoint_a: [
                    atom_pos.x,
                    atom_pos.y,
                    atom_pos.z,
                    PULL_CYLINDER_RADIUS,
                ],
                endpoint_b: [
                    cone_base.x,
                    cone_base.y,
                    cone_base.z,
                    pull.residue_idx as f32,
                ],
                color_a: [PULL_COLOR[0], PULL_COLOR[1], PULL_COLOR[2], 0.0],
                color_b: [PULL_COLOR[0], PULL_COLOR[1], PULL_COLOR[2], 0.0],
            });
        }

        // Cone from cone_base to target (arrow pointing toward mouse)
        if cone_base.distance_squared(target_pos) > 0.001 {
            cones.push(ConeInstance {
                base: [cone_base.x, cone_base.y, cone_base.z, PULL_CONE_RADIUS],
                tip: [
                    target_pos.x,
                    target_pos.y,
                    target_pos.z,
                    pull.residue_idx as f32,
                ],
                color: [PULL_COLOR[0], PULL_COLOR[1], PULL_COLOR[2], 0.0],
                _pad: [0.0, 0.0, 0.0, 0.0],
            });
        }

        (capsules, cones)
    }

    /// Draw pull geometry (cylinder + cone arrow) into the given render pass.
    pub(crate) fn draw<'a>(
        &'a self,
        render_pass: &mut wgpu::RenderPass<'a>,
        bind_groups: &crate::renderer::draw_context::DrawBindGroups<'a>,
    ) {
        self.capsule_pass.draw(render_pass, bind_groups);
        self.cone_pass.draw(render_pass, bind_groups);
    }
}
