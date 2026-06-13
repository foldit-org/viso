//! Exposed-hydrophobic "grease bead" renderer.
//!
//! Renders each flagged exposed-hydrophobic residue as a procedural
//! ray-cast SDF "greasy bead" impostor: one camera-facing billboard quad
//! per residue, anchored at the residue's sidechain, with the bead's body
//! (a central sphere smooth-min'd with a few slowly-boiling satellites)
//! drawn entirely in the [`Shader::Grease`](crate::gpu::Shader::Grease)
//! fragment shader. Unlike the flat-emissive clash card, the bead writes a
//! real `frag_depth` and a real SDF-gradient normal and is lit (warm-yellow
//! greasy PBR with an animated broken highlight), so it sits in the scene as
//! a wet droplet on the sidechain.
//!
//! Mirrors the clash renderer
//! ([`ClashArcRenderer`](crate::renderer::geometry::clash::ClashArcRenderer)):
//! bead specs are externally supplied per residue, resolved from per-entity
//! structural refs to a world-space sidechain anchor every frame, then
//! emitted as one
//! [`GreaseBeadInstance`](crate::renderer::geometry::exposed_hydrophobic::GreaseBeadInstance)
//! per residue. All animation lives in the shader, which reads `camera.time`
//! directly, so the renderer just forwards the resolved anchor and a stable
//! per-bead seed.

use crate::engine::command::ResolvedExposedHydro;
use crate::error::VisoError;
use crate::gpu::{RenderContext, Shader, ShaderComposer};
use crate::renderer::impostor::{ImpostorPass, ShaderDef};

/// Per-instance data for the grease bead impostor.
///
/// Must match the WGSL `GreaseBeadInstance` struct layout exactly (16
/// bytes, 16-byte aligned).
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct GreaseBeadInstance {
    /// xyz = sidechain anchor (world-space), w = per-bead seed.
    pub(crate) center: [f32; 4],
}

/// Renders exposed-hydrophobic residues as procedural greasy-bead
/// impostors.
pub(crate) struct GreaseBeadRenderer {
    pass: ImpostorPass<GreaseBeadInstance>,
}

impl GreaseBeadRenderer {
    /// Create a new grease bead renderer with an empty instance buffer.
    pub(crate) fn new(
        context: &RenderContext,
        layouts: &crate::renderer::PipelineLayouts,
        shader_composer: &mut ShaderComposer,
    ) -> Result<Self, VisoError> {
        let pass = ImpostorPass::new(
            context,
            &ShaderDef {
                label: "Grease Bead",
                shader: Shader::Grease,
            },
            layouts,
            6,
            shader_composer,
        )?;
        Ok(Self { pass })
    }

    /// Emit one bead instance per resolved exposed-hydrophobic residue.
    fn generate_instances(
        beads: &[ResolvedExposedHydro],
    ) -> Vec<GreaseBeadInstance> {
        beads
            .iter()
            .map(|bead| GreaseBeadInstance {
                center: [
                    bead.center.x,
                    bead.center.y,
                    bead.center.z,
                    bead.seed,
                ],
            })
            .collect()
    }

    /// Update bead geometry. Animation is shader-side (driven by
    /// `camera.time`), so this only forwards the resolved anchor and the
    /// per-bead seed.
    pub(crate) fn update(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        beads: &[ResolvedExposedHydro],
    ) {
        let instances = Self::generate_instances(beads);
        let _ = self.pass.write_instances(device, queue, &instances);
    }

    /// GPU buffer sizes: `(label, used_bytes, allocated_bytes)`.
    pub(crate) fn buffer_info(&self) -> Vec<(&'static str, usize, usize)> {
        vec![self.pass.buffer_info("Grease Bead")]
    }

    /// Draw grease beads into the given render pass.
    pub(crate) fn draw<'a>(
        &'a self,
        render_pass: &mut wgpu::RenderPass<'a>,
        bind_groups: &crate::renderer::draw_context::DrawBindGroups<'a>,
    ) {
        self.pass.draw(render_pass, bind_groups);
    }
}

#[cfg(test)]
mod tests {
    use glam::Vec3;

    use super::*;

    fn sample_bead() -> ResolvedExposedHydro {
        ResolvedExposedHydro {
            center: Vec3::new(1.0, 2.0, 3.0),
            seed: 0.5,
        }
    }

    #[test]
    fn one_bead_emits_one_instance() {
        let instances =
            GreaseBeadRenderer::generate_instances(&[sample_bead()]);
        assert_eq!(instances.len(), 1);
    }

    #[test]
    fn instance_carries_center_and_seed() {
        let instances =
            GreaseBeadRenderer::generate_instances(&[sample_bead()]);
        assert_eq!(instances[0].center, [1.0, 2.0, 3.0, 0.5]);
    }

    #[test]
    fn empty_beads_emit_nothing() {
        let instances = GreaseBeadRenderer::generate_instances(&[]);
        assert!(instances.is_empty());
    }
}
