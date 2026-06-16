//! Steric clash lightning renderer.
//!
//! Renders steric clashes between atom pairs as glowing deep-red electric
//! bolts: one camera-facing billboard ribbon per clash spanning the two
//! clashing atoms, with the whole jagged bolt drawn procedurally in the
//! [`Shader::Lightning`] fragment shader. A
//! bright hot-red core blooms over a deep dark semitransparent red
//! envelope; the centerline jag scrolls smoothly with `camera.time` while
//! a fast brightness flicker supplies the electric energy. This reads as a
//! clear "danger" marker without the calm of the constraint band / h-bond
//! capsule.
//!
//! Mirrors the band renderer
//! ([`BandRenderer`](crate::renderer::geometry::band::BandRenderer)):
//! clash specs are externally supplied per pair, resolved from per-entity
//! structural [`ClashEndpoint`](crate::engine::command::ClashEndpoint)s to
//! world-space every frame, then emitted as one
//! [`LightningInstance`]
//! per clash. All animation lives in the shader,
//! which reads `camera.time` directly, so the renderer just forwards the
//! resolved endpoints, the severity, and a stable per-clash seed.

use crate::engine::command::ResolvedClash;
use crate::error::VisoError;
use crate::gpu::{RenderContext, Shader, ShaderComposer};
use crate::renderer::impostor::{ImpostorPass, ShaderDef};

/// Per-instance data for the lightning bolt impostor.
///
/// Must match the WGSL `LightningInstance` struct layout exactly (32 bytes,
/// 16-byte aligned).
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct LightningInstance {
    /// xyz = first clashing atom (world-space), w = raw clash severity.
    pub(crate) endpoint_a: [f32; 4],
    /// xyz = second clashing atom (world-space), w = per-clash seed.
    pub(crate) endpoint_b: [f32; 4],
}

/// Renders steric clashes as procedural lightning-bolt impostors.
pub(crate) struct ClashArcRenderer {
    pass: ImpostorPass<LightningInstance>,
}

impl ClashArcRenderer {
    /// Create a new clash renderer with an empty instance buffer.
    pub(crate) fn new(
        context: &RenderContext,
        layouts: &crate::renderer::PipelineLayouts,
        shader_composer: &mut ShaderComposer,
    ) -> Result<Self, VisoError> {
        let pass = ImpostorPass::new(
            context,
            &ShaderDef {
                label: "Clash Lightning",
                shader: Shader::Lightning,
            },
            layouts,
            6,
            shader_composer,
        )?;
        Ok(Self { pass })
    }

    /// Emit one lightning instance per resolved clash. Degenerate clashes
    /// (coincident endpoints) are skipped so the billboard has a valid
    /// axis.
    fn generate_instances(clashes: &[ResolvedClash]) -> Vec<LightningInstance> {
        let mut instances = Vec::with_capacity(clashes.len());
        for clash in clashes {
            let axis = clash.endpoint_b - clash.endpoint_a;
            if axis.length() < 0.001 {
                continue;
            }
            let a = clash.endpoint_a;
            let b = clash.endpoint_b;
            instances.push(LightningInstance {
                endpoint_a: [a.x, a.y, a.z, clash.severity],
                endpoint_b: [b.x, b.y, b.z, clash.seed],
            });
        }
        instances
    }

    /// Update clash geometry. Animation is shader-side (driven by
    /// `camera.time`), so this only forwards the resolved endpoints,
    /// severity, and per-clash seed.
    pub(crate) fn update(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        clashes: &[ResolvedClash],
    ) {
        let instances = Self::generate_instances(clashes);
        let _ = self.pass.write_instances(device, queue, &instances);
    }

    /// GPU buffer sizes: `(label, used_bytes, allocated_bytes)`.
    pub(crate) fn buffer_info(&self) -> Vec<(&'static str, usize, usize)> {
        vec![self.pass.buffer_info("Clash Lightning")]
    }

    /// Draw clash lightning bolts into the given render pass.
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

    fn sample_clash() -> ResolvedClash {
        ResolvedClash {
            endpoint_a: Vec3::new(0.0, 0.0, 0.0),
            endpoint_b: Vec3::new(4.0, 0.0, 0.0),
            severity: 0.5,
            seed: 0.25,
        }
    }

    #[test]
    fn one_clash_emits_one_instance() {
        let instances = ClashArcRenderer::generate_instances(&[sample_clash()]);
        assert_eq!(instances.len(), 1);
    }

    #[test]
    fn instance_carries_endpoints_severity_and_seed() {
        let instances = ClashArcRenderer::generate_instances(&[sample_clash()]);
        let inst = instances[0];
        assert_eq!(inst.endpoint_a, [0.0, 0.0, 0.0, 0.5]);
        assert_eq!(inst.endpoint_b, [4.0, 0.0, 0.0, 0.25]);
    }

    #[test]
    fn degenerate_clash_emits_nothing() {
        let mut clash = sample_clash();
        clash.endpoint_b = clash.endpoint_a;
        let instances = ClashArcRenderer::generate_instances(&[clash]);
        assert!(instances.is_empty());
    }
}
