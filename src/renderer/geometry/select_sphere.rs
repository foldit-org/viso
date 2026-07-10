//! Transient select-sphere overlay renderer.
//!
//! Draws the host-pushed select-sphere drag preview as a single flat,
//! semi-transparent gray shell: one camera-facing billboard quad ray-cast
//! to an analytic sphere in the [`Shader::SelectSphere`] fragment shader,
//! unlit. The host supplies a resolved world-space centre and radius, so
//! there is no per-frame reference resolution: `update` emits exactly one
//! instance when a sphere is set and zero when it is cleared, which hides
//! the pass for free (the impostor draw early-returns at zero instances).

use crate::engine::command::SelectSphereInfo;
use crate::error::VisoError;
use crate::gpu::{RenderContext, Shader, ShaderComposer};
use crate::options::score_color::{PROVISIONAL_ALPHA, PROVISIONAL_GRAY};
use crate::renderer::impostor::{
    ImpostorPass, SelectSphereInstance, ShaderDef,
};

/// Renders the transient select-sphere drag overlay.
pub(crate) struct SelectSphereRenderer {
    pass: ImpostorPass<SelectSphereInstance>,
}

impl SelectSphereRenderer {
    /// Create a new select-sphere renderer with an empty instance buffer.
    pub(crate) fn new(
        context: &RenderContext,
        layouts: &crate::renderer::PipelineLayouts,
        shader_composer: &mut ShaderComposer,
    ) -> Result<Self, VisoError> {
        let pass = ImpostorPass::new(
            context,
            &ShaderDef {
                label: "Select Sphere",
                shader: Shader::SelectSphere,
            },
            layouts,
            6,
            shader_composer,
        )?;
        Ok(Self { pass })
    }

    /// Build the single overlay instance, tinting it with the shared
    /// provisional-preview gray so the Rust side owns the colour.
    fn instance(sphere: SelectSphereInfo) -> SelectSphereInstance {
        SelectSphereInstance {
            center: [
                sphere.center.x,
                sphere.center.y,
                sphere.center.z,
                sphere.radius,
            ],
            color: [
                PROVISIONAL_GRAY[0],
                PROVISIONAL_GRAY[1],
                PROVISIONAL_GRAY[2],
                PROVISIONAL_ALPHA,
            ],
        }
    }

    /// Upload the overlay: one instance when a sphere is set, none when
    /// it is cleared.
    pub(crate) fn update(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        sphere: Option<SelectSphereInfo>,
    ) {
        let instances: Vec<SelectSphereInstance> =
            sphere.map(Self::instance).into_iter().collect();
        let _ = self.pass.write_instances(device, queue, &instances);
    }

    /// GPU buffer sizes: `(label, used_bytes, allocated_bytes)`.
    pub(crate) fn buffer_info(&self) -> Vec<(&'static str, usize, usize)> {
        vec![self.pass.buffer_info("Select Sphere")]
    }

    /// Draw the overlay into the given render pass.
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

    #[test]
    fn instance_carries_center_radius_and_gray() {
        let inst = SelectSphereRenderer::instance(SelectSphereInfo {
            center: Vec3::new(1.0, 2.0, 3.0),
            radius: 4.0,
        });
        assert_eq!(inst.center, [1.0, 2.0, 3.0, 4.0]);
        assert_eq!(
            inst.color,
            [
                PROVISIONAL_GRAY[0],
                PROVISIONAL_GRAY[1],
                PROVISIONAL_GRAY[2],
                PROVISIONAL_ALPHA,
            ]
        );
    }
}
