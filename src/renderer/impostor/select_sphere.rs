/// Per-instance data for the select-sphere overlay impostor.
/// Must match the WGSL `SelectSphereInstance` struct layout.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct SelectSphereInstance {
    /// xyz = centre (world-space), w = radius.
    pub(crate) center: [f32; 4],
    /// rgb = colour, a = alpha.
    pub(crate) color: [f32; 4],
}
