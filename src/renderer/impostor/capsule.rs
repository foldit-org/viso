/// Per-instance data for capsule impostor.
/// Must match the WGSL CapsuleInstance struct layout exactly.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct CapsuleInstance {
    /// Endpoint A position (xyz), radius (w)
    pub(crate) endpoint_a: [f32; 4],
    /// Endpoint B position (xyz), residue_idx (w) - packed as float
    pub(crate) endpoint_b: [f32; 4],
    /// Color at endpoint A (RGB), emissive glow factor (w)
    pub(crate) color_a: [f32; 4],
    /// Color at endpoint B (RGB), opacity / pulse gate (w)
    pub(crate) color_b: [f32; 4],
}
