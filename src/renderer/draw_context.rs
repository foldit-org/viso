/// Bind groups shared across all molecular draw calls.
pub(crate) struct DrawBindGroups<'a> {
    /// Camera uniform bind group (view-projection, position, etc.).
    pub(crate) camera: &'a wgpu::BindGroup,
    /// Lighting uniform bind group.
    pub(crate) lighting: &'a wgpu::BindGroup,
    /// Per-residue overlay bind group (group 2): selection highlight at
    /// binding 0, non-designable bitset at binding 1. Bound by every
    /// geometry pass; the five score-color shaders read the non-designable
    /// bits to desaturate locked residues toward white.
    pub(crate) selection: &'a wgpu::BindGroup,
    /// Per-residue color override (used by backbone renderer only).
    pub(crate) color: Option<&'a wgpu::BindGroup>,
}
