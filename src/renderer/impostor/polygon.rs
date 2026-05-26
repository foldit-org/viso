/// Per-instance data for extruded polygon impostor (nucleic acid rings).
///
/// Encodes a convex polygon (5 or 6 vertices) extruded by `half_thickness`
/// along its face normal.  The vertex shader procedurally generates top face,
/// bottom face, and side-wall triangles.
///
/// 128 bytes per instance.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct ExtrudedPolygonInstance {
    /// xyz = vertex 0, w = vertex count (5.0 or 6.0)
    pub(crate) v0: [f32; 4],
    /// xyz = vertex 1, w = half_thickness
    pub(crate) v1: [f32; 4],
    /// xyz = vertex 2, w = polygon centroid x (precomputed for the shader)
    pub(crate) v2: [f32; 4],
    /// xyz = vertex 3, w = polygon centroid y (precomputed for the shader)
    pub(crate) v3: [f32; 4],
    /// xyz = vertex 4, w = polygon centroid z (precomputed for the shader)
    pub(crate) v4: [f32; 4],
    /// xyz = vertex 5 (degenerate = centroid for pentagons), w = unused
    pub(crate) v5: [f32; 4],
    /// xyz = face normal, w = residue_idx (packed as float)
    pub(crate) normal: [f32; 4],
    /// xyz = RGB, w = unused
    pub(crate) color: [f32; 4],
}
