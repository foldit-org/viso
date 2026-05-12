#define_import_path viso::lut_cube

// Adobe `.cube` 3D LUT sampling helpers.
//
// The LUT is uploaded as a 3D texture where input R varies along X, input G
// along Y, input B along Z. All inputs are expected in [0, 1] (caller is
// responsible for ensuring the color space is correct).

// Map an input RGB in [0, 1] to normalized texture coordinates in [0, 1],
// offset to texel centers for a LUT of size N.
fn lut_texcoord(rgb01: vec3<f32>, lut_size: u32) -> vec3<f32> {
    // For N texels, center coords are (i + 0.5) / N. If rgb==0, we want i=0;
    // if rgb==1, we want i=N-1.
    let n = max(1.0, f32(lut_size));
    let scaled = clamp(rgb01, vec3<f32>(0.0), vec3<f32>(1.0)) * (n - 1.0);
    return (scaled + vec3<f32>(0.5)) / n;
}

// Apply an Adobe `.cube` LUT stored in a 3D Rgba16Float texture.
fn apply_adobe_cube_lut(
    lut_tex: texture_3d<f16>,
    lut_sampler: sampler,
    rgb01: vec3<f32>,
    lut_size: u32,
) -> vec3<f32> {
    let uvw = lut_texcoord(rgb01, lut_size);
    return textureSample(lut_tex, lut_sampler, uvw).xyz;
}

// Convenience passthrough wrapper for call sites that gate LUT application.
fn apply_adobe_cube_lut_if_enabled(
    enabled: bool,
    lut_tex: texture_3d<f16>,
    lut_sampler: sampler,
    rgb01: vec3<f32>,
    lut_size: u32,
) -> vec3<f32> {
    if (enabled) {
        return apply_adobe_cube_lut(lut_tex, lut_sampler, rgb01, lut_size);
    }
    return rgb01;
}

