// Composite pass - applies SSAO, outlines, fog, tone mapping to the rendered scene

#import viso::fullscreen::{FullscreenVertexOutput, fullscreen_vertex}
#import viso::depth::linearize_depth
#import viso::constants::LUMINANCE_REC709
#import viso::lut_cube::{apply_adobe_cube_lut}

// Load raw depth via textureLoad (bypasses sampler — works on Vulkan and GL/GLES)
fn load_depth(uv: vec2<f32>) -> f32 {
    let dims = vec2<f32>(textureDimensions(depth_texture, 0));
    let texel = vec2<i32>(clamp(uv * dims, vec2<f32>(0.0), dims - 1.0));
    return textureLoad(depth_texture, texel, 0);
}

struct CompositeParams {
    screen_size: vec2<f32>,
    outline_thickness: f32,
    outline_strength: f32,
    ao_strength: f32,
    near: f32,
    far: f32,
    fog_start: f32,
    fog_density: f32,
    normal_outline_strength: f32,
    exposure: f32,
    gamma: f32,
    bloom_intensity: f32,
    _pad: f32,
    _pad2: f32,
    /// `LUT_3D_SIZE` when grading; `0` skips LUT sampling.
    adobe_lut_grid_size: u32,
};

@group(0) @binding(0) var color_texture: texture_2d<f32>;
@group(0) @binding(1) var ssao_texture: texture_2d<f32>;
@group(0) @binding(2) var depth_texture: texture_depth_2d;
@group(0) @binding(3) var tex_sampler: sampler;
@group(0) @binding(4) var depth_sampler: sampler;
@group(0) @binding(5) var<uniform> params: CompositeParams;
@group(0) @binding(6) var normal_texture: texture_2d<f32>;
@group(0) @binding(7) var bloom_texture: texture_2d<f32>;
// Adobe `.cube` 3D LUT (B1 placeholder bind; B2+ samples when `adobe_lut_grid_size > 0`).
@group(0) @binding(8) var adobe_lut_tex: texture_3d<f32>;
@group(0) @binding(9) var adobe_lut_sampler: sampler;

// Full-screen triangle (more efficient than quad - no diagonal edge)
@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> FullscreenVertexOutput {
    return fullscreen_vertex(vertex_index);
}

// Depth-relative edge detection with adaptive threshold
fn detect_depth_edges(uv: vec2<f32>, texel_size: vec2<f32>, thickness: f32) -> f32 {
    let offset = texel_size * thickness;

    let d_c = load_depth(uv);
    let d_t = load_depth(uv + vec2(0.0, -offset.y));
    let d_b = load_depth(uv + vec2(0.0, offset.y));
    let d_l = load_depth(uv + vec2(-offset.x, 0.0));
    let d_r = load_depth(uv + vec2(offset.x, 0.0));

    let l_c = linearize_depth(d_c, params.near, params.far);
    let l_t = linearize_depth(d_t, params.near, params.far);
    let l_b = linearize_depth(d_b, params.near, params.far);
    let l_l = linearize_depth(d_l, params.near, params.far);
    let l_r = linearize_depth(d_r, params.near, params.far);

    let max_diff = max(
        max(abs(l_c - l_t), abs(l_c - l_b)),
        max(abs(l_c - l_l), abs(l_c - l_r))
    );

    let relative_diff = max_diff / max(l_c, 0.1);
    return smoothstep(0.008, 0.03, relative_diff);
}

// Normal-based edge detection: detects outlines where normals differ sharply
// even when depth is continuous (e.g., adjacent helices touching)
fn detect_normal_edges(uv: vec2<f32>, texel_size: vec2<f32>, thickness: f32) -> f32 {
    let offset = texel_size * thickness;

    let n_c = textureSampleLevel(normal_texture, tex_sampler, uv, 0.0).xyz;
    let n_t = textureSampleLevel(normal_texture, tex_sampler, uv + vec2(0.0, -offset.y), 0.0).xyz;
    let n_b = textureSampleLevel(normal_texture, tex_sampler, uv + vec2(0.0, offset.y), 0.0).xyz;
    let n_l = textureSampleLevel(normal_texture, tex_sampler, uv + vec2(-offset.x, 0.0), 0.0).xyz;
    let n_r = textureSampleLevel(normal_texture, tex_sampler, uv + vec2(offset.x, 0.0), 0.0).xyz;

    // Compute maximum normal dissimilarity
    let diff = max(
        max(1.0 - dot(n_c, n_t), 1.0 - dot(n_c, n_b)),
        max(1.0 - dot(n_c, n_l), 1.0 - dot(n_c, n_r))
    );

    // Soft threshold
    return smoothstep(0.3, 0.7, diff);
}

// Khronos PBR Neutral tone mapping
// Preserves hue better than ACES — good for scientific visualization
fn tonemap_pbr_neutral(color: vec3<f32>) -> vec3<f32> {
    let start_compression = 0.8 - 0.04;
    let desaturation = 0.15;

    var x = min(color, vec3<f32>(start_compression));
    let overshoot = max(color - start_compression, vec3<f32>(0.0));

    let overshoot_length = length(overshoot);
    if (overshoot_length > 0.0001) {
        let compressed = 1.0 - exp(-overshoot_length);
        x = x + overshoot * (compressed / overshoot_length);
    }

    // Slight desaturation in highlights
    let lum = dot(x, LUMINANCE_REC709);
    let sat_factor = 1.0 - desaturation * max(lum - start_compression, 0.0) / max(1.0 - start_compression, 0.0001);
    x = mix(vec3<f32>(lum), x, sat_factor);

    return x;
}

@fragment
fn fs_main(in: FullscreenVertexOutput) -> @location(0) vec4<f32> {
    let color = textureSampleLevel(color_texture, tex_sampler, in.uv, 0.0);
    let ao = textureSampleLevel(ssao_texture, tex_sampler, in.uv, 0.0).r;
    let depth = load_depth(in.uv);
    let normal_sample = textureSampleLevel(normal_texture, tex_sampler, in.uv, 0.0);
    let ambient_ratio = normal_sample.w;

    // Background early-out (no outline on empty space)
    if (depth > 0.9999) {
        return vec4<f32>(0.0, 0.0, 0.0, 1.0);
    }

    // === STEP 1: Apply SSAO (darken ambient only) ===
    let adjusted_ao = mix(1.0, ao, params.ao_strength);
    // Only darken the ambient portion of the color, leaving direct lighting intact
    var final_color = color.rgb * mix(1.0, adjusted_ao, ambient_ratio);

    // === STEP 2: Exponential fog from linearized depth ===
    let linear_depth = linearize_depth(depth, params.near, params.far);
    let fog_distance = max(linear_depth - params.fog_start, 0.0);
    let fog_factor = exp(-fog_distance * params.fog_density);
    final_color = final_color * fog_factor;

    // === STEP 3: Apply outlines (depth + normal combined) ===
    let texel_size = 1.0 / params.screen_size;
    let depth_edge = detect_depth_edges(in.uv, texel_size, params.outline_thickness);
    let normal_edge = detect_normal_edges(in.uv, texel_size, params.outline_thickness);
    let combined_edge = max(depth_edge, normal_edge * params.normal_outline_strength);

    let depth_attenuation = 1.0 / (1.0 + linear_depth * 0.002);
    let attenuated_edge = combined_edge * params.outline_strength * depth_attenuation;
    final_color = mix(final_color, vec3<f32>(0.0), attenuated_edge);

    // === STEP 4: Add bloom (before tone mapping, in HDR space) ===
    if (params.bloom_intensity > 0.0) {
        let bloom = textureSampleLevel(bloom_texture, tex_sampler, in.uv, 0.0).rgb;
        final_color = final_color + bloom * params.bloom_intensity;
    }

    // === STEP 5: HDR tone mapping + exposure ===
    final_color = final_color * params.exposure;
    final_color = tonemap_pbr_neutral(final_color);
    final_color = pow(final_color, vec3<f32>(params.gamma));

    if (params.adobe_lut_grid_size > 0u) {
        final_color = apply_adobe_cube_lut(
            adobe_lut_tex,
            adobe_lut_sampler,
            final_color,
            params.adobe_lut_grid_size,
        );
    }

    return vec4<f32>(final_color, color.a);
}
