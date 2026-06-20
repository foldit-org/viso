#import viso::camera::CameraUniform
#import viso::lighting::LightingUniform
#import viso::selection::check_selection
#import viso::highlight::apply_highlight
#import viso::designability::{check_non_designable, apply_designability}
#import viso::shade::{shade_geometry, ShadingResult}
#import viso::constants::MAX_IBL_MIP

fn is_selected(residue_idx: u32) -> bool {
    return check_selection(residue_idx, arrayLength(&selection), selection[residue_idx / 32u]);
}

fn is_non_designable(residue_idx: u32) -> bool {
    return check_non_designable(residue_idx, arrayLength(&non_designable), non_designable[residue_idx / 32u]);
}

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) color: vec4<f32>,
    @location(3) residue_idx: u32,
    @location(4) center_pos: vec3<f32>,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) center_pos: vec3<f32>,
    @location(1) world_position: vec3<f32>,
    @location(2) vertex_color: vec4<f32>,
    @location(3) @interpolate(flat) residue_idx: u32,
};

@group(0) @binding(0) var<uniform> camera: CameraUniform;
@group(1) @binding(0) var<uniform> lighting: LightingUniform;
@group(1) @binding(1) var irradiance_map: texture_cube<f32>;
@group(1) @binding(2) var env_sampler: sampler;
@group(1) @binding(3) var prefiltered_map: texture_cube<f32>;
@group(1) @binding(4) var brdf_lut: texture_2d<f32>;
@group(2) @binding(0) var<storage, read> selection: array<u32>;
@group(2) @binding(1) var<storage, read> non_designable: array<u32>;

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;

    // Expand selected residues outward along normal for 1.4x radius (matches Foldit)
    var position = in.position;
    if (is_selected(in.residue_idx)) {
        position = position + in.normal * 0.24;  // ~1.4x expansion (tube radius ~0.6)
    }

    out.clip_position = camera.view_proj * vec4<f32>(position, 1.0);
    out.center_pos = in.center_pos;
    out.world_position = position;
    out.vertex_color = in.color;
    out.residue_idx = in.residue_idx;
    return out;
}

struct FragOutput {
    @location(0) color: vec4<f32>,
    @location(1) normal: vec4<f32>,
};

@fragment
fn fs_main(in: VertexOutput) -> FragOutput {
    // Per-pixel cylindrical normal: exact outward direction from tube centerline
    let normal = normalize(in.world_position - in.center_pos);
    let view_dir = normalize(camera.position - in.world_position);

    // Highlight (operates on rgb; the vertex alpha passes straight through)
    let alpha = in.vertex_color.a;
    let hovered = camera.hovered_residue >= 0 && u32(camera.hovered_residue) == in.residue_idx;
    let highlighted = apply_highlight(in.vertex_color.rgb, hovered, is_selected(in.residue_idx));
    var base_color = apply_designability(highlighted.xyz, is_non_designable(in.residue_idx));
    let outline_factor = highlighted.w;

    // Pre-sample IBL textures (modules cannot reference bindings)
    let NdotV = max(dot(normal, view_dir), 0.0);
    let R = reflect(-view_dir, normal);
    let irradiance = textureSample(irradiance_map, env_sampler, normal).rgb;
    let prefiltered = textureSampleLevel(prefiltered_map, env_sampler, R,
        lighting.roughness * MAX_IBL_MIP).rgb;
    let brdf = textureSample(brdf_lut, env_sampler, vec2<f32>(NdotV, lighting.roughness)).rg;

    let result = shade_geometry(normal, view_dir, base_color, outline_factor,
        lighting, irradiance, prefiltered, brdf);
    let final_color = result.color;
    let ambient_ratio = result.ambient_ratio;

    var out: FragOutput;
    if (camera.debug_mode == 1u) {
        out.color = vec4<f32>(normal * 0.5 + 0.5, 1.0);
    } else {
        out.color = vec4<f32>(final_color, alpha);
    }
    out.normal = vec4<f32>(normal, ambient_ratio);
    return out;
}
