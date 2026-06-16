// Ray-marched cone impostor for pull arrow rendering
// Cone points from base (atom) toward tip (mouse target)

#import viso::camera::CameraUniform
#import viso::lighting::LightingUniform
#import viso::ray::{intersect_cone, cone_normal}
#import viso::selection::check_selection
#import viso::highlight::apply_highlight
#import viso::designability::{check_non_designable, apply_designability}
#import viso::shade::{shade_geometry, ShadingResult}
#import viso::constants::{MAX_IBL_MIP, BILLBOARD_SCALE}

fn is_selected(residue_idx: u32) -> bool {
    return check_selection(residue_idx, arrayLength(&selection), selection[residue_idx / 32u]);
}

fn is_non_designable(residue_idx: u32) -> bool {
    return check_non_designable(residue_idx, arrayLength(&non_designable), non_designable[residue_idx / 32u]);
}

// Per-instance data for cone
// base: xyz = base position (wide end), w = base radius
// tip: xyz = tip position (point), w = residue_idx (packed as float)
// color: xyz = RGB, w = unused
struct ConeInstance {
    base: vec4<f32>,
    tip: vec4<f32>,
    color: vec4<f32>,
    _pad: vec4<f32>,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_pos: vec3<f32>,
    @location(1) base: vec3<f32>,
    @location(2) tip: vec3<f32>,
    @location(3) base_radius: f32,
    @location(4) color: vec3<f32>,
    @location(5) @interpolate(flat) residue_idx: u32,
};

struct FragOut {
    @builtin(frag_depth) depth: f32,
    @location(0) color: vec4<f32>,
    @location(1) normal: vec4<f32>,
};

@group(0) @binding(0) var<uniform> camera: CameraUniform;
@group(1) @binding(0) var<uniform> lighting: LightingUniform;
@group(1) @binding(1) var irradiance_map: texture_cube<f32>;
@group(1) @binding(2) var env_sampler: sampler;
@group(1) @binding(3) var prefiltered_map: texture_cube<f32>;
@group(1) @binding(4) var brdf_lut: texture_2d<f32>;
@group(2) @binding(0) var<storage, read> selection: array<u32>;
@group(3) @binding(0) var<storage, read> cones: array<ConeInstance>;
@group(2) @binding(1) var<storage, read> non_designable: array<u32>;

@vertex
fn vs_main(
    @builtin(vertex_index) vidx: u32,
    @builtin(instance_index) iidx: u32
) -> VertexOutput {
    let quad = array<vec2<f32>, 6>(
        vec2(-1.0, -1.0), vec2(1.0, -1.0), vec2(-1.0, 1.0),
        vec2(-1.0, 1.0), vec2(1.0, -1.0), vec2(1.0, 1.0)
    );

    let cone = cones[iidx];
    let base = cone.base.xyz;
    let tip = cone.tip.xyz;
    let base_radius = cone.base.w;
    let color = cone.color.xyz;
    let residue_idx = u32(cone.tip.w);

    let center = (base + tip) * 0.5;
    let axis = tip - base;
    let height = length(axis);
    let axis_dir = select(vec3<f32>(0.0, 1.0, 0.0), axis / height, height > 0.0001);

    let to_camera = normalize(camera.position - center);

    var right = cross(axis_dir, to_camera);
    let right_len = length(right);
    if (right_len < 0.001) {
        right = cross(axis_dir, vec3<f32>(0.0, 0.0, 1.0));
        if (length(right) < 0.001) {
            right = vec3<f32>(1.0, 0.0, 0.0);
        }
    }
    right = normalize(right);

    let up = axis_dir;

    // Billboard size needs to encompass the whole cone
    let half_width = base_radius * BILLBOARD_SCALE;
    let half_height = height * 0.5 + base_radius * 0.5;

    let local_uv = quad[vidx];
    let world_offset = right * local_uv.x * half_width + up * local_uv.y * half_height;
    let world_pos = center + world_offset;

    var out: VertexOutput;
    out.clip_position = camera.view_proj * vec4<f32>(world_pos, 1.0);
    out.world_pos = world_pos;
    out.base = base;
    out.tip = tip;
    out.base_radius = base_radius;
    out.color = color;
    out.residue_idx = residue_idx;

    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> FragOut {
    let ray_origin = camera.position;
    let ray_dir = normalize(in.world_pos - camera.position);

    let hit = intersect_cone(ray_origin, ray_dir, in.base, in.tip, in.base_radius);

    if (hit.x < 0.0) {
        discard;
    }

    let t = hit.x;
    let hit_type = hit.z;

    let world_hit = ray_origin + ray_dir * t;

    let normal = cone_normal(world_hit, in.base, in.tip, in.base_radius, hit_type);
    let view_dir = normalize(camera.position - world_hit);

    // Highlight
    let hovered = camera.hovered_residue >= 0 && u32(camera.hovered_residue) == in.residue_idx;
    let highlighted = apply_highlight(in.color, hovered, is_selected(in.residue_idx));
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

    let clip_pos = camera.view_proj * vec4<f32>(world_hit, 1.0);
    let ndc_depth = clip_pos.z / clip_pos.w;

    // SDF edge AA
    let axis = in.tip - in.base;
    let height = length(axis);
    let axis_dir = axis / height;
    let hit_along = dot(world_hit - in.base, axis_dir);
    let radius_at_hit = in.base_radius * (1.0 - hit_along / height);
    let radial_dist = length(world_hit - in.base - axis_dir * hit_along);
    let sdf = radial_dist - radius_at_hit;
    let aa_edge = fwidth(sdf);
    let alpha = smoothstep(aa_edge, -aa_edge, sdf);

    var out: FragOut;
    out.depth = ndc_depth;
    if (camera.debug_mode == 1u) {
        out.color = vec4<f32>(normal * 0.5 + 0.5, alpha);
    } else {
        out.color = vec4<f32>(final_color, alpha);
    }
    out.normal = vec4<f32>(normal, ambient_ratio);
    return out;
}
