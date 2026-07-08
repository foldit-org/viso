// Ray-marched sphere impostors for ball-and-stick rendering
// Each sphere is a billboard quad with per-pixel ray-sphere intersection

#import viso::camera::CameraUniform
#import viso::lighting::LightingUniform
#import viso::ray::intersect_sphere
#import viso::selection::check_selection
#import viso::highlight::apply_highlight
#import viso::designability::{check_non_designable, apply_designability}
#import viso::pulse::{check_pulsing, apply_pulse}
#import viso::shade::{shade_geometry, ShadingResult}
#import viso::constants::{MAX_IBL_MIP, BILLBOARD_SCALE}
#import viso::impostor_types::SphereInstance

fn is_selected(residue_idx: u32) -> bool {
    return check_selection(residue_idx, arrayLength(&selection), selection[residue_idx / 32u]);
}

fn is_non_designable(residue_idx: u32) -> bool {
    return check_non_designable(residue_idx, arrayLength(&non_designable), non_designable[residue_idx / 32u]);
}

fn is_pulsing(residue_idx: u32) -> bool {
    return check_pulsing(residue_idx, arrayLength(&pulse), pulse[residue_idx / 32u]);
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_pos: vec3<f32>,
    @location(1) sphere_center: vec3<f32>,
    @location(2) radius: f32,
    @location(3) color: vec3<f32>,
    @location(4) @interpolate(flat) entity_id: u32,
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
@group(3) @binding(0) var<storage, read> spheres: array<SphereInstance>;
@group(2) @binding(1) var<storage, read> non_designable: array<u32>;
@group(2) @binding(2) var<storage, read> pulse: array<u32>;

@vertex
fn vs_main(
    @builtin(vertex_index) vidx: u32,
    @builtin(instance_index) iidx: u32
) -> VertexOutput {
    let quad = array<vec2<f32>, 6>(
        vec2(-1.0, -1.0), vec2(1.0, -1.0), vec2(-1.0, 1.0),
        vec2(-1.0, 1.0), vec2(1.0, -1.0), vec2(1.0, 1.0)
    );

    let sph = spheres[iidx];
    let center = sph.center.xyz;
    let radius = sph.center.w;
    let color = sph.color.xyz;
    let entity_id = u32(sph.color.w);

    let to_camera = normalize(camera.position - center);

    // Build billboard basis
    var right = cross(to_camera, vec3<f32>(0.0, 1.0, 0.0));
    if (length(right) < 0.001) {
        right = cross(to_camera, vec3<f32>(0.0, 0.0, 1.0));
    }
    right = normalize(right);
    let up = normalize(cross(right, to_camera));

    let half_size = radius * BILLBOARD_SCALE;

    let local_uv = quad[vidx];
    let world_offset = right * local_uv.x * half_size + up * local_uv.y * half_size;
    let world_pos = center + world_offset;

    var out: VertexOutput;
    out.clip_position = camera.view_proj * vec4<f32>(world_pos, 1.0);
    out.world_pos = world_pos;
    out.sphere_center = center;
    out.radius = radius;
    out.color = color;
    out.entity_id = entity_id;

    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> FragOut {
    let ray_origin = camera.position;
    let ray_dir = normalize(in.world_pos - camera.position);

    let t = intersect_sphere(ray_origin, ray_dir, in.sphere_center, in.radius);
    if (t < 0.0) {
        discard;
    }

    let world_hit = ray_origin + ray_dir * t;
    let normal = normalize(world_hit - in.sphere_center);
    let view_dir = normalize(camera.position - world_hit);

    // Highlight
    let hovered = camera.hovered_residue >= 0 && u32(camera.hovered_residue) == in.entity_id;
    let highlighted = apply_highlight(in.color, hovered, is_selected(in.entity_id));
    var base_color = apply_designability(highlighted.xyz, is_non_designable(in.entity_id));
    base_color = apply_pulse(base_color, is_pulsing(in.entity_id), f32(in.entity_id), camera.time);
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
    let sdf = length(in.world_pos - in.sphere_center) - in.radius;
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
