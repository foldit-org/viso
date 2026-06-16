// Ray-marched capsule impostors for sidechain rendering
// Capsules = cylinders with hemispherical caps

#import viso::camera::CameraUniform
#import viso::lighting::LightingUniform
#import viso::ray::{intersect_capsule, capsule_normal}
#import viso::selection::check_selection
#import viso::highlight::apply_highlight
#import viso::designability::{check_non_designable, apply_designability}
#import viso::shade::{shade_geometry, ShadingResult}
#import viso::constants::{MAX_IBL_MIP, BILLBOARD_SCALE, TUBE_RADIUS}
#import viso::impostor_types::CapsuleInstance

fn is_selected(residue_idx: u32) -> bool {
    return check_selection(residue_idx, arrayLength(&selection), selection[residue_idx / 32u]);
}

fn is_non_designable(residue_idx: u32) -> bool {
    return check_non_designable(residue_idx, arrayLength(&non_designable), non_designable[residue_idx / 32u]);
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_pos: vec3<f32>,
    @location(1) endpoint_a: vec3<f32>,
    @location(2) endpoint_b: vec3<f32>,
    @location(3) radius: f32,
    @location(4) color_a: vec3<f32>,
    @location(5) color_b: vec3<f32>,
    @location(6) @interpolate(flat) residue_idx: u32,
    @location(7) @interpolate(flat) emissive: f32,
    @location(8) @interpolate(flat) opacity: f32,
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
@group(3) @binding(0) var<storage, read> capsules: array<CapsuleInstance>;
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

    let cap = capsules[iidx];
    let endpoint_a = cap.endpoint_a.xyz;
    let endpoint_b = cap.endpoint_b.xyz;
    let color_a = cap.color_a.xyz;
    let color_b = cap.color_b.xyz;
    // Residue index packed in endpoint_b.w
    let residue_idx = u32(cap.endpoint_b.w);

    // Use radius from instance data (packed in endpoint_a.w)
    // Fall back to TUBE_RADIUS if instance radius is 0 (legacy compatibility)
    var radius = cap.endpoint_a.w;
    if (radius < 0.001) {
        radius = TUBE_RADIUS;
    }
    // Make selected residues 1.4x larger (matches Foldit SEL_THICKNESS_MULTIPLIER)
    if (is_selected(residue_idx)) {
        radius = radius * 1.4;
    }

    let center = (endpoint_a + endpoint_b) * 0.5;
    let axis = endpoint_b - endpoint_a;
    let seg_length = length(axis);
    let axis_dir = select(vec3<f32>(0.0, 1.0, 0.0), axis / seg_length, seg_length > 0.0001);

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

    let half_width = radius * BILLBOARD_SCALE;
    let half_height = seg_length * 0.5 + radius * BILLBOARD_SCALE;

    let local_uv = quad[vidx];
    let world_offset = right * local_uv.x * half_width + up * local_uv.y * half_height;
    let world_pos = center + world_offset;

    var out: VertexOutput;
    out.clip_position = camera.view_proj * vec4<f32>(world_pos, 1.0);
    out.world_pos = world_pos;
    out.endpoint_a = endpoint_a;
    out.endpoint_b = endpoint_b;
    out.radius = radius;
    out.color_a = color_a;
    out.color_b = color_b;
    out.residue_idx = residue_idx;
    out.emissive = cap.color_a.w;
    out.opacity = cap.color_b.w;

    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> FragOut {
    let ray_origin = camera.position;
    let ray_dir = normalize(in.world_pos - camera.position);

    let hit = intersect_capsule(ray_origin, ray_dir, in.endpoint_a, in.endpoint_b, in.radius);

    if (hit.x < 0.0) {
        discard;
    }

    let t = hit.x;
    let axis_param = hit.y;
    let hit_type = hit.z;

    let world_hit = ray_origin + ray_dir * t;

    let normal = capsule_normal(world_hit, in.endpoint_a, in.endpoint_b, hit_type);
    let view_dir = normalize(camera.position - world_hit);

    // Interpolate base color along capsule axis
    let interp_color = mix(in.color_a, in.color_b, axis_param);

    // Highlight
    let hovered = camera.hovered_residue >= 0 && u32(camera.hovered_residue) == in.residue_idx;
    let highlighted = apply_highlight(interp_color, hovered, is_selected(in.residue_idx));
    var base_color = apply_designability(highlighted.xyz, is_non_designable(in.residue_idx));
    let outline_factor = highlighted.w;

    // Pre-sample IBL textures (modules cannot reference bindings)
    let NdotV = max(dot(normal, view_dir), 0.0);
    let R = reflect(-view_dir, normal);
    let irradiance = textureSample(irradiance_map, env_sampler, normal).rgb;
    let prefiltered = textureSampleLevel(prefiltered_map, env_sampler, R,
        lighting.roughness * MAX_IBL_MIP).rgb;
    let brdf = textureSample(brdf_lut, env_sampler, vec2<f32>(NdotV, lighting.roughness)).rg;

    // Pulse glow for structural bonds (opacity > 0): a bright wavefront
    // travels from donor (A) to acceptor (B) at 0.5 Hz with fade-in at
    // the donor end and fade-out at the acceptor end, plus a rest period
    // between cycles to avoid a hard jump.
    var pulse = 0.0;
    if (in.opacity > 0.0) {
        // Cycle phase [0, 1). Travel occupies [0, 0.7), rest is [0.7, 1).
        let phase = fract(camera.time * 0.25);
        let travel = 0.7;
        if (phase < travel) {
            let pulse_pos = phase / travel; // normalize to [0, 1]
            let dist = abs(axis_param - pulse_pos);
            let raw = exp(-dist * dist * 40.0);
            // Fade in from donor end, fade out toward acceptor end.
            let fade_in = smoothstep(0.0, 0.15, pulse_pos);
            let fade_out = smoothstep(1.0, 0.85, pulse_pos);
            pulse = raw * fade_in * fade_out;
        }
        base_color = mix(base_color, vec3<f32>(1.0), pulse * 0.6);
    }

    let result = shade_geometry(normal, view_dir, base_color, outline_factor,
        lighting, irradiance, prefiltered, brdf);
    // Mix shaded result with emissive self-illumination.
    let final_color = mix(result.color, base_color * 1.5, in.emissive);
    let ambient_ratio = result.ambient_ratio;

    let clip_pos = camera.view_proj * vec4<f32>(world_hit, 1.0);
    let ndc_depth = clip_pos.z / clip_pos.w;

    // SDF edge AA
    let pa = in.world_pos - in.endpoint_a;
    let ba = in.endpoint_b - in.endpoint_a;
    let h = clamp(dot(pa, ba) / dot(ba, ba), 0.0, 1.0);
    let sdf = length(pa - ba * h) - in.radius;
    let aa_edge = fwidth(sdf);
    let edge_alpha = smoothstep(aa_edge, -aa_edge, sdf);
    // Opacity of 0 means use full edge alpha (legacy/default behavior
    // for sidechains etc. that don't set opacity). Nonzero values scale
    // the alpha for semi-transparency, boosted in the pulse region.
    let boosted_opacity = in.opacity + pulse * 0.3 * (1.0 - in.opacity);
    let alpha = select(edge_alpha, edge_alpha * boosted_opacity, in.opacity > 0.0);

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
