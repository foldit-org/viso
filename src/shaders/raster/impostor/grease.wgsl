// Procedural "grease bead" impostor for exposed-hydrophobic markers.
//
// One instance per flagged residue. Each bead is ray-cast in the fragment
// shader on a single camera-facing billboard quad: a central sphere
// smooth-min'd (polynomial smin) with a few smaller satellite spheres whose
// offsets slowly animate with camera.time (the viscous "boil"). The normal
// is taken from the SDF gradient and a real frag_depth is written from the
// hit point (same depth-writing/lit flavor as the sphere impostor, NOT the
// flat-emissive lightning card).
//
// The bead is shaded GREASY: a warm-yellow low-roughness PBR base, a normal
// perturbed by an animated fBm (the crawling broken highlight that reads as
// wet oil instead of yellow plastic), a fresnel rim, a low-roughness
// prefiltered-cubemap reflection along R (the soft wet reflection), a warm
// translucent interior, and an emissive highlight pushed > 1.0 so it blooms.

#import viso::camera::CameraUniform
#import viso::lighting::{LightingUniform, fresnel_schlick}
#import viso::cavity::{hash13, noise3d}
#import viso::shade::{shade_geometry, ShadingResult}
#import viso::constants::{MAX_IBL_MIP, BILLBOARD_SCALE}

// ── Tunable look constants (GREASE_*) ─────────────────────────────────────
//
// These are the runtime-tuning knobs for the bead's appearance. They are
// pure look parameters; the geometry/animation wiring reads them directly.

// Bead bounding/body radius in world units (Angstroms). The central sphere
// radius; the billboard quad is sized off this.
const GREASE_BEAD_RADIUS: f32 = 0.9;
// Warm-yellow greasy base albedo.
const GREASE_BASE_COLOR: vec3<f32> = vec3<f32>(0.92, 0.72, 0.18);
// Warm translucent interior tint added where the bead is thin / grazing,
// to read as a viscous droplet rather than an opaque ball.
const GREASE_INTERIOR_COLOR: vec3<f32> = vec3<f32>(0.85, 0.55, 0.10);
// Surface roughness for the PBR base. Low = wet/shiny.
const GREASE_ROUGHNESS: f32 = 0.12;
// Number of orbiting satellite spheres smooth-min'd into the body.
const GREASE_SATELLITE_COUNT: i32 = 3;
// Satellite sphere radius as a fraction of the bead radius.
const GREASE_SATELLITE_RADIUS_FRAC: f32 = 0.55;
// Satellite orbit distance from the center as a fraction of the bead radius.
const GREASE_SATELLITE_ORBIT_FRAC: f32 = 0.55;
// Polynomial smooth-min blend width (world units) fusing satellites into the
// body. Larger = gloopier, more fused; smaller = more lumpy.
const GREASE_SMIN_K: f32 = 0.45;
// Angular speed (rad/sec) of the satellite "boil" orbit.
const GREASE_BOIL_SPEED: f32 = 0.6;
// Spatial frequency of the surface-highlight fBm perturbation.
const GREASE_NOISE_FREQ: f32 = 4.5;
// Scroll speed of the highlight fBm (the crawling broken reflection).
const GREASE_NOISE_SCROLL: f32 = 0.35;
// Strength of the normal perturbation from the highlight fBm.
const GREASE_NOISE_STRENGTH: f32 = 0.35;
// Fresnel rim strength (wet edge glow).
const GREASE_FRESNEL_STRENGTH: f32 = 0.8;
// Strength of the soft prefiltered-cubemap wet reflection along R.
const GREASE_REFLECTION_STRENGTH: f32 = 0.6;
// Emissive glow color (HDR; > 1.0 components bloom).
const GREASE_GLOW_COLOR: vec3<f32> = vec3<f32>(1.8, 1.25, 0.4);
// Emissive glow strength scaling GREASE_GLOW_COLOR on the bright highlight.
const GREASE_GLOW_STRENGTH: f32 = 0.55;
// Ray-march iteration count and hit epsilon for the SDF cast.
const GREASE_MARCH_STEPS: i32 = 48;
const GREASE_MARCH_EPS: f32 = 0.002;

// ── Instance data (twin of the Rust GreaseBeadInstance) ───────────────────
//
// center: xyz = world-space sidechain anchor, w = per-bead seed.
struct GreaseBeadInstance {
    center: vec4<f32>,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_pos: vec3<f32>,
    @location(1) bead_center: vec3<f32>,
    @location(2) @interpolate(flat) seed: f32,
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
@group(3) @binding(0) var<storage, read> beads: array<GreaseBeadInstance>;

// Polynomial smooth minimum (iq): blends two distances with a soft seam of
// width k.
fn smin(a: f32, b: f32, k: f32) -> f32 {
    let h = clamp(0.5 + 0.5 * (b - a) / k, 0.0, 1.0);
    return mix(b, a, h) - k * h * (1.0 - h);
}

// fractional Brownian motion over the shared value-noise field.
fn fbm(p: vec3<f32>) -> f32 {
    var sum = 0.0;
    var amp = 0.5;
    var freq = 1.0;
    for (var i = 0; i < 4; i = i + 1) {
        sum = sum + amp * (noise3d(p * freq) * 2.0 - 1.0);
        amp = amp * 0.5;
        freq = freq * 2.0;
    }
    return sum;
}

// Bead SDF in bead-local space (origin at the central sphere center). The
// central sphere is smooth-min'd with GREASE_SATELLITE_COUNT satellites that
// orbit slowly with camera.time, giving the viscous boil.
fn bead_sdf(local: vec3<f32>, seed: f32) -> f32 {
    var d = length(local) - GREASE_BEAD_RADIUS;
    let sat_r = GREASE_BEAD_RADIUS * GREASE_SATELLITE_RADIUS_FRAC;
    let orbit = GREASE_BEAD_RADIUS * GREASE_SATELLITE_ORBIT_FRAC;
    let t = camera.time * GREASE_BOIL_SPEED;
    for (var i = 0; i < GREASE_SATELLITE_COUNT; i = i + 1) {
        let fi = f32(i);
        // Decorrelate each satellite's phase by the bead seed + index.
        let phase = seed * 23.7 + fi * 2.3994;
        let a = t + phase;
        let b = t * 0.7 + phase * 1.7;
        let dir = normalize(vec3<f32>(
            cos(a),
            sin(b),
            cos(a * 0.6 + b * 0.4),
        ));
        let sat_center = dir * orbit;
        let sat = length(local - sat_center) - sat_r;
        d = smin(d, sat, GREASE_SMIN_K);
    }
    return d;
}

// SDF gradient (central differences) → surface normal.
fn bead_normal(local: vec3<f32>, seed: f32) -> vec3<f32> {
    let e = vec2<f32>(0.0015, 0.0);
    let nx = bead_sdf(local + e.xyy, seed) - bead_sdf(local - e.xyy, seed);
    let ny = bead_sdf(local + e.yxy, seed) - bead_sdf(local - e.yxy, seed);
    let nz = bead_sdf(local + e.yyx, seed) - bead_sdf(local - e.yyx, seed);
    return normalize(vec3<f32>(nx, ny, nz));
}

@vertex
fn vs_main(
    @builtin(vertex_index) vidx: u32,
    @builtin(instance_index) iidx: u32,
) -> VertexOutput {
    let quad = array<vec2<f32>, 6>(
        vec2(-1.0, -1.0), vec2(1.0, -1.0), vec2(-1.0, 1.0),
        vec2(-1.0, 1.0), vec2(1.0, -1.0), vec2(1.0, 1.0)
    );

    let bead = beads[iidx];
    let center = bead.center.xyz;

    let to_camera = normalize(camera.position - center);
    var right = cross(to_camera, vec3<f32>(0.0, 1.0, 0.0));
    if (length(right) < 0.001) {
        right = cross(to_camera, vec3<f32>(0.0, 0.0, 1.0));
    }
    right = normalize(right);
    let up = normalize(cross(right, to_camera));

    // The body can bulge to ~bead radius + satellite orbit + satellite
    // radius; size the billboard so it always covers the full silhouette.
    let extent = GREASE_BEAD_RADIUS
        * (1.0 + GREASE_SATELLITE_ORBIT_FRAC + GREASE_SATELLITE_RADIUS_FRAC);
    let half_size = extent * BILLBOARD_SCALE;

    let local_uv = quad[vidx];
    let world_offset =
        right * local_uv.x * half_size + up * local_uv.y * half_size;
    let world_pos = center + world_offset;

    var out: VertexOutput;
    out.clip_position = camera.view_proj * vec4<f32>(world_pos, 1.0);
    out.world_pos = world_pos;
    out.bead_center = center;
    out.seed = bead.center.w;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> FragOut {
    let ray_origin = camera.position;
    let ray_dir = normalize(in.world_pos - camera.position);

    // March the SDF in bead-local space. Start at the near intersection of
    // the ray with the bounding sphere so we don't waste steps in empty
    // space, then sphere-trace until a hit.
    let oc = ray_origin - in.bead_center;
    let bound_r = GREASE_BEAD_RADIUS
        * (1.0 + GREASE_SATELLITE_ORBIT_FRAC + GREASE_SATELLITE_RADIUS_FRAC);
    let b = dot(oc, ray_dir);
    let c = dot(oc, oc) - bound_r * bound_r;
    let disc = b * b - c;
    if (disc < 0.0) {
        discard;
    }
    let t_near = max(-b - sqrt(disc), 0.0);
    let t_far = -b + sqrt(disc);

    var t = t_near;
    var hit = false;
    for (var i = 0; i < GREASE_MARCH_STEPS; i = i + 1) {
        let local = oc + ray_dir * t;
        let d = bead_sdf(local, in.seed);
        if (d < GREASE_MARCH_EPS) {
            hit = true;
            break;
        }
        t = t + d;
        if (t > t_far) {
            break;
        }
    }
    if (!hit) {
        discard;
    }

    let world_hit = ray_origin + ray_dir * t;
    let local_hit = world_hit - in.bead_center;
    var normal = bead_normal(local_hit, in.seed);
    let view_dir = normalize(camera.position - world_hit);

    // Animated fBm normal perturbation: the crawling broken highlight that
    // separates wet oil from yellow plastic.
    let np = local_hit * GREASE_NOISE_FREQ
        + vec3<f32>(camera.time * GREASE_NOISE_SCROLL, in.seed * 9.1, 0.0);
    let bump = vec3<f32>(
        fbm(np),
        fbm(np + vec3<f32>(11.3, 0.0, 5.7)),
        fbm(np + vec3<f32>(0.0, 7.1, 13.9)),
    );
    normal = normalize(normal + bump * GREASE_NOISE_STRENGTH);

    // PBR base. Force the greasy low roughness regardless of the global
    // lighting roughness so the wet look is consistent.
    var grease_lighting = lighting;
    grease_lighting.roughness = GREASE_ROUGHNESS;

    let NdotV = max(dot(normal, view_dir), 0.0);
    let R = reflect(-view_dir, normal);
    let irradiance = textureSample(irradiance_map, env_sampler, normal).rgb;
    let prefiltered = textureSampleLevel(prefiltered_map, env_sampler, R,
        GREASE_ROUGHNESS * MAX_IBL_MIP).rgb;
    let brdf = textureSample(brdf_lut, env_sampler,
        vec2<f32>(NdotV, GREASE_ROUGHNESS)).rg;

    let result: ShadingResult = shade_geometry(normal, view_dir,
        GREASE_BASE_COLOR, 0.0, grease_lighting, irradiance, prefiltered, brdf);
    var color = result.color;

    // Fresnel rim: wet edge glow.
    let fresnel = fresnel_schlick(NdotV, vec3<f32>(0.04));
    color = color + fresnel * GREASE_FRESNEL_STRENGTH;

    // Soft wet reflection: extra prefiltered-cubemap sample along R weighted
    // toward grazing angles.
    let refl_weight = (1.0 - NdotV) * GREASE_REFLECTION_STRENGTH;
    color = color + prefiltered * refl_weight;

    // Warm translucent interior at grazing/thin angles.
    color = color + GREASE_INTERIOR_COLOR * (1.0 - NdotV) * 0.3;

    // Emissive bloom highlight: concentrated where the perturbed highlight
    // is bright (a broken specular crawl). Pushed > 1.0 so it blooms warm.
    let highlight = pow(max(fbm(np * 1.7) * 0.5 + 0.5, 0.0), 3.0);
    color = color + GREASE_GLOW_COLOR * highlight * GREASE_GLOW_STRENGTH;

    let clip_pos = camera.view_proj * vec4<f32>(world_hit, 1.0);
    let ndc_depth = clip_pos.z / clip_pos.w;

    var out: FragOut;
    out.depth = ndc_depth;
    if (camera.debug_mode == 1u) {
        out.color = vec4<f32>(normal * 0.5 + 0.5, 1.0);
    } else {
        out.color = vec4<f32>(color, 1.0);
    }
    out.normal = vec4<f32>(normal, result.ambient_ratio);
    return out;
}
