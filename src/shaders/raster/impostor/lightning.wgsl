// Procedural lightning bolt for steric clash markers.
//
// One instance per clash. The whole jagged glowing bolt is drawn in the
// fragment shader on a single camera-facing billboard ribbon spanning the
// two clashing atoms: a bright hot-red electric core that blooms, wrapped
// in a deep dark semitransparent red envelope that fades to transparent at
// the ribbon edges. The bolt's centerline is an fBm-displaced curve that
// scrolls smoothly with camera.time (no position snaps); the "electric"
// energy lives in a fast brightness flicker, not in the geometry.

#import viso::camera::CameraUniform
#import viso::cavity::{hash13, noise3d}

// ── Tunable look constants ────────────────────────────────────────────
//
// Spatial frequency of the centerline fBm along the bolt axis. Higher =
// more wiggles between the two atoms.
const BOLT_FREQ: f32 = 6.0;
// Axial scroll speed of the centerline (cycles/sec-ish). The whole jag
// pattern slides smoothly along the bolt at this rate.
const BOLT_SCROLL: f32 = 1.3;
// Peak lateral centerline offset as a fraction of the ribbon half-width,
// before the severity scale. Worse clashes jag wider. Kept comfortably
// inside BOLT_EDGE_FADE_START so the jagged centerline plus its halo stay
// within the faded-in region and never touch the quad's lateral edge.
const BOLT_AMPLITUDE_FRAC: f32 = 0.45;
// Fixed bolt half-thickness in world units (Angstroms). This is a flat
// ribbon width, NOT an SDF radius.
const BOLT_HALF_THICKNESS: f32 = 0.55;
// Extra ribbon margin (Angstroms) added past the endpoints along the axis
// so the bolt's glow does not clip at the atoms.
const BOLT_AXIAL_MARGIN: f32 = 0.25;
// Flicker rate of the core brightness (Hz-ish). Drives the electric snap.
const BOLT_FLICKER_HZ: f32 = 9.0;
// Width of the bright core as a fraction of the half-thickness. Smaller =
// thinner, sharper electric line.
const BOLT_CORE_WIDTH_FRAC: f32 = 0.10;
// Width of the glow envelope as a fraction of the half-thickness. Tight
// enough that the deep-red halo is a thin band hugging the centerline and
// fades to ~0 well before the quad's lateral edge.
const BOLT_GLOW_WIDTH_FRAC: f32 = 0.22;
// Weight of the glow envelope in the final alpha. The halo is a faint band,
// not a fill; the bright core dominates the opacity near the centerline.
const BOLT_GLOW_ALPHA: f32 = 0.35;
// Fragments below this composed alpha are discarded. Raised well above a
// hairline so the faint background is culled outright: this both erases the
// visible billboard rectangle and stops the quad writing depth where it is
// effectively empty (discarded fragments do not write depth).
const BOLT_DISCARD_ALPHA: f32 = 0.03;
// Lateral edge window: |u| at which alpha begins fading in toward the
// centerline. Beyond this (toward |u| = 1) alpha is forced to 0 so the
// quad's vertical sides are guaranteed invisible regardless of glow math.
const BOLT_EDGE_FADE_START: f32 = 0.85;
// Axial end-fade zone (in v-units from each end) over which the bolt fades
// to 0 at v = 0 and v = 1, so it does not terminate in a hard rectangle
// edge. Kept small so the bolt still visually reaches both atoms.
const BOLT_AXIAL_FADE: f32 = 0.06;
// Hot-red core color in HDR. Above 1.0 so it crosses the bloom threshold
// and blooms RED (not white).
const BOLT_CORE_COLOR: vec3<f32> = vec3<f32>(2.2, 0.45, 0.16);
// Deep dark red envelope color, matching the legacy clash palette.
const BOLT_ENVELOPE_COLOR: vec3<f32> = vec3<f32>(0.643, 0.0, 0.0);
// How much the severity (squashed to 0..1) boosts the centerline jag
// amplitude: amplitude scales over [1 - SEVERITY_GAIN .. 1] * base.
const BOLT_SEVERITY_GAIN: f32 = 0.5;

// Soft-squash knob for the unbounded rosetta clash severity (raw per-pair
// LJ repulsion, which can reach tens to hundreds): severity is mapped into
// 0..1 via 1 - exp(-severity / SCALE). Tuned visually.
const BOLT_SEVERITY_SCALE: f32 = 20.0;

// ── Instance data (twin of the Rust LightningInstance) ────────────────
//
// endpoint_a: xyz = first clashing atom, w = clash severity (raw).
// endpoint_b: xyz = second clashing atom, w = per-clash seed.
struct LightningInstance {
    endpoint_a: vec4<f32>,
    endpoint_b: vec4<f32>,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    // u across the ribbon thickness in [-1, 1], v along the axis in [0, 1].
    @location(0) local_uv: vec2<f32>,
    @location(1) @interpolate(flat) severity: f32,
    @location(2) @interpolate(flat) seed: f32,
};

struct FragOut {
    @location(0) color: vec4<f32>,
    @location(1) normal: vec4<f32>,
};

@group(0) @binding(0) var<uniform> camera: CameraUniform;
@group(3) @binding(0) var<storage, read> bolts: array<LightningInstance>;

// fractional Brownian motion: a few octaves of value noise summed at
// halving amplitude / doubling frequency. Sampled along a 1D path mapped
// into 3D so we reuse the shared noise3d field.
fn fbm(p: vec3<f32>) -> f32 {
    var sum = 0.0;
    var amp = 0.5;
    var freq = 1.0;
    for (var i = 0; i < 5; i = i + 1) {
        sum = sum + amp * (noise3d(p * freq) * 2.0 - 1.0);
        amp = amp * 0.5;
        freq = freq * 2.0;
    }
    return sum;
}

// Lateral centerline offset (in [-1, 1] ribbon space) at axial position v.
// Two layered fBm passes: a large low-frequency sway plus a fine high-
// frequency crackle. Scrolls smoothly with time so the jag slides without
// resetting. `amp_frac` already folds in the severity scaling.
fn centerline(v: f32, seed: f32, amp_frac: f32) -> f32 {
    let scroll = camera.time * BOLT_SCROLL;
    let sway = fbm(vec3<f32>(v * BOLT_FREQ + scroll, seed * 17.3, 0.0));
    let crackle = fbm(vec3<f32>(v * BOLT_FREQ * 3.0 + scroll * 1.7, seed * 5.1, 11.0));
    return clamp((sway * 0.7 + crackle * 0.3) * amp_frac, -1.0, 1.0);
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

    let bolt = bolts[iidx];
    let endpoint_a = bolt.endpoint_a.xyz;
    let endpoint_b = bolt.endpoint_b.xyz;

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

    let half_width = BOLT_HALF_THICKNESS;
    let half_height = seg_length * 0.5 + BOLT_AXIAL_MARGIN;

    let local_uv = quad[vidx];
    let world_offset = right * local_uv.x * half_width + up * local_uv.y * half_height;
    let world_pos = center + world_offset;

    var out: VertexOutput;
    out.clip_position = camera.view_proj * vec4<f32>(world_pos, 1.0);
    // Map v from quad-space [-1, 1] to axial [0, 1].
    out.local_uv = vec2<f32>(local_uv.x, local_uv.y * 0.5 + 0.5);
    out.severity = bolt.endpoint_a.w;
    out.seed = bolt.endpoint_b.w;

    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> FragOut {
    let u = in.local_uv.x; // [-1, 1] across thickness
    let v = in.local_uv.y; // [0, 1] along axis

    // Soft-squash the unbounded raw severity into 0..1, then map to the
    // centerline amplitude fraction. Worse clashes jag wider.
    let raw = max(in.severity, 0.0);
    let severity_norm = clamp(1.0 - exp(-raw / BOLT_SEVERITY_SCALE), 0.0, 1.0);
    let amp_frac = BOLT_AMPLITUDE_FRAC
        * ((1.0 - BOLT_SEVERITY_GAIN) + BOLT_SEVERITY_GAIN * severity_norm);

    // Distance from this fragment to the jagged centerline at this v.
    let c = centerline(v, in.seed, amp_frac);
    let dist = abs(u - c);

    // Bright core: a thin inverse-exponential falloff around the centerline.
    let core_w = BOLT_CORE_WIDTH_FRAC;
    let core = exp(-(dist * dist) / (core_w * core_w));

    // Glow envelope: a wider, softer falloff for the deep-red halo.
    let glow_w = BOLT_GLOW_WIDTH_FRAC;
    let glow = exp(-(dist * dist) / (glow_w * glow_w));

    // Fast brightness flicker for electric energy. The geometry stays
    // continuous; only the core intensity snaps.
    let flick_n = noise3d(vec3<f32>(camera.time * BOLT_FLICKER_HZ, in.seed * 3.7, 0.0));
    let flicker = 0.7 + 0.3 * flick_n;

    // Compose color: hot core blends over the deep-red envelope.
    let core_intensity = core * flicker;
    let color = BOLT_ENVELOPE_COLOR * glow + BOLT_CORE_COLOR * core_intensity;

    // Alpha: the core dominates at the centerline and a faint halo hugs it;
    // away from the centerline alpha falls to ~0 so each bolt reads as a
    // thin glowing line with transparent surroundings.
    var alpha = clamp(glow * BOLT_GLOW_ALPHA + core_intensity, 0.0, 1.0);
    // Lateral edge window: force alpha to 0 approaching the quad's sides so
    // the vertical edges are invisible no matter what the glow math yields.
    alpha = alpha * smoothstep(1.0, BOLT_EDGE_FADE_START, abs(u));
    // Axial end fade: ramp from 0 at each end (v = 0 and v = 1) up to 1 in
    // the interior so the bolt does not finish in a hard top/bottom edge.
    alpha = alpha
        * smoothstep(0.0, BOLT_AXIAL_FADE, v)
        * smoothstep(1.0, 1.0 - BOLT_AXIAL_FADE, v);
    if (alpha < BOLT_DISCARD_ALPHA) {
        discard;
    }

    var out: FragOut;
    out.color = vec4<f32>(color, alpha);
    // Lightning is not a lit surface; write a neutral camera-facing normal
    // and a mid ambient ratio into the meta target.
    out.normal = vec4<f32>(0.0, 0.0, 1.0, 1.0);
    return out;
}
