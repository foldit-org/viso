// Cavity lava-lamp displacement, shared between the isosurface front-face
// pass and the back-face depth pre-pass so both displace cavity geometry
// identically (Beer-Lambert thickness depends on front/back agreement).

#define_import_path viso::cavity

// Mirrors `renderer::geometry::isosurface::isosurface_kind` in Rust.
// Keep these in sync — they're the per-vertex source-kind discriminator.
const ISO_KIND_SURFACE: u32 = 0u;
const ISO_KIND_CAVITY: u32 = 1u;
const ISO_KIND_DENSITY: u32 = 2u;

// ── Lava-lamp displacement helpers ────────────────────────────────────
//
// Cheap 3D value noise used as a SPATIAL liveliness map (no time
// component) so the noise field stays static and the cavity doesn't
// drift. The time-varying motion comes from the sum-of-sines block.

fn hash13(p: vec3<f32>) -> f32 {
    let q = vec3<f32>(
        dot(p, vec3<f32>(127.1, 311.7,  74.7)),
        dot(p, vec3<f32>(269.5, 183.3, 246.1)),
        dot(p, vec3<f32>(113.5, 271.9, 124.6)),
    );
    return fract(sin(q.x + q.y + q.z) * 43758.5453);
}

fn noise3d(p: vec3<f32>) -> f32 {
    let i = floor(p);
    let f = fract(p);
    let u = f * f * (3.0 - 2.0 * f);  // smoothstep interpolation

    let n000 = hash13(i + vec3<f32>(0.0, 0.0, 0.0));
    let n100 = hash13(i + vec3<f32>(1.0, 0.0, 0.0));
    let n010 = hash13(i + vec3<f32>(0.0, 1.0, 0.0));
    let n110 = hash13(i + vec3<f32>(1.0, 1.0, 0.0));
    let n001 = hash13(i + vec3<f32>(0.0, 0.0, 1.0));
    let n101 = hash13(i + vec3<f32>(1.0, 0.0, 1.0));
    let n011 = hash13(i + vec3<f32>(0.0, 1.0, 1.0));
    let n111 = hash13(i + vec3<f32>(1.0, 1.0, 1.0));

    let nx00 = mix(n000, n100, u.x);
    let nx10 = mix(n010, n110, u.x);
    let nx01 = mix(n001, n101, u.x);
    let nx11 = mix(n011, n111, u.x);
    let nxy0 = mix(nx00, nx10, u.y);
    let nxy1 = mix(nx01, nx11, u.y);
    return mix(nxy0, nxy1, u.z);
}

/// Compute the lava-lamp displacement for a cavity vertex.
///
/// Three motion modes layered:
///   1. Surface undulation: sum of sines along the vertex normal. Both
///      AMPLITUDE and PHASE come from spatial noise — amplitude varies
///      where the surface is "lively", phase varies so different
///      regions peak at different times. No linear wavefronts, no
///      lockstep.
///   2. Radial breath: whole cavity grows/shrinks from its centroid.
///      Frequency and phase are derived from noise sampled at the
///      cavity centroid, so each cavity breathes on its own rhythm.
///   3. (No drift — all noise inputs are spatial, no time component.
///      Time only enters through the sine arguments.)
///
/// Returns the world-space offset to add to the rest position.
fn cavity_displacement(
    rest_pos: vec3<f32>,
    normal: vec3<f32>,
    cavity_center: vec3<f32>,
    time: f32,
) -> vec3<f32> {
    let TAU = 6.28318530718;

    // Per-region amplitude for the surface undulation.
    let liveliness = noise3d(rest_pos * 0.5);
    let surf_amp = mix(0.08, 0.30, liveliness);

    // Per-region PHASES for the surface undulation. Sampling noise
    // (instead of dot products) eliminates the planar-wavefront look
    // that linear phases give — neighboring regions wobble independently.
    let phase_a = noise3d(rest_pos * 0.3 + vec3<f32>(13.7,  0.0,  0.0)) * TAU;
    let phase_b = noise3d(rest_pos * 0.4 + vec3<f32>( 0.0, 41.2,  0.0)) * TAU;
    let phase_c = noise3d(rest_pos * 0.6 + vec3<f32>( 0.0,  0.0, 27.5)) * TAU;
    let surf_wave =
          sin(time * 1.10 + phase_a) * 0.5
        + sin(time * 1.65 + phase_b) * 0.3
        + sin(time * 0.75 + phase_c) * 0.2;

    // Per-cavity phase + frequency for the radial breath. Both come
    // from noise sampled at the centroid, so two cavities at different
    // positions get independent breathing rhythms. Frequencies span
    // [0.60, 1.50] rad/s (periods ~4.2s to ~10.5s).
    let cavity_phase = noise3d(cavity_center * 0.7) * TAU;
    let cavity_freq  = 0.60 + noise3d(cavity_center * 0.9 + vec3<f32>(7.7)) * 0.90;

    // Radial breath — whole cavity inflates/deflates from its centroid.
    let to_center = rest_pos - cavity_center;
    let radial_dir = to_center / max(length(to_center), 0.0001);
    let radial = sin(time * cavity_freq + cavity_phase) * 0.15;

    return normal * surf_wave * surf_amp + radial_dir * radial;
}
