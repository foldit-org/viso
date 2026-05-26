//! Spline math and frame computation for backbone geometry.
//!
//! Pure Vec3 -> Vec3 transforms.

use glam::Vec3;
use molex::SSType;

/// A point along the spline with position, tangent, and frame vectors.
#[derive(Clone, Copy)]
pub(crate) struct SplinePoint {
    pub(crate) pos: Vec3,
    pub(crate) tangent: Vec3,
    pub(crate) normal: Vec3,
    pub(crate) binormal: Vec3,
}

/// Position + tangent only -- the pre-frame intermediate along a spline.
///
/// A completed [`SplinePoint`] (with orthonormal normal/binormal) is
/// produced from a slice of these by [`rmf_frames`]. Carrying this
/// distinct type keeps the invalid "frame with zeroed normal/binormal"
/// state out of the mesh path entirely.
#[derive(Clone, Copy)]
pub(crate) struct SplineTrace {
    pub(crate) pos: Vec3,
    pub(crate) tangent: Vec3,
}

/// Build position+tangent traces from spline positions and tangents.
pub(crate) fn build_traces(
    spline: &[Vec3],
    tangents: &[Vec3],
) -> Vec<SplineTrace> {
    spline
        .iter()
        .zip(tangents.iter())
        .map(|(&pos, &tangent)| SplineTrace { pos, tangent })
        .collect()
}

/// Phantom (ghost) neighbor control points for the span starting at
/// control point `j` (ending at `j + 1`).
///
/// Catmull-Rom needs `points[j-1]` and `points[j+2]`; at the chain
/// boundaries those don't exist and are mirror-extrapolated
/// (`2*endpoint - adjacent`) so the curve stays symmetric. Returns
/// `(p0, p3)` -- the outer pair flanking `points[j]`/`points[j+1]`.
fn ghost_neighbors(points: &[Vec3], j: usize) -> (Vec3, Vec3) {
    let n = points.len();
    let p0 = if j == 0 {
        points[0] * 2.0 - points[1]
    } else {
        points[j - 1]
    };
    let p3 = if j + 2 >= n {
        points[n - 1] * 2.0 - points[n - 2]
    } else {
        points[j + 2]
    };
    (p0, p3)
}

/// Catmull-Rom cubic basis at parameter `t in [0, 1]` for one span's four
/// control points (`p1`/`p2` are the span endpoints, `p0`/`p3` flank).
fn catmull_rom_basis(p0: Vec3, p1: Vec3, p2: Vec3, p3: Vec3, t: f32) -> Vec3 {
    let t2 = t * t;
    let t3 = t2 * t;
    0.5 * ((2.0 * p1)
        + (-p0 + p2) * t
        + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * t2
        + (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * t3)
}

/// Map spline sample `i` (of `total_spline`) to the bracketing per-
/// residue indices `(r0, r1)` and the blend parameter `t in [0, 1]`
/// between them.
///
/// Shared core of
/// [`interpolate_profiles`](super::profile::interpolate_profiles) and
/// [`interpolate_per_residue_normals`](super::path::interpolate_per_residue_normals);
/// each applies its own pairwise blend (plain lerp vs. the hemisphere-
/// aligned normal lerp) on top of this identical index math.
pub(crate) fn residue_bracket(
    i: usize,
    total_spline: usize,
    n_residues: usize,
) -> (usize, usize, f32) {
    let frac = i as f32 / (total_spline - 1).max(1) as f32;
    let rf = frac * (n_residues - 1) as f32;
    let r0 = (rf.floor() as usize).min(n_residues - 1);
    let r1 = (r0 + 1).min(n_residues - 1);
    (r0, r1, rf - r0 as f32)
}

/// Evaluate a single point on the Catmull-Rom spline.
///
/// `span` selects which pair of control points (0..n-2), and `t` is the
/// parameter within that span (0.0 = start control point, 1.0 = end).
/// Uses phantom-point extrapolation (`ghost_neighbors`) for boundary
/// spans.
pub(crate) fn catmull_rom_eval(
    points: &[Vec3],
    span: usize,
    t: f32,
) -> Option<Vec3> {
    let n = points.len();
    if n < 2 || span >= n - 1 {
        return None;
    }

    let (p0, p3) = ghost_neighbors(points, span);
    let p1 = points[span];
    let p2 = points[span + 1];
    Some(catmull_rom_basis(p0, p1, p2, p3, t))
}

/// Cubic Hermite evaluation with explicit endpoint positions and tangents.
///
/// `t in [0, 1]`. Tangent magnitudes are in "position units per unit `t`",
/// so their scale must be commensurate with the knot spacing (roughly one
/// knot interval) or the curve will overshoot.
fn hermite(p0: Vec3, m0: Vec3, p1: Vec3, m1: Vec3, t: f32) -> Vec3 {
    let t2 = t * t;
    let t3 = t2 * t;
    let h00 = 2.0 * t3 - 3.0 * t2 + 1.0;
    let h10 = t3 - 2.0 * t2 + t;
    let h01 = -2.0 * t3 + 3.0 * t2;
    let h11 = t3 - t2;
    p0 * h00 + m0 * h10 + p1 * h01 + m1 * h11
}

/// Emit `segments_per_span` Catmull-Rom samples for the span starting at
/// control point `j` (ending just before control point `j + 1`). Uses
/// phantom endpoints (mirror extrapolation, via [`ghost_neighbors`]) at
/// chain boundaries.
fn append_catmull_rom_span(
    points: &[Vec3],
    j: usize,
    segments_per_span: usize,
    out: &mut Vec<Vec3>,
) {
    let (p0, p3) = ghost_neighbors(points, j);
    let p1 = points[j];
    let p2 = points[j + 1];

    for s in 0..segments_per_span {
        let t = s as f32 / segments_per_span as f32;
        out.push(catmull_rom_basis(p0, p1, p2, p3, t));
    }
}

/// 4-tap filtered midpoint between `points[j]` and `points[j + 1]`.
///
/// Uses a cardinal-B-spline-like 4-tap kernel with extension coefficient
/// 0.25, so the midpoint sits slightly outside the straight chord
/// midpoint to track curvature. Reflects at chain boundaries.
fn filtered_midpoint(points: &[Vec3], j: usize) -> Vec3 {
    let (ca_prev, ca_next) = ghost_neighbors(points, j);
    let ca_j = points[j];
    let ca_j1 = points[j + 1];
    // 0.5*((1 + 0.25)*(ca[j] + ca[j+1]) - 0.25*(ca[j-1] + ca[j+2]))
    //   = 0.625*(ca[j] + ca[j+1]) - 0.125*(ca[j-1] + ca[j+2])
    (ca_j + ca_j1) * 0.625 - (ca_prev + ca_next) * 0.125
}

/// Emit `segments_per_span` samples for the span starting at control
/// point `j` using a dual cubic Hermite with a filtered midpoint knot.
///
/// The span `CA[j] -> CA[j+1]` is split in half by a 4-tap filtered
/// midpoint, and each half becomes a cubic Hermite segment. Tangents at
/// all four knots (`CA[j-1]`-ish, `CA[j]`, mid, `CA[j+1]`) are derived from
/// Catmull-Rom central differences on the interleaved CA/midpoint
/// sequence, so their magnitudes are naturally proportional to the
/// local knot spacing -- no hand-tuned scalars. Inserting the midpoint
/// halves each cubic span, which keeps each cubic "honest" and
/// eliminates the square-fold overshoot Catmull-Rom produces when it
/// tries to interpolate a rotating helical CA sequence across a full
/// residue.
fn append_dual_hermite_span(
    points: &[Vec3],
    j: usize,
    segments_per_span: usize,
    out: &mut Vec<Vec3>,
) {
    let ca_j = points[j];
    let ca_j1 = points[j + 1];
    let mid = filtered_midpoint(points, j);

    // Midpoints of the flanking spans. At chain boundaries, reflect
    // through the adjacent CA so the tangents stay symmetric.
    let mid_prev = if j == 0 {
        ca_j * 2.0 - mid
    } else {
        filtered_midpoint(points, j - 1)
    };
    let mid_next = if j + 2 >= points.len() {
        ca_j1 * 2.0 - mid
    } else {
        filtered_midpoint(points, j + 1)
    };

    // Catmull-Rom tangents on the interleaved sequence
    //     [..., mid_prev, CA[j], mid, CA[j+1], mid_next, ...]
    // which gives each tangent magnitude ~ half the two-knot chord,
    // i.e. naturally proportional to the local spacing. C1 continuous
    // across both the CA and the midpoint joins by construction.
    let tangent_j = (mid - mid_prev) * 0.5;
    let tangent_mid = (ca_j1 - ca_j) * 0.5;
    let tangent_j1 = (mid_next - mid) * 0.5;

    // Split samples evenly between the two half-Hermites. For odd
    // `segments_per_span`, the first half gets the extra sample.
    let k1 = segments_per_span.div_ceil(2);
    let k2 = segments_per_span - k1;

    for s in 0..k1 {
        let t = s as f32 / k1 as f32;
        out.push(hermite(ca_j, tangent_j, mid, tangent_mid, t));
    }
    for s in 0..k2 {
        let t = s as f32 / k2 as f32;
        out.push(hermite(mid, tangent_mid, ca_j1, tangent_j1, t));
    }
}

/// Spline a chain of control points, using dual Hermite interpolation
/// with filtered midpoint knots (see [`append_dual_hermite_span`]) for
/// spans where both endpoints are helix residues, and standard
/// Catmull-Rom elsewhere.
///
/// Sample layout: `segments_per_span` samples per span plus one final
/// endpoint.
pub(crate) fn helix_aware_spline(
    points: &[Vec3],
    ss_types: &[SSType],
    segments_per_span: usize,
) -> Vec<Vec3> {
    let n = points.len();
    if n < 2 {
        return points.to_vec();
    }
    if n < 3 {
        return linear_interpolate(points, segments_per_span);
    }

    let mut result = Vec::with_capacity((n - 1) * segments_per_span + 1);
    for j in 0..n - 1 {
        let both_helix = ss_types.get(j).copied() == Some(SSType::Helix)
            && ss_types.get(j + 1).copied() == Some(SSType::Helix);
        if both_helix {
            append_dual_hermite_span(points, j, segments_per_span, &mut result);
        } else {
            append_catmull_rom_span(points, j, segments_per_span, &mut result);
        }
    }
    result.push(points[n - 1]);
    result
}

/// Spline a chain of control points using dual Hermite interpolation
/// with filtered midpoint knots for every span.
///
/// Sample layout: `segments_per_span` samples per span plus one final
/// endpoint.
pub(crate) fn dual_hermite_spline(
    points: &[Vec3],
    segments_per_span: usize,
) -> Vec<Vec3> {
    let n = points.len();
    if n < 2 {
        return points.to_vec();
    }
    if n < 3 {
        return linear_interpolate(points, segments_per_span);
    }

    let mut result = Vec::with_capacity((n - 1) * segments_per_span + 1);
    for j in 0..n - 1 {
        append_dual_hermite_span(points, j, segments_per_span, &mut result);
    }
    result.push(points[n - 1]);
    result
}

/// Linear interpolation fallback for short point sequences.
pub(crate) fn linear_interpolate(
    points: &[Vec3],
    segments_per_span: usize,
) -> Vec<Vec3> {
    let mut result = Vec::new();
    for i in 0..points.len() - 1 {
        for j in 0..segments_per_span {
            let t = j as f32 / segments_per_span as f32;
            result.push(points[i].lerp(points[i + 1], t));
        }
    }
    if let Some(&last) = points.last() {
        result.push(last);
    }
    result
}

/// The single construction site of the position-only `SplinePoint`
/// shell. Every normal/binormal is filled by the frame solver
/// ([`compute_rmf`]) before any consumer sees the result, so a
/// half-built frame never escapes this module.
fn shells_from_traces(traces: &[SplineTrace]) -> Vec<SplinePoint> {
    traces
        .iter()
        .map(|t| SplinePoint {
            pos: t.pos,
            tangent: t.tangent,
            normal: Vec3::ZERO,
            binormal: Vec3::ZERO,
        })
        .collect()
}

/// Build rotation-minimizing frames from position+tangent traces.
///
/// `seed` is the chain-roll seed -- the first residue's peptide-plane
/// normal. When `Some`, it feeds [`compute_rmf`]'s frame-0 projection so
/// the whole chain's roll is fixed by backbone geometry rather than a
/// world axis; `None` lets `compute_rmf` fall back to a world axis. This
/// is the typed replacement for mutating a zeroed `frames[0].normal`.
pub(crate) fn rmf_frames(
    traces: &[SplineTrace],
    seed: Option<Vec3>,
) -> Vec<SplinePoint> {
    let mut frames = shells_from_traces(traces);
    if let (Some(seed), Some(first)) = (seed, frames.first_mut()) {
        first.normal = seed;
    }
    compute_rmf(&mut frames);
    frames
}

/// Compute Rotation Minimizing Frames using the double reflection method
/// (Wang et al. 2008).
pub(crate) fn compute_rmf(points: &mut [SplinePoint]) {
    if points.is_empty() {
        return;
    }

    let t0 = points[0].tangent;
    // Seed the chain roll from `points[0].normal` (a chain-geometry
    // quantity supplied by the caller) projected perpendicular to the
    // first tangent. Falls back to a world axis only when no usable seed
    // is given, so the roll is fixed by geometry rather than by which
    // world octant the chain happens to load in.
    let seed = points[0].normal - t0 * t0.dot(points[0].normal);
    let n0 = if seed.length_squared() > 1e-6 {
        seed.normalize()
    } else {
        let arbitrary = if t0.x.abs() < 0.9 { Vec3::X } else { Vec3::Y };
        t0.cross(arbitrary).normalize()
    };
    let b0 = t0.cross(n0).normalize();

    points[0].normal = n0;
    points[0].binormal = b0;

    for i in 0..points.len() - 1 {
        let x_i = points[i].pos;
        let x_i1 = points[i + 1].pos;
        let t_i = points[i].tangent;
        let t_i1 = points[i + 1].tangent;
        let r_i = points[i].normal;

        let v1 = x_i1 - x_i;
        let c1 = v1.dot(v1);

        if c1 < 1e-10 {
            points[i + 1].normal = r_i;
            points[i + 1].binormal = points[i].binormal;
            continue;
        }

        // First reflection
        let r_i_l = r_i - (2.0 / c1) * v1.dot(r_i) * v1;
        let t_i_l = t_i - (2.0 / c1) * v1.dot(t_i) * v1;

        // Second reflection
        let v2 = t_i1 - t_i_l;
        let c2 = v2.dot(v2);

        let r_i1 = if c2 < 1e-10 {
            r_i_l
        } else {
            r_i_l - (2.0 / c2) * v2.dot(r_i_l) * v2
        };

        // Ensure orthonormality
        let r_i1 = (r_i1 - t_i1 * t_i1.dot(r_i1)).normalize();
        let s_i1 = t_i1.cross(r_i1).normalize();

        points[i + 1].normal = r_i1;
        points[i + 1].binormal = s_i1;
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    const TOL: f32 = 1e-4;

    fn approx_eq(a: Vec3, b: Vec3) -> bool {
        (a - b).length() < TOL
    }

    #[test]
    fn linear_two_points() {
        let pts = vec![Vec3::ZERO, Vec3::new(10.0, 0.0, 0.0)];
        let result = linear_interpolate(&pts, 4);
        assert_eq!(result.len(), 5); // 4 segments + endpoint
        assert!(approx_eq(result[0], pts[0]));
        assert!(approx_eq(*result.last().unwrap(), pts[1]));
        // midpoint at t=0.5 (index 2)
        assert!(approx_eq(result[2], Vec3::new(5.0, 0.0, 0.0)));
    }

    #[test]
    fn rmf_orthonormal() {
        // Build a curved path
        let n = 20;
        let mut points: Vec<SplinePoint> = (0..n)
            .map(|i| {
                let t = i as f32 * 0.3;
                SplinePoint {
                    pos: Vec3::new(t.cos() * 5.0, t.sin() * 5.0, t * 2.0),
                    tangent: Vec3::ZERO,
                    normal: Vec3::ZERO,
                    binormal: Vec3::ZERO,
                }
            })
            .collect();
        // Compute tangents via finite differences
        for i in 0..n {
            let prev = if i == 0 { 0 } else { i - 1 };
            let next = (i + 1).min(n - 1);
            points[i].tangent =
                (points[next].pos - points[prev].pos).normalize();
        }
        compute_rmf(&mut points);

        for (i, p) in points.iter().enumerate() {
            assert!(
                (p.normal.length() - 1.0).abs() < TOL,
                "frame {i}: normal not unit"
            );
            assert!(
                (p.binormal.length() - 1.0).abs() < TOL,
                "frame {i}: binormal not unit"
            );
            assert!(
                p.normal.dot(p.binormal).abs() < TOL,
                "frame {i}: n*b != 0"
            );
            assert!(p.tangent.dot(p.normal).abs() < TOL, "frame {i}: t*n != 0");
        }
    }

    /// The typed `rmf_frames(&[SplineTrace], seed)` entry point must
    /// produce frames byte-identical to the pre-refactor path of
    /// "zeroed `SplinePoint` shells, `frames[0].normal = seed`,
    /// `compute_rmf`". This pins the `SplineTrace` introduction as
    /// behavior-preserving.
    #[test]
    fn rmf_frames_matches_legacy_shell_path() {
        let n = 16;
        let pos: Vec<Vec3> = (0..n)
            .map(|i| {
                let t = i as f32 * 0.27;
                Vec3::new(t.cos() * 4.0, t.sin() * 3.0, t * 1.7)
            })
            .collect();
        let mut tangents = vec![Vec3::ZERO; n];
        for (i, t) in tangents.iter_mut().enumerate() {
            let prev = if i == 0 { 0 } else { i - 1 };
            let next = (i + 1).min(n - 1);
            *t = (pos[next] - pos[prev]).normalize();
        }
        let seed = Vec3::new(0.3, 0.8, -0.5).normalize();

        // Legacy path: zeroed shells, seed mutated into frame 0.
        let mut legacy: Vec<SplinePoint> = pos
            .iter()
            .zip(tangents.iter())
            .map(|(&p, &t)| SplinePoint {
                pos: p,
                tangent: t,
                normal: Vec3::ZERO,
                binormal: Vec3::ZERO,
            })
            .collect();
        legacy[0].normal = seed;
        compute_rmf(&mut legacy);

        // Typed path through the new intermediate.
        let traces: Vec<SplineTrace> = pos
            .iter()
            .zip(tangents.iter())
            .map(|(&p, &t)| SplineTrace { pos: p, tangent: t })
            .collect();
        let typed = rmf_frames(&traces, Some(seed));

        assert_eq!(typed.len(), legacy.len());
        for (i, (a, b)) in typed.iter().zip(legacy.iter()).enumerate() {
            assert!(approx_eq(a.pos, b.pos), "frame {i}: pos differs");
            assert!(
                approx_eq(a.tangent, b.tangent),
                "frame {i}: tangent differs"
            );
            assert!(
                approx_eq(a.normal, b.normal),
                "frame {i}: normal differs ({:?} vs {:?})",
                a.normal,
                b.normal,
            );
            assert!(
                approx_eq(a.binormal, b.binormal),
                "frame {i}: binormal differs"
            );
            assert!(
                (a.normal.length() - 1.0).abs() < TOL
                    && a.normal.dot(a.tangent).abs() < TOL,
                "frame {i}: not orthonormal"
            );
        }
    }

    #[test]
    fn rmf_seed_is_rotation_equivariant() {
        // The RMF chain roll must be fixed by chain geometry, not by a
        // world axis. Feeding the same control points (and the same
        // geometry-derived seed normal) in two world orientations must
        // produce frames that differ only by that rotation.
        let n = 12;
        let base_pos: Vec<Vec3> = (0..n)
            .map(|i| {
                let t = i as f32 * 0.3;
                Vec3::new(t.cos() * 5.0, t.sin() * 5.0, t * 2.0)
            })
            .collect();
        let seed = Vec3::new(0.2, 0.9, 0.3).normalize();
        let r = glam::Quat::from_axis_angle(
            Vec3::new(1.0, 2.0, 3.0).normalize(),
            0.7,
        );

        let build = |xf: &dyn Fn(Vec3) -> Vec3| -> Vec<SplinePoint> {
            let mut pts: Vec<SplinePoint> = base_pos
                .iter()
                .map(|&p| SplinePoint {
                    pos: xf(p),
                    tangent: Vec3::ZERO,
                    normal: Vec3::ZERO,
                    binormal: Vec3::ZERO,
                })
                .collect();
            for i in 0..n {
                let prev = if i == 0 { 0 } else { i - 1 };
                let next = (i + 1).min(n - 1);
                pts[i].tangent = (pts[next].pos - pts[prev].pos).normalize();
            }
            // The seed is a chain-geometry quantity, so it rotates with
            // the chain.
            pts[0].normal = xf(seed) - xf(Vec3::ZERO);
            compute_rmf(&mut pts);
            pts
        };

        let id = build(&|p| p);
        let rot = build(&|p| r * p);

        for i in 0..n {
            let mapped = r * id[i].normal;
            assert!(
                (mapped - rot[i].normal).length() < 1e-3,
                "frame {i}: R*normal {mapped:?} != rotated normal {:?}",
                rot[i].normal,
            );
        }
    }

    #[test]
    fn rmf_empty_and_single() {
        // Empty input
        let mut empty: Vec<SplinePoint> = vec![];
        compute_rmf(&mut empty);

        // Single point
        let mut single = vec![SplinePoint {
            pos: Vec3::ZERO,
            tangent: Vec3::Z,
            normal: Vec3::ZERO,
            binormal: Vec3::ZERO,
        }];
        compute_rmf(&mut single);
        // Should not panic -- that's the test
    }

    /// T0-NA-A regression: the nucleic-acid ribbon path now goes through
    /// `rmf_frames(&traces, seed)`, not raw Frenet. On a straight-then-
    /// curved chain the old Frenet path recomputed an arbitrary world
    /// axis per straight sample and flipped the `-curv` normal 180deg
    /// through the inflection. RMF must instead stay orthonormal, keep a
    /// fixed roll across the straight run, and never flip hemisphere
    /// between adjacent samples through the bend.
    #[test]
    fn na_rmf_no_flip_on_straight_then_curved() {
        // 8 straight samples along +Z, then a quarter circle in the
        // Z-X plane -- a strong inflection at the join.
        let mut pos: Vec<Vec3> =
            (0..8).map(|i| Vec3::new(0.0, 0.0, i as f32)).collect();
        let r = 5.0;
        for k in 1..=12 {
            let a = (k as f32 / 12.0) * std::f32::consts::FRAC_PI_2;
            pos.push(Vec3::new(r * (1.0 - a.cos()), 0.0, 7.0 + r * a.sin()));
        }
        let n = pos.len();
        let traces: Vec<SplineTrace> = (0..n)
            .map(|i| {
                let prev = if i == 0 { 0 } else { i - 1 };
                let next = (i + 1).min(n - 1);
                SplineTrace {
                    pos: pos[i],
                    tangent: (pos[next] - pos[prev]).normalize(),
                }
            })
            .collect();

        // Seed analogous to a first base-ring normal (perp-ish to t0).
        let frames = rmf_frames(&traces, Some(Vec3::X));

        for (i, f) in frames.iter().enumerate() {
            assert!(
                (f.normal.length() - 1.0).abs() < TOL
                    && (f.binormal.length() - 1.0).abs() < TOL,
                "frame {i}: not unit length"
            );
            assert!(
                f.tangent.dot(f.normal).abs() < TOL,
                "frame {i}: normal not perpendicular to tangent"
            );
        }
        for i in 1..frames.len() {
            assert!(
                frames[i].normal.dot(frames[i - 1].normal) > 0.0,
                "frame {}->{i}: 180deg ribbon flip (the old Frenet defect)",
                i - 1,
            );
        }
        // Across the straight run the tangent is constant, so a
        // rotation-minimizing frame must not roll at all.
        for i in 1..7 {
            assert!(
                frames[i].normal.dot(frames[0].normal) > 0.999,
                "frame {i}: roll drift on the straight segment"
            );
        }
    }
}
