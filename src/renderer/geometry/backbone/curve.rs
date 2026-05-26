//! Backbone curve helpers split out of `spline.rs`: backbone-atom
//! projection, B-spline smoothing, and sliding-window centroids.
//!
//! These are non-frame curve utilities (no RMF/Frenet state); the frame
//! math stays in `spline.rs`.

use glam::Vec3;

use super::spline::{catmull_rom_eval, linear_interpolate};

/// Project backbone N and C atom positions onto the Catmull-Rom spline
/// defined by CA control points.
///
/// Returns `(n_positions, c_positions)` -- one per residue -- sitting on
/// the rendered backbone curve. The span fractions below are empirically
/// tuned to place C/N plausibly on the CA spline; they are *not* exact
/// `bond_length / 3.8 A` ratios (those would be ~0.40 / ~0.61) -- the
/// rendered curve is not a straight CA->CA chord, so the visually
/// correct fractions differ from the idealized geometric ones.
///
/// Edge residues (first N, last C in each chain) use the raw backbone
/// positions since they fall outside the spline's control-point range.
pub(crate) fn project_backbone_atoms(
    backbone_chains: &[crate::renderer::entity_topology::ProteinBackboneChain],
) -> (Vec<Vec3>, Vec<Vec3>) {
    /// Fraction of CA->CA span where C sits (CA->C / total).
    const C_FRAC: f32 = 0.35;
    /// Fraction of CA->CA span where N sits (measured from previous CA).
    const N_FRAC: f32 = 0.66;

    let mut all_n = Vec::new();
    let mut all_c = Vec::new();

    for chain in backbone_chains {
        let n_res = chain.residue_count();
        if n_res == 0 {
            continue;
        }

        for i in 0..n_res {
            // N position: on the spline at N_FRAC into span (i-1 -> i).
            let n_pos = if i == 0 || chain.ca().len() < 2 {
                chain.n()[i] // raw N for first residue
            } else {
                catmull_rom_eval(chain.ca(), i - 1, N_FRAC)
                    .unwrap_or_else(|| chain.n()[i])
            };

            // C position: on the spline at C_FRAC into span (i -> i+1).
            let c_pos = if i >= n_res - 1 || chain.ca().len() < 2 {
                chain.c()[i] // raw C for last residue
            } else {
                catmull_rom_eval(chain.ca(), i, C_FRAC)
                    .unwrap_or_else(|| chain.c()[i])
            };

            all_n.push(n_pos);
            all_c.push(c_pos);
        }
    }

    (all_n, all_c)
}

/// Cubic B-spline (smooth approximation, does not pass through control points).
/// Used for helix axis smoothing.
pub(crate) fn cubic_bspline(
    points: &[Vec3],
    segments_per_span: usize,
) -> Vec<Vec3> {
    let n = points.len();
    if n < 2 {
        return points.to_vec();
    }
    if n < 4 {
        return linear_interpolate(points, segments_per_span);
    }

    let mut result = Vec::new();

    fn b0(t: f32) -> f32 {
        (1.0 - t).powi(3) / 6.0
    }
    fn b1(t: f32) -> f32 {
        (3.0 * t.powi(3) - 6.0 * t.powi(2) + 4.0) / 6.0
    }
    fn b2(t: f32) -> f32 {
        (-3.0 * t.powi(3) + 3.0 * t.powi(2) + 3.0 * t + 1.0) / 6.0
    }
    fn b3(t: f32) -> f32 {
        t.powi(3) / 6.0
    }

    let mut padded = Vec::with_capacity(n + 2);
    padded.push(points[0] * 2.0 - points[1]);
    padded.extend_from_slice(points);
    padded.push(points[n - 1] * 2.0 - points[n - 2]);

    for i in 0..n - 1 {
        let p0 = padded[i];
        let p1 = padded[i + 1];
        let p2 = padded[i + 2];
        let p3 = padded[i + 3];

        for j in 0..segments_per_span {
            let t = j as f32 / segments_per_span as f32;
            let pos = p0 * b0(t) + p1 * b1(t) + p2 * b2(t) + p3 * b3(t);
            result.push(pos);
        }
    }

    result.push(points[n - 1]);
    result
}

/// Sliding-window centroids of the CA positions (window ~ one helix
/// turn). Used as a helix-axis approximation for the radial-normal
/// blend, but it is a plain centroid of *all* CAs, not a fitted axis --
/// the name reflects what it computes.
pub(crate) fn sliding_window_centroids(ca_positions: &[Vec3]) -> Vec<Vec3> {
    let n = ca_positions.len();
    let window = 4; // ~one helix turn

    let mut centers = Vec::with_capacity(n);
    for i in 0..n {
        let start = i.saturating_sub(window / 2);
        let end = (i + window / 2 + 1).min(n);
        let mut sum = Vec3::ZERO;
        for pos in &ca_positions[start..end] {
            sum += *pos;
        }
        centers.push(sum / (end - start) as f32);
    }
    centers
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
    fn bspline_last_point_matches() {
        let pts: Vec<Vec3> = (0..6)
            .map(|i| Vec3::new(i as f32, (i as f32 * 0.5).sin(), 0.0))
            .collect();
        let result = cubic_bspline(&pts, 4);
        assert!(approx_eq(*result.last().unwrap(), *pts.last().unwrap()));
    }

    #[test]
    fn bspline_few_points_linear() {
        let pts = vec![Vec3::ZERO, Vec3::X, Vec3::new(2.0, 0.0, 0.0)];
        let result = cubic_bspline(&pts, 4);
        let linear = linear_interpolate(&pts, 4);
        assert_eq!(result.len(), linear.len());
    }

    #[test]
    fn helix_axis_preserves_count() {
        let pts: Vec<Vec3> = (0..10)
            .map(|i| {
                let t = i as f32 * 0.5;
                Vec3::new(t.cos(), t.sin(), t * 1.5)
            })
            .collect();
        let axis = sliding_window_centroids(&pts);
        assert_eq!(axis.len(), pts.len());
    }
}
