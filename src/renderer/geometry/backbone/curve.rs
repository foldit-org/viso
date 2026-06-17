//! Backbone curve helpers split out of `spline.rs`: B-spline smoothing
//! and sliding-window centroids.
//!
//! These are non-frame curve utilities (no RMF/Frenet state); the frame
//! math stays in `spline.rs`.

use glam::Vec3;

use super::spline::linear_interpolate;

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
