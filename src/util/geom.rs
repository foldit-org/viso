//! Pure geometric-math primitives shared across the engine.
//!
//! Leaf module: depends on nothing in `renderer/` or `engine/`, so any
//! layer may depend on it downward. Holds math that is not specific to
//! one primitive renderer.

use glam::Vec3;

/// Newell's-method plane normal for a ring of coplanar positions.
///
/// Returns the zero vector for a degenerate (collinear / coincident)
/// ring. The sign is fixed by the cyclic traversal order of the input;
/// callers that need cross-base sign coherence apply their own
/// hemisphere-alignment pass on top of this.
#[must_use]
pub(crate) fn newell_normal(positions: &[Vec3]) -> Vec3 {
    let n = positions.len();
    let mut normal = Vec3::ZERO;
    for i in 0..n {
        let curr = positions[i];
        let next = positions[(i + 1) % n];
        normal.x += (curr.y - next.y) * (curr.z + next.z);
        normal.y += (curr.z - next.z) * (curr.x + next.x);
        normal.z += (curr.x - next.x) * (curr.y + next.y);
    }
    normal.normalize_or_zero()
}

/// Tangents along a polyline via central differences.
///
/// Interior samples use the symmetric `p[i+1] - p[i-1]`; the endpoints
/// fall back to the one-sided forward/backward difference. Each result
/// is normalized (zero for a degenerate coincident pair). A lone point
/// has no defined tangent and yields a single zero vector.
#[must_use]
pub(crate) fn central_difference_tangents(spline: &[Vec3]) -> Vec<Vec3> {
    let n = spline.len();
    if n <= 1 {
        return vec![Vec3::ZERO; n];
    }
    (0..n)
        .map(|i| {
            if i == 0 {
                (spline[1] - spline[0]).normalize_or_zero()
            } else if i == n - 1 {
                (spline[i] - spline[i - 1]).normalize_or_zero()
            } else {
                (spline[i + 1] - spline[i - 1]).normalize_or_zero()
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_central_difference_empty() {
        let tangents = central_difference_tangents(&[]);
        assert!(tangents.is_empty());
    }

    #[test]
    fn test_central_difference_single_point() {
        // Regression: a lone point used to index spline[1] and panic.
        let tangents = central_difference_tangents(&[Vec3::new(1.0, 2.0, 3.0)]);
        assert_eq!(tangents, vec![Vec3::ZERO]);
    }

    #[test]
    fn test_central_difference_colinear_unit_tangents() {
        // Three points along +X: every tangent should be the +X unit vector.
        let tangents = central_difference_tangents(&[
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(2.0, 0.0, 0.0),
        ]);
        assert_eq!(tangents.len(), 3);
        for t in tangents {
            assert!((t - Vec3::X).length() < 1e-6, "expected +X, got {t}");
        }
    }
}
