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
/// is normalized (zero for a degenerate coincident pair).
#[must_use]
pub(crate) fn central_difference_tangents(spline: &[Vec3]) -> Vec<Vec3> {
    let n = spline.len();
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
