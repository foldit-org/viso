//! Easing functions for animation interpolation.
//!
//! Provides various easing curves for smooth visual transitions.
//! All functions are designed for <100ns evaluation time.

/// Easing function variants for animation curves.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) enum EasingFunction {
    /// Linear interpolation (no easing).
    Linear,
    /// Quadratic ease-in (slow start, fast end).
    QuadraticIn,
    /// Quadratic ease-out (fast start, slow end).
    QuadraticOut,
    /// Cubic Hermite interpolation with configurable control points.
    /// Formula: c1·3t(1-t)² + c2·3(1-t)t² + t³
    CubicHermite {
        /// First control point.
        c1: f32,
        /// Second control point.
        c2: f32,
    },
}

impl EasingFunction {
    /// Default easing function: CubicHermite with c1=0.33, c2=1.0 for natural
    /// ease-out feel.
    pub(crate) const DEFAULT: EasingFunction =
        EasingFunction::CubicHermite { c1: 0.33, c2: 1.0 };

    /// Evaluate the easing function at time t.
    ///
    /// Input t is clamped to [0.0, 1.0].
    /// Returns the eased value, also in [0.0, 1.0].
    ///
    /// Target performance: <100ns
    #[inline]
    #[must_use]
    pub(crate) fn evaluate(&self, t: f32) -> f32 {
        // Clamp input to [0, 1]
        let t = t.clamp(0.0, 1.0);

        match self {
            EasingFunction::Linear => t,
            EasingFunction::QuadraticIn => t * t,
            EasingFunction::QuadraticOut => {
                let omt = 1.0 - t;
                1.0 - omt * omt
            }
            EasingFunction::CubicHermite { c1, c2 } => {
                // f(t) = c0(1-t)³ + c1·3t(1-t)² + c2·3(1-t)t² + c3·t³
                // where c0=0.0, c3=1.0
                // Simplified: c1·3t(1-t)² + c2·3(1-t)t² + t³
                let omt = 1.0 - t;
                c1 * 3.0 * t * omt * omt + c2 * 3.0 * omt * t * t + t * t * t
            }
        }
    }
}

impl Default for EasingFunction {
    #[inline]
    fn default() -> Self {
        Self::DEFAULT
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_linear_endpoints() {
        let linear = EasingFunction::Linear;
        assert_eq!(linear.evaluate(0.0), 0.0);
        assert_eq!(linear.evaluate(0.5), 0.5);
        assert_eq!(linear.evaluate(1.0), 1.0);
    }

    #[test]
    fn test_cubic_hermite_endpoints() {
        let hermite = EasingFunction::CubicHermite { c1: 0.33, c2: 1.0 };
        assert_eq!(hermite.evaluate(0.0), 0.0);
        assert!((hermite.evaluate(1.0) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_cubic_hermite_ease_out_shape() {
        // With c1=0.33, c2=1.0, the curve should have ease-out characteristics:
        // early progress (t=0.25) should yield a result > 0.25 (faster early
        // movement)
        let hermite = EasingFunction::CubicHermite { c1: 0.33, c2: 1.0 };
        let result_at_quarter = hermite.evaluate(0.25);
        assert!(
            result_at_quarter > 0.25,
            "Ease-out should have value > 0.25 at t=0.25, got \
             {result_at_quarter}"
        );
    }

    #[test]
    fn test_input_clamping() {
        let linear = EasingFunction::Linear;

        // Test negative input clamps to 0
        assert_eq!(linear.evaluate(-0.5), 0.0);

        // Test input > 1 clamps to 1
        assert_eq!(linear.evaluate(1.5), 1.0);

        // Also test with other easing functions
        let hermite = EasingFunction::CubicHermite { c1: 0.33, c2: 1.0 };
        assert_eq!(hermite.evaluate(-0.5), 0.0);
        assert!((hermite.evaluate(1.5) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_quadratic_in() {
        let quad_in = EasingFunction::QuadraticIn;
        assert_eq!(quad_in.evaluate(0.0), 0.0);
        assert_eq!(quad_in.evaluate(0.5), 0.25); // 0.5² = 0.25
        assert_eq!(quad_in.evaluate(1.0), 1.0);
    }

    #[test]
    fn test_quadratic_out() {
        let quad_out = EasingFunction::QuadraticOut;
        assert_eq!(quad_out.evaluate(0.0), 0.0);
        assert_eq!(quad_out.evaluate(0.5), 0.75); // 1 - (1-0.5)² = 0.75
        assert_eq!(quad_out.evaluate(1.0), 1.0);
    }

    #[test]
    fn test_default_is_cubic_hermite() {
        let default_easing = EasingFunction::default();
        assert_eq!(default_easing, EasingFunction::DEFAULT);
        assert_eq!(
            default_easing,
            EasingFunction::CubicHermite { c1: 0.33, c2: 1.0 }
        );
    }
}
