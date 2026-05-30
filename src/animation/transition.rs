//! Transition describes how to animate from the current state to a new target.

use std::time::Duration;

use crate::util::easing::EasingFunction;

/// A single phase of an animation sequence.
///
/// Each phase has its own easing, duration, and lerp range. The runner
/// evaluates phases sequentially — when one phase's duration expires,
/// the next begins.
#[derive(Debug, Clone)]
pub(crate) struct AnimationPhase {
    /// Easing curve for this phase.
    pub(crate) easing: EasingFunction,
    /// Duration of this phase.
    pub(crate) duration: Duration,
    /// Start of the global lerp range (0.0–1.0) this phase covers.
    pub(crate) lerp_start: f32,
    /// End of the global lerp range (0.0–1.0) this phase covers.
    pub(crate) lerp_end: f32,
    /// Whether sidechains should be visible during this phase.
    pub(crate) include_sidechains: bool,
}

/// Describes how to animate from current state to a new target.
///
/// Consumers construct transitions via preset constructors:
/// [`snap()`](Self::snap), [`smooth()`](Self::smooth), or
/// [`cascade()`](Self::cascade).
#[derive(Clone)]
pub struct Transition {
    /// Animation phases (single or multi-phase).
    pub(crate) phases: Vec<AnimationPhase>,
    /// Debug name.
    pub(crate) name: &'static str,
    /// Whether the animator should allow backbone size changes.
    /// When false, size mismatches cause an instant snap.
    pub allows_size_change: bool,
    /// Whether to suppress initial sidechain GPU uploads.
    /// Used by multi-phase behaviors that hide sidechains in phase 1.
    pub suppress_initial_sidechains: bool,
}

impl Transition {
    /// Total duration across all phases.
    #[must_use]
    pub fn total_duration(&self) -> Duration {
        self.phases.iter().map(|p| p.duration).sum()
    }

    /// Instant snap with no animation. Allows size changes.
    #[must_use]
    pub fn snap() -> Self {
        Self {
            phases: vec![AnimationPhase {
                easing: EasingFunction::Linear,
                duration: Duration::ZERO,
                lerp_start: 0.0,
                lerp_end: 1.0,
                include_sidechains: true,
            }],

            name: "snap",
            allows_size_change: true,
            suppress_initial_sidechains: false,
        }
    }

    /// Standard smooth interpolation (300ms, cubic hermite ease-out).
    #[must_use]
    pub fn smooth() -> Self {
        Self {
            phases: vec![AnimationPhase {
                easing: EasingFunction::DEFAULT,
                duration: Duration::from_millis(300),
                lerp_start: 0.0,
                lerp_end: 1.0,
                include_sidechains: true,
            }],

            name: "smooth",
            allows_size_change: false,
            suppress_initial_sidechains: false,
        }
    }

    /// Staggered per-residue delays for wave-like effects.
    ///
    /// Staggered per-residue delays for wave-like effects.
    ///
    /// Per-residue staggering is not yet integrated into the runner.
    #[must_use]
    pub fn cascade(base: Duration, _delay_per_residue: Duration) -> Self {
        Self {
            phases: vec![AnimationPhase {
                easing: EasingFunction::QuadraticOut,
                duration: base,
                lerp_start: 0.0,
                lerp_end: 1.0,
                include_sidechains: true,
            }],
            name: "cascade",
            allows_size_change: false,
            suppress_initial_sidechains: false,
        }
    }

    /// Single-phase eased interpolation over the full lerp range. The
    /// animation player drives each of its steps with one of these, timed
    /// by the step's own duration. Equal-length buffers only
    /// (`allows_size_change` is false).
    #[must_use]
    pub(crate) fn eased(
        duration: Duration,
        easing: EasingFunction,
        include_sidechains: bool,
    ) -> Self {
        Self {
            phases: vec![AnimationPhase {
                easing,
                duration,
                lerp_start: 0.0,
                lerp_end: 1.0,
                include_sidechains,
            }],
            name: "eased",
            allows_size_change: false,
            suppress_initial_sidechains: false,
        }
    }

    /// Allow backbone size changes during animation.
    #[must_use]
    pub fn allowing_size_change(mut self) -> Self {
        self.allows_size_change = true;
        self
    }

    /// Suppress initial sidechain GPU uploads.
    #[must_use]
    pub fn suppressing_initial_sidechains(mut self) -> Self {
        self.suppress_initial_sidechains = true;
        self
    }

    /// Linear easing for testing.
    #[cfg(test)]
    pub(crate) fn linear(duration: Duration) -> Self {
        Self {
            phases: vec![AnimationPhase {
                easing: EasingFunction::Linear,
                duration,
                lerp_start: 0.0,
                lerp_end: 1.0,
                include_sidechains: true,
            }],

            name: "linear",
            allows_size_change: false,
            suppress_initial_sidechains: false,
        }
    }
}

impl Default for Transition {
    fn default() -> Self {
        Self::smooth()
    }
}

impl std::fmt::Debug for Transition {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Transition")
            .field("name", &self.name)
            .field("phases", &self.phases.len())
            .field("allows_size_change", &self.allows_size_change)
            .field(
                "suppress_initial_sidechains",
                &self.suppress_initial_sidechains,
            )
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_snap_transition() {
        let t = Transition::snap();
        assert_eq!(t.name, "snap");
        assert!(t.allows_size_change);
        assert!(!t.suppress_initial_sidechains);
        assert_eq!(t.total_duration(), Duration::ZERO);
    }

    #[test]
    fn test_smooth_transition() {
        let t = Transition::smooth();
        assert_eq!(t.name, "smooth");
        assert!(!t.allows_size_change);
        assert!(!t.suppress_initial_sidechains);
        assert_eq!(t.total_duration(), Duration::from_millis(300));
    }

    #[test]
    fn test_default_is_smooth() {
        let t = Transition::default();
        assert_eq!(t.name, "smooth");
    }

    #[test]
    fn test_builder_methods() {
        let t = Transition::smooth()
            .allowing_size_change()
            .suppressing_initial_sidechains();
        assert!(t.allows_size_change);
        assert!(t.suppress_initial_sidechains);
    }
}
