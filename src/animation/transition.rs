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
/// [`snap()`](Self::snap), [`smooth()`](Self::smooth),
/// [`collapse_ease_expand()`](Self::collapse_ease_expand),
/// or [`cascade()`](Self::cascade).
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
    /// Whether this transition animates across an atom-count (topology)
    /// change by sequencing collapse -> backbone-ease -> expand. When set,
    /// the engine defers adopting the new snapshot and drives the three
    /// segments itself (see the morph sequencer) rather than running a
    /// single in-place interpolation. Only [`collapse_ease_expand`] sets it.
    ///
    /// [`collapse_ease_expand`]: Self::collapse_ease_expand
    pub morphs_topology: bool,
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
            morphs_topology: false,
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
            morphs_topology: false,
        }
    }

    /// Animate across an atom-count (topology) change in three segments:
    /// collapse the changed residues' old sidechains into their stub, ease
    /// the backbone A->B, then expand the changed residues' new sidechains
    /// out of the stub. Used for mutations.
    ///
    /// This sets [`morphs_topology`](Self::morphs_topology): the engine
    /// drives the three segments itself (deferring snapshot adoption to the
    /// collapsed waypoint, where the atom-count change is invisible) rather
    /// than running one in-place interpolation. `ease` is zero for a
    /// backbone-fixed mutation, reducing the sequence to collapse + expand.
    /// The phases below mirror the three segment durations so
    /// [`total_duration`](Self::total_duration) and the non-sequenced
    /// fallback both stay sensible.
    #[must_use]
    pub fn collapse_ease_expand(
        collapse: Duration,
        ease: Duration,
        expand: Duration,
    ) -> Self {
        let total_secs = (collapse + ease + expand).as_secs_f32();
        let (c_end, e_end) = if total_secs == 0.0 {
            (1.0 / 3.0, 2.0 / 3.0)
        } else {
            let c = collapse.as_secs_f32() / total_secs;
            (c, c + ease.as_secs_f32() / total_secs)
        };
        Self {
            phases: vec![
                AnimationPhase {
                    easing: EasingFunction::QuadraticIn,
                    duration: collapse,
                    lerp_start: 0.0,
                    lerp_end: c_end,
                    include_sidechains: true,
                },
                AnimationPhase {
                    easing: EasingFunction::DEFAULT,
                    duration: ease,
                    lerp_start: c_end,
                    lerp_end: e_end,
                    include_sidechains: true,
                },
                AnimationPhase {
                    easing: EasingFunction::QuadraticOut,
                    duration: expand,
                    lerp_start: e_end,
                    lerp_end: 1.0,
                    include_sidechains: true,
                },
            ],
            name: "collapse-ease-expand",
            allows_size_change: true,
            suppress_initial_sidechains: false,
            morphs_topology: true,
        }
    }

    /// The three segment durations `(collapse, ease, expand)` of a
    /// [`collapse_ease_expand`](Self::collapse_ease_expand) transition, or
    /// `None` for any other transition. The morph sequencer reads these to
    /// time its collapse / backbone-ease / expand segments.
    #[must_use]
    pub fn morph_durations(&self) -> Option<(Duration, Duration, Duration)> {
        if !self.morphs_topology || self.phases.len() != 3 {
            return None;
        }
        Some((
            self.phases[0].duration,
            self.phases[1].duration,
            self.phases[2].duration,
        ))
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
            morphs_topology: false,
        }
    }

    /// Single-phase eased interpolation over the full lerp range. The
    /// morph sequencer drives each of its segments with one of these,
    /// timed by the segment's own duration. Equal-length buffers only
    /// (`allows_size_change` is false); not a topology morph.
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
            morphs_topology: false,
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
            morphs_topology: false,
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
            .field("morphs_topology", &self.morphs_topology)
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

    #[test]
    fn test_collapse_ease_expand_phases() {
        let t = Transition::collapse_ease_expand(
            Duration::from_millis(200),
            Duration::from_millis(200),
            Duration::from_millis(100),
        );
        assert_eq!(t.phases.len(), 3);
        assert!(t.morphs_topology);
        assert!(t.allows_size_change);
        assert_eq!(t.total_duration(), Duration::from_millis(500));
        // Collapse covers [0, 0.4), ease [0.4, 0.8), expand [0.8, 1.0].
        assert!((t.phases[0].lerp_end - 0.4).abs() < 0.01);
        assert!((t.phases[1].lerp_start - 0.4).abs() < 0.01);
        assert!((t.phases[2].lerp_start - 0.8).abs() < 0.01);
        assert_eq!(
            t.morph_durations(),
            Some((
                Duration::from_millis(200),
                Duration::from_millis(200),
                Duration::from_millis(100),
            ))
        );
    }

    #[test]
    fn test_morph_durations_none_for_non_morph() {
        assert!(Transition::smooth().morph_durations().is_none());
        assert!(Transition::snap().morph_durations().is_none());
        assert!(!Transition::smooth().morphs_topology);
    }

    #[test]
    fn test_collapse_ease_expand_zero_ease() {
        // Backbone-fixed mutation: ease = 0 reduces to collapse + expand.
        let t = Transition::collapse_ease_expand(
            Duration::from_millis(150),
            Duration::ZERO,
            Duration::from_millis(150),
        );
        assert_eq!(t.total_duration(), Duration::from_millis(300));
        assert_eq!(t.phases[1].duration, Duration::ZERO);
        assert!((t.phases[0].lerp_end - 0.5).abs() < 0.01);
        assert!((t.phases[2].lerp_start - 0.5).abs() < 0.01);
    }
}
