//! Animation runner: tracks phase-based progress for a single entity.

use std::time::Duration;

use web_time::Instant;

use super::transition::{AnimationPhase, Transition};

/// Executes phase-based progress for a single entity's animation.
///
/// The runner tracks start time + phases and computes `eased_t` on
/// demand. Per-atom interpolation is done by the caller using the
/// eased value; the runner stays purely about *when*, not *what*.
pub(crate) struct AnimationRunner {
    start_time: Instant,
    phases: Vec<AnimationPhase>,
    total_duration: Duration,
}

impl AnimationRunner {
    /// Start a new animation from the given transition.
    pub(crate) fn new(transition: &Transition) -> Self {
        Self {
            start_time: Instant::now(),
            phases: transition.phases.clone(),
            total_duration: transition.total_duration(),
        }
    }

    /// Create with explicit start time (for testing).
    #[cfg(test)]
    pub(crate) fn with_start_time(
        start_time: Instant,
        transition: &Transition,
    ) -> Self {
        Self {
            start_time,
            phases: transition.phases.clone(),
            total_duration: transition.total_duration(),
        }
    }

    /// Normalized progress 0.0..=1.0.
    pub(crate) fn progress(&self, now: Instant) -> f32 {
        let elapsed = now.saturating_duration_since(self.start_time);
        if self.total_duration.is_zero() {
            1.0
        } else {
            (elapsed.as_secs_f32() / self.total_duration.as_secs_f32()).min(1.0)
        }
    }

    /// Whether the animation has reached completion.
    #[cfg(test)]
    pub(crate) fn is_complete(&self, now: Instant) -> bool {
        self.progress(now) >= 1.0
    }

    /// Eased interpolation value for the given raw progress, respecting
    /// the phase sequence.
    pub(crate) fn eased_t(&self, raw_t: f32) -> f32 {
        if raw_t >= 1.0 {
            return 1.0;
        }
        match self.current_phase(raw_t) {
            Some((phase, local_t)) => {
                let local_eased = phase.easing.evaluate(local_t);
                phase.lerp_start
                    + local_eased * (phase.lerp_end - phase.lerp_start)
            }
            None => 1.0,
        }
    }

    /// Whether sidechains should be drawn at the given raw progress.
    ///
    /// Multi-phase behaviors hide sidechains during the backbone-lerp
    /// phase so new atoms don't flash at their final positions.
    pub(crate) fn should_include_sidechains(&self, raw_t: f32) -> bool {
        if raw_t >= 1.0 {
            return true;
        }
        self.current_phase(raw_t)
            .is_none_or(|(phase, _)| phase.include_sidechains)
    }

    fn current_phase(&self, raw_t: f32) -> Option<(&AnimationPhase, f32)> {
        if self.phases.is_empty() {
            return None;
        }

        let total_secs = self.total_duration.as_secs_f32();
        let mut cumulative = 0.0;

        for (i, phase) in self.phases.iter().enumerate() {
            let phase_frac = if total_secs == 0.0 {
                1.0 / self.phases.len() as f32
            } else {
                phase.duration.as_secs_f32() / total_secs
            };

            let is_last = i == self.phases.len() - 1;

            if raw_t < cumulative + phase_frac || is_last {
                let local_t = if phase_frac > 0.0 {
                    ((raw_t - cumulative) / phase_frac).clamp(0.0, 1.0)
                } else {
                    1.0
                };
                return Some((phase, local_t));
            }

            cumulative += phase_frac;
        }

        self.phases.last().map(|p| (p, 1.0f32))
    }
}

impl std::fmt::Debug for AnimationRunner {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AnimationRunner")
            .field("duration", &self.total_duration)
            .field("phases", &self.phases.len())
            .finish_non_exhaustive()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn runner_progress() {
        let transition = Transition::linear(Duration::from_millis(100));
        let start = Instant::now();
        let runner = AnimationRunner::with_start_time(start, &transition);

        assert!((runner.progress(start) - 0.0).abs() < 0.01);
        let mid = start + Duration::from_millis(50);
        assert!((runner.progress(mid) - 0.5).abs() < 0.01);
        let end = start + Duration::from_millis(100);
        assert!((runner.progress(end) - 1.0).abs() < 0.01);
        let past = start + Duration::from_millis(200);
        assert!((runner.progress(past) - 1.0).abs() < 0.01);
    }

    #[test]
    fn runner_eased_t_linear() {
        let transition = Transition::linear(Duration::from_millis(100));
        let start = Instant::now();
        let runner = AnimationRunner::with_start_time(start, &transition);

        assert!((runner.eased_t(0.0) - 0.0).abs() < 0.01);
        assert!((runner.eased_t(0.5) - 0.5).abs() < 0.01);
        assert!((runner.eased_t(1.0) - 1.0).abs() < 0.01);
    }

    #[test]
    fn runner_snap() {
        let transition = Transition::snap();
        let runner = AnimationRunner::new(&transition);

        assert!(runner.is_complete(Instant::now()));
    }

    #[test]
    fn should_include_sidechains() {
        use crate::util::easing::EasingFunction;

        // A transition whose first phase hides sidechains and whose second
        // shows them, exercising the runner's per-phase visibility gate.
        let transition = Transition {
            phases: vec![
                AnimationPhase {
                    easing: EasingFunction::QuadraticOut,
                    duration: Duration::from_millis(300),
                    lerp_start: 0.0,
                    lerp_end: 1.0,
                    include_sidechains: false,
                },
                AnimationPhase {
                    easing: EasingFunction::Linear,
                    duration: Duration::from_millis(200),
                    lerp_start: 1.0,
                    lerp_end: 1.0,
                    include_sidechains: true,
                },
            ],
            name: "test-hidden-then-shown",
            allows_size_change: false,
        };
        let runner = AnimationRunner::new(&transition);

        assert!(!runner.should_include_sidechains(0.0));
        assert!(!runner.should_include_sidechains(0.3));
        assert!(runner.should_include_sidechains(0.7));
        assert!(runner.should_include_sidechains(1.0));
    }
}
