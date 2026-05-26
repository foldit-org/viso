//! Per-entity animator: interpolates `EntityPositions` between a
//! per-entity start snapshot and a target reference.
//!
//! Keyed on [`EntityId`] and writes directly into [`EntityPositions`].
//! There is no flat aggregation — the animator is a pure interpolation
//! layer over the engine's per-entity position buffers.

use glam::Vec3;
use molex::entity::molecule::id::EntityId;
use rustc_hash::FxHashMap;
use web_time::Instant;

use super::runner::AnimationRunner;
use super::transition::Transition;
use crate::engine::positions::EntityPositions;

/// Per-entity animation state: the timing runner plus the start/target
/// position snapshots it interpolates between.
struct EntityAnimationState {
    runner: AnimationRunner,
    start: Vec<Vec3>,
    target: Vec<Vec3>,
}

/// Drives per-entity position interpolation.
pub(crate) struct StructureAnimator {
    runners: FxHashMap<EntityId, EntityAnimationState>,
}

impl StructureAnimator {
    /// Animator with default settings.
    pub(crate) fn new() -> Self {
        Self {
            runners: FxHashMap::default(),
        }
    }

    /// Whether any entity is currently animating.
    pub(crate) fn is_animating(&self) -> bool {
        !self.runners.is_empty()
    }

    /// Start or replace the animation for a single entity.
    ///
    /// `start` is the current visible positions (animator's
    /// interpolation origin); `target` is the final positions the
    /// animation ends at. Both are expected to have matching lengths
    /// (the standalone app publishes only generation bumps where the
    /// atom set is stable per entity; size changes snap without
    /// animation).
    pub(crate) fn animate_entity(
        &mut self,
        entity_id: EntityId,
        mut start: Vec<Vec3>,
        target: Vec<Vec3>,
        transition: &Transition,
    ) {
        let size_mismatch = start.len() != target.len();
        if size_mismatch {
            if !transition.allows_size_change {
                // Plain interpolation transition on different-shaped
                // arrays: caller is expected to have snapped positions;
                // remove any stale runner.
                let _ = self.runners.remove(&entity_id);
                return;
            }
            // Size-change-aware transition (e.g. collapse_expand for
            // mutation animations). Positions are already snapped to
            // target by the caller; install a runner so the
            // transition's phases (sidechain-visibility timing) still
            // play through. Use `target` as both endpoints so per-frame
            // position writes are no-ops.
            start.clone_from(&target);
        } else if start == target {
            let _ = self.runners.remove(&entity_id);
            return;
        }
        let runner = AnimationRunner::new(transition);
        let _ = self.runners.insert(
            entity_id,
            EntityAnimationState {
                runner,
                start,
                target,
            },
        );
    }

    /// Advance animations for the current frame and write interpolated
    /// positions into `positions`. Returns `true` if any entity's
    /// position buffer was written.
    pub(crate) fn update(
        &mut self,
        now: Instant,
        positions: &mut EntityPositions,
    ) -> bool {
        if self.runners.is_empty() {
            return false;
        }
        let mut any_written = false;
        let mut completed = Vec::new();
        for (&eid, state) in &mut self.runners {
            let t = state.runner.progress(now);
            if t >= 1.0 {
                positions.set(eid, state.target.clone());
                completed.push(eid);
                any_written = true;
                continue;
            }
            let eased = state.runner.eased_t(t);
            let lerped: Vec<Vec3> = state
                .start
                .iter()
                .zip(state.target.iter())
                .map(|(s, e)| s.lerp(*e, eased))
                .collect();
            positions.set(eid, lerped);
            any_written = true;
        }
        for eid in completed {
            let _ = self.runners.remove(&eid);
        }
        any_written
    }

    /// Whether sidechains should be drawn this frame. Returns `false`
    /// if any currently-running entity is in a phase that hides
    /// sidechains.
    pub(crate) fn should_include_sidechains(&self) -> bool {
        let now = Instant::now();
        for state in self.runners.values() {
            let t = state.runner.progress(now);
            if !state.runner.should_include_sidechains(t) {
                return false;
            }
        }
        true
    }
}

impl Default for StructureAnimator {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Debug for StructureAnimator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("StructureAnimator")
            .field("runners", &self.runners.len())
            .finish_non_exhaustive()
    }
}
