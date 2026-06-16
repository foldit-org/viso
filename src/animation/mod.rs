//! Animation system for smooth structural transitions.

pub(crate) mod animator;
pub(crate) mod runner;
pub(crate) mod sequence;
pub(crate) mod state;
pub(crate) mod transition;

pub(crate) use animator::StructureAnimator;
pub(crate) use sequence::{build_animation, Advance, AnimationPlayer};
pub(crate) use state::AnimationState;
