//! Animation system for smooth structural transitions.

pub(crate) mod animator;
/// Topology-morph sequencer: deferred collapse/ease/expand for mutations.
pub(crate) mod morph;
pub(crate) mod runner;
pub(crate) mod state;
pub(crate) mod transition;

pub(crate) use animator::StructureAnimator;
pub(crate) use state::AnimationState;
