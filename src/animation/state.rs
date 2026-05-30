//! Grouped animation state: structural animator, trajectory player,
//! and pending per-entity transitions.

use std::collections::HashMap;
use std::path::Path;

use molex::adapters::dcd::dcd_file_to_frames;
use web_time::Instant;

use super::StructureAnimator;
use crate::animation::transition::Transition;
use crate::engine::annotations::EntityAnnotations;
use crate::engine::positions::EntityPositions;
use crate::engine::scene::Scene;
use crate::engine::trajectory::{
    pick_trajectory_target, TrajectoryFrame, TrajectoryPlayer,
};

/// Grouped animation fields.
pub(crate) struct AnimationState {
    /// Per-entity interpolation runner.
    pub(crate) animator: StructureAnimator,
    /// Multi-frame trajectory player, if loaded.
    pub(crate) trajectory_player: Option<TrajectoryPlayer>,
    /// Transitions pending from the last host mutation, keyed on raw
    /// entity id. Consumed by the engine's sync pipeline when the new
    /// snapshot arrives. Private so callers (including `VisoApp`)
    /// cannot poke this map directly — they go through
    /// `queue_entity_transition` / `clear_pending_transitions` on
    /// [`crate::VisoEngine`].
    pending_transitions: HashMap<u32, Transition>,
    /// In-flight topology-morph sequence, when a mutation is mid-animation.
    /// `Some` only between the start of a deferred collapse/ease/expand and
    /// its swap to the new snapshot; dropped on completion. Driven by the
    /// engine via [`super::morph`]; reset for free with this struct on
    /// scene reset.
    pub(crate) morph: Option<super::morph::MorphSequence>,
}

impl AnimationState {
    /// Create a new `AnimationState` with default values.
    pub(crate) fn new() -> Self {
        Self {
            animator: StructureAnimator::new(),
            trajectory_player: None,
            pending_transitions: HashMap::new(),
            morph: None,
        }
    }

    /// The staged transition for entity `id`, but only if it is a topology
    /// morph (the kind the engine drives as a deferred segment sequence).
    /// Read-only; does not consume the staged transition.
    pub(crate) fn morph_transition(&self, id: u32) -> Option<&Transition> {
        self.pending_transitions
            .get(&id)
            .filter(|t| t.morphs_topology)
    }

    /// Stage a transition for entity `id` on the next sync.
    pub(crate) fn queue_transition(&mut self, id: u32, transition: Transition) {
        let _ = self.pending_transitions.insert(id, transition);
    }

    /// Bulk-replace the pending transitions map.
    pub(crate) fn set_pending_transitions(
        &mut self,
        transitions: HashMap<u32, Transition>,
    ) {
        self.pending_transitions = transitions;
    }

    /// Merge `transitions` into the pending map, incoming entries
    /// winning on key collision.
    pub(crate) fn merge_pending_transitions(
        &mut self,
        transitions: HashMap<u32, Transition>,
    ) {
        self.pending_transitions.extend(transitions);
    }

    /// Drop all pending transitions.
    pub(crate) fn clear_pending_transitions(&mut self) {
        self.pending_transitions.clear();
    }

    /// Drain the pending transitions map (used by the sync pipeline
    /// when applying a new Assembly snapshot).
    pub(crate) fn take_pending_transitions(
        &mut self,
    ) -> HashMap<u32, Transition> {
        std::mem::take(&mut self.pending_transitions)
    }

    /// Tick the structural animator. Returns `true` if any positions
    /// were written.
    pub(crate) fn tick(
        &mut self,
        now: Instant,
        positions: &mut EntityPositions,
    ) -> bool {
        self.animator.update(now, positions)
    }

    /// Advance the trajectory player, returning the per-entity frame
    /// update (if any).
    pub(crate) fn advance_trajectory(
        &mut self,
        now: Instant,
    ) -> Option<TrajectoryFrame> {
        let player = self.trajectory_player.as_mut()?;
        player.tick(now)
    }

    /// Create a trajectory player from parsed DCD frames.
    pub(crate) fn load_trajectory(
        &mut self,
        player: TrajectoryPlayer,
        num_frames: usize,
        num_atoms: usize,
    ) {
        let duration_secs = num_frames as f64 / 30.0;
        self.trajectory_player = Some(player);
        log::info!(
            "Trajectory loaded: {num_frames} frames, {num_atoms} atoms, \
             ~{duration_secs:.1}s at 30fps",
        );
    }

    /// Open a DCD trajectory file, bind it to the first visible protein
    /// entity in `scene`, and begin playback. Logs and returns silently
    /// on any failure (file open, no visible protein, atom-count
    /// mismatch).
    pub(crate) fn load_trajectory_from_path(
        &mut self,
        path: &Path,
        scene: &Scene,
        annotations: &EntityAnnotations,
    ) {
        let (header, frames) = match dcd_file_to_frames(path) {
            Ok(r) => r,
            Err(e) => {
                log::error!("Failed to load DCD trajectory: {e}");
                return;
            }
        };

        let Some((entity_id, atom_count, atom_index_map)) =
            pick_trajectory_target(scene, annotations)
        else {
            log::error!("No protein structure loaded — cannot play trajectory");
            return;
        };

        if (header.num_atoms as usize) < atom_count {
            log::error!(
                "DCD atom count ({}) is less than protein atom count ({})",
                header.num_atoms,
                atom_count,
            );
            return;
        }

        let num_atoms = header.num_atoms as usize;
        let num_frames = frames.len();
        let player =
            TrajectoryPlayer::new(frames, num_atoms, entity_id, atom_index_map);
        self.load_trajectory(player, num_frames, num_atoms);
    }

    /// Toggle trajectory playback (play/pause). No-op if no trajectory
    /// loaded.
    pub(crate) fn toggle_trajectory(&mut self) {
        if let Some(ref mut player) = self.trajectory_player {
            player.toggle_playback();
            let state = if player.is_playing() {
                "playing"
            } else {
                "paused"
            };
            log::info!(
                "Trajectory {state} (frame {}/{})",
                player.current_frame(),
                player.total_frames()
            );
        }
    }
}
