//! Assembly sync + scene → renderer pipeline.
//!
//! [`SyncPipeline`] (in [`pipeline`]) owns the main-thread sync logic:
//! rederiving viso-side state from [`Assembly`] snapshots, building
//! full-rebuild requests for the background mesh processor, consuming
//! processor output, and keeping GPU buffers in sync with
//! [`super::scene::Scene`]. Each associated function takes disjoint
//! borrows of the engine's sub-structs so the pipeline is expressible
//! without routing through `&mut self` on [`VisoEngine`].
//!
//! This module holds [`VisoEngine`]'s thin dispatcher methods — one-line
//! forwards into [`SyncPipeline`]. Neither `&Assembly` nor
//! `&MoleculeEntity` appear in render-path signatures — the sync layer
//! is the only place they're read, and it produces render-ready derived
//! state downstream consumers use.
//!
//! [`Assembly`]: molex::Assembly

mod pipeline;

use std::collections::HashMap;

use molex::{Assembly, SSType};
pub(crate) use pipeline::SyncPipeline;

use super::trajectory::TrajectoryFrame;
use super::VisoEngine;
use crate::animation::transition::Transition;

impl VisoEngine {
    /// Poll the consumer, then submit a full-rebuild request to the
    /// background mesh processor using the current pending transitions.
    pub(crate) fn sync_now(&mut self) {
        SyncPipeline::sync_now(
            &mut self.scene,
            &mut self.annotations,
            &self.options,
            &mut self.gpu,
            &mut self.animation,
        );
        self.resolve_and_upload_bond_connections();
    }

    /// Sync scene data to renderers with per-entity transitions.
    ///
    /// Entities in the map animate with their transition; entities
    /// not in the map snap. Pass an empty map for a non-animated sync.
    pub fn sync_scene_to_renderers(
        &mut self,
        entity_transitions: HashMap<u32, Transition>,
    ) {
        SyncPipeline::submit_full_rebuild(
            &mut self.scene,
            &self.annotations,
            &self.options,
            &mut self.gpu,
            &mut self.animation,
            entity_transitions,
        );
        self.resolve_and_upload_bond_connections();
    }

    /// Apply any pending scene data from the background `SceneProcessor`.
    pub fn apply_pending_scene(&mut self) {
        let applied = SyncPipeline::apply_pending_scene(
            &mut self.scene,
            &self.annotations,
            &self.options,
            &mut self.gpu,
            &mut self.animation,
        );
        // A consumed mesh just finalized positions and lifted the sheet
        // offsets onto each EntityView; re-resolve the bond capsules so a
        // sheet-residue SG / backbone hbond lands on the drawn geometry.
        if applied {
            self.resolve_and_upload_bond_connections();
        }
    }

    /// Apply any pending animation frame from the background thread.
    pub(crate) fn apply_pending_animation(&mut self) {
        let _ = self.gpu.apply_pending_animation();
    }

    /// Submit an animation frame to the background thread using the
    /// engine's current interpolated positions.
    pub(crate) fn submit_animation_frame(&self) {
        SyncPipeline::submit_animation_frame(
            &self.scene,
            &self.options,
            &self.gpu,
            &self.animation,
        );
    }

    /// Apply a trajectory frame's atom-index updates to
    /// [`super::positions::EntityPositions`].
    pub(crate) fn apply_trajectory_frame(&mut self, frame: &TrajectoryFrame) {
        SyncPipeline::apply_trajectory_frame(&mut self.scene, frame);
    }

    /// Rederive viso-side state from an `Assembly` snapshot.
    pub(crate) fn sync_from_assembly(&mut self, assembly: &Assembly) {
        SyncPipeline::sync_from_assembly(
            &mut self.scene,
            &mut self.annotations,
            &self.options,
            assembly,
        );
    }

    /// Concatenated SS across all Cartoon protein entities, in
    /// assembly order. Used by the `SelectSegment` command path.
    pub(crate) fn concatenated_cartoon_ss(&self) -> Vec<SSType> {
        SyncPipeline::concatenated_cartoon_ss(&self.scene, &self.annotations)
    }

    /// Re-run the full-rebuild path after a display/color option
    /// change that affects ball-and-stick rendering.
    pub(crate) fn refresh_ball_and_stick(&mut self) {
        self.sync_scene_to_renderers(HashMap::new());
    }

    /// Recompute per-chain backbone colors and upload them immediately.
    /// Used by display-option changes that affect backbone tint but
    /// don't invalidate mesh geometry.
    pub(crate) fn recompute_backbone_colors(&mut self) {
        SyncPipeline::recompute_backbone_colors(
            &mut self.scene,
            &self.annotations,
            &self.options,
            &mut self.gpu,
        );
    }
}
