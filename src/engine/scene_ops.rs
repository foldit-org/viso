//! Scene operations callable directly on [`VisoEngine`].
//!
//! These are the typed public surface for consumers that drive
//! scene-level state (camera, focus, visibility, trajectory, lipid
//! display) without going through the input intake methods.

use std::collections::HashMap;

use molex::entity::molecule::id::EntityId;
use molex::MoleculeType;

use super::focus::Focus;
use super::VisoEngine;

impl VisoEngine {
    /// Animate the camera to fit the currently focused element.
    pub fn recenter_camera(&mut self) {
        self.fit_camera_to_focus();
    }

    /// Toggle turntable auto-rotation.
    pub fn toggle_auto_rotate(&mut self) {
        let _ = self.camera_controller.toggle_auto_rotate();
    }

    /// Force turntable auto-rotation on or off.
    pub fn set_auto_rotate(&mut self, on: bool) {
        if self.camera_controller.is_auto_rotating() != on {
            let _ = self.camera_controller.toggle_auto_rotate();
        }
    }

    /// Cycle focus: Session → Entity₁ → … → EntityN → Session, and
    /// refit the camera.
    pub fn cycle_focus(&mut self) {
        let _ = self.annotations_mut().cycle_focus();
        self.fit_camera_to_focus();
    }

    /// Reset focus to the all-entities view and refit the camera.
    pub fn reset_focus(&mut self) {
        self.annotations.focus = Focus::All;
        self.fit_camera_to_focus();
    }

    /// Focus a specific entity and refit the camera to it.
    pub fn focus_entity(&mut self, id: EntityId) {
        self.annotations.focus = Focus::Entity(id);
        self.fit_camera_to_focus();
    }

    /// Toggle visibility of a specific entity.
    pub fn toggle_entity_visibility(&mut self, id: EntityId) {
        let currently_visible = self.is_entity_visible(id.raw());
        self.set_entity_visible(id.raw(), !currently_visible);
        self.sync_scene_to_renderers(HashMap::new());
    }

    /// Clear the current residue selection.
    pub fn clear_selection(&mut self) {
        let _ = self.gpu.pick.clear_selection();
    }

    /// Toggle trajectory playback (play / pause).
    pub fn toggle_trajectory(&mut self) {
        self.animation.toggle_trajectory();
    }

    /// Cycle lipid display mode (coarse ↔ ball-and-stick).
    pub fn cycle_lipid_mode(&mut self) {
        self.options.display.overrides.lipid_mode =
            Some(if self.options.display.lipid_ball_and_stick() {
                crate::options::LipidMode::Coarse
            } else {
                crate::options::LipidMode::BallAndStick
            });
        self.refresh_ball_and_stick();
    }

    /// Toggle per-type visibility (Ion / Water / Solvent). Updates the
    /// matching `options.display.show_*` flag, broadcasts the new
    /// value to every entity of `mol_type`, and resyncs the scene.
    /// Unknown molecule types are accepted but no-op.
    pub fn toggle_type_visibility(&mut self, mol_type: MoleculeType) {
        let flag: Option<&mut bool> = match mol_type {
            MoleculeType::Ion => Some(&mut self.options.display.show_ions),
            MoleculeType::Water => Some(&mut self.options.display.show_waters),
            MoleculeType::Solvent => {
                Some(&mut self.options.display.show_solvent)
            }
            _ => None,
        };
        let Some(flag) = flag else {
            return;
        };
        let next = !*flag;
        *flag = next;
        self.apply_type_visibility(mol_type, next);
        self.sync_scene_to_renderers(HashMap::new());
    }

    /// Set per-type visibility (Ion / Water / Solvent) to `visible`.
    /// Updates the matching `options.display.show_*` flag, broadcasts
    /// the new value to every entity of `mol_type`, and resyncs the
    /// scene. Unknown molecule types are accepted but no-op.
    pub fn set_type_visibility(
        &mut self,
        mol_type: MoleculeType,
        visible: bool,
    ) {
        let flag: Option<&mut bool> = match mol_type {
            MoleculeType::Ion => Some(&mut self.options.display.show_ions),
            MoleculeType::Water => Some(&mut self.options.display.show_waters),
            MoleculeType::Solvent => {
                Some(&mut self.options.display.show_solvent)
            }
            _ => None,
        };
        let Some(flag) = flag else {
            return;
        };
        *flag = visible;
        self.apply_type_visibility(mol_type, visible);
        self.sync_scene_to_renderers(HashMap::new());
    }
}
