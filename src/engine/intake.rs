//! Pointer / scroll / modifier intake and click-expansion helpers on
//! [`VisoEngine`].
//!
//! These are the typed public surface for consumers that route raw
//! pointer events into the engine. The engine reports back a classified
//! [`ClickEvent`](crate::input::ClickEvent) on release; consumers map
//! the event onto their own selection state.

use molex::entity::molecule::id::EntityId;

use super::VisoEngine;
use crate::input::click::{ClickEvent, ClickPattern, Modifiers};
use crate::input::click_state::ClickResult;
use crate::input::MouseButton;
use crate::renderer::picking::PickTarget;

impl VisoEngine {
    /// Feed a pointer-motion event. Updates the cursor position used
    /// by GPU picking, and — when the primary mouse button is held on
    /// a non-pickable area — produces camera rotate / pan side-effects.
    pub fn feed_pointer_motion(&mut self, x: f32, y: f32) {
        self.gpu.cursor_pos = (x, y);
        let (delta_x, delta_y) = self.input_state.handle_mouse_position(x, y);

        if self.mouse_pressed && self.input_state.mouse_down_target.is_none() {
            let delta = glam::Vec2::new(delta_x, delta_y);
            if delta.length_squared() > 1.0 {
                self.input_state.mark_dragging();
            }
            if self.shift_pressed {
                self.camera_controller.pan(delta);
            } else {
                self.camera_controller.rotate(delta);
            }
        }
    }

    /// Feed a pointer-button event. On press, records the target
    /// under the cursor for later drag / click classification. On
    /// release, runs the multi-click classifier and returns a
    /// [`ClickEvent`] when the release resolves to a click (single,
    /// double, triple, or empty-area). Returns `None` on press, on
    /// non-left buttons, and on releases that classify as drag-end
    /// rather than a click.
    ///
    /// On `Double` / `Triple`, the [`ClickEvent::expansion`] is
    /// populated by walking the current secondary-structure /
    /// backbone-chain caches; consumers writing to their own
    /// selection store don't need scene-graph knowledge to do
    /// segment / chain selection.
    pub fn feed_pointer_button(
        &mut self,
        button: MouseButton,
        pressed: bool,
    ) -> Option<ClickEvent> {
        if button != MouseButton::Left {
            return None;
        }

        let hovered = self.gpu.pick.hovered_target;

        if pressed {
            self.input_state.handle_mouse_down(hovered);
            self.mouse_pressed = true;
            return None;
        }

        // Release.
        self.mouse_pressed = false;
        let click = self.input_state.process_mouse_up(hovered);

        let modifiers = Modifiers {
            shift: self.shift_pressed,
        };
        match click {
            ClickResult::NoAction => None,
            ClickResult::SingleClick { target } => {
                let expansion = single_click_expansion(self, target);
                Some(ClickEvent {
                    pattern: ClickPattern::Single,
                    target,
                    modifiers,
                    expansion,
                })
            }
            ClickResult::DoubleClick { target } => {
                let expansion = segment_expansion(self, target);
                Some(ClickEvent {
                    pattern: ClickPattern::Double,
                    target,
                    modifiers,
                    expansion,
                })
            }
            ClickResult::TripleClick { target } => {
                let expansion = chain_expansion(self, target);
                Some(ClickEvent {
                    pattern: ClickPattern::Triple,
                    target,
                    modifiers,
                    expansion,
                })
            }
            ClickResult::ClearSelection => Some(ClickEvent {
                pattern: ClickPattern::Empty,
                target: PickTarget::None,
                modifiers,
                expansion: Vec::new(),
            }),
        }
    }

    /// Feed a scroll delta. Positive zooms in; negative zooms out.
    pub fn feed_scroll(&mut self, delta: f32) {
        self.camera_controller.zoom(delta);
    }

    /// Feed a shift-modifier state change.
    pub fn feed_modifiers(&mut self, shift: bool) {
        self.shift_pressed = shift;
    }

    /// Whether the primary mouse button is currently held, per the
    /// engine's pointer intake. Consumers driving an external drag
    /// (e.g. pull/band) read this to decide when their drag opens.
    #[must_use]
    pub fn mouse_pressed(&self) -> bool {
        self.mouse_pressed
    }

    /// Whether the shift modifier is currently held.
    #[must_use]
    pub fn shift_pressed_state(&self) -> bool {
        self.shift_pressed
    }

    /// Release the primary mouse-button state without triggering
    /// click classification. Used by consumers that intercept a
    /// drag (pull/band) and need viso's pointer intake to forget the
    /// in-flight press.
    pub fn release_mouse_state(&mut self) {
        self.mouse_pressed = false;
    }
}

// ── Click-expansion helpers ──

/// Per-entity residue list for a single-click on `target`. Non-residue
/// targets (atom picks, background) produce an empty expansion.
fn single_click_expansion(
    engine: &VisoEngine,
    target: PickTarget,
) -> Vec<(EntityId, u32)> {
    let PickTarget::Residue(flat) = target else {
        return Vec::new();
    };
    engine
        .gpu
        .pick
        .flat_to_entity_residue(flat)
        .into_iter()
        .collect()
}

/// Per-entity residue list for the SS segment under `target`. Empty
/// for non-residue targets and when no offsets are published yet.
fn segment_expansion(
    engine: &VisoEngine,
    target: PickTarget,
) -> Vec<(EntityId, u32)> {
    let flat = target.as_residue_i32();
    if flat < 0 {
        return Vec::new();
    }
    let ss = engine.concatenated_cartoon_ss();
    engine.gpu.pick.residues_in_segment(flat, &ss)
}

/// Per-entity residue list for the chain under `target`. Empty for
/// non-residue targets and when no offsets are published yet.
fn chain_expansion(
    engine: &VisoEngine,
    target: PickTarget,
) -> Vec<(EntityId, u32)> {
    let flat = target.as_residue_i32();
    if flat < 0 {
        return Vec::new();
    }
    let chains = engine.gpu.renderers.backbone.cached_chains();
    engine.gpu.pick.residues_in_chain(flat, chains)
}
