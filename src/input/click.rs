//! Click-event side-output produced by the engine's pointer intake.
//!
//! The mouse-button intake on [`VisoEngine`](crate::VisoEngine) returns
//! a `ClickEvent` on release when the multi-click classifier resolves
//! the gesture into a click. Consumers map [`ClickPattern`] +
//! [`PickTarget`] onto their own selection policy.

use molex::entity::molecule::id::EntityId;

use crate::renderer::picking::PickTarget;

/// A classified click gesture produced by the engine on mouse release.
///
/// Consumers consume `ClickEvent` and decide what (if anything) it
/// should do to their own selection / focus / view state. The engine
/// itself no longer mutates selection in response to clicks; it just
/// reports what happened.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ClickEvent {
    /// Multi-click multiplicity (single / double / triple) or an
    /// empty-area click.
    pub pattern: ClickPattern,
    /// The pick target under the cursor at release time. For
    /// [`ClickPattern::Empty`] this is [`PickTarget::None`].
    pub target: PickTarget,
    /// Modifier keys held at release time.
    pub modifiers: Modifiers,
    /// Residues that this click pattern selects, computed by the
    /// engine against the current scene. Empty for
    /// [`ClickPattern::Empty`] and for clicks that resolve to a
    /// target with no entity owner (atom picks, non-protein hits).
    /// Per-entity grouping mirrors the consumer-side selection store
    /// shape: `(entity, residue_in_entity)` pairs, where
    /// `residue_in_entity` is the entity-local 0-based residue index.
    pub expansion: Vec<(EntityId, u32)>,
}

/// Click multiplicity, as classified by the engine's multi-click
/// state machine.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ClickPattern {
    /// A single click on a pickable target.
    Single,
    /// A second click within the multi-click window on the same target.
    Double,
    /// A third click within the multi-click window on the same target.
    Triple,
    /// Click on non-pickable area; consumers typically interpret as
    /// clear.
    Empty,
}

/// Modifier-key state captured alongside a [`ClickEvent`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Modifiers {
    /// Whether the shift key was held at click time.
    pub shift: bool,
}

/// What a click should do to a selection store, classified from a
/// [`ClickEvent`] without reference to the store's current contents.
///
/// Consumers apply the action to their own per-store API:
/// [`SelectionStore`](crate::app::SelectionStore) flushes into the
/// engine's GPU residue space; foldit-core's `App` routes through
/// `select_residue` / `toggle_residue` / `clear_selection`.
#[derive(Debug, Clone)]
pub enum ClickSelectionAction {
    /// Clear all selection.
    Clear,
    /// Clear and replace the selection with these residues.
    Replace(Vec<(EntityId, u32)>),
    /// Toggle these residues against the current selection.
    Toggle(Vec<(EntityId, u32)>),
}

/// Classify a [`ClickEvent`] into the abstract selection action it
/// requests. Pure: depends only on the event, not on any current
/// selection state.
///
/// Empty-pattern clicks clear. Shift-held clicks toggle the
/// expansion; plain clicks replace the selection with the expansion.
/// An empty expansion on a non-Empty pattern flows through as an
/// empty `Toggle` / `Replace`, which is a no-op / clear respectively
/// on the consumer side; the classifier itself does not collapse
/// these into [`ClickSelectionAction::Clear`].
#[must_use]
pub fn classify_click_for_selection(
    click: &ClickEvent,
) -> ClickSelectionAction {
    if matches!(click.pattern, ClickPattern::Empty) {
        return ClickSelectionAction::Clear;
    }
    if click.modifiers.shift {
        ClickSelectionAction::Toggle(click.expansion.clone())
    } else {
        ClickSelectionAction::Replace(click.expansion.clone())
    }
}
