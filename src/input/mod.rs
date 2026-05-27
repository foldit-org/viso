//! Input handling: event types, state machines, and the key-binding
//! map that hosts use to dispatch physical-key events to the engine.

/// Platform-agnostic mouse-button identifier.
pub(crate) mod button;
/// Classified click-event side-output of pointer intake.
pub(crate) mod click;
/// Internal click-pattern classifier and drag-detection state.
pub(crate) mod click_state;
/// Platform-agnostic input events.
pub(crate) mod event;
/// Key-binding map (physical key string → engine action).
pub(crate) mod key_bindings;

pub use button::MouseButton;
pub use click::{
    classify_click_for_selection, ClickEvent, ClickPattern,
    ClickSelectionAction, Modifiers,
};
pub use event::InputEvent;
pub use key_bindings::{KeyAction, KeyBindings};
