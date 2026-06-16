//! Input handling: event types, state machines, and the key-binding
//! map that hosts use to dispatch physical-key events to the engine.

pub(crate) mod button;
pub(crate) mod click;
pub(crate) mod click_state;
pub(crate) mod event;
pub(crate) mod key_bindings;

pub use button::MouseButton;
pub use click::{
    classify_click_for_selection, ClickEvent, ClickPattern,
    ClickSelectionAction, Modifiers,
};
pub use event::InputEvent;
pub use key_bindings::{KeyAction, KeyBindings};
