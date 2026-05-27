use super::button::MouseButton;

/// Platform-agnostic input events.
///
/// Hosts that prefer a single enum over the individual `feed_*` methods
/// on [`VisoEngine`](crate::VisoEngine) can pattern-match on this and
/// forward each variant to the corresponding intake method.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum InputEvent {
    /// Cursor moved to absolute screen position.
    CursorMoved {
        /// Horizontal position in physical pixels.
        x: f32,
        /// Vertical position in physical pixels.
        y: f32,
    },
    /// Mouse button pressed or released.
    MouseButton {
        /// Which button changed.
        button: MouseButton,
        /// `true` for press, `false` for release.
        pressed: bool,
    },
    /// Scroll wheel (positive = zoom in).
    Scroll {
        /// Scroll amount (positive = zoom in, negative = zoom out).
        delta: f32,
    },
    /// Modifier key state changed.
    ModifiersChanged {
        /// Whether the shift key is held.
        shift: bool,
    },
}
