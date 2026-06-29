# Handling Input

Viso does not own an input event loop or a command vocabulary. The host
owns its windowing layer, decodes platform events, and calls typed
methods on `VisoEngine`. There is no `InputProcessor`, no `VisoCommand`,
and no `engine.execute(...)`.

Two surfaces handle input:

- **Pointer and scroll**: `feed_pointer_motion`, `feed_pointer_button`,
  `feed_scroll`, and `feed_modifiers` on `VisoEngine`
  (`src/engine/intake.rs`). The button feed returns a classified
  `ClickEvent` on release, which the host turns into a selection change.
- **Keyboard**: a `KeyBindings` table mapping physical-key strings to
  engine methods, dispatched with `KeyBindings::dispatch(key, engine)`
  (`src/input/key_bindings.rs`).

## Pointer Intake

```rust
impl VisoEngine {
    pub fn feed_pointer_motion(&mut self, x: f32, y: f32);
    pub fn feed_pointer_button(
        &mut self,
        button: MouseButton,
        pressed: bool,
    ) -> Option<ClickEvent>;
    pub fn feed_scroll(&mut self, delta: f32);
    pub fn feed_modifiers(&mut self, shift: bool);
}
```

`feed_pointer_motion` updates the cursor position used by GPU picking.
While the primary button is held over a non-pickable area, it also drives
the camera: shift held pans, otherwise it rotates.

`feed_pointer_button` records the target under the cursor on press and
runs the multi-click classifier on release. It returns `None` on press,
on non-left buttons, and on releases that classify as the end of a drag.
It returns `Some(ClickEvent)` when a release resolves to a single,
double, triple, or empty-area click.

`feed_scroll` zooms (positive in, negative out). `feed_modifiers`
records the shift state used by both the drag branch and click
classification.

### Wiring (winit example)

```rust
WindowEvent::CursorMoved { position, .. } => {
    engine.feed_pointer_motion(position.x as f32, position.y as f32);
}

WindowEvent::MouseWheel { delta, .. } => {
    let scroll = match delta {
        MouseScrollDelta::LineDelta(_, y) => y,
        MouseScrollDelta::PixelDelta(p) => p.y as f32 * 0.01,
    };
    engine.feed_scroll(scroll);
}

WindowEvent::ModifiersChanged(mods) => {
    engine.feed_modifiers(mods.state().shift_key());
}

WindowEvent::MouseInput { button, state, .. } => {
    let pressed = state == ElementState::Pressed;
    if let Some(click) = engine.feed_pointer_button(button.into(), pressed) {
        apply_click(&mut engine, &click);
    }
}
```

## ClickEvent

`feed_pointer_button` returns a `ClickEvent` (`src/input/click.rs`):

```rust
pub struct ClickEvent {
    pub pattern: ClickPattern,           // Single | Double | Triple | Empty
    pub target: PickTarget,              // hit under the cursor at release
    pub modifiers: Modifiers,            // { shift: bool }
    pub expansion: Vec<(EntityId, u32)>, // residues this click selects
}
```

The engine computes `expansion` for you against the current scene: a
single click expands to the clicked residue, a double click to its
secondary-structure segment, a triple click to its chain. Each entry is
`(entity, entity_local_residue_index)`, the same shape a host-side
selection store holds. `Empty` clicks (background) carry an empty
expansion.

## Applying a Click to Selection

The engine no longer mutates selection in response to clicks; it reports
what happened and lets the host decide. `classify_click_for_selection`
maps a `ClickEvent` to an abstract action without consulting the current
selection:

```rust
use viso::{classify_click_for_selection, ClickSelectionAction};

fn apply_click(engine: &mut VisoEngine, click: &ClickEvent) {
    match classify_click_for_selection(click) {
        ClickSelectionAction::Clear => store.clear(),
        ClickSelectionAction::Replace(residues) => store.replace(residues),
        ClickSelectionAction::Toggle(residues) => store.toggle(residues),
    }
    engine.set_selection(&store.as_btreemap());
}
```

`Empty` clears. Shift-held clicks toggle the expansion against the
store; plain clicks replace it. After mutating its own store the host
pushes the new selection to the engine with
`engine.set_selection(&BTreeMap<EntityId, BTreeSet<u32>>)`, which the
engine flattens into its GPU residue space. See
[GPU Picking and Selection](../deep-dives/picking-and-selection.md) for
how the selection reaches the shaders.

## Keyboard Input

Keyboard handling goes through `KeyBindings`, a table from physical-key
strings to engine actions. Forward the key as winit's `KeyCode` debug
string (or the DOM `KeyboardEvent.code`) and call `dispatch`:

```rust
WindowEvent::KeyboardInput { event, .. } => {
    if event.state == ElementState::Pressed {
        if let PhysicalKey::Code(code) = event.physical_key {
            bindings.dispatch(&format!("{code:?}"), &mut engine);
        }
    }
}
```

`dispatch` looks up the key, runs the bound closure against the engine,
and returns whether anything matched. Each binding calls an engine
method directly; there is no intermediate command type.

`KeyBindings` is a standalone type, not attached to any input processor.
Build the default table with `KeyBindings::default()`, an empty one with
`KeyBindings::empty()`, and add or replace entries with
`insert(key, action)`. The table holds `Box<dyn Fn(&mut VisoEngine)>`
closures, so it is not serde-serializable and cannot be loaded from TOML.

### Default bindings

| Key | Engine method |
|-----|---------------|
| `KeyQ` | `recenter_camera` |
| `KeyT` | `toggle_trajectory` |
| `Tab` | `cycle_focus` (steps through visible focusable entities) |
| `KeyR` | `toggle_auto_rotate` |
| `Backquote` | `reset_focus` (back to all-entities) |
| `Escape` | `clear_selection` |
| `KeyI` | `toggle_type_visibility(Ion)` |
| `KeyU` | `toggle_type_visibility(Water)` |
| `KeyO` | `toggle_type_visibility(Solvent)` |
| `KeyL` | `cycle_lipid_mode` |

## Driving the Engine Directly

The pointer and keyboard surfaces are conveniences. A host can call the
underlying engine methods itself; for example to recenter from a button
in its own UI:

```rust
engine.recenter_camera();
engine.toggle_auto_rotate();
engine.set_selection(&selection);
```
