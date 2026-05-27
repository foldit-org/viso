//! Key-binding map: physical key strings to engine actions.
//!
//! Hosts forward keyboard events as physical-key strings (winit's
//! `KeyCode` debug format / DOM `KeyboardEvent.code`) and call
//! [`KeyBindings::dispatch`] to run the matching engine method.

use std::collections::BTreeMap;

use molex::MoleculeType;

use crate::engine::VisoEngine;

/// Action invoked when a bound key fires. Receives a mutable reference
/// to the engine so the action can mutate any engine state.
///
/// Boxed so the default table can hold closures that capture per-entry
/// arguments (e.g. the [`MoleculeType`] for an ambient-type toggle)
/// without one tiny adapter `fn` per entry.
pub type KeyAction = Box<dyn Fn(&mut VisoEngine)>;

/// Maps physical key strings (e.g. `"KeyQ"`, `"Tab"`, `"Escape"`) to
/// engine actions.
///
/// The default table covers viso's built-in keybindings (camera
/// recenter, focus cycle, trajectory toggle, ambient-type toggles,
/// lipid mode cycle, escape-to-clear). Hosts can extend the table with
/// [`Self::insert`] for additional custom keys.
pub struct KeyBindings {
    table: BTreeMap<String, KeyAction>,
}

impl KeyBindings {
    /// Empty binding table with no entries.
    #[must_use]
    pub fn empty() -> Self {
        Self {
            table: BTreeMap::new(),
        }
    }

    /// Look up `key` and, if bound, invoke the action against `engine`.
    /// Returns `true` if the key was matched and the action ran.
    pub fn dispatch(&self, key: &str, engine: &mut VisoEngine) -> bool {
        let Some(action) = self.table.get(key) else {
            return false;
        };
        action(engine);
        true
    }

    /// Insert or replace a binding for `key`.
    pub fn insert(&mut self, key: String, action: KeyAction) {
        let _ = self.table.insert(key, action);
    }

    /// Whether `key` has a binding.
    #[must_use]
    pub fn contains(&self, key: &str) -> bool {
        self.table.contains_key(key)
    }
}

impl Default for KeyBindings {
    /// The standard viso keybinding table:
    ///
    /// | Key        | Action                              |
    /// |------------|-------------------------------------|
    /// | `KeyQ`     | Recenter the camera on focus        |
    /// | `KeyT`     | Toggle trajectory playback          |
    /// | `Tab`      | Cycle focus through entities        |
    /// | `KeyR`     | Toggle turntable auto-rotate        |
    /// | `Backquote`| Reset focus to all-entities         |
    /// | `Escape`   | Clear residue selection             |
    /// | `KeyI`     | Toggle ion visibility               |
    /// | `KeyU`     | Toggle water visibility             |
    /// | `KeyO`     | Toggle solvent visibility           |
    /// | `KeyL`     | Cycle lipid display mode            |
    fn default() -> Self {
        let entries: [(&str, KeyAction); 10] = [
            ("KeyQ", Box::new(VisoEngine::recenter_camera)),
            ("KeyT", Box::new(VisoEngine::toggle_trajectory)),
            ("Tab", Box::new(VisoEngine::cycle_focus)),
            ("KeyR", Box::new(VisoEngine::toggle_auto_rotate)),
            ("Backquote", Box::new(VisoEngine::reset_focus)),
            ("Escape", Box::new(VisoEngine::clear_selection)),
            (
                "KeyI",
                Box::new(|e| e.toggle_type_visibility(MoleculeType::Ion)),
            ),
            (
                "KeyU",
                Box::new(|e| e.toggle_type_visibility(MoleculeType::Water)),
            ),
            (
                "KeyO",
                Box::new(|e| e.toggle_type_visibility(MoleculeType::Solvent)),
            ),
            ("KeyL", Box::new(VisoEngine::cycle_lipid_mode)),
        ];
        let table = entries
            .into_iter()
            .map(|(k, v)| (k.to_owned(), v))
            .collect();
        Self { table }
    }
}
