//! Per-entity residue selection store used by standalone hosts
//! (viewer + web).

use std::collections::{BTreeMap, BTreeSet};

use molex::entity::molecule::id::EntityId;

use crate::input::click::{
    classify_click_for_selection, ClickEvent, ClickSelectionAction,
};
use crate::VisoEngine;

/// Per-entity residue selection store used by standalone hosts
/// (viewer + web).
///
/// Mirrors the consumer-side selection that foldit-core's
/// `App.selection` owns. Each click event produced by
/// [`VisoEngine::feed_pointer_button`] is fed through
/// [`Self::apply_click`], which mutates the store and re-flattens
/// into the engine's GPU residue space via
/// [`VisoEngine::set_selection`].
#[derive(Default)]
pub struct SelectionStore {
    residues: BTreeMap<EntityId, BTreeSet<u32>>,
}

impl SelectionStore {
    /// Empty store.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Apply a click-event to the selection store and re-flatten the
    /// result into `engine`'s GPU residue space.
    pub fn apply_click(&mut self, engine: &mut VisoEngine, click: &ClickEvent) {
        match classify_click_for_selection(click) {
            ClickSelectionAction::Clear => {
                self.residues.clear();
            }
            ClickSelectionAction::Replace(residues) => {
                self.residues.clear();
                for (entity, residue) in residues {
                    let _ = self
                        .residues
                        .entry(entity)
                        .or_default()
                        .insert(residue);
                }
            }
            ClickSelectionAction::Toggle(residues) => {
                for (entity, residue) in residues {
                    self.toggle(entity, residue);
                }
            }
        }
        self.flush(engine);
    }

    fn toggle(&mut self, entity: EntityId, residue: u32) {
        let set = self.residues.entry(entity).or_default();
        if !set.insert(residue) {
            let _ = set.remove(&residue);
            if set.is_empty() {
                let _ = self.residues.remove(&entity);
            }
        }
    }

    /// Re-flatten the per-entity selection into the engine's flat
    /// residue space. No-op before an offsets map has been published
    /// by the first full rebuild (entities with no offset entry are
    /// dropped).
    fn flush(&self, engine: &mut VisoEngine) {
        let mut flat: Vec<i32> = Vec::new();
        {
            let offsets = engine.entity_residue_offsets();
            for (eid, residues) in &self.residues {
                let Some(&base) = offsets.get(eid) else {
                    continue;
                };
                for r in residues {
                    flat.push((base + *r) as i32);
                }
            }
        }
        engine.set_selection(flat);
    }
}
