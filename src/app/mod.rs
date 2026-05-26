//! Standalone-application layer for viso.
//!
//! In the real deployment (`foldit`), the host application owns the
//! [`Assembly`] and pushes the latest snapshot to viso via
//! [`VisoEngine::set_assembly`]. In standalone deployments
//! (`feature = "viewer"`, `"gui"`, `"web"`), viso plays that same host
//! role: this module owns [`VisoApp`] (the authoritative `Assembly`)
//! and the executable layer that wraps it — the winit [`viewer`], the
//! wry-webview [`gui`] panel, and the wasm `web` entry.
//!
//! Each [`VisoApp`] mutation method takes `&mut VisoEngine` so the new
//! `Assembly` snapshot can be pushed via [`VisoEngine::set_assembly`]
//! and viso-side bookkeeping (animation transitions, camera fit,
//! per-entity annotations) can be updated atomically alongside it.
//! `VisoEngine` has no mutation surface for structural state — the
//! host (in standalone: [`VisoApp`]) is the sole owner of the
//! authoritative `Assembly`.

#[cfg(feature = "viewer")]
pub mod viewer;

#[cfg(feature = "gui")]
pub mod gui;

#[cfg(all(feature = "web", target_arch = "wasm32"))]
pub mod web;

use std::collections::HashMap;
use std::sync::Arc;

use molex::ops::edit::AssemblyEdit;
use molex::{Assembly, MoleculeEntity, MoleculeType, SSType};

use crate::animation::transition::Transition;
use crate::error::VisoError;
use crate::VisoEngine;

/// Owns the authoritative [`Assembly`] in standalone deployments.
///
/// Produced by [`VisoApp::new_empty`], [`VisoApp::from_entities`],
/// [`VisoApp::from_bytes`], or [`VisoApp::from_file`]. Standalone
/// entry points (viewer, gui, web) hold a `VisoApp` alongside their
/// `VisoEngine`; mutation methods take `&mut VisoEngine` and push the
/// new snapshot via [`VisoEngine::set_assembly`].
pub struct VisoApp {
    assembly: Assembly,
}

impl VisoApp {
    // ── Construction ───────────────────────────────────────────────

    /// An empty app — a zero-entity `Assembly`. The caller pushes the
    /// initial snapshot to viso via [`Self::publish`] (or
    /// [`VisoEngine::set_assembly`] directly) after constructing the
    /// engine.
    #[must_use]
    pub fn new_empty() -> Self {
        Self::from_entities(Vec::new())
    }

    /// App seeded with the given entities.
    #[must_use]
    pub fn from_entities(entities: Vec<MoleculeEntity>) -> Self {
        Self {
            assembly: Assembly::new(entities),
        }
    }

    /// Parse structure bytes with the given format hint (`"cif"`,
    /// `"pdb"`, or `"bcif"`) and seed the app with the parsed
    /// entities.
    ///
    /// # Errors
    ///
    /// Returns [`VisoError::StructureLoad`] if parsing fails or the
    /// format hint is unsupported.
    pub fn from_bytes(
        bytes: &[u8],
        format_hint: &str,
    ) -> Result<Self, VisoError> {
        let entities = parse_structure_bytes(bytes, format_hint)?;
        Ok(Self::from_entities(entities))
    }

    /// Parse a structure file (`.cif` / `.pdb` / `.bcif`) and seed
    /// the app with the parsed entities.
    ///
    /// # Errors
    ///
    /// Returns [`VisoError::StructureLoad`] if the file cannot be
    /// read or parsed.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn from_file(path: &str) -> Result<Self, VisoError> {
        let entities = molex::adapters::pdb::structure_file_to_entities(
            std::path::Path::new(path),
        )
        .map_err(|e| VisoError::StructureLoad(e.to_string()))?;
        Ok(Self::from_entities(entities))
    }

    // ── Read accessors ─────────────────────────────────────────────

    /// Read-only access to the app's current `Assembly`.
    ///
    /// Exposed for panel/bridge code that needs to iterate entities
    /// (molecule type, label, chain id). The engine's per-frame sync
    /// ultimately consumes this same assembly via the triple buffer.
    #[must_use]
    pub fn assembly(&self) -> &Assembly {
        &self.assembly
    }

    // ── Publish helper ─────────────────────────────────────────────

    /// Push the current `Assembly` snapshot to the engine. Library
    /// hosts call this once after constructing both the app and the
    /// engine to seed viso's first sync; subsequent mutations route
    /// through the [`VisoApp`] mutation methods, which publish
    /// internally.
    pub fn publish(&self, engine: &mut VisoEngine) {
        engine.set_assembly(Arc::new(self.assembly.clone()));
    }

    // ── Lifecycle mutations ────────────────────────────────────────

    /// Load entities into the scene. Returns the assigned entity IDs.
    /// If `fit_camera` is true, fits the viewer to the combined
    /// bounding sphere after loading.
    pub fn load_entities(
        &mut self,
        engine: &mut VisoEngine,
        entities: Vec<MoleculeEntity>,
        fit_camera: bool,
    ) -> Vec<u32> {
        let was_empty = self.assembly.entities().is_empty()
            && !engine.animation.animator.is_animating();

        let ids: Vec<u32> = entities.iter().map(|e| e.id().raw()).collect();
        // Rebuild the assembly in one shot rather than calling
        // `add_entity` per entity: every `&mut Assembly` mutation
        // re-runs DSSP + H-bond detection over *all* entities, so
        // the incremental form is O(N²) for a load of N entities and
        // wedges the main thread for minutes on large complexes.
        let mut combined: Vec<MoleculeEntity> = self
            .assembly
            .entities()
            .iter()
            .map(|e| MoleculeEntity::clone(e))
            .collect();
        combined.extend(entities);
        self.assembly = Assembly::new(combined);

        apply_type_visibility(self, engine);

        let mut entity_transitions: HashMap<u32, Transition> = HashMap::new();
        if !was_empty {
            for &id in &ids {
                let _ = entity_transitions.insert(id, Transition::snap());
            }
        }
        engine.queue_entity_transitions(entity_transitions);

        self.publish(engine);
        engine.sync_now();

        if fit_camera {
            engine.fit_session_camera();
        }
        ids
    }

    /// Replace the current scene with `entities`. All existing
    /// entities are removed from the assembly first. Fits the camera
    /// to the new combined bounding sphere.
    pub fn replace_scene(
        &mut self,
        engine: &mut VisoEngine,
        entities: Vec<MoleculeEntity>,
    ) -> Vec<u32> {
        self.remove_all_internal(engine);
        self.load_entities(engine, entities, true)
    }

    /// Remove all entities from the scene.
    pub fn clear_scene(&mut self, engine: &mut VisoEngine) {
        self.remove_all_internal(engine);
        engine.clear_pending_transitions();
        self.publish(engine);
        engine.sync_now();
    }

    fn remove_all_internal(&mut self, engine: &mut VisoEngine) {
        // Skip the per-entity `remove_entity` loop: each call re-runs
        // DSSP + H-bond detection over the remaining entities. A bulk
        // replace with an empty `Assembly::new` costs one recompute
        // (and that recompute walks zero entities).
        self.assembly = Assembly::new(Vec::new());
        engine.reset_scene_local_state();
    }

    /// Replace one or more entities with new [`MoleculeEntity`] data.
    /// Each entity is matched by its id. Per-entity behavior overrides
    /// are used when set, otherwise `default_transition` is applied.
    pub fn update_entities(
        &mut self,
        engine: &mut VisoEngine,
        updated: Vec<MoleculeEntity>,
        default_transition: &Transition,
    ) {
        // Bulk rebuild (see `load_entities` comment): every
        // `&mut Assembly` mutation re-runs DSSP + H-bond detection over
        // all entities, so looping `remove_entity` + `add_entity` is
        // O(N²) over the assembly.
        let mut updated_by_id: HashMap<u32, MoleculeEntity> =
            updated.into_iter().map(|e| (e.id().raw(), e)).collect();

        let current: Vec<MoleculeEntity> = self
            .assembly
            .entities()
            .iter()
            .map(|e| MoleculeEntity::clone(e))
            .collect();
        let mut combined: Vec<MoleculeEntity> =
            Vec::with_capacity(current.len());
        let mut entity_transitions: HashMap<u32, Transition> = HashMap::new();
        for entity in current {
            let raw_id = entity.id().raw();
            if let Some(new_entity) = updated_by_id.remove(&raw_id) {
                let _ = entity_transitions
                    .insert(raw_id, default_transition.clone());
                combined.push(new_entity);
            } else {
                combined.push(entity);
            }
        }

        if entity_transitions.is_empty() {
            return;
        }
        self.assembly = Assembly::new(combined);
        engine.queue_entity_transitions(entity_transitions);
        self.publish(engine);
        engine.sync_now();
    }

    /// Update a single entity by ID.
    ///
    /// # Errors
    ///
    /// Returns [`VisoError::StructureLoad`] if no entity with the
    /// given ID exists.
    pub fn update_entity(
        &mut self,
        engine: &mut VisoEngine,
        entity: MoleculeEntity,
        transition: Transition,
    ) -> Result<(), VisoError> {
        let raw_id = entity.id().raw();
        if self.assembly.entity(entity.id()).is_none() {
            return Err(VisoError::StructureLoad(format!(
                "Entity {raw_id} not found"
            )));
        }
        let entity_id = entity.id();
        let _ = self.assembly.apply_edits(&[
            AssemblyEdit::RemoveEntity { entity: entity_id },
            AssemblyEdit::AddEntity { entity },
        ]);

        engine.queue_entity_transition(raw_id, transition);
        self.publish(engine);
        engine.sync_now();
        Ok(())
    }

    /// Reconcile the scene with `entities`. New IDs are added; IDs
    /// missing from the input are removed; existing IDs with changed
    /// data are replaced. `default_transition` applies to new and
    /// updated entities that lack a per-entity behavior.
    pub fn sync_entities(
        &mut self,
        engine: &mut VisoEngine,
        entities: Vec<MoleculeEntity>,
        default_transition: &Transition,
    ) {
        use std::collections::HashSet;

        let incoming_ids: HashSet<u32> =
            entities.iter().map(|e| e.id().raw()).collect();
        let current_raw_ids: HashSet<u32> = self
            .assembly
            .entities()
            .iter()
            .map(|e| e.id().raw())
            .collect();

        for &raw_id in &current_raw_ids {
            if !incoming_ids.contains(&raw_id) {
                if let Some(eid) = engine.entity_id(raw_id) {
                    engine.clear_entity_behavior(eid);
                }
            }
        }

        // Bulk rebuild (see `load_entities` comment): per-entity
        // `remove_entity`/`add_entity` would force an O(N²) DSSP +
        // H-bond recompute over the full assembly.
        let mut entity_transitions: HashMap<u32, Transition> = HashMap::new();
        for entity in &entities {
            let raw_id = entity.id().raw();
            let _ =
                entity_transitions.insert(raw_id, default_transition.clone());
        }
        self.assembly = Assembly::new(entities);

        engine.queue_entity_transitions(entity_transitions);
        self.publish(engine);
        engine.sync_now();
    }

    /// Remove an entity by ID. No-op if the entity is not present.
    pub fn remove_entity(&mut self, engine: &mut VisoEngine, id: u32) {
        let Some(eid) = self
            .assembly
            .entities()
            .iter()
            .map(|e| e.id())
            .find(|e| e.raw() == id)
        else {
            return;
        };
        engine.clear_entity_behavior(eid);
        let _ = self
            .assembly
            .apply_edit(&AssemblyEdit::RemoveEntity { entity: eid });
        self.publish(engine);
        engine.sync_now();
    }

    /// Set visibility for a specific entity. For ambient types
    /// (water, ion, solvent), also syncs the corresponding display
    /// option so the renderer safety net stays consistent.
    pub fn set_entity_visible(
        &mut self,
        engine: &mut VisoEngine,
        id: u32,
        visible: bool,
    ) {
        let eid = self
            .assembly
            .entities()
            .iter()
            .map(|e| e.id())
            .find(|e| e.raw() == id);
        let Some(eid) = eid else {
            return;
        };
        let mol_type = match self.assembly.entity(eid) {
            Some(e) => e.molecule_type(),
            None => return,
        };
        match mol_type {
            MoleculeType::Water => {
                engine.options.display.show_waters = visible;
            }
            MoleculeType::Ion => {
                engine.options.display.show_ions = visible;
            }
            MoleculeType::Solvent => {
                engine.options.display.show_solvent = visible;
            }
            _ => {}
        }
        engine.set_entity_visible(id, visible);
        engine.sync_scene_to_renderers(HashMap::new());
    }

    /// Set per-residue scores for a specific entity. Scores drive
    /// color-by-score visualization. Pass `None` to clear.
    pub fn set_per_residue_scores(
        &mut self,
        engine: &mut VisoEngine,
        id: u32,
        scores: Option<Vec<f64>>,
    ) {
        engine.set_per_residue_scores(id, scores);
        engine.sync_scene_to_renderers(HashMap::new());
    }

    /// Set an SS override for an entity (used for puzzle annotations).
    pub fn set_ss_override(
        &mut self,
        engine: &mut VisoEngine,
        id: u32,
        ss: Vec<SSType>,
    ) {
        engine.set_ss_override(id, ss);
        engine.sync_scene_to_renderers(HashMap::new());
    }
}

/// Apply ambient-type visibility defaults (water, ion, solvent) to
/// every entity currently in the assembly, using the engine's current
/// display options.
fn apply_type_visibility(app: &VisoApp, engine: &mut VisoEngine) {
    for entity in app.assembly.entities() {
        let raw_id = entity.id().raw();
        let visible = match entity.molecule_type() {
            MoleculeType::Water => engine.options.display.show_waters,
            MoleculeType::Ion => engine.options.display.show_ions,
            MoleculeType::Solvent => engine.options.display.show_solvent,
            _ => true,
        };
        engine.set_entity_visible(raw_id, visible);
    }
}

/// Parse structure bytes as in the bootstrap loader.
fn parse_structure_bytes(
    bytes: &[u8],
    format_hint: &str,
) -> Result<Vec<MoleculeEntity>, VisoError> {
    let hint = format_hint.to_ascii_lowercase();
    let hint = hint.trim_start_matches('.');
    match hint {
        "cif" | "mmcif" => {
            let text = std::str::from_utf8(bytes).map_err(|e| {
                VisoError::StructureLoad(format!("Invalid UTF-8 in CIF: {e}"))
            })?;
            molex::adapters::cif::mmcif_str_to_entities(text)
                .map_err(|e| VisoError::StructureLoad(e.to_string()))
        }
        "pdb" | "ent" => {
            let text = std::str::from_utf8(bytes).map_err(|e| {
                VisoError::StructureLoad(format!("Invalid UTF-8 in PDB: {e}"))
            })?;
            molex::adapters::pdb::pdb_str_to_entities(text)
                .map_err(|e| VisoError::StructureLoad(e.to_string()))
        }
        "bcif" => molex::adapters::bcif::bcif_to_entities(bytes)
            .map_err(|e| VisoError::StructureLoad(e.to_string())),
        other => Err(VisoError::StructureLoad(format!(
            "Unsupported format '{other}'. Use 'cif', 'pdb', or 'bcif'."
        ))),
    }
}
