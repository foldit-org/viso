//! Platform-agnostic bridge between the viso engine and viso-ui.
//!
//! Contains the IPC action types, parsing, entity serialization, structure
//! parsing, and the shared bridge JavaScript that both native (wry) and web
//! (wasm) hosts inject into viso-ui.

use crate::engine::focus::Focus;
use crate::VisoEngine;

pub(crate) mod dispatch;

// Panel layout model

/// Panel axis: right sidebar or bottom bar.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PanelAxis {
    /// Panel docked to the right edge (landscape orientation).
    Right,
    /// Panel docked to the bottom edge (portrait orientation).
    Bottom,
}

impl PanelAxis {
    /// Determine axis from window dimensions.
    #[must_use]
    pub fn from_dimensions(width: u32, height: u32) -> Self {
        if height > width {
            Self::Bottom
        } else {
            Self::Right
        }
    }

    /// Returns `"portrait"` or `"landscape"` for the viso-ui orientation
    /// event.
    #[must_use]
    pub fn orientation_str(self) -> &'static str {
        match self {
            Self::Bottom => "portrait",
            Self::Right => "landscape",
        }
    }
}

/// Default panel size in CSS (logical) pixels.
///
/// Both web and desktop hosts use the same logical value.  The desktop
/// host converts to physical pixels via `scale_factor` at the point of
/// `set_bounds()`.
pub(crate) const DEFAULT_PANEL_SIZE: u32 = 340;

/// Minimum panel size in CSS pixels for resize.
pub(crate) const MIN_PANEL_SIZE: u32 = 220;

/// Maximum panel size in CSS pixels for resize.
pub(crate) const MAX_PANEL_SIZE: u32 = 700;

/// Panel size when collapsed (just the toggle arrow strip), in CSS
/// pixels.
#[cfg(target_arch = "wasm32")]
pub const COLLAPSED_SIZE: u32 = 32;

// UiAction

/// Actions sent from the viso-ui WASM app to the host engine.
#[derive(Debug)]
pub enum UiAction {
    /// Set a single option field: `options[section][field] = value`.
    SetOption {
        /// Top-level section key (e.g. `"lighting"`).
        path: String,
        /// Field key within the section (e.g. `"roughness"`).
        field: String,
        /// New JSON value.
        value: serde_json::Value,
    },
    /// Load a structure file by path.
    LoadFile {
        /// Filesystem path to the `.cif` or `.pdb` file.
        path: String,
    },
    /// Fetch a PDB structure by ID from a remote database.
    FetchPdb {
        /// 4-character PDB identifier (e.g. `"1crn"`).
        id: String,
        /// Source database: `"rcsb"` or `"pdb-redo"`.
        source: String,
    },
    /// Open a native file dialog to pick a local structure file.
    OpenFileDialog,
    /// Toggle the panel between pinned and unpinned.
    TogglePanel,
    /// Resize the panel along its current axis.
    ResizePanel {
        /// New panel size in physical pixels (width or height depending
        /// on axis).
        size: u32,
    },
    /// A key press forwarded from the webview (e.g. Tab on Windows).
    KeyPress {
        /// Physical key name matching winit's `KeyCode` debug format.
        key: String,
    },
    /// Clear all per-entity display overrides for an entity.
    ClearEntityAppearance {
        /// Entity ID to clear.
        entity_id: u32,
    },
    /// Set a density map display parameter.
    SetDensityOption {
        /// Density map ID.
        id: u32,
        /// Field name (`"sigma"`, `"opacity"`, `"color_r"`, etc.).
        field: String,
        /// New JSON value for the field.
        value: serde_json::Value,
    },
    /// Remove a density map from the scene.
    RemoveDensityMap {
        /// Density map ID.
        id: u32,
    },
    /// Set molecular surface type for an entity (or remove it).
    SetEntitySurface {
        /// Entity ID.
        entity_id: u32,
        /// Surface kind: `"none"`, `"gaussian"`, or `"ses"`.
        kind: String,
    },
    /// Set a molecular surface display parameter.
    SetSurfaceOption {
        /// Entity ID.
        entity_id: u32,
        /// Field name (`"color_r"`, `"color_g"`, `"color_b"`, `"opacity"`).
        field: String,
        /// New value.
        value: serde_json::Value,
    },
    /// Toggle visibility of a density map.
    ToggleDensityVisibility {
        /// Density map ID.
        id: u32,
    },
    /// Set a field on a per-entity appearance override.
    SetEntityAppearance {
        /// Entity ID.
        entity_id: u32,
        /// Field name.
        field: String,
        /// New JSON value (null clears the override for that field).
        value: serde_json::Value,
    },
    /// Focus a specific entity by ID and fit the camera to it.
    FocusEntity {
        /// Entity identifier.
        id: u32,
    },
    /// Toggle visibility of a specific entity.
    ToggleEntityVisibility {
        /// Entity identifier.
        id: u32,
    },
}

// IPC parsing

/// Parse an IPC message from viso-ui into a [`UiAction`].
pub(crate) fn parse_action(msg: &serde_json::Value) -> Option<UiAction> {
    let action = msg.get("action")?.as_str()?;
    match action {
        "set_option" => {
            let path = msg.get("path")?.as_str()?.to_owned();
            let field = msg.get("field")?.as_str()?.to_owned();
            let value = msg.get("value")?.clone();
            Some(UiAction::SetOption { path, field, value })
        }
        "load_file" => {
            let path = msg.get("path")?.as_str()?.to_owned();
            Some(UiAction::LoadFile { path })
        }
        "fetch_pdb" => {
            let id = msg.get("id")?.as_str()?.to_owned();
            let source = msg.get("source")?.as_str()?.to_owned();
            Some(UiAction::FetchPdb { id, source })
        }
        "open_file_dialog" => Some(UiAction::OpenFileDialog),
        "key" => {
            let key = msg.get("key")?.as_str()?.to_owned();
            Some(UiAction::KeyPress { key })
        }
        "toggle_panel" => Some(UiAction::TogglePanel),
        "resize_panel" => {
            let size =
                msg.get("size").or_else(|| msg.get("width"))?.as_u64()? as u32;
            Some(UiAction::ResizePanel { size })
        }
        "focus_entity" => {
            let id = msg.get("id")?.as_u64()? as u32;
            Some(UiAction::FocusEntity { id })
        }
        "toggle_entity_visibility" => {
            let id = msg.get("id")?.as_u64()? as u32;
            Some(UiAction::ToggleEntityVisibility { id })
        }
        "clear_entity_option" | "clear_entity_appearance" => {
            let entity_id = msg.get("entity_id")?.as_u64()? as u32;
            Some(UiAction::ClearEntityAppearance { entity_id })
        }
        "set_entity_surface" => {
            let entity_id = msg.get("entity_id")?.as_u64()? as u32;
            let kind = msg.get("kind")?.as_str()?.to_owned();
            Some(UiAction::SetEntitySurface { entity_id, kind })
        }
        "set_surface_option" => {
            let entity_id = msg.get("entity_id")?.as_u64()? as u32;
            let field = msg.get("field")?.as_str()?.to_owned();
            let value = msg.get("value")?.clone();
            Some(UiAction::SetSurfaceOption {
                entity_id,
                field,
                value,
            })
        }
        "set_density_option" => {
            let id = msg.get("id")?.as_u64()? as u32;
            let field = msg.get("field")?.as_str()?.to_owned();
            let value = msg.get("value")?.clone();
            Some(UiAction::SetDensityOption { id, field, value })
        }
        "remove_density_map" => {
            let id = msg.get("id")?.as_u64()? as u32;
            Some(UiAction::RemoveDensityMap { id })
        }
        "toggle_density_visibility" => {
            let id = msg.get("id")?.as_u64()? as u32;
            Some(UiAction::ToggleDensityVisibility { id })
        }
        "set_entity_appearance" | "set_entity_option" => {
            let entity_id = msg.get("entity_id")?.as_u64()? as u32;
            let field = msg.get("field")?.as_str()?.to_owned();
            let value = msg.get("value")?.clone();
            Some(UiAction::SetEntityAppearance {
                entity_id,
                field,
                value,
            })
        }
        _ => None,
    }
}

// JS escaping

/// Escape a string for safe embedding in a JavaScript single-quoted
/// literal.
pub(crate) fn escape_for_js(s: &str) -> String {
    s.replace('\\', "\\\\").replace('\'', "\\'")
}

// Entity summaries

/// Build a JSON-serializable summary of all entities for the viso-ui
/// panel.
pub(crate) fn entity_summaries(engine: &VisoEngine) -> Vec<serde_json::Value> {
    use molex::MoleculeType;

    use crate::options::DrawingMode;

    let focus = engine.focus();
    engine
        .assembly()
        .entities()
        .iter()
        .map(|entity| {
            let eid = entity.id();
            let raw_id = eid.raw();
            let mol_type = match entity.molecule_type() {
                MoleculeType::Protein => "Protein",
                MoleculeType::DNA => "DNA",
                MoleculeType::RNA => "RNA",
                MoleculeType::Water => "Water",
                MoleculeType::Ion => "Ion",
                MoleculeType::Cofactor => "Cofactor",
                MoleculeType::Solvent => "Solvent",
                MoleculeType::Lipid => "Lipid",
                MoleculeType::Ligand => "Ligand",
            };
            let chain_ids: Vec<String> = entity
                .pdb_chain_id()
                .map_or_else(Vec::new, |cid| vec![cid.to_owned()]);
            let focused =
                matches!(focus, Focus::Entity(eid) if eid.raw() == raw_id);
            let ovr = engine.entity_appearance(eid);
            let resolved_display = ovr.map_or_else(
                || engine.options().display.clone(),
                |o| o.to_display_options(&engine.options().display),
            );
            let has_overrides = ovr.is_some_and(|o| !o.is_empty());

            let effective_mode =
                ovr.and_then(|o| o.drawing_mode).unwrap_or_else(|| {
                    if engine.options().display.drawing_mode()
                        == DrawingMode::Cartoon
                    {
                        DrawingMode::default_for(entity.molecule_type())
                    } else {
                        engine.options().display.drawing_mode()
                    }
                });

            let mut entry = serde_json::json!({
                "id": raw_id,
                "molecule_type": mol_type,
                "label": entity.label(),
                "visible": engine.is_entity_visible(raw_id),
                "atom_count": entity.atom_count(),
                "chain_ids": chain_ids,
                "focused": focused,
                "focusable": entity.is_focusable(),
                "drawing_mode": effective_mode,
                "color_scheme": resolved_display.backbone_color_scheme(),
                "show_sidechains": resolved_display.show_sidechains(),
                "surface": effective_surface_kind(engine, raw_id),
                "surface_color": effective_surface_color(engine, raw_id),
                "helix_style": resolved_display.helix_style(),
                "sheet_style": resolved_display.sheet_style(),
                "show_cavities": resolved_display.show_cavities(),
                "sidechain_color_mode": resolved_display.sidechain_color_mode(),
                "na_color_mode": resolved_display.na_color_mode(),
                "lipid_mode": resolved_display.lipid_mode(),
                "show_hydrogens": resolved_display.show_hydrogens(),
                "has_overrides": has_overrides,
            });
            if let Some(ovr_val) = ovr {
                if let Ok(ovr_json) = serde_json::to_value(ovr_val) {
                    entry["appearance_overrides"] = ovr_json;
                }
            }
            entry
        })
        .collect()
}

/// Effective surface kind string for an entity, accounting for per-entity
/// overrides (including invisible opt-outs) and global fallback.
fn effective_surface_kind(engine: &VisoEngine, raw: u32) -> &'static str {
    use crate::engine::surface::SurfaceKind;
    use crate::options::SurfaceKindOption;

    if let Some(eid) = engine.entity_id(raw) {
        if let Some(s) = engine.annotations.surfaces.get(&eid) {
            if !s.visible {
                return "none";
            }
            return match s.kind {
                SurfaceKind::Gaussian => "gaussian",
                SurfaceKind::Ses => "ses",
            };
        }
    }
    // No per-entity entry — fall back to global
    match engine.options.display.surface_kind() {
        SurfaceKindOption::Gaussian => "gaussian",
        SurfaceKindOption::Ses => "ses",
        SurfaceKindOption::None => "none",
    }
}

/// Effective surface color for an entity (per-entity or global default).
fn effective_surface_color(engine: &VisoEngine, raw: u32) -> serde_json::Value {
    if let Some(eid) = engine.entity_id(raw) {
        if let Some(s) = engine.annotations.surfaces.get(&eid) {
            if s.visible {
                return serde_json::json!([
                    s.color[0], s.color[1], s.color[2], s.color[3]
                ]);
            }
        }
    }
    serde_json::json!(null)
}

// Density summaries

/// Build a JSON-serializable summary of all density maps for the viso-ui
/// panel.
pub(crate) fn density_summaries(engine: &VisoEngine) -> Vec<serde_json::Value> {
    engine
        .density
        .all_entries()
        .map(|(id, entry)| {
            serde_json::json!({
                "id": id,
                "visible": entry.visible,
                "threshold": entry.threshold,
                "dmin": entry.map.dmin,
                "dmax": entry.map.dmax,
                "color": [entry.color[0], entry.color[1], entry.color[2]],
                "opacity": entry.opacity,
            })
        })
        .collect()
}

// File parsing

/// Result of parsing a file — either a structure or a density map.
#[allow(dead_code)]
pub(crate) enum ParsedFile {
    /// Molecular structure (protein, ligand, etc.).
    Structure(Vec<molex::MoleculeEntity>),
    /// Electron density map (CCP4/MRC format).
    Density(molex::entity::surface::Density),
}

/// Returns `true` if the extension indicates a density map format.
pub(crate) fn is_density_extension(ext: &str) -> bool {
    let lower = ext.to_ascii_lowercase();
    let trimmed = lower.trim_start_matches('.');
    matches!(trimmed, "mrc" | "map" | "ccp4")
}

/// Parse a file from in-memory bytes, auto-detecting structure vs density
/// by extension.
///
/// `format_hint` should be the file extension (e.g. `"cif"`, `"mrc"`),
/// with or without a leading dot.
#[allow(dead_code)]
pub(crate) fn parse_file_bytes(
    bytes: &[u8],
    format_hint: &str,
) -> Result<ParsedFile, String> {
    let hint = format_hint.to_ascii_lowercase();
    let hint = hint.trim_start_matches('.');
    match hint {
        "mrc" | "map" | "ccp4" => {
            let map = molex::adapters::mrc::mrc_to_density(bytes)
                .map_err(|e| format!("Density parse error: {e}"))?;
            Ok(ParsedFile::Density(map))
        }
        "cif" | "mmcif" => {
            let text = std::str::from_utf8(bytes)
                .map_err(|e| format!("Invalid UTF-8 in CIF: {e}"))?;
            let entities = molex::adapters::cif::mmcif_str_to_entities(text)
                .map_err(|e| format!("CIF parse error: {e}"))?;
            Ok(ParsedFile::Structure(entities))
        }
        "pdb" | "ent" => {
            let text = std::str::from_utf8(bytes)
                .map_err(|e| format!("Invalid UTF-8 in PDB: {e}"))?;
            let entities = molex::adapters::pdb::pdb_str_to_entities(text)
                .map_err(|e| format!("PDB parse error: {e}"))?;
            Ok(ParsedFile::Structure(entities))
        }
        "bcif" => {
            let entities = molex::adapters::bcif::bcif_to_entities(bytes)
                .map_err(|e| format!("BCIF parse error: {e}"))?;
            Ok(ParsedFile::Structure(entities))
        }
        other => Err(format!(
            "Unsupported format '{other}'. Use 'cif', 'pdb', 'bcif', 'mrc', \
             'map', or 'ccp4'."
        )),
    }
}

/// Parse a structure file from in-memory bytes (structure formats only).
///
/// `format_hint` should be the file extension (e.g. `"cif"`, `"pdb"`),
/// with or without a leading dot.
#[cfg(target_arch = "wasm32")]
pub fn parse_structure_bytes(
    bytes: &[u8],
    format_hint: &str,
) -> Result<Vec<molex::MoleculeEntity>, String> {
    match parse_file_bytes(bytes, format_hint)? {
        ParsedFile::Structure(entities) => Ok(entities),
        ParsedFile::Density(_) => {
            Err("Expected structure file, got density map".into())
        }
    }
}

// Bridge JavaScript

/// JavaScript bridge injected into viso-ui that defines the
/// `window.__viso_push_*` functions and dispatches `CustomEvent`s.
///
/// Both native (wry) and web (wasm) hosts use this same core script.
/// The native host appends additional platform-specific code for Tab key
/// forwarding and `addEventListener` replay.
pub(crate) const BRIDGE_JS: &str = r"
(function() {
    var pending = {};
    function dispatch(name, json) {
        window.dispatchEvent(new CustomEvent(name, { detail: json }));
    }
    function makePush(key, eventName) {
        window['__viso_push_' + key] = function(json) {
            pending[key] = json;
            dispatch(eventName, json);
        };
    }
    makePush('schema', 'viso-schema');
    makePush('options', 'viso-options');
    makePush('stats', 'viso-stats');
    makePush('panel_pinned', 'viso-panel-pinned');
    makePush('load_status', 'viso-load-status');
    makePush('scene_entities', 'viso-scene-entities');
    makePush('orientation', 'viso-orientation');
    makePush('panel_size', 'viso-panel-size');
    makePush('density_maps', 'viso-density-maps');

    // Allow late listeners (e.g. dioxus WASM) to replay any values
    // that were pushed before they registered.
    window.__viso_replay_pending = function() {
        for (var k in pending) {
            var fn = window['__viso_push_' + k];
            if (fn) fn(pending[k]);
        }
    };

    // Replay any early pushes
    if (window.__viso_early) {
        var e = window.__viso_early;
        for (var k in e) {
            if (window['__viso_push_' + k]) {
                window['__viso_push_' + k](e[k]);
            }
        }
        delete window.__viso_early;
    }
})();
";
