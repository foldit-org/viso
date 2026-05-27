//! Platform-agnostic dispatch of engine-level [`UiAction`]s.
//!
//! Both the native (wry) GUI and the web (wasm) entry points call
//! [`dispatch_engine_action`] to handle every action that is purely an
//! engine mutation. Platform-specific actions (panel layout, file
//! dialogs, key forwarding, file load, fetch) are returned to the
//! caller unchanged via `Some(action)`.
//!
//! Each platform supplies a [`UiHost`] implementation that knows how
//! to ship a `(key, json)` pair back to viso-ui — wry `evaluate_script`
//! on native, an iframe `eval` on web.

use crate::bridge::{self, UiAction};
use crate::options::VisoOptions;
use crate::VisoEngine;

/// State-push surface a host (native or web) provides to the dispatcher.
pub(crate) trait UiHost {
    /// Push a raw JSON string to viso-ui under the given key.
    ///
    /// Implementations are responsible for any platform-specific
    /// escaping or buffering before delivery.
    fn push(&self, key: &str, json: &str);
}

/// Dispatch an engine-level UI action.
///
/// Returns `None` when the action was fully handled. Returns
/// `Some(action)` when the action is platform-specific and the caller
/// must handle it (panel layout, file dialogs, key forwarding, file
/// load, fetch).
pub(crate) fn dispatch_engine_action(
    engine: &mut VisoEngine,
    action: UiAction,
    host: &dyn UiHost,
) -> Option<UiAction> {
    match action {
        UiAction::SetOption { path, field, value } => {
            apply_set_option(engine, &path, &field, value);
            push_options(engine, host);
            // Surface kind/opacity and other display options affect
            // the per-entity summary projections.
            push_scene_entities(engine, host);
            None
        }
        UiAction::FocusEntity { id } => {
            if let Some(eid) = engine.entity_id(id) {
                engine.focus_entity(eid);
            }
            push_scene_entities(engine, host);
            None
        }
        UiAction::ToggleEntityVisibility { id } => {
            if let Some(eid) = engine.entity_id(id) {
                engine.toggle_entity_visibility(eid);
            }
            push_scene_entities(engine, host);
            None
        }
        UiAction::ClearEntityAppearance { entity_id } => {
            if let Some(eid) = engine.entity_id(entity_id) {
                engine.clear_entity_appearance(eid);
            }
            push_scene_entities(engine, host);
            None
        }
        UiAction::SetEntityAppearance {
            entity_id,
            field,
            value,
        } => {
            apply_entity_appearance_field(engine, entity_id, &field, &value);
            push_scene_entities(engine, host);
            None
        }
        UiAction::SetEntitySurface { entity_id, kind } => {
            if let Some(eid) = engine.entity_id(entity_id) {
                let default_color = [0.7, 0.7, 0.7, 0.35];
                let mut view = engine.annotations_mut();
                match kind.as_str() {
                    "gaussian" => {
                        view.add_gaussian_surface(eid, default_color);
                    }
                    "ses" => {
                        view.add_ses_surface(eid, default_color);
                    }
                    _ => {
                        view.remove_surface(eid);
                    }
                }
            }
            push_scene_entities(engine, host);
            None
        }
        UiAction::SetSurfaceOption {
            entity_id,
            field,
            value,
        } => {
            if let (Some(eid), Some(v)) =
                (engine.entity_id(entity_id), value.as_f64())
            {
                let ch = match field.as_str() {
                    "color_r" => Some(0),
                    "color_g" => Some(1),
                    "color_b" => Some(2),
                    "opacity" => Some(3),
                    _ => None,
                };
                if let Some(ch) = ch {
                    engine
                        .annotations_mut()
                        .set_surface_color_channel(eid, ch, v as f32);
                }
            }
            push_scene_entities(engine, host);
            None
        }
        UiAction::SetDensityOption { id, field, value } => {
            apply_density_option(engine, id, &field, &value);
            push_density_maps(engine, host);
            None
        }
        UiAction::RemoveDensityMap { id } => {
            engine.density_mut().remove(id);
            push_density_maps(engine, host);
            None
        }
        UiAction::ToggleDensityVisibility { id } => {
            let vis = engine.density.get(id).is_some_and(|e| !e.visible);
            engine.density_mut().set_visible(id, vis);
            push_density_maps(engine, host);
            None
        }
        // Platform-specific — return to caller.
        passthrough @ (UiAction::TogglePanel
        | UiAction::ResizePanel { .. }
        | UiAction::OpenFileDialog
        | UiAction::KeyPress { .. }
        | UiAction::LoadFile { .. }
        | UiAction::FetchPdb { .. }) => Some(passthrough),
    }
}

// ── State push helpers ──────────────────────────────────────────────────

/// Serialize and push the current options state.
pub(crate) fn push_options(engine: &VisoEngine, host: &dyn UiHost) {
    let json = serde_json::to_string(&engine.options).unwrap_or_default();
    host.push("options", &json);
}

/// Serialize and push the current scene entity summaries.
pub(crate) fn push_scene_entities(engine: &VisoEngine, host: &dyn UiHost) {
    let summaries = bridge::entity_summaries(engine);
    let json = serde_json::to_string(&summaries).unwrap_or_default();
    host.push("scene_entities", &json);
}

/// Serialize and push the current density map summaries.
pub(crate) fn push_density_maps(engine: &VisoEngine, host: &dyn UiHost) {
    let maps = bridge::density_summaries(engine);
    let json = serde_json::to_string(&maps).unwrap_or_default();
    host.push("density_maps", &json);
}

// ── Engine mutators ─────────────────────────────────────────────────────

/// Apply a `SetOption` patch by serializing options, mutating the JSON,
/// and deserializing back. Falls back to inserting the field on the
/// parent object when the field is absent (e.g. `Option::is_none`
/// fields skipped during serialization).
fn apply_set_option(
    engine: &mut VisoEngine,
    path: &str,
    field: &str,
    value: serde_json::Value,
) {
    let mut opts = engine.options.clone();
    let Ok(mut root) = serde_json::to_value(&opts) else {
        log::warn!("Failed to serialize options to JSON");
        return;
    };
    patch_options_json(&mut root, path, field, value);
    match serde_json::from_value::<VisoOptions>(root) {
        Ok(updated) => opts = updated,
        Err(e) => {
            log::warn!("Options deserialization failed: {e}");
        }
    }
    engine.set_options(opts);
}

/// Apply a `{path, field, value}` patch to a serialized [`VisoOptions`].
fn patch_options_json(
    root: &mut serde_json::Value,
    path: &str,
    field: &str,
    value: serde_json::Value,
) {
    let pointer = format!(
        "/{}",
        path.split('.')
            .chain(std::iter::once(field))
            .collect::<Vec<_>>()
            .join("/")
    );
    if let Some(target) = root.pointer_mut(&pointer) {
        *target = value;
        return;
    }
    let parent_pointer = format!("/{}", path.replace('.', "/"));
    let Some(parent) = root.pointer_mut(&parent_pointer) else {
        log::warn!("Option path not found: {pointer}");
        return;
    };
    let Some(obj) = parent.as_object_mut() else {
        log::warn!("Option parent is not an object: {parent_pointer}");
        return;
    };
    let _ = obj.insert(field.to_owned(), value);
}

/// Apply a single density option field.
fn apply_density_option(
    engine: &mut VisoEngine,
    id: u32,
    field: &str,
    value: &serde_json::Value,
) {
    match field {
        "threshold" => {
            if let Some(v) = value.as_f64() {
                engine.density_mut().set_threshold(id, v as f32);
            }
        }
        "opacity" => {
            if let Some(v) = value.as_f64() {
                engine.density_mut().set_opacity(id, v as f32);
            }
        }
        "color_r" | "color_g" | "color_b" => {
            if let Some(v) = value.as_f64() {
                let mut color =
                    engine.density.get(id).map_or([0.3, 0.5, 0.8], |e| e.color);
                match field {
                    "color_r" => color[0] = v as f32,
                    "color_g" => color[1] = v as f32,
                    "color_b" => color[2] = v as f32,
                    _ => {}
                }
                engine.density_mut().set_color(id, color);
            }
        }
        _ => log::warn!("Unknown density field: {field}"),
    }
}

/// Apply a single per-entity appearance override field.
fn apply_entity_appearance_field(
    engine: &mut VisoEngine,
    entity_id: u32,
    field: &str,
    value: &serde_json::Value,
) {
    let Some(eid) = engine.entity_id(entity_id) else {
        return;
    };
    let mut ovr = engine.entity_appearance(eid).cloned().unwrap_or_default();
    if let Err(unknown) = ovr.apply_json_field(field, value) {
        log::warn!("Unknown entity appearance field: {unknown}");
        return;
    }
    if ovr.is_empty() {
        engine.clear_entity_appearance(eid);
    } else {
        engine.set_entity_appearance(eid, ovr);
    }
}
