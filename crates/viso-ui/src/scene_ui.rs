//! Scene panel UI showing all entities in the current scene.

use std::collections::HashSet;

use dioxus::prelude::*;
use serde_json::Value;

use crate::bridge;

/// Scene panel: global appearance, entity list, density maps.
///
/// `expanded_ids` tracks which entity rows have their options panel open.
/// It's owned by the parent `app()` so it survives tab switches.
#[component]
pub fn ScenePanel(
    scene_entities: Signal<Option<Value>>,
    expanded_ids: Signal<HashSet<u64>>,
    density_maps: Signal<Option<Value>>,
    options: Signal<Option<Value>>,
) -> Element {
    let entities = scene_entities.read();
    let list = entities.as_ref().and_then(Value::as_array);

    let density = density_maps.read();
    let density_list = density.as_ref().and_then(Value::as_array);

    rsx! {
        div { class: "scene-panel",
            {global_appearance_section(&options.read())}
            if let Some(items) = list {
                if items.is_empty() {
                    div { class: "scene-empty", "No entities in scene" }
                } else {
                    for item in items.iter() {
                        {entity_row(item, expanded_ids)}
                    }
                }
            } else {
                div { class: "scene-empty", "Waiting for scene data..." }
            }
            if let Some(maps) = density_list {
                if !maps.is_empty() {
                    for map_item in maps.iter() {
                        {density_row(map_item)}
                    }
                }
            }
        }
    }
}

// ── Global Appearance ──────────────────────────────────────────────────────

/// Read a string field from `options.display`.
fn display_str<'a>(
    opts: Option<&'a Value>,
    field: &str,
    default: &'a str,
) -> &'a str {
    opts.and_then(|o| o.get("display"))
        .and_then(|d| d.get(field))
        .and_then(Value::as_str)
        .unwrap_or(default)
}

/// Read a bool field from `options.display`.
fn display_bool(opts: Option<&Value>, field: &str, default: bool) -> bool {
    opts.and_then(|o| o.get("display"))
        .and_then(|d| d.get(field))
        .and_then(Value::as_bool)
        .unwrap_or(default)
}

/// Read an f64 field from `options.display`.
fn display_f64(opts: Option<&Value>, field: &str, default: f64) -> f64 {
    opts.and_then(|o| o.get("display"))
        .and_then(|d| d.get(field))
        .and_then(Value::as_f64)
        .unwrap_or(default)
}

/// Read a string field from `options.display.bonds.<bond_type>`.
fn bond_str<'a>(
    opts: Option<&'a Value>,
    bond_type: &str,
    field: &str,
    default: &'a str,
) -> &'a str {
    opts.and_then(|o| o.get("display"))
        .and_then(|d| d.get("bonds"))
        .and_then(|b| b.get(bond_type))
        .and_then(|bt| bt.get(field))
        .and_then(Value::as_str)
        .unwrap_or(default)
}

/// Read a bool field from `options.display.bonds.<bond_type>`.
fn bond_bool(
    opts: Option<&Value>,
    bond_type: &str,
    field: &str,
    default: bool,
) -> bool {
    opts.and_then(|o| o.get("display"))
        .and_then(|d| d.get("bonds"))
        .and_then(|b| b.get(bond_type))
        .and_then(|bt| bt.get(field))
        .and_then(Value::as_bool)
        .unwrap_or(default)
}

/// Global appearance card at the top of the Scene panel.
fn global_appearance_section(options: &Option<Value>) -> Element {
    let opts = options.as_ref();

    let drawing_mode = display_str(opts, "drawing_mode", "cartoon").to_owned();
    let color_scheme = display_str(opts, "color_scheme", "entity").to_owned();
    let show_sidechains = display_bool(opts, "show_sidechains", true);
    let surface_kind = display_str(opts, "surface_kind", "none").to_owned();
    let surface_opacity = display_f64(opts, "surface_opacity", 0.35);
    let show_cavities = display_bool(opts, "show_cavities", false);
    let helix_style = display_str(opts, "helix_style", "ribbon").to_owned();
    let sheet_style = display_str(opts, "sheet_style", "ribbon").to_owned();

    let show_hbonds = bond_bool(opts, "hydrogen_bonds", "visible", true);
    let hbond_style =
        bond_str(opts, "hydrogen_bonds", "style", "solid").to_owned();
    let show_disulfides = bond_bool(opts, "disulfide_bonds", "visible", true);
    let disulfide_style =
        bond_str(opts, "disulfide_bonds", "style", "solid").to_owned();

    rsx! {
        div { class: "entity-options global-appearance",
            div {
                style: "padding:4px 12px; font-weight:600; font-size:12px; \
                        color:#ccc; border-bottom:1px solid #444;",
                "Global Appearance"
            }
            {global_select(
                "Drawing Mode", "drawing_mode", &drawing_mode,
                &[("cartoon", "Cartoon"), ("stick", "Stick"),
                  ("thin_stick", "Thin Stick"), ("ball_and_stick", "Ball & Stick")],
            )}
            {global_select(
                "Color", "color_scheme", &color_scheme,
                &[("entity", "Entity"),
                  ("secondary_structure", "Secondary Structure"),
                  ("residue_index", "Residue Index"),
                  ("b_factor", "B-Factor"),
                  ("hydrophobicity", "Hydrophobicity"),
                  ("score", "Score"), ("score_relative", "Score (Rel)"),
                  ("solid", "Solid")],
            )}
            {global_select(
                "Surface", "surface_kind", &surface_kind,
                &[("none", "None"), ("gaussian", "Gaussian"), ("ses", "SES")],
            )}
            if surface_kind != "none" {
                {global_slider("Opacity", "surface_opacity", surface_opacity, 0.0, 1.0, 0.01)}
            }
            {global_toggle("Cavities", "show_cavities", show_cavities)}
            {global_select(
                "Helix Style", "helix_style", &helix_style,
                &[("ribbon", "Ribbon"), ("tube", "Tube"), ("cylinder", "Cylinder")],
            )}
            {global_select(
                "Sheet Style", "sheet_style", &sheet_style,
                &[("ribbon", "Ribbon"), ("tube", "Tube")],
            )}
            {global_toggle("Sidechains", "show_sidechains", show_sidechains)}

            // ── Structural bonds ──
            {bond_toggle("H-Bonds", "display.bonds.hydrogen_bonds", "visible", show_hbonds)}
            if show_hbonds {
                {bond_select(
                    "H-Bond Style", "display.bonds.hydrogen_bonds", "style", &hbond_style,
                    &[("solid", "Solid"), ("dashed", "Dashed"), ("stippled", "Stippled")],
                )}
            }
            {bond_toggle("Disulfides", "display.bonds.disulfide_bonds", "visible", show_disulfides)}
            if show_disulfides {
                {bond_select(
                    "Disulfide Style", "display.bonds.disulfide_bonds", "style", &disulfide_style,
                    &[("solid", "Solid"), ("dashed", "Dashed"), ("stippled", "Stippled")],
                )}
            }
        }
    }
}

/// Dropdown for a global display option field.
///
/// Sends `set_option("display", field, value)` — the same mechanism
/// used for all other options sections.
fn global_select(
    label: &str,
    field: &str,
    current: &str,
    choices: &[(&str, &str)],
) -> Element {
    let field = field.to_owned();
    let options: Vec<(String, String)> = choices
        .iter()
        .map(|(v, l)| ((*v).to_owned(), (*l).to_owned()))
        .collect();
    let current = current.to_owned();
    rsx! {
        div { class: "entity-option-row",
            label { class: "entity-option-label", "{label}" }
            select {
                class: "entity-option-select",
                value: "{current}",
                onchange: move |evt: Event<FormData>| {
                    bridge::send_set_option(
                        "display",
                        &field,
                        &Value::String(evt.value()),
                    );
                },
                for (value, display_label) in options.iter() {
                    option {
                        value: "{value}",
                        selected: *value == current,
                        "{display_label}"
                    }
                }
            }
        }
    }
}

/// Slider for a global display option field.
fn global_slider(
    label: &str,
    field: &str,
    value: f64,
    min: f64,
    max: f64,
    _step: f64,
) -> Element {
    let field_slider = field.to_owned();
    let field_input = field.to_owned();
    let val_str = format!("{value:.2}");
    let min_str = format!("{min:.2}");
    let max_str = format!("{max:.2}");
    rsx! {
        div { class: "entity-option-row",
            label { class: "entity-option-label", "{label}" }
            input {
                r#type: "range",
                style: "flex:1; min-width:60px;",
                min: "{min_str}",
                max: "{max_str}",
                step: "any",
                value: "{val_str}",
                oninput: move |evt: Event<FormData>| {
                    if let Ok(v) = evt.value().parse::<f64>() {
                        bridge::send_set_option(
                            "display",
                            &field_slider,
                            &serde_json::json!(v),
                        );
                    }
                },
            }
            input {
                r#type: "number",
                style: "width:60px; font-size:13px; padding:4px 6px; \
                        background:#333; color:#eee; border:1px solid #666; \
                        border-radius:3px;",
                min: "{min_str}",
                max: "{max_str}",
                step: "0.01",
                value: "{val_str}",
                onchange: move |evt: Event<FormData>| {
                    if let Ok(v) = evt.value().parse::<f64>() {
                        bridge::send_set_option(
                            "display",
                            &field_input,
                            &serde_json::json!(v),
                        );
                    }
                },
            }
        }
    }
}

/// Checkbox toggle for a global display option boolean field.
fn global_toggle(label: &str, field: &str, current: bool) -> Element {
    let field = field.to_owned();
    rsx! {
        div { class: "entity-option-row",
            label { class: "entity-option-label", "{label}" }
            input {
                r#type: "checkbox",
                checked: current,
                onchange: move |evt: Event<FormData>| {
                    let checked = evt.value() == "true";
                    bridge::send_set_option(
                        "display",
                        &field,
                        &Value::Bool(checked),
                    );
                },
            }
        }
    }
}

/// Checkbox toggle for a bond option (uses dotted path).
fn bond_toggle(label: &str, path: &str, field: &str, current: bool) -> Element {
    let path = path.to_owned();
    let field = field.to_owned();
    rsx! {
        div { class: "entity-option-row",
            label { class: "entity-option-label", "{label}" }
            input {
                r#type: "checkbox",
                checked: current,
                onchange: move |evt: Event<FormData>| {
                    let checked = evt.value() == "true";
                    bridge::send_set_option(
                        &path,
                        &field,
                        &Value::Bool(checked),
                    );
                },
            }
        }
    }
}

/// Dropdown for a bond option (uses dotted path).
fn bond_select(
    label: &str,
    path: &str,
    field: &str,
    current: &str,
    choices: &[(&str, &str)],
) -> Element {
    let path = path.to_owned();
    let field = field.to_owned();
    let options: Vec<(String, String)> = choices
        .iter()
        .map(|(v, l)| ((*v).to_owned(), (*l).to_owned()))
        .collect();
    let current = current.to_owned();
    rsx! {
        div { class: "entity-option-row",
            label { class: "entity-option-label", "{label}" }
            select {
                class: "entity-option-select",
                value: "{current}",
                onchange: move |evt: Event<FormData>| {
                    bridge::send_set_option(
                        &path,
                        &field,
                        &Value::String(evt.value()),
                    );
                },
                for (value, display_label) in options.iter() {
                    option {
                        value: "{value}",
                        selected: *value == current,
                        "{display_label}"
                    }
                }
            }
        }
    }
}

// ── Entity row ─────────────────────────────────────────────────────────────

/// Render a single entity row with optional expandable options.
fn entity_row(
    entity: &Value,
    mut expanded_ids: Signal<HashSet<u64>>,
) -> Element {
    let id = entity.get("id").and_then(Value::as_u64).unwrap_or(0);
    let mol_type = entity
        .get("molecule_type")
        .and_then(Value::as_str)
        .unwrap_or("Unknown");
    let visible = entity
        .get("visible")
        .and_then(Value::as_bool)
        .unwrap_or(true);
    let atom_count = entity
        .get("atom_count")
        .and_then(Value::as_u64)
        .unwrap_or(0);
    let focused = entity
        .get("focused")
        .and_then(Value::as_bool)
        .unwrap_or(false);
    let chain_ids = entity
        .get("chain_ids")
        .and_then(Value::as_array)
        .map(|arr| {
            arr.iter()
                .filter_map(Value::as_str)
                .collect::<Vec<_>>()
                .join(", ")
        })
        .unwrap_or_default();

    // Resolved effective values from entity_summaries.
    let drawing_mode = entity
        .get("drawing_mode")
        .and_then(Value::as_str)
        .unwrap_or("cartoon")
        .to_owned();
    let helix_style = entity
        .get("helix_style")
        .and_then(Value::as_str)
        .unwrap_or("ribbon")
        .to_owned();
    let sheet_style = entity
        .get("sheet_style")
        .and_then(Value::as_str)
        .unwrap_or("ribbon")
        .to_owned();
    let show_sidechains = entity
        .get("show_sidechains")
        .and_then(Value::as_bool)
        .unwrap_or(true);
    let color_scheme = entity
        .get("color_scheme")
        .and_then(Value::as_str)
        .unwrap_or("chain")
        .to_owned();
    let surface_kind = entity
        .get("surface")
        .and_then(Value::as_str)
        .unwrap_or("none")
        .to_owned();
    let has_overrides = entity
        .get("has_overrides")
        .and_then(Value::as_bool)
        .unwrap_or(false);

    // Check which fields have per-entity overrides (non-null in
    // appearance_overrides object).
    let ovr = entity.get("appearance_overrides");
    let has_drawing_mode_ovr = ovr
        .and_then(|o| o.get("drawing_mode"))
        .is_some_and(|v| !v.is_null());
    let has_color_ovr = ovr
        .and_then(|o| o.get("color_scheme"))
        .is_some_and(|v| !v.is_null());
    let has_helix_ovr = ovr
        .and_then(|o| o.get("helix_style"))
        .is_some_and(|v| !v.is_null());
    let has_sheet_ovr = ovr
        .and_then(|o| o.get("sheet_style"))
        .is_some_and(|v| !v.is_null());
    let has_sc_ovr = ovr
        .and_then(|o| o.get("show_sidechains"))
        .is_some_and(|v| !v.is_null());
    let has_surface_ovr = ovr
        .and_then(|o| o.get("surface_kind"))
        .is_some_and(|v| !v.is_null());

    let is_cartoon_capable = matches!(mol_type, "Protein" | "DNA" | "RNA");
    let is_protein = mol_type == "Protein";

    let is_expanded = expanded_ids.read().contains(&id);

    let type_class = match mol_type {
        "Protein" => "entity-type-protein",
        "DNA" => "entity-type-dna",
        "RNA" => "entity-type-rna",
        _ => "entity-type-ligand",
    };

    let row_class = if focused {
        format!("entity-row focused {type_class}")
    } else {
        format!("entity-row {type_class}")
    };

    let label = if chain_ids.is_empty() {
        mol_type.to_string()
    } else {
        format!("{mol_type} {chain_ids}")
    };

    let subtitle = format!("{atom_count} atoms");

    let opacity_class = if visible { "" } else { " entity-hidden" };

    rsx! {
        div {
            key: "{id}",
            class: "{row_class}{opacity_class}",
            div { class: "entity-type-indicator" }
            div {
                class: "entity-info",
                onclick: move |_| {
                    bridge::send_focus_entity(id);
                },
                div { class: "entity-label", "{label}" }
                div { class: "entity-subtitle", "{subtitle}" }
            }
            div { class: "entity-actions",
                // Expand/collapse toggle
                button {
                        class: "entity-action-btn entity-expand-btn",
                        title: if is_expanded { "Collapse options" } else { "Expand options" },
                        onclick: move |_| {
                            let mut ids = expanded_ids.write();
                            if ids.contains(&id) {
                                ids.remove(&id);
                            } else {
                                ids.insert(id);
                            }
                        },
                        svg {
                            width: "14",
                            height: "14",
                            view_box: "0 0 24 24",
                            fill: "none",
                            stroke: "currentColor",
                            stroke_width: "2",
                            stroke_linecap: "round",
                            stroke_linejoin: "round",
                            if is_expanded {
                                path { d: "M18 15l-6-6-6 6" }
                            } else {
                                path { d: "M6 9l6 6 6-6" }
                            }
                        }
                    }
                button {
                    class: "entity-action-btn",
                    title: if visible { "Hide" } else { "Show" },
                    onclick: move |_| {
                        bridge::send_toggle_entity_visibility(id);
                    },
                    svg {
                        width: "16",
                        height: "16",
                        view_box: "0 0 24 24",
                        fill: "none",
                        stroke: "currentColor",
                        stroke_width: "2",
                        stroke_linecap: "round",
                        stroke_linejoin: "round",
                        if visible {
                            path { d: "M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z" }
                            circle { cx: "12", cy: "12", r: "3" }
                        } else {
                            path { d: "M17.94 17.94A10.07 10.07 0 0 1 12 20c-7 0-11-8-11-8a18.45 18.45 0 0 1 5.06-5.94" }
                            path { d: "M9.9 4.24A9.12 9.12 0 0 1 12 4c7 0 11 8 11 8a18.5 18.5 0 0 1-2.16 3.19" }
                            line { x1: "1", y1: "1", x2: "23", y2: "23" }
                        }
                    }
                }
                button {
                    class: "entity-action-btn entity-remove-btn",
                    title: "Remove",
                    onclick: move |_| {
                        bridge::send_remove_entity(id);
                    },
                    svg {
                        width: "14",
                        height: "14",
                        view_box: "0 0 24 24",
                        fill: "none",
                        stroke: "currentColor",
                        stroke_width: "2",
                        stroke_linecap: "round",
                        stroke_linejoin: "round",
                        line { x1: "18", y1: "6", x2: "6", y2: "18" }
                        line { x1: "6", y1: "6", x2: "18", y2: "18" }
                    }
                }
            }
        }
        // Expanded entity options panel
        if is_expanded {
            div { class: "entity-options",
                // Drawing mode (with Default option for inheritance)
                if is_cartoon_capable {
                    {entity_appearance_select(
                        id, "Drawing Mode", "drawing_mode", &drawing_mode,
                        has_drawing_mode_ovr,
                        &[("cartoon", "Cartoon"), ("stick", "Stick"),
                          ("thin_stick", "Thin Stick"), ("ball_and_stick", "Ball & Stick")],
                    )}
                    {entity_appearance_select(
                        id, "Color", "color_scheme", &color_scheme,
                        has_color_ovr,
                        &[("entity", "Entity"),
                          ("secondary_structure", "Secondary Structure"),
                          ("residue_index", "Residue Index"), ("solid", "Solid")],
                    )}
                }
                // Surface dropdown
                {entity_appearance_select(
                    id, "Surface", "surface_kind", &surface_kind,
                    has_surface_ovr,
                    &[("none", "None"), ("gaussian", "Gaussian"), ("ses", "SES")],
                )}
                if surface_kind != "none" {
                    {entity_opacity_slider(id, entity)}
                }
                // Cartoon sub-options (helix/sheet/sidechains)
                if is_cartoon_capable && is_protein && drawing_mode == "cartoon" {
                    {entity_appearance_select(
                        id, "Helix Style", "helix_style", &helix_style,
                        has_helix_ovr,
                        &[("ribbon", "Ribbon"), ("tube", "Tube"), ("cylinder", "Cylinder")],
                    )}
                    {entity_appearance_select(
                        id, "Sheet Style", "sheet_style", &sheet_style,
                        has_sheet_ovr,
                        &[("ribbon", "Ribbon"), ("tube", "Tube")],
                    )}
                    {entity_appearance_toggle(
                        id, "Sidechains", "show_sidechains",
                        show_sidechains, has_sc_ovr,
                    )}
                }
                if has_overrides {
                    div { class: "entity-option-row",
                        button {
                            class: "entity-reset-btn",
                            onclick: move |_| {
                                bridge::send_clear_entity_option(id);
                            },
                            "Reset to defaults"
                        }
                    }
                }
            }
        }
    }
}

/// Dropdown for a per-entity appearance field with "Default" (inherit) option.
fn entity_appearance_select(
    id: u64,
    label: &str,
    field: &str,
    current: &str,
    has_override: bool,
    choices: &[(&str, &str)],
) -> Element {
    let field = field.to_owned();
    let options: Vec<(String, String)> = choices
        .iter()
        .map(|(v, l)| ((*v).to_owned(), (*l).to_owned()))
        .collect();
    // If there's no per-entity override, show "Default" as selected;
    // otherwise show the effective value.
    let selected_value = if has_override {
        current.to_owned()
    } else {
        String::new()
    };
    let current_display = current.to_owned();
    rsx! {
        div { class: "entity-option-row",
            label { class: "entity-option-label", "{label}" }
            select {
                class: "entity-option-select",
                value: "{selected_value}",
                onchange: move |evt: Event<FormData>| {
                    let val = evt.value();
                    if val.is_empty() {
                        // "Default" → clear override
                        bridge::send_set_entity_appearance(
                            id,
                            &field,
                            &Value::Null,
                        );
                    } else {
                        bridge::send_set_entity_appearance(
                            id,
                            &field,
                            &Value::String(val),
                        );
                    }
                },
                option {
                    value: "",
                    selected: !has_override,
                    "Default"
                }
                for (value, display_label) in options.iter() {
                    option {
                        value: "{value}",
                        selected: has_override && *value == current_display,
                        "{display_label}"
                    }
                }
            }
        }
    }
}

/// Per-entity opacity slider.
fn entity_opacity_slider(id: u64, entity: &Value) -> Element {
    let value = entity
        .get("surface_color")
        .and_then(Value::as_array)
        .and_then(|a| a.get(3))
        .and_then(Value::as_f64)
        .unwrap_or(0.35);
    let val_str = format!("{value:.2}");
    rsx! {
        div { class: "entity-option-row",
            label { class: "entity-option-label", "Opacity" }
            input {
                r#type: "range",
                style: "flex:1; min-width:60px;",
                min: "0",
                max: "1",
                step: "any",
                value: "{val_str}",
                oninput: move |evt: Event<FormData>| {
                    if let Ok(v) = evt.value().parse::<f64>() {
                        bridge::send_set_entity_appearance(
                            id,
                            "surface_opacity",
                            &serde_json::json!(v),
                        );
                    }
                },
            }
            input {
                r#type: "number",
                style: "width:60px; font-size:13px; padding:4px 6px; \
                        background:#333; color:#eee; border:1px solid #666; \
                        border-radius:3px;",
                min: "0",
                max: "1",
                step: "0.01",
                value: "{val_str}",
                onchange: move |evt: Event<FormData>| {
                    if let Ok(v) = evt.value().parse::<f64>() {
                        bridge::send_set_entity_appearance(
                            id,
                            "surface_opacity",
                            &serde_json::json!(v),
                        );
                    }
                },
            }
        }
    }
}

/// Per-entity boolean toggle with Default/On/Off.
fn entity_appearance_toggle(
    id: u64,
    label: &str,
    field: &str,
    current: bool,
    has_override: bool,
) -> Element {
    let field = field.to_owned();
    let display_val = if !has_override {
        "default"
    } else if current {
        "on"
    } else {
        "off"
    };
    rsx! {
        div { class: "entity-option-row",
            label { class: "entity-option-label", "{label}" }
            select {
                class: "entity-option-select",
                value: "{display_val}",
                onchange: move |evt: Event<FormData>| {
                    let val = evt.value();
                    match val.as_str() {
                        "on" => bridge::send_set_entity_appearance(
                            id,
                            &field,
                            &Value::Bool(true),
                        ),
                        "off" => bridge::send_set_entity_appearance(
                            id,
                            &field,
                            &Value::Bool(false),
                        ),
                        _ => bridge::send_set_entity_appearance(
                            id,
                            &field,
                            &Value::Null,
                        ),
                    }
                },
                option { value: "default", selected: !has_override, "Default" }
                option { value: "on", selected: has_override && current, "On" }
                option { value: "off", selected: has_override && !current, "Off" }
            }
        }
    }
}

// ── Density ────────────────────────────────────────────────────────────────

/// Render a single density map row with inline controls.
fn density_row(map_val: &Value) -> Element {
    let id = map_val.get("id").and_then(Value::as_u64).unwrap_or(0);
    let visible = map_val
        .get("visible")
        .and_then(Value::as_bool)
        .unwrap_or(true);
    let threshold = map_val
        .get("threshold")
        .and_then(Value::as_f64)
        .unwrap_or(0.0);
    let dmax = map_val.get("dmax").and_then(Value::as_f64).unwrap_or(1.0);
    let color = map_val.get("color").and_then(Value::as_array);
    let (cr, cg, cb) = color.map_or((0.3, 0.5, 0.8), |arr| {
        (
            arr.first().and_then(Value::as_f64).unwrap_or(0.3),
            arr.get(1).and_then(Value::as_f64).unwrap_or(0.5),
            arr.get(2).and_then(Value::as_f64).unwrap_or(0.8),
        )
    });
    let opacity = map_val
        .get("opacity")
        .and_then(Value::as_f64)
        .unwrap_or(0.35);

    let opacity_class = if visible { "" } else { " entity-hidden" };
    // Slider range: 0 to dmax (no negatives)
    let slider_max = dmax.max(0.01);
    let step = (slider_max / 100.0).max(0.01);

    rsx! {
        div {
            key: "density-{id}",
            class: "entity-row entity-type-ligand{opacity_class}",
            div { class: "entity-type-indicator" }
            div { class: "entity-info",
                div { class: "entity-label", "Density {id}" }
            }
            div { class: "entity-actions",
                button {
                    class: "entity-action-btn",
                    title: if visible { "Hide" } else { "Show" },
                    onclick: move |_| {
                        bridge::send_toggle_density_visibility(id);
                    },
                    svg {
                        width: "16",
                        height: "16",
                        view_box: "0 0 24 24",
                        fill: "none",
                        stroke: "currentColor",
                        stroke_width: "2",
                        stroke_linecap: "round",
                        stroke_linejoin: "round",
                        if visible {
                            path { d: "M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z" }
                            circle { cx: "12", cy: "12", r: "3" }
                        } else {
                            path { d: "M17.94 17.94A10.07 10.07 0 0 1 12 20c-7 0-11-8-11-8a18.45 18.45 0 0 1 5.06-5.94" }
                            path { d: "M9.9 4.24A9.12 9.12 0 0 1 12 4c7 0 11 8 11 8a18.5 18.5 0 0 1-2.16 3.19" }
                            line { x1: "1", y1: "1", x2: "23", y2: "23" }
                        }
                    }
                }
                button {
                    class: "entity-action-btn entity-remove-btn",
                    title: "Remove",
                    onclick: move |_| {
                        bridge::send_remove_density_map(id);
                    },
                    svg {
                        width: "14",
                        height: "14",
                        view_box: "0 0 24 24",
                        fill: "none",
                        stroke: "currentColor",
                        stroke_width: "2",
                        stroke_linecap: "round",
                        stroke_linejoin: "round",
                        line { x1: "18", y1: "6", x2: "6", y2: "18" }
                        line { x1: "6", y1: "6", x2: "18", y2: "18" }
                    }
                }
            }
        }
        // Density controls (always expanded)
        div { class: "entity-options",
            {density_slider_row(id, "Thresh", "threshold", threshold, 0.0, slider_max, step)}
            {density_slider_row(id, "R", "color_r", cr, 0.0, 1.0, 0.01)}
            {density_slider_row(id, "G", "color_g", cg, 0.0, 1.0, 0.01)}
            {density_slider_row(id, "B", "color_b", cb, 0.0, 1.0, 0.01)}
            {density_slider_row(id, "A", "opacity", opacity, 0.0, 1.0, 0.01)}
        }
    }
}

/// Density control row: label + range slider + number input.
fn density_slider_row(
    id: u64,
    label: &str,
    field: &str,
    value: f64,
    min: f64,
    max: f64,
    _step: f64,
) -> Element {
    let field_slider = field.to_owned();
    let field_input = field.to_owned();
    let val_str = format!("{value:.2}");
    let min_str = format!("{min:.2}");
    let max_str = format!("{max:.2}");
    rsx! {
        div {
            style: "display:flex; align-items:center; gap:6px; padding:2px 12px;",
            label {
                style: "font-size:11px; color:#999; min-width:40px; text-align:right;",
                "{label}"
            }
            input {
                r#type: "range",
                style: "flex:1; min-width:60px;",
                min: "{min_str}",
                max: "{max_str}",
                step: "any",
                value: "{val_str}",
                onchange: move |evt: Event<FormData>| {
                    if let Ok(v) = evt.value().parse::<f64>() {
                        bridge::send_set_density_option(
                            id,
                            &field_slider,
                            &serde_json::json!(v),
                        );
                    }
                },
            }
            input {
                r#type: "number",
                style: "width:60px; font-size:13px; padding:4px 6px; \
                        background:#333; color:#eee; border:1px solid #666; \
                        border-radius:3px;",
                min: "{min_str}",
                max: "{max_str}",
                step: "0.01",
                value: "{val_str}",
                onchange: move |evt: Event<FormData>| {
                    if let Ok(v) = evt.value().parse::<f64>() {
                        bridge::send_set_density_option(
                            id,
                            &field_input,
                            &serde_json::json!(v),
                        );
                    }
                },
            }
        }
    }
}
