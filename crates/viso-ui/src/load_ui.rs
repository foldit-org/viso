//! Load panel UI for fetching PDB structures and opening local files.

use dioxus::prelude::*;
use serde_json::Value;

use crate::bridge;

/// Load panel: PDB fetch form + local file browser + status line.
#[component]
pub fn LoadPanel(load_status: Signal<Option<Value>>) -> Element {
    let mut pdb_id = use_signal(|| String::new());
    let mut source = use_signal(|| "rcsb".to_string());

    let status = load_status.read();
    let is_loading = status
        .as_ref()
        .and_then(|v| v.get("status"))
        .and_then(Value::as_str)
        == Some("loading");

    rsx! {
        div { class: "load-panel",
            // ── Fetch Structure ──
            div { class: "load-section",
                div { class: "load-section-title", "Fetch Structure" }
                div { class: "field-row",
                    label { class: "field-label", "PDB ID" }
                    div { class: "fetch-row",
                        input {
                            r#type: "text",
                            class: "pdb-input",
                            placeholder: "e.g. 1crn",
                            maxlength: "4",
                            value: "{pdb_id}",
                            oninput: move |evt: Event<FormData>| {
                                pdb_id.set(evt.value().to_string());
                            },
                        }
                        button {
                            class: "fetch-btn",
                            disabled: is_loading
                                || pdb_id.read().trim().is_empty(),
                            onclick: move |_| {
                                let id = pdb_id.read().clone();
                                let src = source.read().clone();
                                bridge::send_fetch_pdb(&id, &src);
                            },
                            if is_loading { "Fetching..." } else { "Fetch" }
                        }
                    }
                }
                div { class: "field-row",
                    label { class: "field-label", "Source" }
                    div { class: "source-radio",
                        label {
                            input {
                                r#type: "radio",
                                name: "pdb-source",
                                value: "rcsb",
                                checked: *source.read() == "rcsb",
                                onchange: move |_| source.set("rcsb".into()),
                            }
                            " RCSB"
                        }
                        label {
                            input {
                                r#type: "radio",
                                name: "pdb-source",
                                value: "pdb-redo",
                                checked: *source.read() == "pdb-redo",
                                onchange: move |_| source.set("pdb-redo".into()),
                            }
                            " PDB-REDO"
                        }
                    }
                }
            }

            // ── Open Local File ──
            div { class: "load-section",
                div { class: "load-section-title", "Open Local File" }
                button {
                    class: "browse-btn",
                    disabled: is_loading,
                    onclick: move |_| {
                        bridge::send_open_file_dialog();
                    },
                    "Browse..."
                }
                div { class: "field-label", style: "margin-top: 6px;",
                    "Accepts .cif, .pdb, .cube (color LUT)"
                }
            }

            // ── Status line ──
            {render_status(&status)}
        }
    }
}

/// Render the status message line.
fn render_status(status: &Option<Value>) -> Element {
    let Some(val) = status else {
        return rsx! {};
    };

    let status_str = val.get("status").and_then(Value::as_str).unwrap_or("");
    let message = val.get("message").and_then(Value::as_str).unwrap_or("");

    if message.is_empty() {
        return rsx! {};
    }

    let class = match status_str {
        "loading" => "load-status loading",
        "loaded" => "load-status loaded",
        "error" => "load-status error",
        _ => "load-status",
    };

    rsx! {
        div { class: "{class}", "{message}" }
    }
}
