//! Wry webview child of the winit window.
//!
//! Creates a [`wry::WebView`] positioned at the right edge of the window,
//! loads the viso-ui WASM bundle via a custom `viso://` protocol, and
//! bridges IPC between the Dioxus web app and the native engine.

use std::borrow::Cow;
use std::sync::mpsc;

use rust_embed::RustEmbed;
use wry::http::header::CONTENT_TYPE;
use wry::http::Response;
use wry::{dpi, Rect, WebView, WebViewBuilder};

use crate::bridge::{self, PanelAxis, UiAction};
use crate::options::VisoOptions;

/// Embedded viso-ui dist output (built by `trunk build`).
#[derive(RustEmbed)]
#[folder = "crates/viso-ui/dist/"]
struct UiAssets;

/// Default panel size in physical pixels (re-exported from bridge).
pub const PANEL_WIDTH: u32 = bridge::DEFAULT_PANEL_SIZE;

/// Create the wry webview as a child of the given window.
///
/// Returns `(webview, action_rx)` — the receiver yields [`UiAction`]s
/// from the WASM app.
///
/// # Errors
///
/// Returns [`wry::Error`] if the webview fails to build.
pub fn create_webview<W: wry::raw_window_handle::HasWindowHandle>(
    window: &W,
    window_width: u32,
    window_height: u32,
    panel_width: u32,
) -> Result<(WebView, mpsc::Receiver<UiAction>), wry::Error> {
    // If the embedded dist is just the placeholder (no WASM files), skip the
    // webview entirely so the user doesn't see a blank white panel.
    let has_wasm = UiAssets::iter().any(|name| name.ends_with(".wasm"));
    if !has_wasm {
        log::warn!(
            "viso-ui was not built (no .wasm in embedded assets). Run `trunk \
             build` in crates/viso-ui/ to enable the GUI panel."
        );
        return Err(wry::Error::MessageSender);
    }

    let (tx, rx) = mpsc::channel();

    let axis = PanelAxis::from_dimensions(window_width, window_height);
    let bounds = panel_bounds(window_width, window_height, panel_width, axis);

    let init_js = format!("{}{}", bridge::BRIDGE_JS, BRIDGE_JS_NATIVE_ADDENDUM);

    let webview = WebViewBuilder::new()
        .with_bounds(bounds)
        .with_transparent(true)
        .with_custom_protocol("viso".into(), |_id, request| {
            let path = request.uri().path();
            // Default to index.html for the root path.
            let path = if path == "/" {
                "index.html"
            } else {
                &path[1..]
            };

            match UiAssets::get(path) {
                Some(asset) => {
                    let mime = mime_guess::from_path(path)
                        .first_or_octet_stream()
                        .to_string();
                    Response::builder()
                        .header(CONTENT_TYPE, mime)
                        .body(Cow::from(asset.data.to_vec()))
                        .unwrap_or_else(|_| {
                            Response::new(Cow::from(Vec::new()))
                        })
                }
                None => Response::builder()
                    .status(404)
                    .body(Cow::from(Vec::new()))
                    .unwrap_or_else(|_| Response::new(Cow::from(Vec::new()))),
            }
        })
        .with_url(dev_or_embedded_url())
        .with_initialization_script(&init_js)
        .with_ipc_handler(move |req| {
            let body = req.body();
            if let Ok(msg) = serde_json::from_str::<serde_json::Value>(body) {
                if let Some(action) = bridge::parse_action(&msg) {
                    let _ = tx.send(action);
                }
            }
        })
        .build_as_child(window)?;

    Ok((webview, rx))
}

/// Compute the [`Rect`] for the pinned panel along the given axis.
#[must_use]
pub fn panel_bounds(
    window_width: u32,
    window_height: u32,
    panel_size: u32,
    axis: PanelAxis,
) -> Rect {
    match axis {
        PanelAxis::Right => {
            let x = window_width.saturating_sub(panel_size);
            Rect {
                position: dpi::Position::Physical(dpi::PhysicalPosition::new(
                    x as i32, 0,
                )),
                size: dpi::Size::Physical(dpi::PhysicalSize::new(
                    panel_size.min(window_width),
                    window_height,
                )),
            }
        }
        PanelAxis::Bottom => {
            // macOS NSView coordinates: origin at bottom-left, y-up.
            // y=0 places the panel at the bottom of the window.
            Rect {
                position: dpi::Position::Physical(dpi::PhysicalPosition::new(
                    0, 0,
                )),
                size: dpi::Size::Physical(dpi::PhysicalSize::new(
                    window_width,
                    panel_size.min(window_height),
                )),
            }
        }
    }
}

/// Compute the [`Rect`] for the floating (unpinned) panel, inset by
/// `margin` on all sides, along the given axis.
#[must_use]
pub fn panel_bounds_floating(
    window_width: u32,
    window_height: u32,
    panel_size: u32,
    margin: u32,
    axis: PanelAxis,
) -> Rect {
    match axis {
        PanelAxis::Right => {
            let x = window_width.saturating_sub(panel_size + margin);
            Rect {
                position: dpi::Position::Physical(dpi::PhysicalPosition::new(
                    x as i32,
                    margin as i32,
                )),
                size: dpi::Size::Physical(dpi::PhysicalSize::new(
                    panel_size.min(window_width),
                    window_height.saturating_sub(margin * 2),
                )),
            }
        }
        PanelAxis::Bottom => {
            let y = window_height.saturating_sub(panel_size + margin);
            Rect {
                position: dpi::Position::Physical(dpi::PhysicalPosition::new(
                    margin as i32,
                    y as i32,
                )),
                size: dpi::Size::Physical(dpi::PhysicalSize::new(
                    window_width.saturating_sub(margin * 2),
                    panel_size.min(window_height),
                )),
            }
        }
    }
}

/// Push the current orientation to the webview.
pub fn push_orientation(webview: &WebView, axis: PanelAxis) {
    safe_push(webview, "orientation", axis.orientation_str());
}

/// Push the Options JSON schema to the webview (call once after creation).
pub fn push_schema(webview: &WebView, options: &VisoOptions) {
    let schema = schemars::schema_for!(VisoOptions);
    let json = serde_json::to_string(&schema).unwrap_or_default();
    safe_push(webview, "schema", &bridge::escape_for_js(&json));

    push_options(webview, options);
}

/// Push the current Options state to the webview.
pub fn push_options(webview: &WebView, options: &VisoOptions) {
    let json = serde_json::to_string(options).unwrap_or_default();
    safe_push(webview, "options", &bridge::escape_for_js(&json));
}

/// Push stats (FPS + GPU buffer sizes) to the webview.
pub fn push_stats(
    webview: &WebView,
    fps: f32,
    buffers: &[(&str, usize, usize)],
) {
    let buffer_list: Vec<serde_json::Value> = buffers
        .iter()
        .map(|(name, used, alloc)| {
            serde_json::json!({
                "name": name,
                "used": used,
                "allocated": alloc,
            })
        })
        .collect();
    let json = serde_json::json!({ "fps": fps, "buffers": buffer_list });
    safe_push(webview, "stats", &bridge::escape_for_js(&json.to_string()));
}

/// Push the scene entity list to the webview.
pub fn push_scene_entities(webview: &WebView, entities_json: &str) {
    safe_push(
        webview,
        "scene_entities",
        &bridge::escape_for_js(entities_json),
    );
}

/// Push the density map list to the webview.
pub fn push_density_maps(webview: &WebView, density_json: &str) {
    safe_push(
        webview,
        "density_maps",
        &bridge::escape_for_js(density_json),
    );
}

/// Push a load-status event to the webview.
///
/// `status` is one of `"loading"`, `"loaded"`, or `"error"`.
/// `message` is a human-readable description.
pub fn push_load_status(webview: &WebView, status: &str, message: &str) {
    let json = serde_json::json!({ "status": status, "message": message });
    safe_push(
        webview,
        "load_status",
        &bridge::escape_for_js(&json.to_string()),
    );
}

/// Push the current panel size (CSS pixels) to the webview so the
/// resize-drag logic uses the host's authoritative value.
pub fn push_panel_size_css(webview: &WebView, size_css: u32) {
    safe_push(webview, "panel_size", &format!("{size_css}"));
}

/// Push the panel pinned state to the webview.
pub fn push_panel_pinned(webview: &WebView, pinned: bool) {
    let val = if pinned { "true" } else { "false" };
    safe_push(webview, "panel_pinned", val);
}

/// Push a raw JSON string to viso-ui under the given key.
///
/// Wrapper that escapes for JS embedding and routes through the
/// internal `safe_push`. Used by the platform-agnostic dispatcher via
/// the in-crate `UiHost` trait.
pub(crate) fn push_raw(webview: &WebView, key: &str, json: &str) {
    safe_push(webview, key, &bridge::escape_for_js(json));
}

/// Call `window.__viso_push_{key}(value)`, buffering on
/// `window.__viso_early` if the bridge script hasn't loaded yet.
///
/// On Windows (WebView2), `evaluate_script` can fire before the
/// initialization script has run.  This wrapper stores the value so
/// the bridge script can replay it on init.
fn safe_push(webview: &WebView, key: &str, escaped_value: &str) {
    let js = format!(
        "if(window.__viso_push_{key}){{window.__viso_push_{key}('\
         {escaped_value}')}}else{{window.__viso_early=window.\
         __viso_early||{{}};window.__viso_early.{key}='{escaped_value}'}}"
    );
    let _ = webview.evaluate_script(&js);
}

// Internals

/// Native-specific addendum to the shared bridge JavaScript.
///
/// Handles Tab key forwarding (WebView2 intercepts Tab for its own focus
/// navigation) and replays pending data when the WASM app adds
/// `addEventListener` calls late.
const BRIDGE_JS_NATIVE_ADDENDUM: &str = r"
(function() {
    // Forward Tab to the native side so the engine can cycle focus.
    document.addEventListener('keydown', function(e) {
        if (e.key === 'Tab') {
            e.preventDefault();
            window.ipc.postMessage(JSON.stringify({ action: 'key', key: 'Tab' }));
        }
    });

    // When the WASM app adds a listener, replay buffered data.
    var origAdd = EventTarget.prototype.addEventListener;
    EventTarget.prototype.addEventListener = function(type, fn, opts) {
        origAdd.call(this, type, fn, opts);
        if (this === window && type.startsWith('viso-') && window.__viso_replay_pending) {
            window.__viso_replay_pending();
        }
    };
})();
";

/// Use the trunk dev server when `VISO_UI_DEV` is set, otherwise load
/// from the embedded assets via the custom protocol.
///
/// For hot-reload during UI development, run `trunk serve` in
/// `crates/viso-ui/` and launch viso with `VISO_UI_DEV=1 cargo run`.
fn dev_or_embedded_url() -> &'static str {
    if std::env::var("VISO_UI_DEV").is_ok() {
        log::info!("VISO_UI_DEV set — loading UI from trunk dev server");
        "http://localhost:8080/"
    } else {
        "viso://localhost/"
    }
}
