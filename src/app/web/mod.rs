//! WASM entry point for running Viso in a browser via WebGPU.
//!
//! Sets up the engine, wires the viso-ui IPC bridge (same protocol as the
//! native wry path), and runs a `requestAnimationFrame` render loop.

mod input;

use std::cell::RefCell;
use std::rc::Rc;

use wasm_bindgen::prelude::*;
use wasm_bindgen::JsCast;
pub use wasm_bindgen_rayon::init_thread_pool;
use web_sys::HtmlCanvasElement;

use crate::app::{SelectionStore, VisoApp};
use crate::bridge::dispatch::{self, UiHost};
use crate::bridge::{self, PanelAxis, UiAction};
use crate::gpu::RenderContext;
use crate::input::KeyBindings;
use crate::options::VisoOptions;
use crate::VisoEngine;

/// [`UiHost`] adapter that funnels dispatcher pushes through the
/// iframe `eval` transport already used by [`push_to_ui`].
struct WebHost;

impl UiHost for WebHost {
    fn push(&self, key: &str, json: &str) {
        push_to_ui(key, json);
    }
}

// ---------------------------------------------------------------------------
// Shared handles
// ---------------------------------------------------------------------------

/// Thread-local engine state shared between the rAF loop, input
/// listeners, and the IPC bridge.
type EngineHandle = Rc<RefCell<VisoEngine>>;

/// Thread-local handle to the standalone-host [`VisoApp`]. The web
/// build plays the same host role the native GUI does — owning
/// `Assembly` + publisher and exposing the mutation surface.
type AppHandle = Rc<RefCell<VisoApp>>;

/// Thread-local handle to the consumer-side selection mirror that
/// receives click events from [`VisoEngine::feed_pointer_button`]
/// and flushes the result back to the engine's flat residue space.
type SelectionHandle = Rc<RefCell<SelectionStore>>;

// ---------------------------------------------------------------------------
// Initialization
// ---------------------------------------------------------------------------

/// Initialize the WASM environment: console logging and panic hook.
///
/// Call this once before [`start`].
#[wasm_bindgen]
pub fn init() {
    console_error_panic_hook::set_once();
    let _ = console_log::init_with_level(log::Level::Info);
}

// ---------------------------------------------------------------------------
// Engine start
// ---------------------------------------------------------------------------

/// Create and start a `VisoEngine` rendering to the given canvas.
///
/// This also installs the IPC bridge so that viso-ui (running in the same
/// page or an iframe) can communicate with the engine via the same
/// `window.ipc.postMessage` / `window.__viso_push_*` protocol used
/// natively.
///
/// # Errors
///
/// Returns a `JsValue` error string if engine initialization fails.
#[wasm_bindgen]
pub async fn start(canvas: HtmlCanvasElement) -> Result<(), JsValue> {
    let dpr = web_sys::window()
        .map(|w| w.device_pixel_ratio())
        .unwrap_or(1.0);

    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let width = (f64::from(canvas.client_width()) * dpr) as u32;
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let height = (f64::from(canvas.client_height()) * dpr) as u32;
    let width = width.max(1);
    let height = height.max(1);

    // Set the canvas drawing buffer to match the CSS layout size *before*
    // creating the wgpu surface.  Without this the buffer stays at the HTML
    // default (often 300×150) and the surface texture is mismatched — giving
    // a blank canvas until a resize event forces synchronization.
    canvas.set_width(width);
    canvas.set_height(height);

    let surface_target = wgpu::SurfaceTarget::Canvas(canvas.clone());
    let mut context = RenderContext::new(surface_target, (width, height))
        .await
        .map_err(|e| JsValue::from_str(&format!("GPU init failed: {e}")))?;

    // 2x supersampling on low-DPI displays, matching native viewer behavior
    if dpr < 2.0 {
        context.render_scale = 2;
    }

    let app = VisoApp::new_empty();
    let mut engine = VisoEngine::new(context, VisoOptions::default())
        .map_err(|e| JsValue::from_str(&format!("Engine init failed: {e}")))?;
    app.publish(&mut engine);

    let engine = Rc::new(RefCell::new(engine));
    let app = Rc::new(RefCell::new(app));
    let keybindings = Rc::new(KeyBindings::default());
    let selection = Rc::new(RefCell::new(SelectionStore::new()));

    // Track panel axis and push orientation on init.  On resize, if the
    // axis changed, push the new orientation and re-apply layout.
    let panel_axis = Rc::new(RefCell::new(current_axis()));
    let panel_collapsed = Rc::new(RefCell::new(false));
    let panel_size = Rc::new(RefCell::new(bridge::DEFAULT_PANEL_SIZE));

    // Install the IPC bridge so viso-ui can talk to us.
    // The bridge must be installed on the iframe's contentWindow (not the
    // parent) because viso-ui runs inside the iframe and has its own
    // window object.  We defer the initial push until the iframe loads.
    install_ipc_bridge(
        Rc::clone(&app),
        Rc::clone(&engine),
        Rc::clone(&panel_axis),
        Rc::clone(&panel_collapsed),
        Rc::clone(&panel_size),
    );

    push_to_ui("orientation", panel_axis.borrow().orientation_str());
    push_to_ui("panel_size", &format!("{}", bridge::DEFAULT_PANEL_SIZE));
    {
        let axis = Rc::clone(&panel_axis);
        let collapsed = Rc::clone(&panel_collapsed);
        let size = Rc::clone(&panel_size);
        let cb = Closure::<dyn FnMut()>::new(move || {
            let new_axis = current_axis();
            let mut cur = axis.borrow_mut();
            if new_axis != *cur {
                *cur = new_axis;
                push_to_ui("orientation", new_axis.orientation_str());
                apply_web_layout(new_axis, *collapsed.borrow(), *size.borrow());
            }
        });
        let _ = web_sys::window()
            .expect("no window")
            .add_event_listener_with_callback(
                "resize",
                cb.as_ref().unchecked_ref(),
            );
        cb.forget();
    }

    // Wire up canvas input and resize handling
    input::attach_input_listeners(
        &canvas,
        Rc::clone(&engine),
        Rc::clone(&keybindings),
        Rc::clone(&selection),
    );
    input::attach_resize_observer(&canvas, Rc::clone(&engine));

    // Start the render loop
    request_animation_frame_loop(Rc::clone(&engine), canvas);

    Ok(())
}

/// Load a structure from bytes into the running engine.
///
/// Called from JS after fetching a file (e.g. from RCSB).
#[wasm_bindgen]
pub fn load_structure(
    engine_holder: &JsValue,
    bytes: &[u8],
    format_hint: &str,
) -> Result<(), JsValue> {
    // This is called from JS via the IPC bridge, not directly.
    // The actual implementation is in handle_ipc_action.
    let _ = (engine_holder, bytes, format_hint);
    Err(JsValue::from_str(
        "Use the viso-ui load panel or call window.__viso_load_bytes()",
    ))
}

// ---------------------------------------------------------------------------
// IPC bridge (viso-ui ↔ engine)
// ---------------------------------------------------------------------------

/// Install the IPC bridge between the parent page (engine) and the
/// viso-ui iframe.
///
/// - Creates the `ipc.postMessage` handler that routes actions into the engine.
/// - Waits for the `ui-panel` iframe to load, then injects `ipc` and the
///   `BRIDGE_JS` push functions into the iframe's `contentWindow` so that
///   viso-ui (which runs inside the iframe) can send/receive messages.
/// - Also installs `__viso_load_bytes` on the parent window for direct
///   byte-loading from JS.
/// Shared panel state for the web host.
#[derive(Clone)]
struct WebPanelState {
    axis: Rc<RefCell<PanelAxis>>,
    collapsed: Rc<RefCell<bool>>,
    size: Rc<RefCell<u32>>,
}

fn install_ipc_bridge(
    app: AppHandle,
    engine: EngineHandle,
    axis: Rc<RefCell<PanelAxis>>,
    collapsed: Rc<RefCell<bool>>,
    size: Rc<RefCell<u32>>,
) {
    log::info!("[bridge] install_ipc_bridge: starting");
    let window = web_sys::window().expect("no global window");
    let panel = WebPanelState {
        axis,
        collapsed,
        size,
    };

    // -- Build the ipc object with a postMessage handler --
    let eng = Rc::clone(&engine);
    let app_for_msg = Rc::clone(&app);
    let p = panel.clone();
    let on_message = Closure::<dyn FnMut(String)>::new(move |json: String| {
        log::info!("[bridge] ipc.postMessage received: {json}");
        handle_ipc_action(&app_for_msg, &eng, &p, &json);
    });

    let ipc = js_sys::Object::new();
    let _ = js_sys::Reflect::set(
        &ipc,
        &JsValue::from_str("postMessage"),
        on_message.as_ref(),
    );
    on_message.forget();

    // Install ipc on the parent window (for __viso_load_bytes and
    // direct JS callers).
    let _ = js_sys::Reflect::set(&window, &JsValue::from_str("ipc"), &ipc);
    log::info!("[bridge] ipc.postMessage installed on parent window");

    // -- window.__viso_load_bytes(bytes, formatHint) --
    let eng = Rc::clone(&engine);
    let app_for_load = Rc::clone(&app);
    let load_bytes = Closure::<dyn FnMut(Vec<u8>, String)>::new(
        move |bytes: Vec<u8>, hint: String| match bridge::parse_file_bytes(
            &bytes, &hint,
        ) {
            Ok(bridge::ParsedFile::Structure(entities)) => {
                let mut a = app_for_load.borrow_mut();
                let mut e = eng.borrow_mut();
                let _ids = a.replace_scene(&mut e, entities);
                push_load_status("loaded", "Structure loaded");
                push_scene_entities(&e);
            }
            Ok(bridge::ParsedFile::Density(map)) => {
                let mut e = eng.borrow_mut();
                let _id = e.density_mut().load(map);
                push_load_status("loaded", "Density map loaded");
            }
            Err(msg) => {
                log::error!("load failed: {msg}");
                push_load_status("error", &msg);
            }
        },
    );
    let _ = js_sys::Reflect::set(
        &window,
        &JsValue::from_str("__viso_load_bytes"),
        load_bytes.as_ref(),
    );
    load_bytes.forget();
    log::info!("[bridge] __viso_load_bytes installed on parent window");

    // -- Wire up the iframe once it loads --
    let eng_for_iframe = Rc::clone(&engine);
    let ipc_ref = js_sys::Reflect::get(&window, &JsValue::from_str("ipc"))
        .expect("ipc not on window");
    let on_load = Closure::<dyn FnMut()>::new(move || {
        log::info!("[bridge] iframe 'load' event fired");
        setup_iframe_bridge(&ipc_ref, &eng_for_iframe);
    });

    let document = window.document().expect("no document");
    if let Some(iframe) = document.get_element_by_id("ui-panel") {
        log::info!("[bridge] found ui-panel iframe element");
        let iframe: web_sys::HtmlIFrameElement = iframe.unchecked_into();

        // Attach load listener for future loads / reloads.
        let _ = iframe.add_event_listener_with_callback(
            "load",
            on_load.as_ref().unchecked_ref(),
        );
        log::info!("[bridge] attached 'load' listener to iframe");

        // If the iframe already loaded (likely — it's a small HTML page
        // while the WASM took a while), set up immediately.
        if iframe.content_window().is_some() {
            log::info!("[bridge] iframe already loaded, setting up bridge now");
            let ipc2 = js_sys::Reflect::get(
                &web_sys::window().expect("window"),
                &JsValue::from_str("ipc"),
            )
            .expect("ipc");
            setup_iframe_bridge(&ipc2, &engine);
        }
    } else {
        log::warn!("[bridge] ui-panel iframe element NOT found in DOM");
    }
    on_load.forget();
    log::info!("[bridge] install_ipc_bridge: done");
}

/// Inject the IPC bridge into the viso-ui iframe's contentWindow.
fn setup_iframe_bridge(ipc: &JsValue, engine: &EngineHandle) {
    log::info!("[bridge] setup_iframe_bridge: starting");
    let Some(iframe_win) = ui_iframe_window() else {
        log::warn!(
            "[bridge] setup_iframe_bridge: iframe contentWindow NOT accessible"
        );
        return;
    };
    log::info!("[bridge] setup_iframe_bridge: got iframe contentWindow");

    // Install `window.ipc` on the iframe so bridge.rs can call
    // `window.ipc.postMessage(json)`.
    let _ = js_sys::Reflect::set(&iframe_win, &JsValue::from_str("ipc"), ipc);
    log::info!("[bridge] setup_iframe_bridge: ipc installed on iframe window");

    // Run BRIDGE_JS inside the iframe context — this installs the
    // `__viso_push_*` functions and CustomEvent dispatchers on the
    // iframe's window.
    eval_in(&iframe_win, bridge::BRIDGE_JS);
    log::info!("[bridge] setup_iframe_bridge: BRIDGE_JS eval'd in iframe");

    // Verify the push functions exist on the iframe window
    let has_push = js_sys::Reflect::get(
        &iframe_win,
        &JsValue::from_str("__viso_push_schema"),
    )
    .map(|v| v.is_function())
    .unwrap_or(false);
    log::info!(
        "[bridge] setup_iframe_bridge: __viso_push_schema on iframe = {}",
        has_push
    );

    // Now push the initial data into the (now-ready) iframe.
    log::info!("[bridge] setup_iframe_bridge: pushing initial schema+options");
    push_schema_and_options(&engine.borrow());

    // Schedule retries — dioxus WASM inside the iframe may not have
    // registered its event listeners yet.  __viso_replay_pending
    // re-dispatches all stored values.
    eval_in(
        &iframe_win,
        "setTimeout(function(){if(window.__viso_replay_pending)window.\
         __viso_replay_pending()},200);setTimeout(function(){if(window.\
         __viso_replay_pending)window.__viso_replay_pending()},1000);\
         setTimeout(function(){if(window.__viso_replay_pending)window.\
         __viso_replay_pending()},3000);",
    );

    log::info!(
        "[bridge] setup_iframe_bridge: done (retries scheduled at \
         200/1000/3000ms)"
    );
}

/// Get the `contentWindow` of the `ui-panel` iframe, if available.
fn ui_iframe_window() -> Option<JsValue> {
    let document = web_sys::window()?.document()?;
    let el = document.get_element_by_id("ui-panel")?;
    let iframe: web_sys::HtmlIFrameElement = el.unchecked_into();
    iframe.content_window().map(JsValue::from)
}

/// Evaluate a JS string in the context of a target window.
///
/// Retrieves the target window's own `eval` function and calls it, so
/// `window` references inside the script resolve to the target.
fn eval_in(target_window: &JsValue, js: &str) {
    let Ok(eval_fn) =
        js_sys::Reflect::get(target_window, &JsValue::from_str("eval"))
    else {
        log::warn!("[bridge] eval_in: could not get eval from target window");
        return;
    };
    if !eval_fn.is_function() {
        log::warn!("[bridge] eval_in: eval is not a function on target window");
        return;
    }
    let eval_fn: js_sys::Function = eval_fn.unchecked_into();
    match eval_fn.call1(target_window, &JsValue::from_str(js)) {
        Ok(_) => {}
        Err(e) => {
            log::warn!("[bridge] eval_in: eval threw: {e:?}");
        }
    }
}

/// Handle a JSON action from viso-ui (same format as native wry IPC).
fn handle_ipc_action(
    app: &AppHandle,
    engine: &EngineHandle,
    panel: &WebPanelState,
    json: &str,
) {
    let Ok(msg) = serde_json::from_str::<serde_json::Value>(json) else {
        log::warn!("invalid IPC JSON: {json}");
        return;
    };

    let Some(action) = bridge::parse_action(&msg) else {
        let action_str =
            msg.get("action").and_then(|v| v.as_str()).unwrap_or("???");
        log::warn!("unhandled IPC action: {action_str}");
        return;
    };

    let passthrough = {
        let mut eng = engine.borrow_mut();
        dispatch::dispatch_engine_action(&mut eng, action, &WebHost)
    };
    let Some(passthrough) = passthrough else {
        return;
    };
    match passthrough {
        UiAction::FetchPdb { id, source } => {
            let app_clone = Rc::clone(app);
            let eng_clone = Rc::clone(engine);
            wasm_bindgen_futures::spawn_local(async move {
                fetch_and_load(&app_clone, &eng_clone, &id, &source).await;
            });
        }
        UiAction::TogglePanel => {
            let mut collapsed = panel.collapsed.borrow_mut();
            *collapsed = !*collapsed;
            let axis = *panel.axis.borrow();
            let size = *panel.size.borrow();
            apply_web_layout(axis, *collapsed, size);
        }
        UiAction::ResizePanel { size } => {
            let clamped =
                size.clamp(bridge::MIN_PANEL_SIZE, bridge::MAX_PANEL_SIZE);
            *panel.size.borrow_mut() = clamped;
            let axis = *panel.axis.borrow();
            let collapsed = *panel.collapsed.borrow();
            apply_web_layout(axis, collapsed, clamped);
            push_to_ui("panel_size", &format!("{clamped}"));
        }
        // OpenFileDialog/KeyPress/LoadFile are native-only; remaining
        // variants are engine-level and were handled by the dispatcher
        // above.
        _ => {}
    }
}

/// Fetch a structure from RCSB/PDB-REDO and load it.
async fn fetch_and_load(
    app: &AppHandle,
    engine: &EngineHandle,
    id: &str,
    source: &str,
) {
    push_load_status("loading", &format!("Fetching {id}..."));

    let url = match source {
        "pdb-redo" => format!(
            "https://pdb-redo.eu/db/{id}/{id}_final.cif",
            id = id.to_lowercase()
        ),
        _ => {
            format!("https://files.rcsb.org/download/{}.cif", id.to_uppercase())
        }
    };

    let resp = match web_sys::window()
        .expect("no window")
        .fetch_with_str(&url)
        .dyn_into::<js_sys::Promise>()
    {
        Ok(promise) => {
            match wasm_bindgen_futures::JsFuture::from(promise).await {
                Ok(resp) => resp,
                Err(e) => {
                    push_load_status("error", &format!("Fetch failed: {e:?}"));
                    return;
                }
            }
        }
        Err(e) => {
            push_load_status("error", &format!("Fetch failed: {e:?}"));
            return;
        }
    };

    let resp: web_sys::Response = match resp.dyn_into() {
        Ok(r) => r,
        Err(_) => {
            push_load_status("error", "Invalid fetch response");
            return;
        }
    };

    if !resp.ok() {
        push_load_status("error", &format!("HTTP {}", resp.status()));
        return;
    }

    let buf = match resp.array_buffer() {
        Ok(promise) => {
            match wasm_bindgen_futures::JsFuture::from(promise).await {
                Ok(buf) => buf,
                Err(e) => {
                    push_load_status("error", &format!("Read failed: {e:?}"));
                    return;
                }
            }
        }
        Err(e) => {
            push_load_status("error", &format!("Read failed: {e:?}"));
            return;
        }
    };

    let array = js_sys::Uint8Array::new(&buf);
    let bytes = array.to_vec();

    match bridge::parse_structure_bytes(&bytes, "cif") {
        Ok(entities) => {
            let mut a = app.borrow_mut();
            let mut eng = engine.borrow_mut();
            let _ids = a.replace_scene(&mut eng, entities);
            push_load_status("loaded", &format!("Loaded {id}"));
            push_scene_entities(&eng);
        }
        Err(msg) => {
            push_load_status("error", &msg);
        }
    }
}

// ---------------------------------------------------------------------------
// State push (engine → viso-ui via CustomEvents)
// ---------------------------------------------------------------------------

/// Push the options JSON schema + current values to viso-ui.
fn push_schema_and_options(engine: &VisoEngine) {
    let schema = schemars::schema_for!(VisoOptions);
    let json = serde_json::to_string(&schema).unwrap_or_default();
    push_to_ui("schema", &json);

    let opts_json = serde_json::to_string(engine.options()).unwrap_or_default();
    push_to_ui("options", &opts_json);
}

/// Push the scene entity list to viso-ui.
fn push_scene_entities(engine: &VisoEngine) {
    dispatch::push_scene_entities(engine, &WebHost);
}

/// Push a load-status event to viso-ui.
fn push_load_status(status: &str, message: &str) {
    let json = serde_json::json!({ "status": status, "message": message });
    push_to_ui("load_status", &json.to_string());
}

/// Determine the current [`PanelAxis`] from the browser window
/// dimensions.
fn current_axis() -> PanelAxis {
    web_sys::window()
        .map(|w| {
            #[allow(clippy::cast_possible_truncation)]
            let width =
                w.inner_width().ok().and_then(|v| v.as_f64()).unwrap_or(0.0)
                    as u32;
            #[allow(clippy::cast_possible_truncation)]
            let height = w
                .inner_height()
                .ok()
                .and_then(|v| v.as_f64())
                .unwrap_or(0.0) as u32;
            PanelAxis::from_dimensions(width, height)
        })
        .unwrap_or(PanelAxis::Right)
}

/// Apply panel layout to the `ui-panel` iframe element.  Single source
/// of truth for all iframe style changes.
fn apply_web_layout(axis: PanelAxis, collapsed: bool, size: u32) {
    let Some(document) = web_sys::window().and_then(|w| w.document()) else {
        return;
    };
    let Some(el) = document.get_element_by_id("ui-panel") else {
        return;
    };

    let dim = if collapsed {
        format!("{}px", bridge::COLLAPSED_SIZE)
    } else {
        format!("{size}px")
    };

    let style = match axis {
        PanelAxis::Right => format!(
            "transition:width 0.2s \
             ease;width:{dim};height:100%;top:0;right:0;position:fixed"
        ),
        PanelAxis::Bottom => format!(
            "transition:height 0.2s \
             ease;height:{dim};width:100%;bottom:0;left:0;position:fixed"
        ),
    };
    let _ = el.set_attribute("style", &style);
}

/// Dispatch a value to viso-ui via the bridge's
/// `window.__viso_push_{key}(json)` function on the iframe's window.
fn push_to_ui(key: &str, json: &str) {
    let escaped = bridge::escape_for_js(json);
    let js = format!(
        "if(window.__viso_push_{key}){{window.__viso_push_{key}('{escaped}')}}"
    );

    // Eval in the iframe's context so the CustomEvent dispatches on the
    // window that viso-ui is listening on.
    if let Some(iframe_win) = ui_iframe_window() {
        log::info!("[bridge] push_to_ui({key}): targeting iframe window");
        eval_in(&iframe_win, &js);
    } else {
        log::warn!(
            "[bridge] push_to_ui({key}): iframe not available, falling back \
             to parent"
        );
        let _ = js_sys::eval(&js);
    }
}

// ---------------------------------------------------------------------------
// requestAnimationFrame render loop
// ---------------------------------------------------------------------------

fn request_animation_frame_loop(
    engine: EngineHandle,
    canvas: HtmlCanvasElement,
) {
    let closure_holder: Rc<RefCell<Option<Closure<dyn FnMut()>>>> =
        Rc::new(RefCell::new(None));
    let holder_for_closure = Rc::clone(&closure_holder);

    let dt = 1.0 / 60.0_f32;
    // Push stats to viso-ui roughly every 250ms (~15 frames at 60fps).
    let frame_counter = Rc::new(RefCell::new(0u32));
    // Force a surface reconfigure on the first frame.  WebGPU canvases
    // may not display content from textures acquired before the browser
    // has composited the element into the visual tree.  Re-calling
    // engine.resize() after the first rAF (which runs post-composite)
    // ensures the surface context produces displayable textures.
    let needs_initial_resize = Rc::new(RefCell::new(true));

    let cb = Closure::<dyn FnMut()>::new(move || {
        {
            let mut eng = engine.borrow_mut();

            if *needs_initial_resize.borrow() {
                *needs_initial_resize.borrow_mut() = false;
                let dpr = web_sys::window()
                    .map(|w| w.device_pixel_ratio())
                    .unwrap_or(1.0);
                #[allow(
                    clippy::cast_possible_truncation,
                    clippy::cast_sign_loss
                )]
                let w = (f64::from(canvas.client_width()) * dpr) as u32;
                #[allow(
                    clippy::cast_possible_truncation,
                    clippy::cast_sign_loss
                )]
                let h = (f64::from(canvas.client_height()) * dpr) as u32;
                if w > 0 && h > 0 {
                    canvas.set_width(w);
                    canvas.set_height(h);
                    eng.resize(w, h);
                }
            }

            eng.update(dt);
            match eng.render() {
                Ok(()) => {}
                Err(e) => log::error!("render error: {e:?}"),
            }

            let mut count = frame_counter.borrow_mut();
            *count += 1;
            if *count >= 15 {
                *count = 0;
                let fps = eng.fps();
                let json = serde_json::json!({ "fps": fps, "buffers": [] });
                push_to_ui("stats", &json.to_string());
            }
        }
        let holder = holder_for_closure.borrow();
        if let Some(ref cb) = *holder {
            let _ = request_animation_frame(cb);
        }
    });

    let _ = request_animation_frame(&cb);
    *closure_holder.borrow_mut() = Some(cb);
}

fn request_animation_frame(
    callback: &Closure<dyn FnMut()>,
) -> Result<i32, JsValue> {
    web_sys::window()
        .ok_or_else(|| JsValue::from_str("no global window"))?
        .request_animation_frame(callback.as_ref().unchecked_ref())
}
