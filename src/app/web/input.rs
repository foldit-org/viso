//! DOM event listeners for forwarding browser input to the engine.

use std::cell::RefCell;
use std::rc::Rc;

use wasm_bindgen::prelude::*;
use wasm_bindgen::JsCast;
use web_sys::HtmlCanvasElement;

use super::{EngineHandle, SelectionHandle};
use crate::input::{KeyBindings, MouseButton};

fn dpr() -> f32 {
    web_sys::window()
        .map(|w| w.device_pixel_ratio() as f32)
        .unwrap_or(1.0)
}

pub(super) fn attach_input_listeners(
    canvas: &HtmlCanvasElement,
    engine: EngineHandle,
    keybindings: Rc<KeyBindings>,
    selection: SelectionHandle,
) {
    // Mouse move
    {
        let engine = Rc::clone(&engine);
        let cb =
            Closure::<dyn FnMut(_)>::new(move |event: web_sys::MouseEvent| {
                let scale = dpr();
                let x = event.offset_x() as f32 * scale;
                let y = event.offset_y() as f32 * scale;
                let mut eng = engine.borrow_mut();
                eng.set_cursor_pos(x, y);
                eng.feed_pointer_motion(x, y);
            });
        let _ = canvas.add_event_listener_with_callback(
            "mousemove",
            cb.as_ref().unchecked_ref(),
        );
        cb.forget();
    }

    // Mouse down
    {
        let engine = Rc::clone(&engine);
        let cb =
            Closure::<dyn FnMut(_)>::new(move |event: web_sys::MouseEvent| {
                let button = match event.button() {
                    2 => MouseButton::Right,
                    1 => MouseButton::Middle,
                    _ => MouseButton::Left,
                };
                let mut eng = engine.borrow_mut();
                let _ = eng.feed_pointer_button(button, true);
            });
        let _ = canvas.add_event_listener_with_callback(
            "mousedown",
            cb.as_ref().unchecked_ref(),
        );
        cb.forget();
    }

    // Mouse up
    {
        let engine = Rc::clone(&engine);
        let selection = Rc::clone(&selection);
        let cb =
            Closure::<dyn FnMut(_)>::new(move |event: web_sys::MouseEvent| {
                let button = match event.button() {
                    2 => MouseButton::Right,
                    1 => MouseButton::Middle,
                    _ => MouseButton::Left,
                };
                let click = {
                    let mut eng = engine.borrow_mut();
                    eng.feed_pointer_button(button, false)
                };
                if let Some(click) = click {
                    let mut eng = engine.borrow_mut();
                    selection.borrow_mut().apply_click(&mut eng, &click);
                }
            });
        let _ = canvas.add_event_listener_with_callback(
            "mouseup",
            cb.as_ref().unchecked_ref(),
        );
        cb.forget();
    }

    // Wheel
    {
        let engine = Rc::clone(&engine);
        let cb =
            Closure::<dyn FnMut(_)>::new(move |event: web_sys::WheelEvent| {
                event.prevent_default();
                let delta = -(event.delta_y() as f32) * 0.01;
                let mut eng = engine.borrow_mut();
                eng.feed_scroll(delta);
            });
        let opts = web_sys::AddEventListenerOptions::new();
        opts.set_passive(false);
        let _ = canvas
            .add_event_listener_with_callback_and_add_event_listener_options(
                "wheel",
                cb.as_ref().unchecked_ref(),
                &opts,
            );
        cb.forget();
    }

    // Prevent context menu on right-click
    {
        let cb =
            Closure::<dyn FnMut(_)>::new(move |event: web_sys::MouseEvent| {
                event.prevent_default();
            });
        let _ = canvas.add_event_listener_with_callback(
            "contextmenu",
            cb.as_ref().unchecked_ref(),
        );
        cb.forget();
    }

    // Keyboard input — listen on the document so keys work even when
    // the canvas isn't explicitly focused.
    {
        let engine = Rc::clone(&engine);
        let keybindings = Rc::clone(&keybindings);
        let cb = Closure::<dyn FnMut(_)>::new(
            move |event: web_sys::KeyboardEvent| {
                // Let browser shortcuts (Cmd+R, Ctrl+Shift+R, etc.)
                // pass through unintercepted.
                if event.meta_key() || event.ctrl_key() || event.alt_key() {
                    return;
                }

                // Browser KeyboardEvent.code uses the same naming as
                // winit's KeyCode debug format: "KeyQ", "Tab", etc.
                let code = event.code();
                let matched = {
                    let mut eng = engine.borrow_mut();
                    keybindings.dispatch(&code, &mut eng)
                };
                if matched {
                    event.prevent_default();
                }

                // Forward shift state
                engine.borrow_mut().feed_modifiers(event.shift_key());
            },
        );
        let document = web_sys::window()
            .and_then(|w| w.document())
            .expect("no document");
        let _ = document.add_event_listener_with_callback(
            "keydown",
            cb.as_ref().unchecked_ref(),
        );
        cb.forget();
    }

    // Shift key release
    {
        let engine = Rc::clone(&engine);
        let cb = Closure::<dyn FnMut(_)>::new(
            move |event: web_sys::KeyboardEvent| {
                engine.borrow_mut().feed_modifiers(event.shift_key());
            },
        );
        let document = web_sys::window()
            .and_then(|w| w.document())
            .expect("no document");
        let _ = document.add_event_listener_with_callback(
            "keyup",
            cb.as_ref().unchecked_ref(),
        );
        cb.forget();
    }
}

// ---------------------------------------------------------------------------
// Canvas resize handling
// ---------------------------------------------------------------------------

/// Watch the canvas element for size changes and call `engine.resize()`.
pub(super) fn attach_resize_observer(
    canvas: &HtmlCanvasElement,
    engine: EngineHandle,
) {
    let canvas_clone = canvas.clone();
    let cb = Closure::<dyn FnMut(JsValue)>::new(move |_entries: JsValue| {
        let dpr = web_sys::window()
            .map(|w| w.device_pixel_ratio())
            .unwrap_or(1.0);
        let w = (f64::from(canvas_clone.client_width()) * dpr) as u32;
        let h = (f64::from(canvas_clone.client_height()) * dpr) as u32;
        if w > 0 && h > 0 {
            canvas_clone.set_width(w);
            canvas_clone.set_height(h);
            engine.borrow_mut().resize(w, h);
        }
    });

    // Create ResizeObserver via JS interop
    let observer_js =
        js_sys::Function::new_with_args("cb", "return new ResizeObserver(cb)");
    if let Ok(observer) = observer_js.call1(&JsValue::NULL, cb.as_ref()) {
        let observe_fn =
            js_sys::Reflect::get(&observer, &JsValue::from_str("observe"));
        if let Ok(observe_fn) = observe_fn {
            let observe_fn: js_sys::Function = observe_fn.unchecked_into();
            let _ = observe_fn.call1(&observer, canvas);
        }
    }
    cb.forget();
}
