//! GPU-accelerated 3D protein visualization engine built on wgpu.
//!
//! Viso renders protein structures with real-time animation support,
//! post-processing (SSAO, bloom, FXAA), and GPU-based residue picking.
//!
//! # Key types
//!
//! - [`VisoEngine`]: the main rendering engine
//! - [`VisoOptions`](options::VisoOptions): runtime configuration (display,
//!   lighting, camera, colors)
//! - [`VisoError`]: error type
//! - [`KeyBindings`]: physical-key to engine-action dispatch table
//!
//! # Architecture
//!
//! The engine runs a background scene-processor thread that generates mesh
//! data off the main thread, delivering results via a lock-free triple
//! buffer. The main thread uploads prepared geometry to the GPU and
//! orchestrates a multi-pass pipeline: geometry → SSAO → bloom → composite
//! → FXAA.
//!
//! For integration guides and deep dives, see the companion mdBook
//! documentation.

#![deny(deprecated)]

pub(crate) mod animation;
#[cfg(any(feature = "viewer", feature = "web"))]
pub mod app;
pub(crate) mod bridge;
pub(crate) mod camera;
pub(crate) mod engine;
pub(crate) mod error;
pub(crate) mod gpu;
pub(crate) mod input;
pub(crate) mod renderer;
pub(crate) mod util;

/// Runtime display, lighting, camera, and color options.
pub mod options;

/// Bench-only entry points. `#[doc(hidden)]`, not part of the public API.
#[doc(hidden)]
pub mod bench_api;

// Animation (preset constructors only)
pub use animation::transition::Transition;
// Feature-gated
#[cfg(feature = "viewer")]
pub use app::viewer::{Viewer, ViewerBuilder};
#[cfg(any(feature = "viewer", feature = "web"))]
pub use app::VisoApp;
#[cfg(feature = "gui")]
pub use bridge::UiAction;
pub use engine::command::{
    AtomRef, BandInfo, BandTarget, BandType, ClashEndpoint, ClashInfo,
    ExposedHydrophobicInfo, PullInfo, SelectSphereInfo,
};
pub use engine::constraint::PickedResidueAtom;
pub use engine::focus::Focus;
pub use engine::VisoEngine;
pub use error::VisoError;
pub use gpu::render_context::{RenderContext, SurfaceError};
pub use gpu::texture::RenderTarget;
// Input (optional convenience)
pub use input::{
    classify_click_for_selection, ClickEvent, ClickPattern,
    ClickSelectionAction, InputEvent, KeyAction, KeyBindings, Modifiers,
    MouseButton,
};
pub use molex;
// Display overrides + drawing mode enums
pub use options::{DisplayOverrides, DrawingMode, HelixStyle, SheetStyle};
// Picking output
pub use renderer::picking::PickTarget;
#[cfg(all(feature = "web", target_arch = "wasm32"))]
pub use wasm_bindgen_rayon::init_thread_pool;
pub use wgpu;
