//! GPU resource management utilities.
//!
//! Provides wgpu device/surface initialization, dynamic buffer management,
//! lighting, per-residue color storage, and shader composition.

pub(crate) mod dynamic_buffer;
pub(crate) mod lighting;
pub(crate) mod pipeline_helpers;
pub(crate) mod render_context;
pub(crate) mod residue_color;
pub(crate) mod shader_composer;

pub(crate) use render_context::RenderContext;
pub(crate) use shader_composer::{Shader, ShaderComposer};
pub(crate) mod texture;
