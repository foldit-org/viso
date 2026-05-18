//! GPU resource management utilities.
//!
//! Provides wgpu device/surface initialization, dynamic buffer management,
//! lighting, per-residue color storage, and shader composition.

/// 3D `Rgba16Float` LUT texture + sampler from a decoded Adobe `.cube` volume.
pub(crate) mod adobe_cube_lut;
/// Growable GPU buffers with automatic reallocation.
pub(crate) mod dynamic_buffer;
/// GPU lighting uniform, IBL cubemaps, and bind group management.
pub(crate) mod lighting;
/// Shared wgpu boilerplate helpers for screen-space post-process pipelines.
pub(crate) mod pipeline_helpers;
/// wgpu device, surface, and queue initialization.
pub(crate) mod render_context;
/// Per-residue color storage buffer for GPU shaders.
pub(crate) mod residue_color;
/// WGSL shader composition with `#import` support via naga-oil.
pub(crate) mod shader_composer;

pub(crate) use render_context::RenderContext;
pub(crate) use shader_composer::{Shader, ShaderComposer};
/// Framework-agnostic render-target texture abstraction.
pub(crate) mod texture;
