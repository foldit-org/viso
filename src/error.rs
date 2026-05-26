//! Crate-level error types.

use crate::gpu::render_context::RenderContextError;

/// Errors produced by the viso crate.
#[derive(Debug, thiserror::Error)]
pub enum VisoError {
    /// GPU context initialization failure.
    #[error("GPU error: {0}")]
    Gpu(#[from] RenderContextError),
    /// Allocation or device-limit failure building a GPU resource (e.g. 3D
    /// LUT).
    #[error("GPU resource error: {0}")]
    GpuResource(String),
    /// Failed to load a molecular structure file.
    #[error("structure load error: {0}")]
    StructureLoad(String),
    /// Generic I/O failure.
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    /// Failed to spawn a background thread.
    #[error("failed to spawn thread: {0}")]
    ThreadSpawn(#[source] std::io::Error),
    /// TOML options parsing/serialization failure.
    #[error("options parse error: {0}")]
    OptionsParse(String),
    /// Color LUT parse/validation failure (e.g. Adobe `.cube`).
    #[error("color LUT error: {0}")]
    ColorLut(String),
    /// Viewer event-loop failure.
    #[error("viewer error: {0}")]
    Viewer(String),
    /// Shader compilation or composition failure.
    #[error("shader error: {0}")]
    Shader(String),
}
