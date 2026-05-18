//! Crate-level error types.

use std::fmt;

use crate::gpu::render_context::RenderContextError;

/// Errors produced by the viso crate.
#[derive(Debug)]
pub enum VisoError {
    /// GPU context initialization failure.
    Gpu(RenderContextError),
    /// Allocation or device-limit failure building a GPU resource (e.g. 3D
    /// LUT).
    GpuResource(String),
    /// Failed to load a molecular structure file.
    StructureLoad(String),
    /// Generic I/O failure.
    Io(std::io::Error),
    /// Failed to spawn a background thread.
    ThreadSpawn(std::io::Error),
    /// TOML options parsing/serialization failure.
    OptionsParse(String),
    /// Viewer event-loop failure.
    Viewer(String),
    /// Shader compilation or composition failure.
    Shader(String),
}

impl fmt::Display for VisoError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Gpu(e) => write!(f, "GPU error: {e}"),
            Self::GpuResource(msg) => write!(f, "GPU resource error: {msg}"),
            Self::StructureLoad(msg) => {
                write!(f, "structure load error: {msg}")
            }
            Self::Io(e) => write!(f, "I/O error: {e}"),
            Self::ThreadSpawn(e) => {
                write!(f, "failed to spawn thread: {e}")
            }
            Self::OptionsParse(msg) => {
                write!(f, "options parse error: {msg}")
            }
            Self::Viewer(msg) => write!(f, "viewer error: {msg}"),
            Self::Shader(msg) => write!(f, "shader error: {msg}"),
        }
    }
}

impl std::error::Error for VisoError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Gpu(e) => Some(e),
            Self::Io(e) | Self::ThreadSpawn(e) => Some(e),
            _ => None,
        }
    }
}

impl From<RenderContextError> for VisoError {
    fn from(e: RenderContextError) -> Self {
        Self::Gpu(e)
    }
}

impl From<std::io::Error> for VisoError {
    fn from(e: std::io::Error) -> Self {
        Self::Io(e)
    }
}
