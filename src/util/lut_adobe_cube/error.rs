//! Parse and validation errors for Adobe ASCII `.cube` LUT files.

use crate::VisoError;

/// Errors emitted while parsing or validating `.cube` LUT files.
#[allow(dead_code)]
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum LutCubeParseError {
    /// file not contain any `LUT_3D_SIZE` header line.
    MissingLutSize,
    /// header line not formatted as `LUT_3D_SIZE N`.
    InvalidLutSizeLine {
        /// 1-based source line no.
        line: usize,
    },
    /// size outside the supported range.
    InvalidLutSize { size: u32 },
    /// line in RGB data section not three floats.
    MalformedRgbLine {
        /// 1-based source line no.
        line: usize,
    },
    /// number of RGB samples not match `size^3`.
    WrongRgbCount { expected: usize, actual: usize },
    /// Input bytes are not valid UTF-8.
    InvalidUtf8,
}

impl std::fmt::Display for LutCubeParseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::MissingLutSize => {
                write!(f, "LUT cube file is missing LUT_3D_SIZE header")
            }
            Self::InvalidLutSizeLine { line } => {
                write!(f, "invalid LUT_3D_SIZE header (line {line})")
            }
            Self::InvalidLutSize { size } => {
                write!(f, "LUT size {size} is outside supported bounds")
            }
            Self::MalformedRgbLine { line } => {
                write!(f, "malformed RGB sample line (line {line})")
            }
            Self::WrongRgbCount { expected, actual } => write!(
                f,
                "LUT cube has {actual} RGB samples but expected {expected}"
            ),
            Self::InvalidUtf8 => write!(f, "LUT cube file is not valid UTF-8"),
        }
    }
}

impl std::error::Error for LutCubeParseError {}

/// Map to `VisoError::OptionsParse`.
impl From<LutCubeParseError> for VisoError {
    fn from(value: LutCubeParseError) -> Self {
        Self::OptionsParse(value.to_string())
    }
}
