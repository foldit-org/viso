//! Errors for Adobe ASCII `.cube` LUT parsing and validation.

use crate::VisoError;

/// Failure modes while parsing or validating an Adobe ASCII `.cube` LUT.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LutCubeParseError {
    /// No `LUT_3D_SIZE` header line was found after preprocessing.
    MissingLutSize,

    /// A non-metadata line before samples was not exactly `LUT_3D_SIZE N`.
    InvalidLutSizeLine {
        /// 1-based source line number.
        line: usize,
    },

    /// N outside 2..=256 or N³ not fit in usize.
    InvalidLutSize {
        /// Declared or inferred grid length `N`.
        size: u32,
    },

    /// RGB sample line not exactly three finite floats.
    MalformedRgbLine {
        /// 1-based source line number.
        line: usize,
    },

    /// RGB sample count differs from N³.
    WrongRgbCount {
        /// Expected count (`N³`).
        expected: usize,
        /// Actual RGB triplets parsed.
        actual: usize,
    },

    /// Input bytes are not valid UTF-8 (byte entrypoint only).
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

/// Converts into [`VisoError::OptionsParse`] using this error’s formatted
/// message.
impl From<LutCubeParseError> for VisoError {
    fn from(value: LutCubeParseError) -> Self {
        Self::OptionsParse(value.to_string())
    }
}
