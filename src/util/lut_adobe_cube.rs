//! Adobe / DaVinci Resolve ASCII `.cube` LUT parsing (CPU).

use crate::VisoError;

/// In-memory RGB samples for a 3D LUT of edge length [`Self::size`].
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct LutRgbF32Cube3d {
    /// Cube dimension (`N` in `LUT_3D_SIZE N`).
    pub(crate) size: u32,
    /// Flattened RGB triplets; length should equal `size³` after parsing.
    pub(crate) rgb: Vec<[f32; 3]>,
}

#[allow(dead_code)] // Not wired into the render path until `.cube` parsing lands.
impl LutRgbF32Cube3d {
    /// Maximum supported size.
    pub(crate) const MAX_SIZE: u32 = 256;

    /// Build a LUT after validating `size` and `rgb.len() == size³`.
    ///
    /// # Errors
    ///
    /// Returns [`LutCubeParseError`] when `size` is outside `2..=MAX_SIZE`, when
    /// `size³` does not fit in [`usize`], or when the RGB sample count is
    /// wrong.
    pub(crate) fn new(
        size: u32,
        rgb: Vec<[f32; 3]>,
    ) -> Result<Self, LutCubeParseError> {
        if !(2..=Self::MAX_SIZE).contains(&size) {
            return Err(LutCubeParseError::InvalidLutSize { size });
        }

        let expected = expected_lut_sample_count(size)
            .ok_or(LutCubeParseError::InvalidLutSize { size })?;

        let actual = rgb.len();
        if actual != expected {
            return Err(LutCubeParseError::WrongRgbCount { expected, actual });
        }

        Ok(Self { size, rgb })
    }
}

/// Errors emitted while parsing or validating `.cube` LUT files.
#[allow(dead_code)]
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum LutCubeParseError {
    /// size outside the supported range.
    InvalidLutSize { size: u32 },
    /// number of RGB samples not match `size^3`.
    WrongRgbCount { expected: usize, actual: usize },
}

impl std::fmt::Display for LutCubeParseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidLutSize { size } => {
                write!(f, "LUT size {size} is outside supported bounds")
            }
            Self::WrongRgbCount { expected, actual } => write!(
                f,
                "LUT cube has {actual} RGB samples but expected {expected}"
            ),
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

/// Returns `size³` as [`usize`] if fits; otherwise [`None`]
#[must_use]
#[allow(dead_code)] // Not wired into the render path until `.cube` parsing lands.
pub(crate) fn expected_lut_sample_count(size: u32) -> Option<usize> {
    let n = usize::try_from(size).ok()?;
    Some(n.checked_mul(n)?.checked_mul(n)?)
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
#[allow(clippy::expect_used)]
mod tests {
    use super::{expected_lut_sample_count, LutCubeParseError, LutRgbF32Cube3d};

    #[test]
    // check math on tiny LUTs without building large vectors.
    fn expected_lut_sample_count_matches_size_cubed_for_small_sizes() {
        // N=2 ⇒ 2³ = 8 RGB triplets; N=3 ⇒ 27 triplets.
        assert_eq!(expected_lut_sample_count(2), Some(8));
        assert_eq!(expected_lut_sample_count(3), Some(27));
    }

    #[test]
    // check minimal legal LUT: `LUT_3D_SIZE 2` exactly eight RGB rows.
    fn new_accepts_n2_corner_lut() {
        // Eight corners of the RGB cube ordering follows flattening,
        // but `new()` only checks counts not ordering.
        let rgb = vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
        ];

        let lut = LutRgbF32Cube3d::new(2, rgb).expect("valid 2³ LUT");
        assert_eq!(lut.size, 2);
        assert_eq!(lut.rgb.len(), 8);
    }

    #[test]
    // `N=2`, handing in one RGB row fail
    fn new_rejects_wrong_sample_count() {
        let err =
            LutRgbF32Cube3d::new(2, vec![[0.0, 0.0, 0.0]]).expect_err("too few samples");

        assert_eq!(
            err,
            LutCubeParseError::WrongRgbCount {
                expected: 8,
                actual: 1
            }
        );
    }
}
