//! In-memory 3D LUT buffer ([`LutRgbF32Cube3d`]) and [`expected_lut_sample_count`].

use super::LutCubeParseError;

/// In-memory RGB samples for a 3D LUT with edge length [`Self::size`] (`N` in `LUT_3D_SIZE N`).
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct LutRgbF32Cube3d {
    /// Grid dimension `N` (`LUT_3D_SIZE N`).
    pub(crate) size: u32,
    /// Flattened RGB triplets in file order; length must equal `size³` for a valid LUT.
    pub(crate) rgb: Vec<[f32; 3]>,
}

#[allow(dead_code)] // Use by GPU path once 3D LUT textures are connected.
impl LutRgbF32Cube3d {
    /// Maximum supported 'LUT_3D_SIZE' value ('N').
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

/// Returns 'size³' as [`usize`] if fits; otherwise [`None`].
#[must_use]
#[allow(dead_code)] // Used by the parser and `LutRgbF32Cube3d::new`; make public for tests.
pub(crate) fn expected_lut_sample_count(size: u32) -> Option<usize> {
    let n = usize::try_from(size).ok()?;
    n.checked_mul(n)?.checked_mul(n)
}
