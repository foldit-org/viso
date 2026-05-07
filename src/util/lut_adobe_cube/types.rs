//! In-memory 3D LUT buffer ([`LutRgbCube3d`]), lattice indexing for GPU
//! upload, and [`expected_lut_sample_count`].

use bytemuck::bytes_of;

use super::LutCubeParseError;

/// In-memory RGB samples for a 3D LUT with edge length [`Self::size`] (`N` in
/// `LUT_3D_SIZE N`).
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct LutRgbCube3d {
    /// Grid dimension `N` (`LUT_3D_SIZE N`).
    pub(crate) size: u32,
    /// Flattened RGB triplets in file order; length must equal `size³` for a
    /// valid LUT.
    pub(crate) rgb: Vec<[f32; 3]>,
}

/// Maps Adobe `.cube` sample index `k` to 3D lattice coordinates `(x, y, z)`
/// used as texel indices for a `wgpu::TextureDimension::D3` upload.
///
/// Adobe ASCII order: input **R** varies fastest, then **G**, then **B**:
///
/// - `x = k mod N` — input R
/// - `y = (k / N) mod N` — input G
/// - `z = k / N²` — input B
///
/// `rgb[k]` is the output RGB triplet at lattice vertex `(x, y, z)`.
#[cfg_attr(not(test), allow(dead_code))]
#[must_use]
pub(crate) fn lattice_xyz_for_sample_index(
    size: u32,
    k: usize,
) -> Option<(u32, u32, u32)> {
    let n = usize::try_from(size).ok()?;
    let n3 = n.checked_mul(n)?.checked_mul(n)?;
    if k >= n3 {
        return None;
    }
    let x = k % n;
    let y = (k / n) % n;
    let z = k / (n * n);
    Some((
        u32::try_from(x).ok()?,
        u32::try_from(y).ok()?,
        u32::try_from(z).ok()?,
    ))
}

impl LutRgbCube3d {
    /// Maximum supported 'LUT_3D_SIZE' value ('N').
    pub(crate) const MAX_SIZE: u32 = 256;

    /// Build a LUT after validating `size` and `rgb.len() == size³`.
    ///
    /// # Errors
    ///
    /// Returns [`LutCubeParseError`] when `size` is outside `2..=MAX_SIZE`,
    /// when `size³` does not fit in [`usize`], or when the RGB sample count
    /// is wrong.
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

    /// RGBA texels (`A = 1.0`) in **volume upload order**: index `k` matches
    /// `.cube` file sample order and [`lattice_xyz_for_sample_index`].
    ///
    /// Suitable for `TextureFormat::Rgba32Float` and PR2 `queue.write_texture`.
    #[cfg_attr(not(test), allow(dead_code))]
    #[must_use]
    pub(crate) fn rgba_f32_volume_texels(&self) -> Vec<[f32; 4]> {
        self.rgb.iter().map(|c| [c[0], c[1], c[2], 1.0]).collect()
    }

    /// Raw bytes for a full `N×N×N` RGBA32F volume: `16 × N³` bytes,
    /// native-endian `f32`, order identical to
    /// [`Self::rgba_f32_volume_texels`].
    #[cfg_attr(not(test), allow(dead_code))]
    #[must_use]
    pub(crate) fn rgba_bytes_volume_order(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(self.rgb.len().saturating_mul(16));
        for c in &self.rgb {
            let px: [f32; 4] = [c[0], c[1], c[2], 1.0];
            out.extend_from_slice(bytes_of(&px));
        }
        out
    }
}

/// Returns 'size³' as [`usize`] if fits; otherwise [`None`].
#[must_use]
pub(crate) fn expected_lut_sample_count(size: u32) -> Option<usize> {
    let n = usize::try_from(size).ok()?;
    n.checked_mul(n)?.checked_mul(n)
}
