//! Read Adobe / DaVinci style `.cube` 3D LUT files on the CPU, and produce
//! GPU-oriented packed RGBA bytes (`Rgba16Float` upload path for PR2+).
//!
//! Supports a small strict format: `LUT_3D_SIZE N`, then `N×N×N` lines of three
//! RGB numbers. Skips blank lines, `#` comments, `TITLE` / `DOMAIN_*` lines,
//! and an optional UTF-8 BOM at the start of the file.
//!
//! Lattice indexing for 3D texture upload is defined in
//! `crate::util::lut_adobe_cube::types::lattice_xyz_for_sample_index`; see
//! `types.rs` for details.

mod error;
mod parse;
mod types;

pub use error::LutCubeParseError;
pub use parse::{parse_adobe_cube_bytes, parse_adobe_cube_str};
pub use types::{expected_lut_sample_count, LutRgbCube3d};

#[cfg(test)]
#[allow(clippy::unwrap_used)]
#[allow(clippy::expect_used)]
mod tests;
