//! Read Adobe / DaVinci style `.cube` 3D LUT files on the CPU (not GPU yet).
//!
//! Supports a small strict format: `LUT_3D_SIZE N`, then `N×N×N` lines of three
//! RGB numbers. Skips blank lines, `#` comments, `TITLE` / `DOMAIN_*` lines, and
//! an optional UTF-8 BOM at the start of the file.
//!
//! Code lives in `error.rs`, `types.rs`, `parse.rs`; tests in `tests.rs`.

mod error;
mod types;
mod parse;

pub(crate) use error::LutCubeParseError;
// Parse functions are re-exported here for the rest of the crate. The main library
// build does not call them yet, so silence "unused import" until wiring lands.
#[allow(unused_imports)]
pub(crate) use parse::{parse_adobe_cube_bytes, parse_adobe_cube_str};
pub(crate) use types::{expected_lut_sample_count, LutRgbF32Cube3d};

#[cfg(test)]
#[allow(clippy::unwrap_used)]
#[allow(clippy::expect_used)]
mod tests;
