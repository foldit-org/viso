//! Adobe / DaVinci Resolve ASCII `.cube` LUT parsing (CPU).

mod error;
mod types;
mod parse;

pub(crate) use error::LutCubeParseError;
// Re-exported for upcoming callers; only referenced from `#[cfg(test)]` today.
#[allow(unused_imports)]
pub(crate) use parse::{parse_adobe_cube_bytes, parse_adobe_cube_str};
pub(crate) use types::{expected_lut_sample_count, LutRgbF32Cube3d};

#[cfg(test)]
#[allow(clippy::unwrap_used)]
#[allow(clippy::expect_used)]
mod tests;
