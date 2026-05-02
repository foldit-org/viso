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
    n.checked_mul(n)?.checked_mul(n)
}

/// Parse a minimal ASCII `.cube` LUT.
///
/// Define a strict subset: after `LUT_3D_SIZE N`, each
/// non-empty line must be exactly three whitespace-separated floats (`r g b`).
/// Blank lines are skipped. Comment/BOM handling haven't done yet.
///
/// # Errors
///
/// Returns [`LutCubeParseError`] if the text does not match the supported
/// subset.
#[allow(dead_code)] // Called from tests until host wiring lands.
pub(crate) fn parse_adobe_cube_str(input: &str) -> Result<LutRgbF32Cube3d, LutCubeParseError> {
    let mut lut_size: Option<u32> = None;
    let mut rgb: Vec<[f32; 3]> = Vec::new();

    for (idx, raw_line) in input.lines().enumerate() {
        let line_no = idx + 1;
        let line = raw_line.trim();
        if line.is_empty() {
            continue;
        }

        match lut_size {
            None => {
                let n = parse_lut_size_line(line, line_no)?;

                // Reject LUT sizes outside `LutRgbF32Cube3d::new`'s accepted range.
                // `parse_lut_size_line` only ensures the token parses as [`u32`].
                if !(2..=LutRgbF32Cube3d::MAX_SIZE).contains(&n)
                    || expected_lut_sample_count(n).is_none()
                {
                    return Err(LutCubeParseError::InvalidLutSize { size: n });
                }

                lut_size = Some(n);
            }
            Some(lut_sz) => {
                let expected_len = expected_lut_sample_count(lut_sz)
                    .ok_or(LutCubeParseError::InvalidLutSize { size: lut_sz })?;

                if rgb.len() == expected_len {
                    return Err(LutCubeParseError::WrongRgbCount {
                        expected: expected_len,
                        actual: expected_len.saturating_add(1),
                    });
                }

                let triplet = parse_rgb_triplet_line(line, line_no)?;
                rgb.push(triplet);
            }
        }
    }

    let Some(lut_sz) = lut_size else {
        return Err(LutCubeParseError::MissingLutSize);
    };

    LutRgbF32Cube3d::new(lut_sz, rgb)
}

fn parse_lut_size_line(line: &str, line_no: usize) -> Result<u32, LutCubeParseError> {
    let mut tokens = line.split_whitespace();
    let Some(head) = tokens.next() else {
        return Err(LutCubeParseError::InvalidLutSizeLine { line: line_no });
    };

    if head != "LUT_3D_SIZE" {
        return Err(LutCubeParseError::InvalidLutSizeLine { line: line_no });
    }

    let Some(raw_n) = tokens.next() else {
        return Err(LutCubeParseError::InvalidLutSizeLine { line: line_no });
    };

    if tokens.next().is_some() {
        return Err(LutCubeParseError::InvalidLutSizeLine { line: line_no });
    }

    raw_n
        .parse::<u32>()
        .map_err(|_| LutCubeParseError::InvalidLutSizeLine { line: line_no })
}

fn parse_rgb_triplet_line(line: &str, line_no: usize) -> Result<[f32; 3], LutCubeParseError> {
    let mut tokens = line.split_whitespace();
    let r_s = tokens
        .next()
        .ok_or(LutCubeParseError::MalformedRgbLine { line: line_no })?;
    let g_s = tokens
        .next()
        .ok_or(LutCubeParseError::MalformedRgbLine { line: line_no })?;
    let b_s = tokens
        .next()
        .ok_or(LutCubeParseError::MalformedRgbLine { line: line_no })?;

    if tokens.next().is_some() {
        return Err(LutCubeParseError::MalformedRgbLine { line: line_no });
    }

    let r = r_s
        .parse::<f32>()
        .map_err(|_| LutCubeParseError::MalformedRgbLine { line: line_no })?;
    let g = g_s
        .parse::<f32>()
        .map_err(|_| LutCubeParseError::MalformedRgbLine { line: line_no })?;
    let b = b_s
        .parse::<f32>()
        .map_err(|_| LutCubeParseError::MalformedRgbLine { line: line_no })?;

    Ok([r, g, b])
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
#[allow(clippy::expect_used)]
mod tests {
    use super::{
        expected_lut_sample_count, parse_adobe_cube_str, LutCubeParseError, LutRgbF32Cube3d,
    };

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

    #[test]
    fn parse_reports_missing_header() {
        let err = parse_adobe_cube_str("").expect_err("should reject empty input");
        assert_eq!(err, LutCubeParseError::MissingLutSize);
    }

    #[test]
    fn parse_accepts_lut_size_after_blank_lines() {
        let err = parse_adobe_cube_str("\n\nLUT_3D_SIZE 2\n")
            .expect_err("header-only file has no RGB rows yet");
        assert_eq!(
            err,
            LutCubeParseError::WrongRgbCount {
                expected: 8,
                actual: 0
            }
        );
    }

    #[test]
    fn parse_requires_lut_size_before_other_content() {
        let err = parse_adobe_cube_str("0 0 0\nLUT_3D_SIZE 2\n").expect_err("order matters");
        assert_eq!(
            err,
            LutCubeParseError::InvalidLutSizeLine { line: 1 }
        );
    }

    #[test]
    fn parse_header_only_yields_wrong_sample_count() {
        let err = parse_adobe_cube_str("LUT_3D_SIZE 2\n")
            .expect_err("no RGB rows after header");
        assert_eq!(
            err,
            LutCubeParseError::WrongRgbCount {
                expected: 8,
                actual: 0
            }
        );
    }

    #[test]
    fn parse_rejects_incomplete_lut_after_valid_triplets() {
        let err = parse_adobe_cube_str("LUT_3D_SIZE 2\n0 0 0\n").expect_err("incomplete LUT");
        assert_eq!(
            err,
            LutCubeParseError::WrongRgbCount {
                expected: 8,
                actual: 1
            }
        );
    }

    #[test]
    fn parse_rejects_malformed_rgb_line() {
        let err =
            parse_adobe_cube_str("LUT_3D_SIZE 2\nnot_float 0 0\n").expect_err("bad float token");
        assert_eq!(
            err,
            LutCubeParseError::MalformedRgbLine { line: 2 }
        );
    }

    #[test]
    fn parse_rejects_extra_rgb_line_after_lut_is_full() {
        let cube = "\
LUT_3D_SIZE 2
0 0 0
1 0 0
0 1 0
1 1 0
0 0 1
1 0 1
0 1 1
1 1 1
0 0 0
";
        let err = parse_adobe_cube_str(cube).expect_err("unexpected ninth sample row");
        assert_eq!(
            err,
            LutCubeParseError::WrongRgbCount {
                expected: 8,
                actual: 9
            }
        );
    }

    #[test]
    fn parse_succeeds_for_minimal_two_cubed_lut() {
        let src = "LUT_3D_SIZE 2\n\
             0 0 0\n1 0 0\n0 1 0\n1 1 0\n\
             0 0 1\n1 0 1\n0 1 1\n1 1 1\n";

        let lut = parse_adobe_cube_str(src).expect("minimal N=2 LUT must parse");

        let expected = [
            [0.0_f32, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
        ];

        assert_eq!(lut.size, 2);
        assert_eq!(lut.rgb, expected);
    }

    #[test]
    fn parse_skips_blank_lines_between_header_and_samples() {
        let src = "LUT_3D_SIZE 2\n\
             \n\
             0 0 0\n\n1 0 0\n  \n0 1 0\n1 1 0\n0 0 1\n1 0 1\n0 1 1\n1 1 1\n";

        let lut = parse_adobe_cube_str(src).expect("blank lines should not break parse");
        assert_eq!(lut.size, 2);
        assert_eq!(lut.rgb.len(), 8);
        assert_eq!(lut.rgb[0], [0.0, 0.0, 0.0]);
        assert_eq!(lut.rgb[7], [1.0, 1.0, 1.0]);
    }

    #[test]
    fn parse_rejects_rgb_line_with_too_few_tokens() {
        let err = parse_adobe_cube_str("LUT_3D_SIZE 2\n0 0\n").expect_err("need three floats");
        assert_eq!(
            err,
            LutCubeParseError::MalformedRgbLine { line: 2 }
        );
    }

    #[test]
    fn parse_rejects_rgb_line_with_too_many_tokens() {
        let err =
            parse_adobe_cube_str("LUT_3D_SIZE 2\n0 0 0 1\n").expect_err("extra token");
        assert_eq!(
            err,
            LutCubeParseError::MalformedRgbLine { line: 2 }
        );
    }

    #[test]
    fn parse_rejects_lut_size_line_with_trailing_token() {
        let err = parse_adobe_cube_str("LUT_3D_SIZE 2 extra\n")
            .expect_err("header must be exactly two tokens after split");
        assert_eq!(
            err,
            LutCubeParseError::InvalidLutSizeLine { line: 1 }
        );
    }
}
