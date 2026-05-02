//! Line-oriented parsing for Adobe ASCII `.cube` LUT files.

use super::{expected_lut_sample_count, LutCubeParseError, LutRgbF32Cube3d};

/// Parse a minimal ASCII `.cube` LUT.
///
/// After `LUT_3D_SIZE N`, each non-empty line must be exactly three
/// whitespace-separated floats (`r g b`). Blank lines are skipped.
///
/// ASCII `#` comments are supported: a line whose first non-space
/// character is `#` is ignored; otherwise text from the first `#` onward is
/// stripped before parsing.
///
/// Common DaVinci / Adobe header lines `TITLE`, `DOMAIN_MIN`, and `DOMAIN_MAX`
/// are ignored (payload not validated). A leading UTF-8 BOM (`U+FEFF`) is
/// stripped before parsing.
///
/// # Errors
///
/// Returns [`LutCubeParseError`] if the text does not match the supported
/// subset.
#[allow(dead_code)] // Called from tests until host wiring lands.
pub(crate) fn parse_adobe_cube_str(input: &str) -> Result<LutRgbF32Cube3d, LutCubeParseError> {
    let input = input.strip_prefix('\u{FEFF}').unwrap_or(input);

    let mut lut_size: Option<u32> = None;
    let mut rgb: Vec<[f32; 3]> = Vec::new();

    for (idx, raw_line) in input.lines().enumerate() {
        let line_no = idx + 1;
        let trimmed = raw_line.trim();
        if trimmed.is_empty() {
            continue;
        }

        let Some(line) = meaningful_cube_line(trimmed) else {
            continue;
        };

        if is_adobe_cube_metadata_line(line) {
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

/// Parse a `.cube` LUT from UTF-8 bytes (including a leading UTF-8 BOM).
///
/// # Errors
///
/// Returns [`LutCubeParseError::InvalidUtf8`] when `input` is not valid UTF-8.
/// Other errors match [`parse_adobe_cube_str`].
#[allow(dead_code)] // Called from tests until host wiring lands.
pub(crate) fn parse_adobe_cube_bytes(input: &[u8]) -> Result<LutRgbF32Cube3d, LutCubeParseError> {
    let text = std::str::from_utf8(input).map_err(|_| LutCubeParseError::InvalidUtf8)?;
    parse_adobe_cube_str(text)
}

/// Returns the portion of `trimmed_physical_line` that should be parsed, or
/// [`None`] when the line is only a comment.
///
/// `trimmed_physical_line` must be the line after [`str::trim`].
fn meaningful_cube_line(trimmed_physical_line: &str) -> Option<&str> {
    if trimmed_physical_line.starts_with('#') {
        return None;
    }

    let before_hash = trimmed_physical_line
        .split('#')
        .next()
        .unwrap_or("")
        .trim();

    (!before_hash.is_empty()).then_some(before_hash)
}

/// Returns `true` when `meaningful_line` is a known Adobe `.cube` metadata
/// header line (`TITLE`, `DOMAIN_MIN`, `DOMAIN_MAX`).
fn is_adobe_cube_metadata_line(meaningful_line: &str) -> bool {
    let mut tokens = meaningful_line.split_whitespace();
    let Some(head) = tokens.next() else {
        return false;
    };

    matches!(head, "TITLE" | "DOMAIN_MIN" | "DOMAIN_MAX")
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

/// Parses `token` as [`f32`] and rejects NaN and infinity (unsuitable for LUT texels).
fn parse_finite_f32(token: &str, line_no: usize) -> Result<f32, LutCubeParseError> {
    let value = token
        .parse::<f32>()
        .map_err(|_| LutCubeParseError::MalformedRgbLine { line: line_no })?;

    if !value.is_finite() {
        return Err(LutCubeParseError::MalformedRgbLine { line: line_no });
    }

    Ok(value)
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

    let r = parse_finite_f32(r_s, line_no)?;
    let g = parse_finite_f32(g_s, line_no)?;
    let b = parse_finite_f32(b_s, line_no)?;

    Ok([r, g, b])
}
