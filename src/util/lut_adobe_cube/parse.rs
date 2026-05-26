//! Parse `.cube` text line by line. Public API: [`parse_adobe_cube_str`],
//! [`parse_adobe_cube_bytes`].

use super::{expected_lut_sample_count, LutCubeParseError, LutRgbCube3d};

/// Parse a UTF-8 `.cube` string. Skips blanks, `#` comments, `TITLE` /
/// `DOMAIN_*`, strips a leading BOM, then expects `LUT_3D_SIZE N` and `N³` RGB
/// lines (three finite floats each).
///
/// # Errors
///
/// Returns [`LutCubeParseError`] when the text does not match the supported
/// `.cube` subset (missing header, wrong sample count, malformed floats, etc.).
pub(crate) fn parse_adobe_cube_str(
    input: &str,
) -> Result<LutRgbCube3d, LutCubeParseError> {
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

                // Header line is valid syntax only; check allowed N here.
                if !(2..=LutRgbCube3d::MAX_SIZE).contains(&n)
                    || expected_lut_sample_count(n).is_none()
                {
                    return Err(LutCubeParseError::InvalidLutSize { size: n });
                }

                lut_size = Some(n);
            }
            Some(lut_sz) => {
                let expected_len = expected_lut_sample_count(lut_sz).ok_or(
                    LutCubeParseError::InvalidLutSize { size: lut_sz },
                )?;

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

    LutRgbCube3d::new(lut_sz, rgb)
}

/// UTF-8 bytes → [`parse_adobe_cube_str`].
///
/// # Errors
///
/// Returns [`LutCubeParseError::InvalidUtf8`] if `input` is not valid UTF-8,
/// or any error from [`parse_adobe_cube_str`] otherwise.
pub(crate) fn parse_adobe_cube_bytes(
    input: &[u8],
) -> Result<LutRgbCube3d, LutCubeParseError> {
    let text = std::str::from_utf8(input)
        .map_err(|_| LutCubeParseError::InvalidUtf8)?;
    parse_adobe_cube_str(text)
}

/// Strip `#` comments; line must already be [`str::trim`]'d. [`None`] means
/// skip the line.
fn meaningful_cube_line(trimmed_physical_line: &str) -> Option<&str> {
    if trimmed_physical_line.starts_with('#') {
        return None;
    }

    let before_hash =
        trimmed_physical_line.split('#').next().unwrap_or("").trim();

    (!before_hash.is_empty()).then_some(before_hash)
}

/// First word is `TITLE`, `DOMAIN_MIN`, or `DOMAIN_MAX` (skip whole line).
fn is_adobe_cube_metadata_line(meaningful_line: &str) -> bool {
    let mut tokens = meaningful_line.split_whitespace();
    let Some(head) = tokens.next() else {
        return false;
    };

    matches!(head, "TITLE" | "DOMAIN_MIN" | "DOMAIN_MAX")
}

/// Expect exactly `LUT_3D_SIZE N`.
fn parse_lut_size_line(
    line: &str,
    line_no: usize,
) -> Result<u32, LutCubeParseError> {
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

/// One float; reject NaN and infinity.
fn parse_finite_f32(
    token: &str,
    line_no: usize,
) -> Result<f32, LutCubeParseError> {
    let value = token
        .parse::<f32>()
        .map_err(|_| LutCubeParseError::MalformedRgbLine { line: line_no })?;

    if !value.is_finite() {
        return Err(LutCubeParseError::MalformedRgbLine { line: line_no });
    }

    Ok(value)
}

/// One RGB sample: three floats, no extra tokens.
fn parse_rgb_triplet_line(
    line: &str,
    line_no: usize,
) -> Result<[f32; 3], LutCubeParseError> {
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
