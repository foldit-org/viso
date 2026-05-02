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
mod tests {
    use super::{
        expected_lut_sample_count, parse_adobe_cube_bytes, parse_adobe_cube_str,
        LutCubeParseError, LutRgbF32Cube3d,
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
    fn parse_rejects_nan_rgb_value() {
        let err = parse_adobe_cube_str("LUT_3D_SIZE 2\nNaN 0 0\n").expect_err("NaN must fail");

        assert_eq!(
            err,
            LutCubeParseError::MalformedRgbLine { line: 2 }
        );
    }

    #[test]
    fn parse_rejects_infinite_rgb_value() {
        let err = parse_adobe_cube_str("LUT_3D_SIZE 2\ninf 0 0\n").expect_err("inf must fail");

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

    #[test]
    fn parse_accepts_hash_comments_and_inline_hash_on_header() {
        let src = "\
# prelude
LUT_3D_SIZE 2  # grid
# before samples
0 0 0 # black
1 0 0
0 1 0 # mid
1 1 0
#
0 0 1
1 0 1 # etc
0 1 1
1 1 1
# tail
";

        let lut = parse_adobe_cube_str(src).expect("hash comments should parse");
        assert_eq!(lut.size, 2);
        assert_eq!(lut.rgb.len(), 8);
        assert_eq!(lut.rgb[0], [0.0, 0.0, 0.0]);
    }

    #[test]
    fn parse_lut_size_trailing_token_still_fails_when_not_in_comment() {
        let err =
            parse_adobe_cube_str("LUT_3D_SIZE 2 junk # not a comment separator for tokens\n")
                .expect_err("junk before # is still an extra token");

        assert_eq!(
            err,
            LutCubeParseError::InvalidLutSizeLine { line: 1 }
        );
    }

    #[test]
    fn parse_fixture_minimal_n2_without_comments() {
        const SRC: &str = include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/testdata/lut/minimal_n2.cube"
        ));

        let lut = parse_adobe_cube_str(SRC).expect("testdata fixture must parse");
        assert_eq!(lut.size, 2);
        assert_eq!(lut.rgb.len(), 8);
        assert_eq!(lut.rgb[0], [0.0, 0.0, 0.0]);
        assert_eq!(lut.rgb[7], [1.0, 1.0, 1.0]);
    }

    #[test]
    fn parse_fixture_minimal_n2_hash_comments_matches_strict_fixture_rgb() {
        const STRICT: &str = include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/testdata/lut/minimal_n2.cube"
        ));
        const WITH_HASH: &str = include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/testdata/lut/minimal_n2_hash_comments.cube"
        ));

        let lut_strict =
            parse_adobe_cube_str(STRICT).expect("strict LUT fixture must parse");
        let lut_hash =
            parse_adobe_cube_str(WITH_HASH).expect("#-comment LUT fixture must parse");

        assert_eq!(lut_strict.size, lut_hash.size);
        assert_eq!(lut_strict.rgb, lut_hash.rgb);
    }

    #[test]
    fn parse_skips_title_and_domain_lines_before_lut_size() {
        let src = "TITLE \"warm grade\"\n\
             DOMAIN_MIN 0 0 0\n\
             DOMAIN_MAX 1 1 1\n\
             LUT_3D_SIZE 2\n\
             0 0 0\n1 0 0\n0 1 0\n1 1 0\n\
             0 0 1\n1 0 1\n0 1 1\n1 1 1\n";

        let lut = parse_adobe_cube_str(src).expect("TITLE/DOMAIN prefix must parse");
        assert_eq!(lut.size, 2);
        assert_eq!(lut.rgb.len(), 8);
        assert_eq!(lut.rgb[7], [1.0, 1.0, 1.0]);
    }

    #[test]
    fn parse_fixture_minimal_n2_metadata_matches_strict_fixture_rgb() {
        const STRICT: &str = include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/testdata/lut/minimal_n2.cube"
        ));
        const WITH_META: &str = include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/testdata/lut/minimal_n2_metadata.cube"
        ));

        let lut_strict =
            parse_adobe_cube_str(STRICT).expect("strict LUT fixture must parse");
        let lut_meta =
            parse_adobe_cube_str(WITH_META).expect("TITLE/DOMAIN LUT fixture must parse");

        assert_eq!(lut_strict.size, lut_meta.size);
        assert_eq!(lut_strict.rgb, lut_meta.rgb);
    }

    #[test]
    fn parse_accepts_leading_utf8_bom() {
        let inner = "LUT_3D_SIZE 2\n\
             0 0 0\n1 0 0\n0 1 0\n1 1 0\n\
             0 0 1\n1 0 1\n0 1 1\n1 1 1\n";
        let with_bom = format!("{}{}", '\u{FEFF}', inner);

        let lut_plain = parse_adobe_cube_str(inner).expect("plain LUT must parse");
        let lut_bom = parse_adobe_cube_str(&with_bom).expect("BOM-prefixed LUT must parse");

        assert_eq!(lut_plain, lut_bom);
    }

    #[test]
    fn parse_bytes_rejects_invalid_utf8() {
        let err = parse_adobe_cube_bytes(&[0xff, 0xfe, 0xfd]).expect_err("invalid UTF-8");
        assert_eq!(err, LutCubeParseError::InvalidUtf8);
    }

    #[test]
    fn parse_bytes_accepts_utf8_bom_and_matches_str_parse() {
        let inner = "LUT_3D_SIZE 2\n\
             0 0 0\n1 0 0\n0 1 0\n1 1 0\n\
             0 0 1\n1 0 1\n0 1 1\n1 1 1\n";
        let mut bytes = vec![0xef_u8, 0xbb, 0xbf];
        bytes.extend_from_slice(inner.as_bytes());

        let lut_bytes = parse_adobe_cube_bytes(&bytes).expect("UTF-8 BOM bytes must parse");
        let lut_str = parse_adobe_cube_str(inner).expect("plain string must parse");

        assert_eq!(lut_bytes, lut_str);
    }
}
