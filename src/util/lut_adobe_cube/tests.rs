//! Tests for the `lut_adobe_cube` module.

use bytemuck::pod_read_unaligned;
use half::f16;

use super::types::lattice_xyz_for_sample_index;
use super::{
    expected_lut_sample_count, parse_adobe_cube_bytes, parse_adobe_cube_str,
    LutCubeParseError, LutRgbCube3d,
};

#[test]
fn expected_lut_sample_count_matches_size_cubed_for_small_sizes() {
    assert_eq!(expected_lut_sample_count(2), Some(8));
    assert_eq!(expected_lut_sample_count(3), Some(27));
}

#[test]
fn new_accepts_n2_corner_lut() {
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

    let lut = LutRgbCube3d::new(2, rgb).expect("valid 2³ LUT");
    assert_eq!(lut.size, 2);
    assert_eq!(lut.rgb.len(), 8);
}

#[test]
fn new_rejects_wrong_sample_count() {
    let err = LutRgbCube3d::new(2, vec![[0.0, 0.0, 0.0]])
        .expect_err("too few samples");

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
    let err = parse_adobe_cube_str("0 0 0\nLUT_3D_SIZE 2\n")
        .expect_err("order matters");
    assert_eq!(err, LutCubeParseError::InvalidLutSizeLine { line: 1 });
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
    let err = parse_adobe_cube_str("LUT_3D_SIZE 2\n0 0 0\n")
        .expect_err("incomplete LUT");
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
    let err = parse_adobe_cube_str("LUT_3D_SIZE 2\nnot_float 0 0\n")
        .expect_err("bad float token");
    assert_eq!(err, LutCubeParseError::MalformedRgbLine { line: 2 });
}

#[test]
fn parse_rejects_nan_rgb_value() {
    let err = parse_adobe_cube_str("LUT_3D_SIZE 2\nNaN 0 0\n")
        .expect_err("NaN must fail");

    assert_eq!(err, LutCubeParseError::MalformedRgbLine { line: 2 });
}

#[test]
fn parse_rejects_infinite_rgb_value() {
    let err = parse_adobe_cube_str("LUT_3D_SIZE 2\ninf 0 0\n")
        .expect_err("inf must fail");

    assert_eq!(err, LutCubeParseError::MalformedRgbLine { line: 2 });
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
    let err =
        parse_adobe_cube_str(cube).expect_err("unexpected ninth sample row");
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
    let src = "LUT_3D_SIZE 2\n0 0 0\n1 0 0\n0 1 0\n1 1 0\n0 0 1\n1 0 1\n0 1 \
               1\n1 1 1\n";

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
    let src = "LUT_3D_SIZE 2\n\n0 0 0\n\n1 0 0\n  \n0 1 0\n1 1 0\n0 0 1\n1 0 \
               1\n0 1 1\n1 1 1\n";

    let lut =
        parse_adobe_cube_str(src).expect("blank lines should not break parse");
    assert_eq!(lut.size, 2);
    assert_eq!(lut.rgb.len(), 8);
    assert_eq!(lut.rgb[0], [0.0, 0.0, 0.0]);
    assert_eq!(lut.rgb[7], [1.0, 1.0, 1.0]);
}

#[test]
fn parse_rejects_rgb_line_with_too_few_tokens() {
    let err = parse_adobe_cube_str("LUT_3D_SIZE 2\n0 0\n")
        .expect_err("need three floats");
    assert_eq!(err, LutCubeParseError::MalformedRgbLine { line: 2 });
}

#[test]
fn parse_rejects_rgb_line_with_too_many_tokens() {
    let err = parse_adobe_cube_str("LUT_3D_SIZE 2\n0 0 0 1\n")
        .expect_err("extra token");
    assert_eq!(err, LutCubeParseError::MalformedRgbLine { line: 2 });
}

#[test]
fn parse_rejects_lut_size_line_with_trailing_token() {
    let err = parse_adobe_cube_str("LUT_3D_SIZE 2 extra\n")
        .expect_err("header must be exactly two tokens after split");
    assert_eq!(err, LutCubeParseError::InvalidLutSizeLine { line: 1 });
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
    let err = parse_adobe_cube_str(
        "LUT_3D_SIZE 2 junk # not a comment separator for tokens\n",
    )
    .expect_err("junk before # is still an extra token");

    assert_eq!(err, LutCubeParseError::InvalidLutSizeLine { line: 1 });
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
    let lut_hash = parse_adobe_cube_str(WITH_HASH)
        .expect("#-comment LUT fixture must parse");

    assert_eq!(lut_strict.size, lut_hash.size);
    assert_eq!(lut_strict.rgb, lut_hash.rgb);
}

#[test]
fn parse_skips_title_and_domain_lines_before_lut_size() {
    let src = "TITLE \"warm grade\"\nDOMAIN_MIN 0 0 0\nDOMAIN_MAX 1 1 \
               1\nLUT_3D_SIZE 2\n0 0 0\n1 0 0\n0 1 0\n1 1 0\n0 0 1\n1 0 1\n0 \
               1 1\n1 1 1\n";

    let lut =
        parse_adobe_cube_str(src).expect("TITLE/DOMAIN prefix must parse");
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
    let lut_meta = parse_adobe_cube_str(WITH_META)
        .expect("TITLE/DOMAIN LUT fixture must parse");

    assert_eq!(lut_strict.size, lut_meta.size);
    assert_eq!(lut_strict.rgb, lut_meta.rgb);
}

#[test]
fn parse_accepts_leading_utf8_bom() {
    let inner = "LUT_3D_SIZE 2\n0 0 0\n1 0 0\n0 1 0\n1 1 0\n0 0 1\n1 0 1\n0 1 \
                 1\n1 1 1\n";
    let with_bom = format!("{}{}", '\u{FEFF}', inner);

    let lut_plain = parse_adobe_cube_str(inner).expect("plain LUT must parse");
    let lut_bom =
        parse_adobe_cube_str(&with_bom).expect("BOM-prefixed LUT must parse");

    assert_eq!(lut_plain, lut_bom);
}

#[test]
fn parse_bytes_rejects_invalid_utf8() {
    let err =
        parse_adobe_cube_bytes(&[0xff, 0xfe, 0xfd]).expect_err("invalid UTF-8");
    assert_eq!(err, LutCubeParseError::InvalidUtf8);
}

#[test]
fn parse_bytes_accepts_utf8_bom_and_matches_str_parse() {
    let inner = "LUT_3D_SIZE 2\n0 0 0\n1 0 0\n0 1 0\n1 1 0\n0 0 1\n1 0 1\n0 1 \
                 1\n1 1 1\n";
    let mut bytes = vec![0xef_u8, 0xbb, 0xbf];
    bytes.extend_from_slice(inner.as_bytes());

    let lut_bytes =
        parse_adobe_cube_bytes(&bytes).expect("UTF-8 BOM bytes must parse");
    let lut_str = parse_adobe_cube_str(inner).expect("plain string must parse");

    assert_eq!(lut_bytes, lut_str);
}

// CPU and GPU layout matches see `types.rs`
// tests for lattice and rgba helpers matches `minimal_n2` fixture

#[test]
// k=0..7 maps to 8 corners (0,0,0) .. (1,1,1)
// k=8 is out of range
fn lattice_n2_maps_sample_index_k_to_rgb_axes_xyz() {
    let n = 2;
    let corners = [
        (0, 0, 0),
        (1, 0, 0),
        (0, 1, 0),
        (1, 1, 0),
        (0, 0, 1),
        (1, 0, 1),
        (0, 1, 1),
        (1, 1, 1),
    ];
    for (k, want) in corners.iter().enumerate() {
        let got = lattice_xyz_for_sample_index(n, k).expect("in range");
        assert_eq!(got, *want, "k={k}");
    }
    assert!(lattice_xyz_for_sample_index(n, 8).is_none());
}

#[test]
// parse minimal 2³ LUT, assert f32 texel matches first and last corner
fn rgba_volume_helpers_match_minimal_n2_fixture() {
    let src = "LUT_3D_SIZE 2\n0 0 0\n1 0 0\n0 1 0\n1 1 0\n0 0 1\n1 0 1\n0 1 \
               1\n1 1 1\n";
    let lut = parse_adobe_cube_str(src).expect("minimal N=2 LUT");

    let texels = lut.rgba_f32_volume_texels();
    assert_eq!(texels.len(), 8);
    assert_eq!(texels[0], [0.0, 0.0, 0.0, 1.0]);
    assert_eq!(texels[7], [1.0, 1.0, 1.0, 1.0]);

    let bytes = lut.rgba_bytes_volume_order();
    assert_eq!(bytes.len(), 8_usize.saturating_mul(16));

    let first = pod_read_unaligned::<[f32; 4]>(&bytes[0..16]);
    let last = pod_read_unaligned::<[f32; 4]>(&bytes[8 * 16 - 16..8 * 16]);
    assert_eq!(first, [0.0, 0.0, 0.0, 1.0]);
    assert_eq!(last, [1.0, 1.0, 1.0, 1.0]);

    let bytes16 = lut.rgba16f_bytes_volume_order();
    assert_eq!(bytes16.len(), 8_usize.saturating_mul(8));
    let first16 = [
        f16::from_le_bytes([bytes16[0], bytes16[1]]),
        f16::from_le_bytes([bytes16[2], bytes16[3]]),
        f16::from_le_bytes([bytes16[4], bytes16[5]]),
        f16::from_le_bytes([bytes16[6], bytes16[7]]),
    ];
    let last16 = [
        f16::from_le_bytes([bytes16[56], bytes16[57]]),
        f16::from_le_bytes([bytes16[58], bytes16[59]]),
        f16::from_le_bytes([bytes16[60], bytes16[61]]),
        f16::from_le_bytes([bytes16[62], bytes16[63]]),
    ];
    assert_eq!(first16.map(f16::to_f32), [0.0, 0.0, 0.0, 1.0]);
    assert_eq!(last16.map(f16::to_f32), [1.0, 1.0, 1.0, 1.0]);
}

#[test]
// k == N³ is out of range, must return None
fn lattice_rejects_out_of_range_k() {
    assert!(lattice_xyz_for_sample_index(2, 9).is_none());
    assert!(lattice_xyz_for_sample_index(3, 27).is_none());
}
