//! Unit tests for the parent `overrides` module. Kept in a sibling
//! file (via `#[path]` on the `mod tests` declaration) so
//! `overrides.rs` stays under the 800-line `just file-lengths`
//! gate. Test-only code; not part of the public surface.

use super::*;

fn sample_global() -> DisplayOverrides {
    DisplayOverrides {
        drawing_mode: Some(DrawingMode::Cartoon),
        color_scheme: Some(ColorScheme::Entity),
        show_sidechains: Some(true),
        ..Default::default()
    }
}

#[test]
fn default_is_empty() {
    let a = DisplayOverrides::default();
    assert!(a.is_empty());
}

#[test]
fn overlay_inherits_from_base() {
    let base = sample_global();
    let entity = DisplayOverrides::default();
    let overlaid = entity.overlay(&base);
    assert_eq!(overlaid.drawing_mode, Some(DrawingMode::Cartoon));
    assert_eq!(overlaid.show_sidechains, Some(true));
}

#[test]
fn overlay_self_wins() {
    let base = sample_global();
    let entity = DisplayOverrides {
        drawing_mode: Some(DrawingMode::BallAndStick),
        ..Default::default()
    };
    let overlaid = entity.overlay(&base);
    assert_eq!(overlaid.drawing_mode, Some(DrawingMode::BallAndStick));
    // Other fields still inherited
    assert_eq!(overlaid.color_scheme, Some(ColorScheme::Entity));
}

#[test]
fn overlay_is_associative() {
    let a = DisplayOverrides {
        drawing_mode: Some(DrawingMode::Stick),
        ..Default::default()
    };
    let b = DisplayOverrides {
        color_scheme: Some(ColorScheme::BFactor),
        ..Default::default()
    };
    let c = DisplayOverrides {
        show_sidechains: Some(true),
        ..Default::default()
    };
    let lhs = a.overlay(&b).overlay(&c);
    let rhs = a.overlay(&b.overlay(&c));
    assert_eq!(lhs, rhs);
}

#[test]
fn round_trip_serde() {
    let a = sample_global();
    let json = serde_json::to_string(&a).unwrap();
    let b: DisplayOverrides = serde_json::from_str(&json).unwrap();
    assert_eq!(a, b);
}

#[test]
fn legacy_toml_aliases_accepted() {
    // Existing TOML used `backbone_color_scheme` and
    // `backbone_palette_*` keys at the [display] level. After the
    // refactor these flatten into DisplayOverrides via aliases.
    let toml_str = r#"
backbone_color_scheme = "b_factor"
backbone_palette_preset = "viridis"
"#;
    let parsed: DisplayOverrides = toml::from_str(toml_str).unwrap();
    assert_eq!(parsed.color_scheme, Some(ColorScheme::BFactor));
    assert_eq!(parsed.palette_preset, Some(PalettePreset::Viridis),);
}

#[test]
fn to_display_options_patches_all_fields() {
    let base = DisplayOptions::default();
    let ovr = DisplayOverrides {
        drawing_mode: Some(DrawingMode::Stick),
        color_scheme: Some(ColorScheme::BFactor),
        show_sidechains: Some(false),
        ..Default::default()
    };
    let result = ovr.to_display_options(&base);
    assert_eq!(result.drawing_mode(), DrawingMode::Stick);
    assert_eq!(result.backbone_color_scheme(), ColorScheme::BFactor);
    assert!(!result.show_sidechains());
    // Unset fields pass through from base
    assert_eq!(result.helix_style(), base.helix_style());
}

// RenderInvalidation tests

#[test]
fn invalidation_none_is_empty() {
    assert!(RenderInvalidation::NONE.is_empty());
    assert!(!RenderInvalidation::RE_MESH.is_empty());
}

#[test]
fn invalidation_bitor_and_contains() {
    let combined = RenderInvalidation::RE_MESH | RenderInvalidation::RE_COLOR;
    assert!(combined.contains(RenderInvalidation::RE_MESH));
    assert!(combined.contains(RenderInvalidation::RE_COLOR));
    assert!(!combined.contains(RenderInvalidation::RE_SURFACE));
}

#[test]
fn diff_identical_returns_none() {
    let a = sample_global();
    assert_eq!(a.diff(&a), RenderInvalidation::NONE);
}

#[test]
fn diff_drawing_mode_sets_resolve_and_mesh() {
    let a = DisplayOverrides::default();
    let b = DisplayOverrides {
        drawing_mode: Some(DrawingMode::Stick),
        ..Default::default()
    };
    let inv = a.diff(&b);
    assert!(inv.contains(RenderInvalidation::DRAWING_MODE_RESOLVE));
    assert!(inv.contains(RenderInvalidation::RE_MESH));
    assert!(!inv.contains(RenderInvalidation::RE_SURFACE));
}

#[test]
fn diff_color_scheme_sets_color_and_mesh() {
    let a = DisplayOverrides::default();
    let b = DisplayOverrides {
        color_scheme: Some(ColorScheme::BFactor),
        ..Default::default()
    };
    let inv = a.diff(&b);
    assert!(inv.contains(RenderInvalidation::RE_COLOR));
    assert!(inv.contains(RenderInvalidation::RE_MESH));
    assert!(!inv.contains(RenderInvalidation::DRAWING_MODE_RESOLVE));
}

#[test]
fn diff_surface_kind_sets_re_surface() {
    // Previously a bug: per-entity surface_kind changes never
    // triggered surface regeneration. RE_SURFACE must fire.
    let a = DisplayOverrides::default();
    let b = DisplayOverrides {
        surface_kind: Some(SurfaceKindOption::Gaussian),
        ..Default::default()
    };
    let inv = a.diff(&b);
    assert!(inv.contains(RenderInvalidation::RE_SURFACE));
    assert!(!inv.contains(RenderInvalidation::RE_MESH));
}

#[test]
fn diff_surface_opacity_sets_opacity_not_surface() {
    // Global surface opacity is a shader uniform, so an opacity-only
    // change must fire RE_SURFACE_OPACITY (cheap uniform write at global
    // scope) and NOT RE_SURFACE (which would re-mesh).
    let a = DisplayOverrides::default();
    let b = DisplayOverrides {
        surface_opacity: Some(0.5),
        ..Default::default()
    };
    let inv = a.diff(&b);
    assert!(inv.contains(RenderInvalidation::RE_SURFACE_OPACITY));
    assert!(!inv.contains(RenderInvalidation::RE_SURFACE));
    assert!(!inv.contains(RenderInvalidation::RE_MESH));
}

#[test]
fn diff_surface_kind_not_opacity() {
    // The converse split guard: a surface_kind / show_cavities change
    // fires RE_SURFACE but not RE_SURFACE_OPACITY.
    let a = DisplayOverrides::default();
    let b = DisplayOverrides {
        surface_kind: Some(SurfaceKindOption::Gaussian),
        show_cavities: Some(true),
        ..Default::default()
    };
    let inv = a.diff(&b);
    assert!(inv.contains(RenderInvalidation::RE_SURFACE));
    assert!(!inv.contains(RenderInvalidation::RE_SURFACE_OPACITY));
}

#[test]
fn diff_helix_style_sets_lod_and_mesh() {
    let a = DisplayOverrides::default();
    let b = DisplayOverrides {
        helix_style: Some(HelixStyle::Cylinder),
        ..Default::default()
    };
    let inv = a.diff(&b);
    assert!(inv.contains(RenderInvalidation::LOD_REMESH));
    assert!(inv.contains(RenderInvalidation::RE_MESH));
}

#[test]
fn diff_bond_style_sets_mesh_only() {
    let a = DisplayOverrides::default();
    let b = DisplayOverrides {
        show_hbonds: Some(true),
        ..Default::default()
    };
    let inv = a.diff(&b);
    assert!(inv.contains(RenderInvalidation::RE_MESH));
    assert!(!inv.contains(RenderInvalidation::RE_SURFACE));
    assert!(!inv.contains(RenderInvalidation::RE_COLOR));
}

#[test]
fn diff_simultaneous_changes_union() {
    // Regression test for the historical triple-sync bug: multiple
    // concurrent field changes should OR into a single invalidation
    // set with each kind firing at most once (dedup is structural).
    let a = DisplayOverrides::default();
    let b = DisplayOverrides {
        drawing_mode: Some(DrawingMode::Stick),
        color_scheme: Some(ColorScheme::BFactor),
        helix_style: Some(HelixStyle::Tube),
        surface_kind: Some(SurfaceKindOption::Gaussian),
        ..Default::default()
    };
    let inv = a.diff(&b);
    assert!(inv.contains(RenderInvalidation::DRAWING_MODE_RESOLVE));
    assert!(inv.contains(RenderInvalidation::RE_MESH));
    assert!(inv.contains(RenderInvalidation::RE_COLOR));
    assert!(inv.contains(RenderInvalidation::LOD_REMESH));
    assert!(inv.contains(RenderInvalidation::RE_SURFACE));
}

// apply_json_field contract tests

#[test]
fn apply_json_null_clears_field() {
    let mut o = DisplayOverrides {
        color_scheme: Some(ColorScheme::BFactor),
        ..Default::default()
    };
    let r = o.apply_json_field("color_scheme", &serde_json::Value::Null);
    assert!(r.is_ok());
    assert_eq!(o.color_scheme, None);
}

#[test]
fn apply_json_null_clears_bool_field() {
    let mut o = DisplayOverrides {
        show_sidechains: Some(true),
        ..Default::default()
    };
    let r = o.apply_json_field("show_sidechains", &serde_json::Value::Null);
    assert!(r.is_ok());
    assert_eq!(o.show_sidechains, None);
}

#[test]
fn apply_json_null_clears_f32_field() {
    let mut o = DisplayOverrides {
        surface_opacity: Some(0.5),
        ..Default::default()
    };
    let r = o.apply_json_field("surface_opacity", &serde_json::Value::Null);
    assert!(r.is_ok());
    assert_eq!(o.surface_opacity, None);
}

#[test]
fn apply_json_valid_sets_enum_field() {
    let mut o = DisplayOverrides::default();
    let r = o.apply_json_field("color_scheme", &serde_json::json!("b_factor"));
    assert!(r.is_ok());
    assert_eq!(o.color_scheme, Some(ColorScheme::BFactor));
}

#[test]
fn apply_json_valid_sets_bool_field() {
    let mut o = DisplayOverrides::default();
    let r = o.apply_json_field("show_sidechains", &serde_json::json!(true));
    assert!(r.is_ok());
    assert_eq!(o.show_sidechains, Some(true));
}

#[test]
fn apply_json_valid_sets_f32_field() {
    let mut o = DisplayOverrides::default();
    let r = o.apply_json_field("surface_opacity", &serde_json::json!(0.25));
    assert!(r.is_ok());
    assert_eq!(o.surface_opacity, Some(0.25));
}

#[test]
fn apply_json_malformed_enum_errs_and_preserves() {
    // Core regression: a typo'd enum variant must NOT wipe the
    // existing override. It returns Err and leaves the slot intact.
    let mut o = DisplayOverrides {
        color_scheme: Some(ColorScheme::BFactor),
        ..Default::default()
    };
    let r = o.apply_json_field(
        "color_scheme",
        &serde_json::json!("not_a_real_enum_variant"),
    );
    assert_eq!(r, Err("color_scheme"));
    assert_eq!(o.color_scheme, Some(ColorScheme::BFactor));
}

#[test]
fn apply_json_malformed_bool_errs_and_preserves() {
    let mut o = DisplayOverrides {
        show_sidechains: Some(true),
        ..Default::default()
    };
    // A string where a bool is expected must be rejected.
    let r = o.apply_json_field("show_sidechains", &serde_json::json!("yes"));
    assert_eq!(r, Err("show_sidechains"));
    assert_eq!(o.show_sidechains, Some(true));
}

#[test]
fn apply_json_malformed_f32_errs_and_preserves() {
    let mut o = DisplayOverrides {
        surface_opacity: Some(0.5),
        ..Default::default()
    };
    let r = o.apply_json_field("surface_opacity", &serde_json::json!("opaque"));
    assert_eq!(r, Err("surface_opacity"));
    assert_eq!(o.surface_opacity, Some(0.5));
}

#[test]
fn apply_json_unknown_field_errs() {
    let mut o = DisplayOverrides::default();
    let r = o.apply_json_field("not_a_field", &serde_json::json!(true));
    assert_eq!(r, Err("not_a_field"));
}

#[test]
fn apply_json_show_cavities_contract() {
    let mut o = DisplayOverrides {
        show_cavities: Some(true),
        ..Default::default()
    };
    // null clears
    assert!(o
        .apply_json_field("show_cavities", &serde_json::Value::Null)
        .is_ok());
    assert_eq!(o.show_cavities, None);
    // valid sets
    assert!(o
        .apply_json_field("show_cavities", &serde_json::json!(true))
        .is_ok());
    assert_eq!(o.show_cavities, Some(true));
    // malformed errs and preserves
    let r = o.apply_json_field("show_cavities", &serde_json::json!("yes"));
    assert_eq!(r, Err("show_cavities"));
    assert_eq!(o.show_cavities, Some(true));
}

#[test]
fn apply_json_show_hydrogens_contract() {
    let mut o = DisplayOverrides {
        show_hydrogens: Some(true),
        ..Default::default()
    };
    assert!(o
        .apply_json_field("show_hydrogens", &serde_json::Value::Null)
        .is_ok());
    assert_eq!(o.show_hydrogens, None);
    assert!(o
        .apply_json_field("show_hydrogens", &serde_json::json!(true))
        .is_ok());
    assert_eq!(o.show_hydrogens, Some(true));
    let r = o.apply_json_field("show_hydrogens", &serde_json::json!(3));
    assert_eq!(r, Err("show_hydrogens"));
    assert_eq!(o.show_hydrogens, Some(true));
}

#[test]
fn apply_json_sidechain_color_mode_contract() {
    let mut o = DisplayOverrides {
        sidechain_color_mode: Some(SidechainColorMode::Backbone),
        ..Default::default()
    };
    assert!(o
        .apply_json_field("sidechain_color_mode", &serde_json::Value::Null)
        .is_ok());
    assert_eq!(o.sidechain_color_mode, None);
    assert!(o
        .apply_json_field(
            "sidechain_color_mode",
            &serde_json::json!("hydrophobicity")
        )
        .is_ok());
    assert_eq!(
        o.sidechain_color_mode,
        Some(SidechainColorMode::Hydrophobicity)
    );
    let r = o.apply_json_field(
        "sidechain_color_mode",
        &serde_json::json!("not_a_variant"),
    );
    assert_eq!(r, Err("sidechain_color_mode"));
    assert_eq!(
        o.sidechain_color_mode,
        Some(SidechainColorMode::Hydrophobicity)
    );
}

#[test]
fn apply_json_na_color_mode_contract() {
    let mut o = DisplayOverrides {
        na_color_mode: Some(NaColorMode::BaseColor),
        ..Default::default()
    };
    assert!(o
        .apply_json_field("na_color_mode", &serde_json::Value::Null)
        .is_ok());
    assert_eq!(o.na_color_mode, None);
    assert!(o
        .apply_json_field("na_color_mode", &serde_json::json!("uniform"))
        .is_ok());
    assert_eq!(o.na_color_mode, Some(NaColorMode::Uniform));
    let r = o
        .apply_json_field("na_color_mode", &serde_json::json!("not_a_variant"));
    assert_eq!(r, Err("na_color_mode"));
    assert_eq!(o.na_color_mode, Some(NaColorMode::Uniform));
}

#[test]
fn apply_json_lipid_mode_contract() {
    let mut o = DisplayOverrides {
        lipid_mode: Some(LipidMode::Coarse),
        ..Default::default()
    };
    assert!(o
        .apply_json_field("lipid_mode", &serde_json::Value::Null)
        .is_ok());
    assert_eq!(o.lipid_mode, None);
    assert!(o
        .apply_json_field("lipid_mode", &serde_json::json!("ball_and_stick"))
        .is_ok());
    assert_eq!(o.lipid_mode, Some(LipidMode::BallAndStick));
    let r =
        o.apply_json_field("lipid_mode", &serde_json::json!("not_a_variant"));
    assert_eq!(r, Err("lipid_mode"));
    assert_eq!(o.lipid_mode, Some(LipidMode::BallAndStick));
}

#[test]
fn color_scheme_serde_round_trip_does_not_self_brick() {
    // Regression guard: `color_scheme` carries a serde alias
    // (`backbone_color_scheme`). Serialization must emit exactly one key,
    // and the round-trip must deserialize cleanly so a subsequent option
    // set is not bricked by a duplicate flattened field.
    let o = DisplayOverrides {
        color_scheme: Some(ColorScheme::BFactor),
        show_sidechains: Some(true),
        ..Default::default()
    };
    let json = serde_json::to_value(&o).unwrap();
    let obj = json.as_object().unwrap();
    assert!(obj.contains_key("color_scheme"));
    assert!(!obj.contains_key("backbone_color_scheme"));

    let back: DisplayOverrides = serde_json::from_value(json).unwrap();
    assert_eq!(back, o);

    // A subsequent apply on the round-tripped value still works.
    let mut back = back;
    assert!(back
        .apply_json_field("color_scheme", &serde_json::json!("entity"))
        .is_ok());
    assert_eq!(back.color_scheme, Some(ColorScheme::Entity));
}
