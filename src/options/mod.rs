//! Centralized rendering/display options with TOML preset support.
//!
//! All tweakable settings (lighting, post-processing, camera, colors, geometry,
//! display toggles) are consolidated here. Options serialize to/from TOML for
//! view presets stored in `assets/view_presets/`.
//!
//! Key bindings live in [`crate::input::KeyBindings`], not here --
//! they are an input concern, not a rendering option.

mod camera;
mod colors;
mod debug;
mod display;
mod geometry;
mod lighting;
/// Display override bag (used globally and per-entity).
pub mod overrides;
/// Color palette system.
pub mod palette;
mod post_processing;
/// Score-to-color gradient mapping.
pub(crate) mod score_color;

#[cfg(not(target_arch = "wasm32"))]
use std::path::Path;

pub use camera::CameraOptions;
pub use colors::ColorOptions;
pub use debug::DebugOptions;
pub use display::{
    BackboneColorMode, BondOptions, BondSource, BondStyle, BondTypeOptions,
    ColorScheme, DisplayOptions, DrawingMode, HelixStyle, LipidMode,
    NaColorMode, PresentMode, SheetStyle, SidechainColorMode,
    SurfaceKindOption,
};
pub use geometry::{
    lod_params, lod_scaled, select_chain_lod_tier, select_lod_tier,
    CartoonStyle, ChainLod, GeometryOptions,
};
pub use lighting::LightingOptions;
pub use overrides::DisplayOverrides;
pub use palette::{Palette, PaletteMode, PalettePreset};
pub use post_processing::PostProcessingOptions;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::error::VisoError;

/// Top-level options container. All sub-structs use `#[serde(default)]` so
/// partial TOML files (e.g. only overriding `[lighting]`) work correctly.
#[derive(
    Debug, Clone, Serialize, Deserialize, PartialEq, Default, JsonSchema,
)]
#[serde(default)]
pub struct VisoOptions {
    /// Display toggles and coloring modes.
    pub display: DisplayOptions,
    /// Lighting parameters.
    pub lighting: LightingOptions,
    /// Post-processing effect parameters.
    pub post_processing: PostProcessingOptions,
    /// Camera projection and control parameters.
    pub camera: CameraOptions,
    /// Color palette options.
    #[schemars(skip)]
    pub colors: ColorOptions,
    /// Backbone and ligand geometry options.
    pub geometry: GeometryOptions,
    /// Debug visualization options.
    pub debug: DebugOptions,
}

impl VisoOptions {
    /// Generate JSON Schema describing the UI-exposed options.
    #[must_use]
    pub fn json_schema() -> schemars::Schema {
        schemars::schema_for!(VisoOptions)
    }

    /// Load options from a TOML file. Missing fields use defaults.
    ///
    /// # Errors
    ///
    /// Returns [`VisoError::Io`] if the file cannot be read, or
    /// [`VisoError::OptionsParse`] if the TOML content is invalid.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn load(path: &Path) -> Result<Self, VisoError> {
        let content = std::fs::read_to_string(path).map_err(VisoError::Io)?;
        toml::from_str(&content)
            .map_err(|e| VisoError::OptionsParse(e.to_string()))
    }

    /// Save options to a TOML file (pretty-printed).
    ///
    /// # Errors
    ///
    /// Returns [`VisoError::Io`] if the file cannot be written, or
    /// [`VisoError::OptionsParse`] if serialization fails.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn save(&self, path: &Path) -> Result<(), VisoError> {
        let content = toml::to_string_pretty(self)
            .map_err(|e| VisoError::OptionsParse(e.to_string()))?;
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).map_err(VisoError::Io)?;
        }
        std::fs::write(path, content).map_err(VisoError::Io)
    }

    /// Geometry options with display helix/sheet style folded in.
    #[must_use]
    pub fn resolved_geometry(&self) -> GeometryOptions {
        self.geometry
            .resolve_cartoon_style()
            .with_helix_style(self.display.helix_style())
            .with_sheet_style(self.display.sheet_style())
    }

    /// List available preset names (TOML file stems) in a directory.
    #[cfg(not(target_arch = "wasm32"))]
    #[must_use]
    pub fn list_presets(dir: &Path) -> Vec<String> {
        let mut names = Vec::new();
        if let Ok(entries) = std::fs::read_dir(dir) {
            for entry in entries.flatten() {
                let path = entry.path();
                if path.extension().is_none_or(|ext| ext != "toml") {
                    continue;
                }
                if let Some(stem) = path.file_stem().and_then(|s| s.to_str()) {
                    names.push(stem.to_owned());
                }
            }
        }
        names.sort();
        names
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn default_round_trips_through_toml() {
        let opts = VisoOptions::default();
        let toml_str = toml::to_string_pretty(&opts).unwrap();
        let parsed: VisoOptions = toml::from_str(&toml_str).unwrap();
        assert_eq!(opts, parsed);
    }

    #[test]
    fn partial_toml_fills_defaults() {
        let toml_str = r"
[lighting]
shininess = 80.0
";
        let opts: VisoOptions = toml::from_str(toml_str).unwrap();
        assert_eq!(opts.lighting.shininess, 80.0);
        // Everything else should be default
        assert_eq!(opts.lighting.ambient, 0.45);
        assert_eq!(opts.display.lipid_mode(), LipidMode::Coarse);
    }

    #[test]
    fn cofactor_tint_lookup() {
        let colors = ColorOptions::default();
        assert_eq!(colors.cofactor_tint("CLA"), [0.2, 0.7, 0.3]);
        assert_eq!(colors.cofactor_tint("UNKNOWN"), [0.5, 0.5, 0.5]);
    }

    #[test]
    fn schema_has_expected_properties() {
        let schema_value =
            serde_json::to_value(VisoOptions::json_schema()).unwrap();
        let props = schema_value["properties"].as_object().unwrap();

        // UI-exposed sections should be present
        assert!(props.contains_key("lighting"));
        assert!(props.contains_key("post_processing"));
        assert!(props.contains_key("camera"));

        // display is now exposed; colors is still hidden.
        assert!(props.contains_key("display"));
        assert!(!props.contains_key("colors"));

        // Geometry and debug should be present (exposed in UI)
        assert!(props.contains_key("geometry"));
        assert!(props.contains_key("debug"));

        // The un-hidden override fields appear as flat sibling properties of
        // display (proof that #[serde(flatten)] produced flat siblings, not a
        // nested sub-object).
        let display = &props["display"]["properties"];
        assert!(display.get("color_scheme").is_some());
        assert!(display.get("show_cavities").is_some());
        assert!(display.get("surface_opacity").is_some());
        assert!(display.get("drawing_mode").is_some());
        // backbone_color_mode stays hidden via #[schemars(skip)].
        assert!(display.get("backbone_color_mode").is_none());

        // surface_opacity carries minimum/maximum so the GUI renders it as a
        // slider.
        let opacity = &display["surface_opacity"];
        assert_eq!(opacity["minimum"], serde_json::json!(0.0));
        assert_eq!(opacity["maximum"], serde_json::json!(1.0));

        // Lighting should have exposed fields but not skipped ones
        let lighting = &props["lighting"]["properties"];
        assert!(lighting.get("light1_intensity").is_some());
        assert!(lighting.get("ambient").is_some());
        assert!(lighting.get("light1_dir").is_none());
        assert!(lighting.get("specular_intensity").is_none());
    }

    #[test]
    fn display_override_fields_round_trip_through_json() {
        // The real apply boundary (foldit-core app/load.rs) deserializes a
        // VisoOptions from a serde_json::Value, so exercise that path: a
        // payload carrying several flattened display override fields must
        // deserialize losslessly.
        let value = serde_json::json!({
            "display": {
                "drawing_mode": "stick",
                "color_scheme": "b_factor",
                "surface_opacity": 0.5,
                "show_cavities": true,
            }
        });
        let opts: VisoOptions = serde_json::from_value(value).unwrap();
        assert_eq!(opts.display.drawing_mode(), DrawingMode::Stick);
        assert_eq!(opts.display.backbone_color_scheme(), ColorScheme::BFactor);
        assert_eq!(opts.display.surface_opacity(), 0.5);
        assert!(opts.display.show_cavities());
    }
}
