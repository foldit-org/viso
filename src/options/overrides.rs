//! Display override bag — optional visual settings applied at either
//! global or per-entity scope.
//!
//! [`DisplayOverrides`] holds each visual setting as `Option<T>`. `None`
//! means "inherit from the next level up": a per-entity override falls
//! back to the user's global overrides, which fall back to built-in
//! defaults. Same type is used at both scopes.

use schemars::JsonSchema;
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};

use super::display::{
    BondStyle, ColorScheme, DrawingMode, HelixStyle, LipidMode, NaColorMode,
    SheetStyle, SidechainColorMode, SurfaceKindOption,
};
use super::geometry::GeometryOptions;
use super::palette::{PaletteMode, PalettePreset};
use super::DisplayOptions;

// Schema-default providers for the `Option<Enum>` override fields. Each
// returns the Rust `#[default]` variant wrapped in `Some`, so the schema
// emitted for the settings panel advertises the same default the engine
// resolves to when the slot is `None`. The outer `Option` is required:
// the bare-enum form does not satisfy the `skip_serializing_if =
// "Option::is_none"` guard on these fields, so the always-`Some` shape is
// load-bearing rather than reducible.
#[allow(clippy::unnecessary_wraps)]
fn default_drawing_mode() -> Option<DrawingMode> {
    Some(DrawingMode::default())
}
#[allow(clippy::unnecessary_wraps)]
fn default_color_scheme() -> Option<ColorScheme> {
    Some(ColorScheme::default())
}
#[allow(clippy::unnecessary_wraps)]
fn default_surface_kind() -> Option<SurfaceKindOption> {
    Some(SurfaceKindOption::default())
}
#[allow(clippy::unnecessary_wraps)]
fn default_helix_style() -> Option<HelixStyle> {
    Some(HelixStyle::default())
}
#[allow(clippy::unnecessary_wraps)]
fn default_sheet_style() -> Option<SheetStyle> {
    Some(SheetStyle::default())
}
#[allow(clippy::unnecessary_wraps)]
fn default_sidechain_color_mode() -> Option<SidechainColorMode> {
    Some(SidechainColorMode::default())
}
#[allow(clippy::unnecessary_wraps)]
fn default_na_color_mode() -> Option<NaColorMode> {
    Some(NaColorMode::default())
}
#[allow(clippy::unnecessary_wraps)]
fn default_lipid_mode() -> Option<LipidMode> {
    Some(LipidMode::default())
}
#[allow(clippy::unnecessary_wraps)]
fn default_palette_preset() -> Option<PalettePreset> {
    Some(PalettePreset::default())
}
#[allow(clippy::unnecessary_wraps)]
fn default_palette_mode() -> Option<PaletteMode> {
    Some(PaletteMode::default())
}

/// Bag of optional display overrides. Used at both global and per-entity
/// scope; `None` means "inherit from the next level up."
#[derive(
    Debug, Clone, Serialize, Deserialize, PartialEq, Default, JsonSchema,
)]
#[serde(default)]
pub struct DisplayOverrides {
    /// Top-level drawing mode (Cartoon / Stick / BallAndStick).
    #[serde(skip_serializing_if = "Option::is_none")]
    #[schemars(default = "default_drawing_mode")]
    pub drawing_mode: Option<DrawingMode>,
    /// What property drives backbone coloring.
    #[serde(
        skip_serializing_if = "Option::is_none",
        alias = "backbone_color_scheme"
    )]
    #[schemars(default = "default_color_scheme")]
    pub color_scheme: Option<ColorScheme>,
    /// Whether to render amino acid sidechains.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub show_sidechains: Option<bool>,
    /// Molecular surface type (None / Gaussian / SES).
    #[serde(skip_serializing_if = "Option::is_none")]
    #[schemars(default = "default_surface_kind")]
    pub surface_kind: Option<SurfaceKindOption>,
    /// Surface opacity (alpha channel, 0.0–1.0).
    #[serde(skip_serializing_if = "Option::is_none")]
    #[schemars(range(min = 0.0, max = 1.0))]
    pub surface_opacity: Option<f32>,
    /// Whether to render internal cavity meshes.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub show_cavities: Option<bool>,
    /// Helix rendering style within Cartoon mode.
    #[serde(skip_serializing_if = "Option::is_none")]
    #[schemars(default = "default_helix_style")]
    pub helix_style: Option<HelixStyle>,
    /// Sheet rendering style within Cartoon mode.
    #[serde(skip_serializing_if = "Option::is_none")]
    #[schemars(default = "default_sheet_style")]
    pub sheet_style: Option<SheetStyle>,
    /// Sidechain coloring strategy.
    #[serde(skip_serializing_if = "Option::is_none")]
    #[schemars(default = "default_sidechain_color_mode")]
    pub sidechain_color_mode: Option<SidechainColorMode>,
    /// Nucleic acid coloring strategy.
    #[serde(skip_serializing_if = "Option::is_none")]
    #[schemars(default = "default_na_color_mode")]
    pub na_color_mode: Option<NaColorMode>,
    /// Lipid rendering style.
    #[serde(skip_serializing_if = "Option::is_none")]
    #[schemars(default = "default_lipid_mode")]
    pub lipid_mode: Option<LipidMode>,
    /// Whether to render hydrogen atoms.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub show_hydrogens: Option<bool>,
    /// Named color palette preset for backbone coloring.
    #[serde(
        skip_serializing_if = "Option::is_none",
        alias = "backbone_palette_preset"
    )]
    #[schemars(default = "default_palette_preset")]
    pub palette_preset: Option<PalettePreset>,
    /// How backbone palette colors are distributed.
    #[serde(
        skip_serializing_if = "Option::is_none",
        alias = "backbone_palette_mode"
    )]
    #[schemars(default = "default_palette_mode")]
    pub palette_mode: Option<PaletteMode>,

    // --- Structural bond overrides ---
    /// Whether to show hydrogen bonds for this entity.
    #[serde(skip_serializing_if = "Option::is_none")]
    #[schemars(skip)]
    pub show_hbonds: Option<bool>,
    /// Visual style for hydrogen bonds.
    #[serde(skip_serializing_if = "Option::is_none")]
    #[schemars(skip)]
    pub hbond_style: Option<BondStyle>,
    /// Whether to show disulfide bonds for this entity.
    #[serde(skip_serializing_if = "Option::is_none")]
    #[schemars(skip)]
    pub show_disulfides: Option<bool>,
    /// Visual style for disulfide bonds.
    #[serde(skip_serializing_if = "Option::is_none")]
    #[schemars(skip)]
    pub disulfide_style: Option<BondStyle>,
}

/// Classes of rendering work invalidated by an overrides diff.
///
/// A `DisplayOverrides::diff` projects each changed field onto a union
/// of these kinds. The dispatcher fires each kind at most once per
/// `set_options` / `set_entity_appearance` call — dedup is the structural
/// property of the type.
///
/// Bitflag-like u32 with const masks; no macro dependency. Each const
/// covers a single class of GPU work; combinations express multi-kind
/// invalidations (e.g. `drawing_mode` change ⇒ `DRAWING_MODE_RESOLVE |
/// RE_MESH`).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct RenderInvalidation(u32);

impl RenderInvalidation {
    /// No invalidation.
    pub const NONE: Self = Self(0);
    /// Bump entity mesh versions + resync the scene — the catch-all for
    /// any overridable change that reshapes geometry or bonds.
    pub const RE_MESH: Self = Self(1 << 0);
    /// Recompute backbone colors (palette / scheme change).
    pub const RE_COLOR: Self = Self(1 << 1);
    /// Regenerate molecular surfaces (kind / opacity / cavities).
    pub const RE_SURFACE: Self = Self(1 << 2);
    /// Per-chain LOD remesh (backbone style change).
    pub const LOD_REMESH: Self = Self(1 << 3);
    /// Re-resolve per-entity `drawing_mode` (global drawing_mode moved).
    pub const DRAWING_MODE_RESOLVE: Self = Self(1 << 4);

    /// True if no flags set.
    #[must_use]
    pub const fn is_empty(self) -> bool {
        self.0 == 0
    }

    /// True if `self` contains all flags in `other`.
    #[must_use]
    pub const fn contains(self, other: Self) -> bool {
        (self.0 & other.0) == other.0
    }
}

impl std::ops::BitOr for RenderInvalidation {
    type Output = Self;
    fn bitor(self, rhs: Self) -> Self {
        Self(self.0 | rhs.0)
    }
}

impl std::ops::BitOrAssign for RenderInvalidation {
    fn bitor_assign(&mut self, rhs: Self) {
        self.0 |= rhs.0;
    }
}

impl DisplayOverrides {
    /// Overlay `self` on `base`. `self`'s `Some` values win; `None`
    /// fields fall through to `base`.
    ///
    /// Used at both scopes: per-entity overlaid on global (entity's
    /// `Some` wins), and global overlaid on built-in defaults. Same
    /// operation, different layer.
    #[must_use]
    pub fn overlay(&self, base: &Self) -> Self {
        // Exhaustive destructuring — adding a field to DisplayOverrides
        // without updating this walk fails to compile.
        let Self {
            drawing_mode: _,
            color_scheme: _,
            show_sidechains: _,
            surface_kind: _,
            surface_opacity: _,
            show_cavities: _,
            helix_style: _,
            sheet_style: _,
            sidechain_color_mode: _,
            na_color_mode: _,
            lipid_mode: _,
            show_hydrogens: _,
            palette_preset: _,
            palette_mode: _,
            show_hbonds: _,
            hbond_style: _,
            show_disulfides: _,
            disulfide_style: _,
        } = self;
        Self {
            drawing_mode: self.drawing_mode.or(base.drawing_mode),
            color_scheme: self
                .color_scheme
                .clone()
                .or_else(|| base.color_scheme.clone()),
            show_sidechains: self.show_sidechains.or(base.show_sidechains),
            surface_kind: self.surface_kind.or(base.surface_kind),
            surface_opacity: self.surface_opacity.or(base.surface_opacity),
            show_cavities: self.show_cavities.or(base.show_cavities),
            helix_style: self.helix_style.or(base.helix_style),
            sheet_style: self.sheet_style.or(base.sheet_style),
            sidechain_color_mode: self
                .sidechain_color_mode
                .clone()
                .or_else(|| base.sidechain_color_mode.clone()),
            na_color_mode: self
                .na_color_mode
                .clone()
                .or_else(|| base.na_color_mode.clone()),
            lipid_mode: self
                .lipid_mode
                .clone()
                .or_else(|| base.lipid_mode.clone()),
            show_hydrogens: self.show_hydrogens.or(base.show_hydrogens),
            palette_preset: self
                .palette_preset
                .clone()
                .or_else(|| base.palette_preset.clone()),
            palette_mode: self
                .palette_mode
                .clone()
                .or_else(|| base.palette_mode.clone()),
            show_hbonds: self.show_hbonds.or(base.show_hbonds),
            hbond_style: self.hbond_style.or(base.hbond_style),
            show_disulfides: self.show_disulfides.or(base.show_disulfides),
            disulfide_style: self.disulfide_style.or(base.disulfide_style),
        }
    }

    /// Per-field invalidation diff.
    ///
    /// Projects each changed field onto a union of
    /// [`RenderInvalidation`] classes. Used by both the global path
    /// (`DisplayOptions.overrides` diff) and the per-entity path
    /// (`EntityAnnotations.appearance[eid]` diff), producing the same
    /// kind of invalidation set regardless of scope. Dispatchers fire
    /// each kind at most once per call.
    ///
    /// Exhaustive destructuring — adding a field to `DisplayOverrides`
    /// without updating this walk fails to compile.
    #[must_use]
    pub fn diff(&self, new: &Self) -> RenderInvalidation {
        // Destructure to force compile error on new fields.
        let Self {
            drawing_mode: _,
            color_scheme: _,
            show_sidechains: _,
            surface_kind: _,
            surface_opacity: _,
            show_cavities: _,
            helix_style: _,
            sheet_style: _,
            sidechain_color_mode: _,
            na_color_mode: _,
            lipid_mode: _,
            show_hydrogens: _,
            palette_preset: _,
            palette_mode: _,
            show_hbonds: _,
            hbond_style: _,
            show_disulfides: _,
            disulfide_style: _,
        } = self;

        let mut inv = RenderInvalidation::NONE;

        // Drawing mode: per-entity drawing_mode needs resolution +
        // a mesh rebuild (drawing mode can switch between Cartoon / Stick /
        // BallAndStick, each with entirely different meshes).
        if self.drawing_mode != new.drawing_mode {
            inv |= RenderInvalidation::DRAWING_MODE_RESOLVE
                | RenderInvalidation::RE_MESH;
        }

        // Color scheme / palette: mesh regenerates with new colors, and
        // backbone color buffer rebuilds separately.
        if self.color_scheme != new.color_scheme
            || self.palette_preset != new.palette_preset
            || self.palette_mode != new.palette_mode
        {
            inv |= RenderInvalidation::RE_COLOR | RenderInvalidation::RE_MESH;
        }

        // Sidechain coloring: mesh rebuild picks up new sidechain colors.
        if self.sidechain_color_mode != new.sidechain_color_mode {
            inv |= RenderInvalidation::RE_MESH;
        }

        // Sidechain/hydrogen visibility: geometry change, needs remesh.
        if self.show_sidechains != new.show_sidechains
            || self.show_hydrogens != new.show_hydrogens
        {
            inv |= RenderInvalidation::RE_MESH;
        }

        // Surface changes: regen surface mesh + sync.
        if self.surface_kind != new.surface_kind
            || self.surface_opacity != new.surface_opacity
            || self.show_cavities != new.show_cavities
        {
            inv |= RenderInvalidation::RE_SURFACE;
        }

        // Cartoon style: backbone geometry changes -> LOD remesh.
        if self.helix_style != new.helix_style
            || self.sheet_style != new.sheet_style
        {
            inv |= RenderInvalidation::LOD_REMESH | RenderInvalidation::RE_MESH;
        }

        // Nucleic acid coloring mode affects mesh attributes.
        if self.na_color_mode != new.na_color_mode {
            inv |= RenderInvalidation::RE_MESH;
        }

        // Lipid rendering style: different geometry (sphere vs ball-and-stick).
        if self.lipid_mode != new.lipid_mode {
            inv |= RenderInvalidation::RE_MESH;
        }

        // Bond visibility / style: bond geometry change.
        if self.show_hbonds != new.show_hbonds
            || self.hbond_style != new.hbond_style
            || self.show_disulfides != new.show_disulfides
            || self.disulfide_style != new.disulfide_style
        {
            inv |= RenderInvalidation::RE_MESH;
        }

        inv
    }

    /// Apply a single override field from a JSON value.
    ///
    /// `field` is the serde field name (matches a column in the entity
    /// override panel). The value is interpreted against the field's
    /// typed slot:
    ///
    /// - `Null` clears the override (sets the slot to `None`, so the field
    ///   inherits from the next level up). This is the UI's "reset to default"
    ///   signal.
    /// - A valid value for the field sets the slot to `Some(parsed)`.
    /// - A non-null but malformed value (wrong JSON type, or an unrecognised
    ///   enum variant) leaves the slot untouched and returns `Err` — a
    ///   fat-fingered value must not silently wipe an existing override.
    ///
    /// # Errors
    /// Returns `Err(field)` if the field name is not recognised, or if
    /// `value` is non-null and fails to parse into the field's type.
    pub fn apply_json_field<'a>(
        &mut self,
        field: &'a str,
        value: &serde_json::Value,
    ) -> Result<(), &'a str> {
        match field {
            "backbone_color_scheme" | "color_scheme" => {
                self.color_scheme = parse_field(value, field)?;
            }
            "show_sidechains" => {
                self.show_sidechains = parse_bool_field(value, field)?;
            }
            "drawing_mode" => {
                self.drawing_mode = parse_field(value, field)?;
            }
            "helix_style" => {
                self.helix_style = parse_field(value, field)?;
            }
            "sheet_style" => {
                self.sheet_style = parse_field(value, field)?;
            }
            "surface_kind" => {
                self.surface_kind = parse_field(value, field)?;
            }
            "surface_opacity" => {
                self.surface_opacity = parse_f32_field(value, field)?;
            }
            "show_hbonds" => {
                self.show_hbonds = parse_bool_field(value, field)?;
            }
            "hbond_style" => {
                self.hbond_style = parse_field(value, field)?;
            }
            "show_disulfides" => {
                self.show_disulfides = parse_bool_field(value, field)?;
            }
            "disulfide_style" => {
                self.disulfide_style = parse_field(value, field)?;
            }
            "show_cavities" => {
                self.show_cavities = parse_bool_field(value, field)?;
            }
            "show_hydrogens" => {
                self.show_hydrogens = parse_bool_field(value, field)?;
            }
            "sidechain_color_mode" => {
                self.sidechain_color_mode = parse_field(value, field)?;
            }
            "na_color_mode" => {
                self.na_color_mode = parse_field(value, field)?;
            }
            "lipid_mode" => {
                self.lipid_mode = parse_field(value, field)?;
            }
            _ => return Err(field),
        }
        Ok(())
    }

    /// Whether all fields are `None` (no overrides).
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.drawing_mode.is_none()
            && self.color_scheme.is_none()
            && self.show_sidechains.is_none()
            && self.surface_kind.is_none()
            && self.surface_opacity.is_none()
            && self.show_cavities.is_none()
            && self.helix_style.is_none()
            && self.sheet_style.is_none()
            && self.sidechain_color_mode.is_none()
            && self.na_color_mode.is_none()
            && self.lipid_mode.is_none()
            && self.show_hydrogens.is_none()
            && self.palette_preset.is_none()
            && self.palette_mode.is_none()
            && self.show_hbonds.is_none()
            && self.hbond_style.is_none()
            && self.show_disulfides.is_none()
            && self.disulfide_style.is_none()
    }

    /// Produce a [`DisplayOptions`] with these overrides applied on top
    /// of `base`.
    ///
    /// Overlays the overridable bag and propagates the four bond-override
    /// fields into the global `bonds` config. `None` fields leave the
    /// corresponding `base` value untouched.
    #[must_use]
    pub fn to_display_options(&self, base: &DisplayOptions) -> DisplayOptions {
        let mut out = base.clone();
        out.overrides = self.overlay(&base.overrides);
        // Propagate bond-specific overrides into the global bonds config.
        // These four fields project onto a different struct shape than
        // the rest of the overlay (nested BondOptions, not DisplayOverrides).
        if let Some(v) = self.show_hbonds {
            out.bonds.hydrogen_bonds.visible = v;
        }
        if let Some(v) = self.hbond_style {
            out.bonds.hydrogen_bonds.style = v;
        }
        if let Some(v) = self.show_disulfides {
            out.bonds.disulfide_bonds.visible = v;
        }
        if let Some(v) = self.disulfide_style {
            out.bonds.disulfide_bonds.style = v;
        }
        out
    }

    /// Produce a [`GeometryOptions`] by patching helix/sheet style onto
    /// `base`.
    #[must_use]
    pub fn to_geometry_options(
        &self,
        base: &GeometryOptions,
    ) -> GeometryOptions {
        let mut out = base.clone();
        if let Some(helix) = self.helix_style {
            out = out.with_helix_style(helix);
        }
        if let Some(sheet) = self.sheet_style {
            out = out.with_sheet_style(sheet);
        }
        out
    }
}

/// Parse a JSON value into an optional typed override slot.
///
/// `Null` maps to `Ok(None)` (clear the override). A value that
/// deserializes into `T` maps to `Ok(Some(value))`. Anything else
/// (wrong JSON type, unrecognised enum variant) returns `Err(field)`
/// so the caller can reject it without mutating the slot.
fn parse_field<'a, T: DeserializeOwned>(
    value: &serde_json::Value,
    field: &'a str,
) -> Result<Option<T>, &'a str> {
    if value.is_null() {
        return Ok(None);
    }
    serde_json::from_value(value.clone())
        .map(Some)
        .map_err(|_| field)
}

/// Bool variant of [`parse_field`]: `Null` clears, a JSON bool sets,
/// anything else is rejected.
fn parse_bool_field<'a>(
    value: &serde_json::Value,
    field: &'a str,
) -> Result<Option<bool>, &'a str> {
    if value.is_null() {
        return Ok(None);
    }
    value.as_bool().map(Some).ok_or(field)
}

/// f32 variant of [`parse_field`]: `Null` clears, a JSON number sets
/// (narrowed to `f32`), anything else is rejected.
fn parse_f32_field<'a>(
    value: &serde_json::Value,
    field: &'a str,
) -> Result<Option<f32>, &'a str> {
    if value.is_null() {
        return Ok(None);
    }
    value.as_f64().map(|v| Some(v as f32)).ok_or(field)
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
#[path = "overrides_tests.rs"]
mod tests;
