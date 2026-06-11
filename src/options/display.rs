use molex::MoleculeType;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

// ── Structural bond options ──────────────────────────────────────────────

/// Visual style for structural bond rendering (H-bonds, disulfides).
#[derive(
    Debug,
    Clone,
    Copy,
    Serialize,
    Deserialize,
    PartialEq,
    Eq,
    Default,
    JsonSchema,
)]
#[serde(rename_all = "snake_case")]
pub enum BondStyle {
    /// Solid cylinder with hemispherical caps (Foldit default).
    #[default]
    Solid,
    /// Dashed segments along the bond axis.
    Dashed,
    /// Solid cylinder with stipple pattern (fragment discard).
    Stippled,
}

/// How structural bonds of a given type are sourced.
#[derive(
    Debug,
    Clone,
    Copy,
    Serialize,
    Deserialize,
    PartialEq,
    Eq,
    Default,
    JsonSchema,
)]
#[serde(rename_all = "snake_case")]
pub enum BondSource {
    /// Auto-detect from atomic geometry (distance + angle cutoffs).
    Auto,
    /// Only show bonds explicitly provided by the caller.
    #[default]
    Manual,
    /// Auto-detect, but caller-provided bonds take precedence for
    /// overlapping atom pairs.
    Both,
}

/// Display options for a single category of structural bond.
#[derive(
    Debug, Clone, Serialize, Deserialize, PartialEq, Default, JsonSchema,
)]
#[serde(default)]
pub struct BondTypeOptions {
    /// Whether to show this bond type at all.
    pub visible: bool,
    /// Visual rendering style.
    pub style: BondStyle,
    /// Base radius in Angstroms.
    #[schemars(range(min = 0.0, max = 0.5))]
    pub radius: f32,
    /// How bonds are sourced (auto-detect, manual, or both).
    pub source: BondSource,
}

/// Options controlling structural bond visualization (H-bonds,
/// disulfides).
///
/// Colors are configured separately in [`super::ColorOptions`]
/// (`band_hbond`, `band_disulfide`).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, JsonSchema)]
#[serde(default)]
pub struct BondOptions {
    /// Hydrogen bond display settings.
    pub hydrogen_bonds: BondTypeOptions,
    /// Disulfide bond display settings.
    pub disulfide_bonds: BondTypeOptions,
}

impl Default for BondOptions {
    fn default() -> Self {
        Self {
            hydrogen_bonds: BondTypeOptions {
                visible: false,
                style: BondStyle::Solid,
                radius: 0.1,
                source: BondSource::Auto,
            },
            disulfide_bonds: BondTypeOptions {
                visible: false,
                style: BondStyle::Solid,
                radius: 0.15,
                source: BondSource::Auto,
            },
        }
    }
}

use super::palette::{Palette, PaletteMode, PalettePreset};

/// Top-level drawing mode for an entity.
///
/// Determines whether the entity is rendered as a cartoon backbone,
/// stick model, or ball-and-stick model.
#[derive(
    Debug,
    Clone,
    Copy,
    Serialize,
    Deserialize,
    PartialEq,
    Eq,
    Default,
    JsonSchema,
)]
#[serde(rename_all = "snake_case")]
pub enum DrawingMode {
    /// Cartoon backbone with optional sidechains (default for proteins/NA).
    #[default]
    Cartoon,
    /// Bonds as capsules (sidechain thickness), small joint spheres.
    Stick,
    /// Thin bonds with tiny joint spheres (wire-like).
    ThinStick,
    /// Full ball-and-stick with vdW-scaled atom spheres.
    BallAndStick,
}

impl DrawingMode {
    /// Return the appropriate default drawing mode for a molecule type.
    #[must_use]
    pub fn default_for(mol_type: MoleculeType) -> Self {
        match mol_type {
            MoleculeType::Protein | MoleculeType::DNA | MoleculeType::RNA => {
                Self::Cartoon
            }
            _ => Self::BallAndStick,
        }
    }
}

/// Helix rendering style within Cartoon mode.
#[derive(
    Debug,
    Clone,
    Copy,
    Serialize,
    Deserialize,
    PartialEq,
    Eq,
    Default,
    JsonSchema,
)]
#[serde(rename_all = "snake_case")]
pub enum HelixStyle {
    /// Flat ribbon (default).
    #[default]
    Ribbon,
    /// Round tube.
    Tube,
    /// Solid cylinder.
    Cylinder,
}

/// Sheet rendering style within Cartoon mode.
#[derive(
    Debug,
    Clone,
    Copy,
    Serialize,
    Deserialize,
    PartialEq,
    Eq,
    Default,
    JsonSchema,
)]
#[serde(rename_all = "snake_case")]
pub enum SheetStyle {
    /// Flat ribbon (default).
    #[default]
    Ribbon,
    /// Round tube.
    Tube,
}

/// How protein backbone is colored.
///
/// Legacy enum retained for backward compatibility. New code should prefer
/// [`ColorScheme`].
#[derive(
    Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Default, JsonSchema,
)]
#[serde(rename_all = "snake_case")]
pub enum BackboneColorMode {
    /// Color by absolute score.
    Score,
    /// Color by relative score.
    ScoreRelative,
    /// Color by secondary structure type.
    SecondaryStructure,
    /// Each chain gets a distinct color, interpolated blue→red.
    #[default]
    Chain,
}

/// What property drives coloring.
///
/// The scheme determines *what data* maps to color. The palette (a separate
/// field) determines *which colors* are used. Any scheme can be combined with
/// any palette.
#[derive(
    Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Default, JsonSchema,
)]
#[serde(rename_all = "snake_case")]
pub enum ColorScheme {
    /// Each entity gets a distinct color from the palette.
    #[default]
    Entity,
    /// Color by secondary structure type (helix, sheet, coil).
    SecondaryStructure,
    /// N-to-C gradient per chain using the palette.
    ResidueIndex,
    /// Gradient by crystallographic B-factor.
    BFactor,
    /// Gradient by Kyte-Doolittle hydrophobicity scale.
    Hydrophobicity,
    /// Absolute Rosetta energy score.
    Score,
    /// Relative score (5th/95th percentile normalized).
    ScoreRelative,
    /// Single uniform color (uses first color from palette stops).
    Solid,
}

impl From<&BackboneColorMode> for ColorScheme {
    fn from(mode: &BackboneColorMode) -> Self {
        match mode {
            BackboneColorMode::Chain => Self::Entity,
            BackboneColorMode::Score => Self::Score,
            BackboneColorMode::ScoreRelative => Self::ScoreRelative,
            BackboneColorMode::SecondaryStructure => Self::SecondaryStructure,
        }
    }
}

/// How sidechains are colored.
#[derive(
    Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Default, JsonSchema,
)]
#[serde(rename_all = "snake_case")]
pub enum SidechainColorMode {
    /// Color by hydrophobicity.
    Hydrophobicity,
    /// Match the backbone color of the corresponding residue.
    #[default]
    Backbone,
}

/// How nucleic acid backbone is colored.
#[derive(
    Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Default, JsonSchema,
)]
#[serde(rename_all = "snake_case")]
pub enum NaColorMode {
    /// Single uniform color.
    Uniform,
    /// Color each residue's backbone segment to match its base (A/T/G/C/U).
    #[default]
    BaseColor,
}

/// Lipid display style.
#[derive(
    Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Default, JsonSchema,
)]
#[serde(rename_all = "snake_case")]
pub enum LipidMode {
    /// Coarse-grained spheres.
    #[default]
    Coarse,
    /// Full ball-and-stick representation.
    BallAndStick,
}

/// Global surface type option for the Display panel.
///
/// When set to something other than `None`, all entities without a per-entity
/// surface override will get this surface type applied automatically.
#[derive(
    Debug,
    Clone,
    Copy,
    Serialize,
    Deserialize,
    PartialEq,
    Eq,
    Default,
    JsonSchema,
)]
#[serde(rename_all = "snake_case")]
pub enum SurfaceKindOption {
    /// No global surface.
    #[default]
    None,
    /// Smooth Gaussian blob surface.
    Gaussian,
    /// Solvent-excluded / Connolly surface.
    Ses,
}

/// Surface presentation mode.
///
/// Not all modes are supported on every platform. If the requested mode is
/// unavailable, the engine falls back to [`PresentMode::Fifo`] (always
/// supported).
#[derive(
    Debug,
    Clone,
    Copy,
    Serialize,
    Deserialize,
    PartialEq,
    Eq,
    Default,
    JsonSchema,
)]
#[serde(rename_all = "snake_case")]
pub enum PresentMode {
    /// VSync — capped to display refresh rate, no tearing.
    #[default]
    Fifo,
    /// Immediate — lowest latency, may tear.
    Immediate,
    /// Mailbox — low-latency VSync (triple-buffered).
    Mailbox,
}

impl PresentMode {
    /// Convert to the corresponding wgpu present mode.
    #[must_use]
    pub fn to_wgpu(self) -> wgpu::PresentMode {
        match self {
            Self::Fifo => wgpu::PresentMode::Fifo,
            Self::Immediate => wgpu::PresentMode::Immediate,
            Self::Mailbox => wgpu::PresentMode::Mailbox,
        }
    }
}

#[derive(
    Debug, Clone, Serialize, Deserialize, PartialEq, Default, JsonSchema,
)]
#[serde(default)]
#[schemars(title = "Display", inline)]
#[allow(clippy::struct_excessive_bools)]
/// Display toggles and coloring mode selections.
///
/// Combines:
/// - 5 global-only ambient toggles (waters/ions/solvent/present_mode/bonds)
/// - a flattened [`super::DisplayOverrides`] bag of 14 per-entity overridable
///   fields (drawing mode, color scheme, cartoon style, etc.) serialized flat
///   for TOML compatibility
///
/// The same [`super::DisplayOverrides`] type is used at per-entity scope
/// (via `DisplayOverrides::overlay`). `None` at either scope falls through
/// to the next layer (entity → global → built-in defaults).
pub struct DisplayOptions {
    // --- Ambient visibility (type-level toggles, not per-entity) ---
    /// Whether to render water molecules.
    pub show_waters: bool,
    /// Whether to render ion atoms.
    pub show_ions: bool,
    /// Whether to render solvent molecules.
    pub show_solvent: bool,

    // --- Rendering (not per-entity) ---
    /// Surface presentation mode (VSync, immediate, mailbox).
    pub present_mode: PresentMode,

    // --- Structural bonds ---
    /// Structural bond visualization settings (H-bonds, disulfides).
    pub bonds: BondOptions,

    // --- Legacy ---
    /// Backbone coloring strategy (legacy field — prefer
    /// `overrides.color_scheme`).
    #[schemars(skip)]
    pub backbone_color_mode: BackboneColorMode,

    // --- Per-entity overridable fields (flattened for TOML compat) ---
    /// User's global display preferences, expressed as a bag of
    /// overrides. `None` fields fall through to built-in defaults.
    #[serde(flatten)]
    pub overrides: super::DisplayOverrides,
}

impl DisplayOptions {
    /// Top-level drawing mode (Cartoon / Stick / BallAndStick), resolved
    /// against built-in defaults.
    #[must_use]
    pub fn drawing_mode(&self) -> DrawingMode {
        self.overrides.drawing_mode.unwrap_or_default()
    }

    /// What property backbone color maps to, resolved.
    #[must_use]
    pub fn backbone_color_scheme(&self) -> ColorScheme {
        self.overrides.color_scheme.clone().unwrap_or_default()
    }

    /// Whether to render amino acid sidechains, resolved.
    #[must_use]
    pub fn show_sidechains(&self) -> bool {
        self.overrides.show_sidechains.unwrap_or(false)
    }

    /// Global molecular surface type, resolved.
    #[must_use]
    pub fn surface_kind(&self) -> SurfaceKindOption {
        self.overrides.surface_kind.unwrap_or_default()
    }

    /// Global surface opacity (0.0–1.0), resolved.
    #[must_use]
    pub fn surface_opacity(&self) -> f32 {
        self.overrides
            .surface_opacity
            .unwrap_or_else(default_surface_opacity)
    }

    /// Whether to render internal cavity meshes, resolved.
    #[must_use]
    pub fn show_cavities(&self) -> bool {
        self.overrides.show_cavities.unwrap_or(false)
    }

    /// Helix rendering style within Cartoon mode, resolved.
    #[must_use]
    pub fn helix_style(&self) -> HelixStyle {
        self.overrides.helix_style.unwrap_or_default()
    }

    /// Sheet rendering style within Cartoon mode, resolved.
    #[must_use]
    pub fn sheet_style(&self) -> SheetStyle {
        self.overrides.sheet_style.unwrap_or_default()
    }

    /// Sidechain coloring strategy, resolved.
    #[must_use]
    pub fn sidechain_color_mode(&self) -> SidechainColorMode {
        self.overrides
            .sidechain_color_mode
            .clone()
            .unwrap_or_default()
    }

    /// Nucleic acid coloring strategy, resolved.
    #[must_use]
    pub fn na_color_mode(&self) -> NaColorMode {
        self.overrides.na_color_mode.clone().unwrap_or_default()
    }

    /// Lipid rendering style, resolved.
    #[must_use]
    pub fn lipid_mode(&self) -> LipidMode {
        self.overrides.lipid_mode.clone().unwrap_or_default()
    }

    /// Whether to render hydrogen atoms, resolved.
    #[must_use]
    pub fn show_hydrogens(&self) -> bool {
        self.overrides.show_hydrogens.unwrap_or(false)
    }

    /// Named color palette preset for backbone coloring, resolved.
    #[must_use]
    pub fn backbone_palette_preset(&self) -> PalettePreset {
        self.overrides.palette_preset.clone().unwrap_or_default()
    }

    /// How backbone palette colors are distributed, resolved.
    #[must_use]
    pub fn backbone_palette_mode(&self) -> PaletteMode {
        self.overrides.palette_mode.clone().unwrap_or_default()
    }

    /// Whether lipid mode uses full ball-and-stick representation.
    #[must_use]
    pub fn lipid_ball_and_stick(&self) -> bool {
        matches!(self.lipid_mode(), LipidMode::BallAndStick)
    }

    /// Construct a [`Palette`] from the resolved preset/mode fields.
    #[must_use]
    pub fn backbone_palette(&self) -> Palette {
        Palette {
            preset: self.backbone_palette_preset(),
            mode: self.backbone_palette_mode(),
            stops: Vec::new(),
        }
    }
}

/// Default surface opacity for serde deserialization.
fn default_surface_opacity() -> f32 {
    0.35
}
