use glam::Vec3;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use super::display::{HelixStyle, SheetStyle};

/// Cartoon rendering style presets.
///
/// Each preset maps to specific per-SS geometry parameters (width, thickness,
/// roundness). When set to anything other than `Custom`, the preset values
/// override the per-SS fields.
#[derive(
    Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Default, JsonSchema,
)]
#[serde(rename_all = "snake_case")]
pub enum CartoonStyle {
    /// Flat ribbons for helices/sheets, round tubes for coils.
    #[default]
    Ribbon,
    /// Round tubes everywhere, uniform radius.
    Tube,
    /// Cylindrical helices, flat sheets with arrows, round coils.
    Cylindrical,
    /// Custom -- user controls all per-SS parameters directly.
    Custom,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, JsonSchema)]
#[schemars(title = "Geometry", inline)]
#[serde(default)]
/// Geometry detail options for molecular rendering primitives.
pub struct GeometryOptions {
    /// Cartoon rendering style preset.
    #[schemars(title = "Cartoon Style", extend("x-group" = "Style"))]
    pub cartoon_style: CartoonStyle,
    /// Whether to draw arrow heads at the C-terminal end of beta sheets.
    #[schemars(title = "Sheet Arrows", extend("x-group" = "Sheet"))]
    pub sheet_arrows: bool,
    /// Helix ribbon half-width in angstroms.
    #[schemars(title = "Helix Width", range(min = 0.2, max = 3.0), extend("step" = 0.1), extend("x-group" = "Helix"))]
    pub helix_width: f32,
    /// Helix ribbon thickness.
    #[schemars(title = "Helix Thickness", range(min = 0.05, max = 1.0), extend("step" = 0.05), extend("x-group" = "Helix"))]
    pub helix_thickness: f32,
    /// Helix cross-section roundness (0 = flat ribbon, 1 = circular tube).
    #[schemars(title = "Helix Roundness", range(min = 0.0, max = 1.0), extend("step" = 0.05), extend("x-group" = "Helix"))]
    pub helix_roundness: f32,

    /// Sheet ribbon half-width in angstroms.
    #[schemars(title = "Sheet Width", range(min = 0.2, max = 3.0), extend("step" = 0.1), extend("x-group" = "Sheet"))]
    pub sheet_width: f32,
    /// Sheet ribbon thickness.
    #[schemars(title = "Sheet Thickness", range(min = 0.05, max = 1.0), extend("step" = 0.05), extend("x-group" = "Sheet"))]
    pub sheet_thickness: f32,
    /// Sheet cross-section roundness (0 = flat ribbon, 1 = circular tube).
    #[schemars(title = "Sheet Roundness", range(min = 0.0, max = 1.0), extend("step" = 0.05), extend("x-group" = "Sheet"))]
    pub sheet_roundness: f32,

    /// Coil tube width (diameter).
    #[schemars(title = "Coil Width", range(min = 0.1, max = 1.5), extend("step" = 0.05), extend("x-group" = "Coil"))]
    pub coil_width: f32,
    /// Coil tube thickness.
    #[schemars(title = "Coil Thickness", range(min = 0.1, max = 1.5), extend("step" = 0.05), extend("x-group" = "Coil"))]
    pub coil_thickness: f32,
    /// Coil cross-section roundness (0 = flat ribbon, 1 = circular tube).
    #[schemars(title = "Coil Roundness", range(min = 0.0, max = 1.0), extend("step" = 0.05), extend("x-group" = "Coil"))]
    pub coil_roundness: f32,

    /// Nucleic acid backbone ribbon width.
    #[schemars(title = "NA Width", range(min = 0.2, max = 3.0), extend("step" = 0.1), extend("x-group" = "Nucleic Acid"))]
    pub na_width: f32,
    /// Nucleic acid backbone ribbon thickness.
    #[schemars(title = "NA Thickness", range(min = 0.05, max = 1.0), extend("step" = 0.05), extend("x-group" = "Nucleic Acid"))]
    pub na_thickness: f32,
    /// Nucleic acid backbone roundness (0 = flat ribbon, 1 = circular tube).
    #[schemars(title = "NA Roundness", range(min = 0.0, max = 1.0), extend("step" = 0.05), extend("x-group" = "Nucleic Acid"))]
    pub na_roundness: f32,

    /// Spline segments per residue (higher = smoother curves, more GPU cost).
    #[schemars(title = "Spline Detail", range(min = 4, max = 32), extend("step" = 4), extend("x-group" = "Quality"))]
    pub segments_per_residue: usize,
    /// Vertices per cross-section ring (higher = rounder tubes, more GPU
    /// cost).
    #[schemars(title = "Cross-Section Detail", range(min = 4, max = 16), extend("step" = 2), extend("x-group" = "Quality"))]
    pub cross_section_verts: usize,

    /// Solvent sphere radius in angstroms.
    #[schemars(skip)]
    pub solvent_radius: f32,
    /// Ligand atom sphere radius.
    #[schemars(skip)]
    pub ligand_sphere_radius: f32,
    /// Ligand bond cylinder radius.
    #[schemars(skip)]
    pub ligand_bond_radius: f32,
}

impl GeometryOptions {
    /// Return a copy with detail scaled down for the given LOD tier.
    ///
    /// The user's `segments_per_residue` and `cross_section_verts` are treated
    /// as the maximum detail (tier 0). Higher tiers scale down:
    /// - Tier 0: 100% of user settings (32 spr, 16 csv at defaults)
    /// - Tier 1: 50% (16 spr, 8 csv)
    /// - Tier 2: 25% (8 spr, 4 csv)
    /// - Tier 3: 12.5% (4 spr, 4 csv)
    #[must_use]
    pub fn with_lod_tier(&self, tier: u8) -> Self {
        if tier == 0 {
            return self.clone();
        }
        let lod = lod_scaled(
            self.segments_per_residue,
            self.cross_section_verts,
            tier,
        );
        Self {
            segments_per_residue: lod.segments_per_residue,
            cross_section_verts: lod.cross_section_verts,
            ..self.clone()
        }
    }

    /// Clamp detail so the estimated vertex buffer stays under the wgpu 256 MB
    /// max buffer size. Returns `self` unchanged for small structures.
    ///
    /// The backbone vertex buffer scales as
    /// `SPR x CSV x residues x 52 bytes/vertex` (with a 1.15x overhead
    /// factor). For 5 000 residues at default settings (SPR = 8, CSV = 16)
    /// this is ~33 MB -- well within the 256 MB limit. When the estimate
    /// exceeds the limit this method progressively applies LOD tiers
    /// (halving SPR each step) until the buffer fits.
    ///
    /// Beyond ~50 000 residues even the most aggressive tier (SPR = 4,
    /// CSV = 16) may approach the limit, so per-chain LOD
    /// ([`VisoEngine::check_and_submit_lod`](crate::engine::VisoEngine)) is
    /// essential at that scale.
    #[must_use]
    pub fn clamped_for_residues(&self, total_residues: usize) -> Self {
        const MAX_BUFFER_BYTES: usize = 256 * 1024 * 1024;
        const VERTEX_BYTES: usize = 52; // size_of::<BackboneVertex>()
        const OVERHEAD: f64 = 1.15;

        let est = |spr: usize, csv: usize| -> usize {
            (total_residues as f64 * spr as f64 * csv as f64 * OVERHEAD)
                as usize
                * VERTEX_BYTES
        };

        // Try current settings first
        if est(self.segments_per_residue, self.cross_section_verts)
            <= MAX_BUFFER_BYTES
        {
            return self.clone();
        }

        // Progressively reduce until it fits
        for tier in 1..=3u8 {
            let lod = lod_scaled(
                self.segments_per_residue,
                self.cross_section_verts,
                tier,
            );
            if est(lod.segments_per_residue, lod.cross_section_verts)
                <= MAX_BUFFER_BYTES
            {
                return self.with_lod_tier(tier);
            }
        }
        self.with_lod_tier(3)
    }
}

/// Per-chain backbone tessellation detail: segments along the spline per
/// residue and vertices around each cross-section ring.
///
/// Replaces the bare `(usize, usize)` LOD tuple -- the field order was
/// swap-bait across the six signatures that threaded it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ChainLod {
    /// Spline samples generated per residue (curve smoothness).
    pub segments_per_residue: usize,
    /// Vertices around each cross-section ring (tube/ribbon resolution).
    pub cross_section_verts: usize,
}

/// Scale user detail settings down for an LOD tier.
///
/// Only `spr` (segments per residue) is reduced -- `csv` (cross-section
/// vertices) is kept at the user's setting. Reducing csv changes the
/// cross-section shape and normals, making flat ribbons look rounded and
/// incorrectly lit. Reducing spr only makes curves slightly less smooth,
/// which is barely visible at distance.
///
/// - Tier 0: spr 100% (unchanged)
/// - Tier 1: spr 50%
/// - Tier 2: spr 25%
/// - Tier 3: spr 12.5%
#[must_use]
pub fn lod_scaled(max_spr: usize, max_csv: usize, tier: u8) -> ChainLod {
    let segments_per_residue = match tier {
        0 => max_spr,
        1 => (max_spr / 2).max(4),
        2 => (max_spr / 4).max(4),
        _ => (max_spr / 8).max(4),
    };
    ChainLod {
        segments_per_residue,
        cross_section_verts: max_csv,
    }
}

/// Convenience: LOD tier params from defaults (for backward compat).
#[must_use]
pub fn lod_params(tier: u8) -> ChainLod {
    let defaults = GeometryOptions::default();
    lod_scaled(
        defaults.segments_per_residue,
        defaults.cross_section_verts,
        tier,
    )
}

/// Select LOD tier based on absolute camera distance in angstroms.
///
/// Uses absolute distance thresholds -- screen-space occupancy is what
/// matters, not relative bounding-radius ratios.
///
/// - Close (< 150 A):    tier 0 -- 32 spr (full detail)
/// - Medium (150-250 A): tier 1 -- 16 spr (half detail)
/// - Far (250-400 A):    tier 2 -- 8 spr (quarter detail)
/// - Very far (> 400 A): tier 3 -- 4 spr (minimum detail)
#[must_use]
pub fn select_lod_tier(camera_distance: f32, _bounding_radius: f32) -> u8 {
    if camera_distance < 150.0 {
        0
    } else if camera_distance < 250.0 {
        1
    } else if camera_distance < 400.0 {
        2
    } else {
        3
    }
}

/// Select LOD tier for a single chain based on its bounding center distance
/// to the camera eye.
#[must_use]
pub fn select_chain_lod_tier(chain_center: Vec3, camera_eye: Vec3) -> u8 {
    let distance = (chain_center - camera_eye).length();
    select_lod_tier(distance, 0.0)
}

impl Default for GeometryOptions {
    fn default() -> Self {
        Self {
            cartoon_style: CartoonStyle::default(),
            sheet_arrows: true,
            helix_width: 1.4,
            helix_thickness: 0.25,
            helix_roundness: 0.0,
            sheet_width: 1.6,
            sheet_thickness: 0.25,
            sheet_roundness: 0.0,
            coil_width: 0.4,
            coil_thickness: 0.4,
            coil_roundness: 1.0,
            na_width: 1.2,
            na_thickness: 0.25,
            na_roundness: 0.0,
            segments_per_residue: 32,
            cross_section_verts: 16,
            solvent_radius: 0.15,
            ligand_sphere_radius: 0.3,
            ligand_bond_radius: 0.12,
        }
    }
}

impl GeometryOptions {
    /// Return a copy with per-SS parameters resolved from the cartoon style
    /// preset. When `Custom`, returns self unchanged.
    #[must_use]
    pub fn resolve_cartoon_style(&self) -> Self {
        match self.cartoon_style {
            CartoonStyle::Custom => self.clone(),
            CartoonStyle::Ribbon => Self {
                helix_width: 1.4,
                helix_thickness: 0.25,
                helix_roundness: 0.0,
                sheet_width: 1.6,
                sheet_thickness: 0.25,
                sheet_roundness: 0.0,
                coil_width: 0.4,
                coil_thickness: 0.4,
                coil_roundness: 1.0,
                ..self.clone()
            },
            CartoonStyle::Tube => Self {
                helix_width: 0.4,
                helix_thickness: 0.4,
                helix_roundness: 1.0,
                sheet_width: 0.4,
                sheet_thickness: 0.4,
                sheet_roundness: 1.0,
                coil_width: 0.4,
                coil_thickness: 0.4,
                coil_roundness: 1.0,
                ..self.clone()
            },
            CartoonStyle::Cylindrical => Self {
                helix_width: 0.5,
                helix_thickness: 0.5,
                helix_roundness: 1.0,
                sheet_width: 1.6,
                sheet_thickness: 0.25,
                sheet_roundness: 0.0,
                coil_width: 0.3,
                coil_thickness: 0.3,
                coil_roundness: 1.0,
                ..self.clone()
            },
        }
    }

    /// Return a copy with helix parameters patched from a [`HelixStyle`].
    #[must_use]
    pub fn with_helix_style(&self, style: HelixStyle) -> Self {
        match style {
            HelixStyle::Ribbon => Self {
                helix_width: 1.4,
                helix_thickness: 0.25,
                helix_roundness: 0.0,
                ..self.clone()
            },
            HelixStyle::Tube => Self {
                helix_width: 0.4,
                helix_thickness: 0.4,
                helix_roundness: 1.0,
                ..self.clone()
            },
            HelixStyle::Cylinder => Self {
                helix_width: 0.5,
                helix_thickness: 0.5,
                helix_roundness: 1.0,
                ..self.clone()
            },
        }
    }

    /// Return a copy with sheet parameters patched from a [`SheetStyle`].
    #[must_use]
    pub fn with_sheet_style(&self, style: SheetStyle) -> Self {
        match style {
            SheetStyle::Ribbon => Self {
                sheet_width: 1.6,
                sheet_thickness: 0.25,
                sheet_roundness: 0.0,
                ..self.clone()
            },
            SheetStyle::Tube => Self {
                sheet_width: 0.4,
                sheet_thickness: 0.4,
                sheet_roundness: 1.0,
                ..self.clone()
            },
        }
    }
}
