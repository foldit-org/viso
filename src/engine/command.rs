//! Structural reference types for atom-anchored constraints (bands +
//! pull).
//!
//! Bands and pulls anchor to atoms by structural reference
//! ([`crate::engine::command::AtomRef`]) rather than world-space
//! positions. The engine resolves the positions every frame so the
//! renderable geometry tracks animated atoms.

use glam::Vec3;

// ── Constraint payload types ────────────────────────────────────────────

/// Type of constraint band for color coding.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum BandType {
    /// Default band (purple).
    #[default]
    Default,
    /// Backbone-to-backbone band (yellow-orange).
    Backbone,
    /// Disulfide bridge band (yellow-green).
    Disulfide,
    /// Hydrogen bond band (cyan).
    HBond,
}

/// Structural reference to a specific atom.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AtomRef {
    /// 0-based flat residue index.
    pub residue: u32,
    /// PDB atom name ("CA", "CB", "N", etc.).
    pub atom_name: String,
}

/// One end of a band constraint.
#[derive(Debug, Clone, PartialEq)]
pub enum BandTarget {
    /// Attached to a specific atom.
    Atom(AtomRef),
    /// Anchored to a fixed world-space position (space pulls).
    Position(Vec3),
}

/// Information about a constraint band to be rendered.
///
/// Uses structural references ([`AtomRef`]) instead of world-space
/// positions. The engine resolves atom positions each frame from Scene
/// data, so bands auto-track animated atoms.
#[derive(Debug, Clone, PartialEq)]
#[allow(clippy::struct_excessive_bools)]
pub struct BandInfo {
    /// First endpoint — always an atom.
    pub anchor_a: AtomRef,
    /// Second endpoint — atom or fixed position.
    pub anchor_b: BandTarget,
    /// Band strength (affects radius and color intensity, default 1.0).
    pub strength: f32,
    /// Target length for the band (Angstroms, used for type detection if
    /// not specified).
    pub target_length: f32,
    /// Explicit band type (overrides auto-detection from `target_length`).
    pub band_type: Option<BandType>,
    /// Whether the band is in pull mode (attracts).
    pub is_pull: bool,
    /// Whether the band is in push mode (repels).
    pub is_push: bool,
    /// Whether the band is disabled.
    pub is_disabled: bool,
    /// Whether this band was created by a recipe/script (dimmer appearance).
    pub from_script: bool,
}

/// Information about the active pull constraint.
///
/// Uses a structural reference ([`AtomRef`]) for the pulled atom and a
/// screen-space target. The engine resolves atom position from Scene data
/// and unprojecs `screen_target` at atom depth each frame, so the pull
/// auto-tracks the animated atom.
#[derive(Debug, Clone, PartialEq)]
pub struct PullInfo {
    /// The atom being pulled.
    pub atom: AtomRef,
    /// Screen-space drag position (physical pixels).
    pub screen_target: (f32, f32),
}

// ── Resolved types (internal, world-space) ──────────────────────────────

/// Resolved band with world-space positions, ready for the renderer.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct ResolvedBand {
    /// World-space position of first endpoint.
    pub(crate) endpoint_a: Vec3,
    /// World-space position of second endpoint.
    pub(crate) endpoint_b: Vec3,
    /// Whether the band is disabled.
    pub(crate) is_disabled: bool,
    /// Band strength (affects radius and color intensity).
    pub(crate) strength: f32,
    /// Target length for the band (used for type detection).
    pub(crate) target_length: f32,
    /// Residue index for picking (from anchor_a).
    pub(crate) residue_idx: u32,
    /// Whether anchor_b is a fixed position (renders anchor sphere).
    pub(crate) is_space_pull: bool,
    /// Explicit band type.
    pub(crate) band_type: Option<BandType>,
    /// Whether this band was created by a script.
    pub(crate) from_script: bool,
}

/// Resolved pull with world-space positions, ready for the renderer.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct ResolvedPull {
    /// World-space position of the atom being pulled.
    pub(crate) atom_pos: Vec3,
    /// World-space target position.
    pub(crate) target_pos: Vec3,
    /// Residue index for picking.
    pub(crate) residue_idx: u32,
}
