//! Structural reference types for atom-anchored constraints (bands +
//! pull).
//!
//! Bands and pulls anchor to atoms by structural reference
//! ([`crate::engine::command::AtomRef`]) rather than world-space
//! positions. The engine resolves the positions every frame so the
//! renderable geometry tracks animated atoms.

use glam::Vec3;
use molex::entity::molecule::id::EntityId;

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

/// One clashing atom, referenced per-entity.
///
/// Unlike [`AtomRef`] (which uses a flat residue index), a clash endpoint
/// names its owning entity and an entity-local residue. The flat ordering
/// is authoritative only inside viso, so the host cannot compute it; viso
/// resolves these per-entity refs directly against Scene data (the same
/// principle as
/// [`VisoEngine::set_selection`](crate::VisoEngine::set_selection)).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ClashEndpoint {
    /// Owning entity.
    pub entity: EntityId,
    /// Entity-local residue index (0-based).
    pub residue: u32,
    /// PDB atom name ("CA", "CB", "N", etc.).
    pub atom_name: String,
}

/// Information about a steric clash to be rendered as an electric arc.
///
/// Both endpoints are always atoms (clashes are atom-atom). Uses per-entity
/// structural references ([`ClashEndpoint`]) instead of world-space
/// positions, so the engine resolves atom positions each frame from Scene
/// data and clash arcs auto-track animated atoms (mirrors [`BandInfo`]).
#[derive(Debug, Clone, PartialEq)]
pub struct ClashInfo {
    /// First clashing atom.
    pub a: ClashEndpoint,
    /// Second clashing atom.
    pub b: ClashEndpoint,
    /// Clash severity (drives emissive intensity and pulse brightness).
    pub severity: f32,
}

/// A flagged exposed-hydrophobic residue, referenced per-entity.
///
/// Names its owning entity and an entity-local residue (same per-entity
/// principle as [`ClashEndpoint`] /
/// [`VisoEngine::set_selection`](crate::VisoEngine::set_selection)); viso
/// owns the flat ordering, so the host cannot compute it. The engine
/// resolves the residue to a world-space sidechain anchor every frame so
/// the bead marker tracks the residue live.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExposedHydrophobicInfo {
    /// Owning entity.
    pub entity: EntityId,
    /// Entity-local residue index (0-based).
    pub residue: u32,
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

/// Resolved clash with world-space positions, ready for the renderer.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct ResolvedClash {
    /// World-space position of first clashing atom.
    pub(crate) endpoint_a: Vec3,
    /// World-space position of second clashing atom.
    pub(crate) endpoint_b: Vec3,
    /// Clash severity (drives the lightning bolt's jag amplitude).
    pub(crate) severity: f32,
    /// Stable per-clash seed (deterministic hash of the atom pair) that
    /// decorrelates the procedural bolt's jag and flicker between clashes.
    pub(crate) seed: f32,
}

/// Resolved exposed-hydrophobic marker with a world-space anchor, ready
/// for the bead renderer.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct ResolvedExposedHydro {
    /// World-space sidechain anchor (CB if present, else sidechain
    /// centroid, else CA).
    pub(crate) center: Vec3,
    /// Stable per-bead seed (deterministic hash of entity + residue) that
    /// decorrelates the procedural "boil" between beads.
    pub(crate) seed: f32,
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
