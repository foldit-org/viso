//! Structural reference types for atom-anchored render annotations (bands,
//! pulls, clashes, exposed hydrophobics).
//!
//! These types name atoms by structural reference ([`AtomRef`] for a flat
//! residue index, [`ClashEndpoint`] for an entity-local one) rather than
//! by world-space position. The engine re-resolves those references to
//! positions every frame from Scene data, so the geometry auto-tracks
//! animated atoms. viso owns the flat residue ordering, so the host cannot
//! precompute these positions (the same principle as
//! [`VisoEngine::set_selection`](crate::VisoEngine::set_selection)).

use glam::Vec3;
use molex::entity::molecule::id::EntityId;

// Constraint payload types

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

/// A constraint band to be rendered, anchored by [`AtomRef`] (see the
/// module docs for the per-frame resolution contract).
#[derive(Debug, Clone, PartialEq)]
#[allow(clippy::struct_excessive_bools)]
pub struct BandInfo {
    /// First endpoint; always an atom.
    pub anchor_a: AtomRef,
    /// Second endpoint; atom or fixed position.
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
/// Unlike [`AtomRef`] (a flat residue index), a clash endpoint names its
/// owning entity and an entity-local residue (see the module docs for the
/// per-frame resolution contract).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ClashEndpoint {
    /// Owning entity.
    pub entity: EntityId,
    /// Entity-local residue index (0-based).
    pub residue: u32,
    /// PDB atom name ("CA", "CB", "N", etc.).
    pub atom_name: String,
}

/// A steric clash to be rendered as an electric arc between two atoms,
/// each a [`ClashEndpoint`] (see the module docs for the per-frame
/// resolution contract).
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
/// Named like a [`ClashEndpoint`]; the engine resolves it to a world-space
/// sidechain anchor each frame (see the module docs for the contract).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExposedHydrophobicInfo {
    /// Owning entity.
    pub entity: EntityId,
    /// Entity-local residue index (0-based).
    pub residue: u32,
}

/// The active pull constraint: an [`AtomRef`] for the pulled atom plus a
/// screen-space target unprojected at atom depth each frame (see the
/// module docs for the per-frame resolution contract).
#[derive(Debug, Clone, PartialEq)]
pub struct PullInfo {
    /// The atom being pulled.
    pub atom: AtomRef,
    /// Screen-space drag position (physical pixels).
    pub screen_target: (f32, f32),
}

/// A transient select-sphere overlay in world space, pushed by the host
/// while a select-sphere drag gesture is active.
///
/// Unlike the atom-anchored specs above, this carries a resolved
/// world-space centre and radius directly, so the engine forwards it to
/// the renderer with no per-frame reference resolution.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SelectSphereInfo {
    /// Sphere centre, world space, angstroms.
    pub center: Vec3,
    /// Sphere radius, world space, angstroms.
    pub radius: f32,
}

// Resolved types (internal, world-space)

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
