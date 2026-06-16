//! Cross-entity scene render state.
//!
//! [`SceneRenderState`] is the scene-wide derived output of an
//! [`Assembly`](molex::Assembly) snapshot: the per-type
//! [`AtomLink`](molex::AtomLink) lists the viewer draws
//! (hydrogen bonds, disulfides). The links are read straight off
//! [`Assembly::connections`](molex::Assembly::connections) on sync; the
//! actual world-space resolution into [`StructuralBond`] capsules runs in
//! the constraint pass against the live positions and sheet offsets, so an
//! animating ribbon or a flattened sheet-residue SG is tracked rather than
//! frozen on the raw atom coordinate the sync saw.

use glam::Vec3;

use super::entity_view::RibbonBackbone;
use crate::options::{BondOptions, ColorOptions, DrawingMode};
use crate::renderer::geometry::backbone::SheetOffset;
use crate::renderer::geometry::bond::StructuralBond;
use crate::renderer::geometry::sheet_adjust::sheet_offset_at;

// SceneRenderState

/// Cross-entity rendering data derived from [`Assembly`].
///
/// Holds the per-type [`AtomLink`] lists pulled from
/// [`Assembly::connections`](molex::Assembly::connections) on sync. The
/// constraint pass resolves these to [`StructuralBond`] capsules every
/// time positions become final, so the capsules track animated atoms and
/// pick up sheet-flattening offsets that are only known after the mesh
/// returns.
///
/// [`Assembly`]: molex::Assembly
/// [`AtomLink`]: molex::AtomLink
#[derive(Clone, Default)]
pub(crate) struct SceneRenderState {
    /// Disulfide bridge links (SG–SG pairs). Pulled from the
    /// [`ConnectionType::Disulfide`](molex::ConnectionType) entry of
    /// [`Assembly::connections`](molex::Assembly::connections) on sync.
    pub(crate) disulfide_links: Vec<molex::AtomLink>,
    /// Hydrogen-bond links (donor N / acceptor carbonyl C atom pairs).
    /// Pulled from the [`ConnectionType::HBond`](molex::ConnectionType)
    /// entry of [`Assembly::connections`](molex::Assembly::connections)
    /// on sync.
    pub(crate) hbond_links: Vec<molex::AtomLink>,
}

impl SceneRenderState {
    /// Empty scene render state.
    #[must_use]
    pub(crate) fn new() -> Self {
        Self::default()
    }

    /// Rederive scene state from the latest Assembly snapshot: copy the
    /// per-type connection links the constraint pass resolves each frame.
    #[must_use]
    pub(crate) fn from_assembly(assembly: &molex::Assembly) -> Self {
        let connections = assembly.connections();
        Self {
            disulfide_links: connections
                .get(&molex::ConnectionType::Disulfide)
                .cloned()
                .unwrap_or_default(),
            hbond_links: connections
                .get(&molex::ConnectionType::HBond)
                .cloned()
                .unwrap_or_default(),
        }
    }
}

// Helpers

/// Build a hydrogen-bond [`StructuralBond`] capsule between two
/// already-resolved world-space positions, using the hydrogen-bond
/// style, radius, tint, emissive, and opacity. Shared so every
/// hydrogen-bond capsule renders identically.
pub(crate) fn external_hbond_capsule(
    pos_a: Vec3,
    pos_b: Vec3,
    bonds: &BondOptions,
    colors: &ColorOptions,
) -> StructuralBond {
    let opts = &bonds.hydrogen_bonds;
    StructuralBond {
        pos_a,
        pos_b,
        color: tinted(colors.band_hbond),
        radius: opts.radius,
        residue_idx: 0,
        style: opts.style,
        emissive: 0.6,
        opacity: 0.5,
    }
}

/// Build a disulfide [`StructuralBond`] capsule between two
/// already-resolved world-space positions, using the same style, radius,
/// tint, emissive, and opacity the sync-time molex path used. Shared so the
/// per-frame disulfide pass (resolved in the constraint pass) renders
/// identically to the prior sync-resolved disulfides. The cysteine SG atoms
/// are sidechain, so the per-frame resolver re-anchors them onto the
/// sheet-flattened stick when the residue is on a strand.
pub(crate) fn disulfide_capsule(
    pos_a: Vec3,
    pos_b: Vec3,
    bonds: &BondOptions,
    colors: &ColorOptions,
) -> StructuralBond {
    let opts = &bonds.disulfide_bonds;
    StructuralBond {
        pos_a,
        pos_b,
        color: tinted(colors.band_disulfide),
        radius: opts.radius,
        residue_idx: 0,
        style: opts.style,
        emissive: 0.6,
        opacity: 0.5,
    }
}

/// Resolve an atom's *rendered* world-space position, accounting for both
/// Cartoon-mode render transforms so a structural bond attaches to the
/// drawn geometry rather than to the raw atom coordinate.
///
/// In Cartoon mode the structure is not drawn at raw atom positions: the
/// backbone is projected onto the ribbon spline, and beta-strand sidechains
/// are shifted by a per-residue sheet-flattening offset. This resolver
/// applies whichever transform the atom is subject to:
///
/// - **Backbone** N → the ribbon's N control point; carbonyl O or C → the
///   ribbon's C control point (rosetta names the backbone acceptor O, and the
///   ribbon carries only N and C control points per residue, so the carbonyl
///   region maps to C). CA sits on the spline already, so it stays raw. With no
///   ribbon (too short to project) the backbone atom is raw.
/// - **Sidechain** (anything else, including the disulfide SG) → `raw +
///   sheet_offset` when this residue is on a flattened strand, matching the
///   shift the mesh applied to the sidechain sticks; raw otherwise.
/// - **Non-Cartoon / non-protein** → always raw (no render transform).
///
/// `residue` is the entity-local residue index; `sheet_offsets` is this
/// entity's own ascending offset slice
/// ([`EntityView::sheet_offsets`](super::entity_view::EntityView::sheet_offsets)).
/// This is the single positioning function the molex hbond, rosetta hbond,
/// and disulfide bond paths share.
pub(crate) fn rendered_atom_position(
    raw: Vec3,
    drawing_mode: DrawingMode,
    is_protein: bool,
    ribbon: Option<&RibbonBackbone>,
    sheet_offsets: &[SheetOffset],
    residue: u32,
    atom_name: &str,
) -> Vec3 {
    if drawing_mode != DrawingMode::Cartoon || !is_protein {
        return raw;
    }
    match atom_name {
        "N" => ribbon.and_then(|r| r.n_at(residue)).unwrap_or(raw),
        // Backbone carbonyl: the acceptor is named O upstream; the ribbon
        // anchors the carbonyl region at its C control point.
        "O" | "C" => ribbon.and_then(|r| r.c_at(residue)).unwrap_or(raw),
        // CA is essentially on the spline; leave it raw.
        "CA" => raw,
        // Sidechain atom (incl. SG): re-anchor onto the flattened stick.
        _ => {
            raw + sheet_offset_at(sheet_offsets, residue).unwrap_or(Vec3::ZERO)
        }
    }
}

/// Lighten a base color 50% toward white for on-screen pop.
fn tinted(base: [f32; 3]) -> [f32; 3] {
    let w = 0.5;
    [
        base[0] + (1.0 - base[0]) * w,
        base[1] + (1.0 - base[1]) * w,
        base[2] + (1.0 - base[2]) * w,
    ]
}
