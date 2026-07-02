//! Seam-time hydrogen-bond + disulfide connection resolution: resolve
//! [`molex::Assembly::connections`] [`molex::AtomLink`]s to rendered capsule
//! positions whenever positions become final (a sync, a consumed mesh, or
//! an animation/trajectory tick).
//!
//! Unlike the per-frame band/pull/clash/grease resolution in
//! [`crate::engine::constraint`], this path runs only at those
//! position-finalizing seams, so a backbone hbond attaches to the mesh's
//! emitted ribbon anchors and a sheet-residue SG picks up the
//! sheet-flattening offset while a resting scene re-resolves nothing per
//! frame. The ribbon-anchor view and the shared
//! [`rendered_atom_position`] resolver are borrowed from
//! [`crate::engine::constraint`].

use glam::Vec3;
use molex::entity::molecule::id::EntityId;
use molex::{AtomEnd, AtomId, AtomLink};
use rustc_hash::FxHashMap;

use super::annotations::EntityAnnotations;
use super::constraint::build_hbond_ribbons;
use super::entity_view::{EntityView, RibbonBackbone};
use super::scene::Scene;
use super::scene_state::{
    disulfide_capsule, external_hbond_capsule, rendered_atom_position,
};
use crate::options::VisoOptions;
use crate::renderer::geometry::bond::StructuralBond;
use crate::renderer::GpuPipeline;

/// Resolve every hydrogen-bond and disulfide connection link to a
/// render-ready [`StructuralBond`] capsule and re-upload the complete bond
/// buffer.
///
/// Runs whenever positions become final (a sync, a consumed mesh, or an
/// animation/trajectory tick) rather than every frame, so a backbone hbond
/// attaches to the mesh's emitted ribbon anchors and a sheet-residue SG
/// picks up the sheet-flattening offset (both only known after the mesh
/// returns) while a resting scene re-resolves nothing per frame. Each toggle
/// gates its
/// source; the full set is uploaded unconditionally (an empty set clears
/// the buffer rather than leaving a stale one) and the renderer reallocs as
/// needed, so the tracked capacity always equals the live link count.
pub(super) fn resolve_and_upload_bond_connections(
    scene: &Scene,
    annotations: &EntityAnnotations,
    options: &VisoOptions,
    gpu: &mut GpuPipeline,
) {
    // Ribbon-anchor view for Cartoon-mode protein entities: a backbone hbond
    // endpoint resolves to the mesh's emitted N/C anchor so the end lands on
    // the drawn ribbon. Disulfide SG atoms are sidechain and ignore the ribbon.
    let ribbons = build_hbond_ribbons(scene);

    let mut bonds: Vec<StructuralBond> = Vec::new();
    if options.display.bonds.hydrogen_bonds.visible {
        append_link_capsules(
            &mut bonds,
            scene,
            annotations,
            &ribbons,
            &scene.render_state.hbond_links,
            |pos_a, pos_b| {
                external_hbond_capsule(
                    pos_a,
                    pos_b,
                    &options.display.bonds,
                    &options.colors,
                )
            },
        );
    }
    if options.display.bonds.disulfide_bonds.visible {
        append_link_capsules(
            &mut bonds,
            scene,
            annotations,
            &ribbons,
            &scene.render_state.disulfide_links,
            |pos_a, pos_b| {
                disulfide_capsule(
                    pos_a,
                    pos_b,
                    &options.display.bonds,
                    &options.colors,
                )
            },
        );
    }
    let _ = gpu.renderers.bond.update(
        &gpu.context.device,
        &gpu.context.queue,
        &bonds,
    );
}

/// Resolve each [`AtomLink`]'s two endpoints to rendered world-space
/// positions and push the built capsule. A link is dropped if either
/// endpoint fails to resolve (exactly as clashes skip) or if either
/// endpoint's owning entity is hidden (see [`connection_end_visible`]).
fn append_link_capsules(
    out: &mut Vec<StructuralBond>,
    scene: &Scene,
    annotations: &EntityAnnotations,
    ribbons: &FxHashMap<EntityId, RibbonBackbone<'_>>,
    links: &[AtomLink],
    build: impl Fn(Vec3, Vec3) -> StructuralBond,
) {
    for link in links {
        if !connection_end_visible(annotations, &link.a)
            || !connection_end_visible(annotations, &link.b)
        {
            continue;
        }
        let Some(pos_a) = resolve_connection_end(scene, ribbons, &link.a)
        else {
            continue;
        };
        let Some(pos_b) = resolve_connection_end(scene, ribbons, &link.b)
        else {
            continue;
        };
        out.push(build(pos_a, pos_b));
    }
}

/// Whether a connection endpoint's owning entity is currently drawn. An
/// [`AtomEnd::Atom`] consults the per-entity visibility overlay (missing
/// entry means visible, matching every other resolve path); an
/// [`AtomEnd::Anchor`] is a fixed world point with no owning entity and is
/// always treated as visible.
fn connection_end_visible(
    annotations: &EntityAnnotations,
    end: &AtomEnd,
) -> bool {
    match end {
        AtomEnd::Anchor(_) => true,
        AtomEnd::Atom(atom) => annotations.is_visible(atom.entity),
    }
}

/// Resolve one connection endpoint ([`AtomEnd`]) to its rendered
/// world-space position.
///
/// An [`AtomEnd::Atom`] resolves through the shared
/// [`rendered_atom_position`] resolver: raw position from `scene.positions`
/// for the atom's `(entity, index)`, then the Cartoon-mode render
/// transform keyed on the atom's role within its residue (backbone N → the
/// ribbon's N control point; carbonyl C / O → the ribbon's C control
/// point; CA stays raw; any sidechain atom, including the cysteine SG,
/// picks up the per-residue sheet-flattening offset). An
/// [`AtomEnd::Anchor`] is a fixed world-space point passed straight
/// through. Returns `None` if the atom's entity or position is absent (the
/// link is dropped, exactly as clashes skip).
fn resolve_connection_end(
    scene: &Scene,
    ribbons: &FxHashMap<EntityId, RibbonBackbone<'_>>,
    end: &AtomEnd,
) -> Option<Vec3> {
    match end {
        AtomEnd::Anchor(pos) => Some(*pos),
        AtomEnd::Atom(atom) => resolve_connection_atom(scene, ribbons, *atom),
    }
}

/// Resolve a connection's [`AtomId`] endpoint to its rendered world-space
/// position.
///
/// The raw coordinate comes from the entity's displayed-frame snapshot for
/// `(entity, index)` (the positions the drawn mesh was built from), so the
/// capsule pairs with the same-frame ribbon/sheet transform; the
/// entity-local residue comes from `topology.atom_residue_index`. The atom's
/// role name (which keys the Cartoon render transform) is derived from the
/// atom's offset within its residue's canonical `N, CA, C, O, sidechain...`
/// layout. Falls back to the live position when the entity view is absent,
/// or to the raw position when the residue index is absent.
fn resolve_connection_atom(
    scene: &Scene,
    ribbons: &FxHashMap<EntityId, RibbonBackbone<'_>>,
    atom: AtomId,
) -> Option<Vec3> {
    let Some(state) = scene.entity_state.get(&atom.entity) else {
        // No view yet: read the live position directly (no render
        // transform applies without a view).
        return scene
            .positions
            .get(atom.entity)
            .and_then(|slice| slice.get(atom.index as usize).copied());
    };
    let raw = super::constraint::displayed_positions(state, scene, atom.entity)
        .and_then(|slice| slice.get(atom.index as usize).copied())?;
    let Some(residue) = state
        .topology
        .atom_residue_index
        .get(atom.index as usize)
        .copied()
    else {
        return Some(raw);
    };
    Some(rendered_atom_position(
        raw,
        state.drawing_mode,
        state.topology.is_protein(),
        ribbons.get(&atom.entity),
        &state.sheet_offsets,
        residue,
        backbone_role_name(state, residue, atom.index),
    ))
}

/// The role name [`rendered_atom_position`] keys its Cartoon transform on,
/// derived from the atom's offset within its residue's canonical
/// `N, CA, C, O, sidechain...` layout: offset 0 → "N", 1 → "CA", 2 → "C",
/// 3 → "O", everything else → a sidechain marker. Any name the resolver
/// does not special-case ("N"/"CA"/"C"/"O") routes to the sidechain arm,
/// so "SC" stands in for every sidechain atom (including the cysteine SG).
fn backbone_role_name(
    state: &EntityView,
    residue: u32,
    atom_index: u32,
) -> &'static str {
    let Some(range) = state.topology.residue_atom_ranges.get(residue as usize)
    else {
        return "SC";
    };
    match atom_index.checked_sub(range.start) {
        Some(0) => "N",
        Some(1) => "CA",
        Some(2) => "C",
        Some(3) => "O",
        _ => "SC",
    }
}
