use glam::Vec3;

use super::prepared::{
    BallAndStickInstances, CachedBackbone, CachedEntityMesh, FullRebuildEntity,
    NucleicAcidInstances,
};
use crate::options::{
    ChainLod, ColorOptions, DisplayOptions, DrawingMode, GeometryOptions,
    NaColorMode, SidechainColorMode,
};
use crate::renderer::entity_topology::{EntityTopology, SidechainLayout};
use crate::renderer::geometry::backbone::SheetOffset;
use crate::renderer::geometry::sheet_adjust::{
    adjust_bonds_for_sheet, adjust_sidechains_for_sheet,
};
use crate::renderer::geometry::{
    BackboneRenderer, BallAndStickRenderer, NucleicAcidRenderer,
    SidechainRenderer, SidechainView,
};

// ---------------------------------------------------------------------------
// Sidechain capsule instance helper
// ---------------------------------------------------------------------------

/// Resolve sidechain atom positions and backbone-bond anchor positions
/// from a layout against an entity's interpolated `positions` slice.
///
/// An out-of-range index means the topology layout and the position
/// slice have desynced upstream (the same invariant class as
/// [`ProteinBackboneIndices::resolve`](crate::renderer::entity_topology::ProteinBackboneIndices::resolve)),
/// not recoverable data -- fail loudly rather than substituting an origin
/// point that would then participate in centroid/offset math as if real.
#[allow(clippy::panic)]
fn resolve_sidechain_atoms(
    layout: &SidechainLayout,
    positions: &[Vec3],
) -> (Vec<Vec3>, Vec<(Vec3, u32)>) {
    let at = |idx: u32, role: &str| -> Vec3 {
        match positions.get(idx as usize) {
            Some(&p) => p,
            None => panic!(
                "sidechain {role} atom index {idx} out of range for {} \
                 positions: topology/position desync",
                positions.len(),
            ),
        }
    };
    let sidechain_positions = layout
        .atom_indices
        .iter()
        .map(|&idx| at(idx, "atom"))
        .collect();
    let backbone_bonds = layout
        .backbone_bonds
        .iter()
        .map(|&(ca_atom_idx, layout_idx)| (at(ca_atom_idx, "CA"), layout_idx))
        .collect();
    (sidechain_positions, backbone_bonds)
}

/// Derive the renderer-facing sidechain view from a topology slice and
/// interpolated atom positions, then apply sheet-surface adjustment
/// against the fitted sheet-plane offsets.
#[allow(clippy::too_many_arguments)]
fn generate_sidechain_bytes(
    topology: &EntityTopology,
    positions: &[Vec3],
    per_residue_colors: Option<&[[f32; 3]]>,
    sheet_offsets: &[SheetOffset],
    colors: &ColorOptions,
    display: &DisplayOptions,
) -> (Vec<u8>, u32) {
    let layout = &topology.sidechain_layout;
    if layout.atom_indices.is_empty() {
        return (Vec::new(), 0);
    }
    // Backbone->sidechain bonds use CA position (resolved from positions)
    // + an index into the sidechain layout.
    let (sidechain_positions, backbone_bonds) =
        resolve_sidechain_atoms(layout, positions);

    let adjusted_positions = adjust_sidechains_for_sheet(
        &sidechain_positions,
        &layout.residue_indices,
        sheet_offsets,
    );
    let adjusted_bonds = adjust_bonds_for_sheet(
        &backbone_bonds,
        &layout.residue_indices,
        sheet_offsets,
    );
    let view = SidechainView {
        positions: &adjusted_positions,
        bonds: &layout.bonds,
        backbone_bonds: &adjusted_bonds,
        hydrophobicity: &layout.hydrophobicity,
        residue_indices: &layout.residue_indices,
    };
    let backbone_colors = (display.sidechain_color_mode()
        == SidechainColorMode::Backbone)
        .then_some(per_residue_colors)
        .flatten();
    let insts = SidechainRenderer::generate_instances(
        &view,
        None,
        Some((colors.hydrophobic_sidechain, colors.hydrophilic_sidechain)),
        backbone_colors,
    );
    let count = insts.len() as u32;
    (bytemuck::cast_slice(&insts).to_vec(), count)
}

// ---------------------------------------------------------------------------
// Ball-and-stick / nucleic acid helper
// ---------------------------------------------------------------------------

/// Generate ball-and-stick + nucleic acid instance bytes for a single
/// entity.
///
/// BnS pick IDs are emitted with a 0 base offset; `mesh_concat` applies
/// the global offset during concatenation.
fn generate_non_backbone_bytes(
    entity: &FullRebuildEntity,
    display: &DisplayOptions,
    colors: &ColorOptions,
) -> (BallAndStickInstances, NucleicAcidInstances, u32) {
    let (bns_spheres, bns_capsules) =
        BallAndStickRenderer::generate_entity_instances(
            &entity.topology,
            &entity.positions,
            display,
            Some(colors),
            0,
            entity.drawing_mode,
            entity.per_residue_colors.as_deref(),
        );
    let rings = if entity.topology.is_nucleic_acid() {
        entity.topology.resolve_rings(&entity.positions)
    } else {
        Vec::new()
    };
    let (na_stems, na_rings) = NucleicAcidRenderer::generate_instances(&rings);
    let bns_atoms = if bns_spheres.is_empty() && bns_capsules.is_empty() {
        0
    } else {
        entity.topology.atom_elements.len() as u32
    };
    (
        BallAndStickInstances {
            sphere_instances: bytemuck::cast_slice(&bns_spheres).to_vec(),
            sphere_count: bns_spheres.len() as u32,
            capsule_instances: bytemuck::cast_slice(&bns_capsules).to_vec(),
            capsule_count: bns_capsules.len() as u32,
        },
        NucleicAcidInstances {
            stem_instances: bytemuck::cast_slice(&na_stems).to_vec(),
            stem_count: na_stems.len() as u32,
            ring_instances: bytemuck::cast_slice(&na_rings).to_vec(),
            ring_count: na_rings.len() as u32,
        },
        bns_atoms,
    )
}

// ---------------------------------------------------------------------------
// Entity mesh generation
// ---------------------------------------------------------------------------

/// Generate mesh for a single entity.
///
/// `per_chain_lod`, when `Some`, carries this entity's own per-chain LOD
/// tiers (already sliced to the entity's chains by the caller); `None`
/// uses the global `geometry` detail for every chain. Position-animation
/// frames pass `None`; only the camera-distance LOD remesh passes `Some`.
pub(super) fn generate_entity_mesh(
    entity: &FullRebuildEntity,
    display: &DisplayOptions,
    colors: &ColorOptions,
    geometry: &GeometryOptions,
    per_chain_lod: Option<&[ChainLod]>,
) -> CachedEntityMesh {
    let skip_backbone = entity.drawing_mode != DrawingMode::Cartoon;
    let topology = &entity.topology;

    let backbone_mesh = if skip_backbone {
        BackboneRenderer::generate_mesh_colored(
            &[],
            &[],
            None,
            None,
            geometry,
            None,
            None,
            None,
            None,
        )
    } else {
        let is_na = topology.is_nucleic_acid();
        let protein_chains = if is_na {
            Vec::new()
        } else {
            topology.protein_backbone_chains(&entity.positions)
        };
        let na_chains = if is_na {
            topology.na_backbone_chain_positions(&entity.positions)
        } else {
            Vec::new()
        };

        // Residue-parallel with the P-atom stream (built per residue,
        // not per resolvable ring) so a skipped/modified base doesn't
        // shift every later base's backbone color.
        let na_base_colors: &[[f32; 3]] =
            if is_na && display.na_color_mode() == NaColorMode::BaseColor {
                &topology.na_residue_base_colors
            } else {
                &[]
            };
        let na_colors_ref =
            (!na_base_colors.is_empty()).then_some(na_base_colors);

        let na_seeds: Vec<Option<Vec3>> = if is_na {
            topology.na_chain_seed_normals(&entity.positions)
        } else {
            Vec::new()
        };
        let na_seeds_ref =
            (!na_seeds.is_empty()).then_some(na_seeds.as_slice());

        let na_guides: Vec<Vec3> = if is_na {
            topology.na_residue_guide_dirs(&entity.positions)
        } else {
            Vec::new()
        };
        let na_guides_ref =
            (!na_guides.is_empty()).then_some(na_guides.as_slice());

        let ss_slice = entity
            .ss_override
            .as_deref()
            .or_else(|| Some(topology.ss_types.as_slice()))
            .filter(|s| !s.is_empty());

        BackboneRenderer::generate_mesh_colored(
            &protein_chains,
            &na_chains,
            ss_slice,
            entity.per_residue_colors.as_deref(),
            geometry,
            per_chain_lod,
            na_colors_ref,
            na_seeds_ref,
            na_guides_ref,
        )
    };

    let (sidechain_instances, sidechain_instance_count) = if skip_backbone {
        (Vec::new(), 0)
    } else {
        generate_sidechain_bytes(
            topology,
            &entity.positions,
            entity.per_residue_colors.as_deref(),
            &backbone_mesh.sheet_offsets,
            colors,
            display,
        )
    };
    let (bns, na, bns_atom_count) =
        generate_non_backbone_bytes(entity, display, colors);

    let residue_count = if topology.is_protein() {
        topology
            .protein_backbone_layout
            .iter()
            .map(|seg| seg.ca.len() as u32)
            .sum()
    } else {
        0
    };

    CachedEntityMesh {
        backbone: CachedBackbone {
            verts: bytemuck::cast_slice(&backbone_mesh.vertices).to_vec(),
            tube_inds: backbone_mesh.tube_indices,
            ribbon_inds: backbone_mesh.ribbon_indices,
            vert_count: backbone_mesh.vertices.len() as u32,
            sheet_offsets: backbone_mesh.sheet_offsets,
            chain_ranges: backbone_mesh.chain_ranges,
        },
        sidechain_instances,
        sidechain_instance_count,
        bns,
        na,
        residue_count,
        bns_atom_count,
        entity_id: entity.id,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A sidechain atom index past the position slice means a
    /// topology/position desync; resolution must fail loudly rather than
    /// substitute `Vec3::ZERO`, which would enter centroid math as a real
    /// origin-point atom.
    #[test]
    #[should_panic(expected = "topology/position desync")]
    fn resolve_sidechain_atoms_panics_on_out_of_range() {
        let mut layout = SidechainLayout::empty();
        layout.atom_indices = vec![0, 99];
        let positions = vec![Vec3::ZERO; 3];
        let _ = resolve_sidechain_atoms(&layout, &positions);
    }
}
