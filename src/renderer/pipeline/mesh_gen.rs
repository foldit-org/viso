use std::sync::Arc;

use glam::Vec3;
use molex::entity::molecule::id::EntityId;
use molex::SSType;
use rustc_hash::FxHashMap;

use super::prepared::{
    BackboneMeshData, BallAndStickInstances, CachedBackbone, CachedEntityMesh,
    FullRebuildEntity, NucleicAcidInstances, PreparedAnimationFrame,
};
use crate::engine::positions::EntityPositions;
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
pub(super) fn generate_entity_mesh(
    entity: &FullRebuildEntity,
    display: &DisplayOptions,
    colors: &ColorOptions,
    geometry: &GeometryOptions,
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
            None,
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
        entity_id: *entity.id,
    }
}

// ---------------------------------------------------------------------------
// Animation-frame regeneration (backbone + optional sidechains)
// ---------------------------------------------------------------------------

/// Cached per-scene inputs threaded into each animation frame.
pub(super) struct AnimationFrameCache {
    /// Per-entity topology snapshots (same Arcs the main thread holds).
    pub topologies: FxHashMap<EntityId, Arc<EntityTopology>>,
    /// Per-entity drawing-mode + SS-override + sidechain color lookup.
    pub entity_meta: FxHashMap<EntityId, EntityMetaSnapshot>,
    /// Concatenated SS types for Cartoon-mode entities only (feeds the
    /// backbone mesh).
    pub cartoon_ss_types: Option<Vec<SSType>>,
    /// Concatenated per-residue colors for Cartoon-mode entities only.
    pub cartoon_per_residue_colors: Option<Vec<[f32; 3]>>,
    /// Concatenated NA base colors for Cartoon-mode NA entities.
    pub cartoon_na_base_colors: Option<Vec<[f32; 3]>>,
    /// Hydrophobic / hydrophilic sidechain color pair for the
    /// Hydrophobicity sidechain mode. Captured from the global
    /// [`crate::options::ColorOptions`] at rebuild time.
    pub sidechain_palette: ([f32; 3], [f32; 3]),
    /// Rendering order of entities, captured at the last `FullRebuild`.
    pub entity_order: Vec<EntityId>,
}

/// Per-entity metadata the animator needs when regenerating a frame.
#[derive(Clone)]
pub(super) struct EntityMetaSnapshot {
    pub drawing_mode: DrawingMode,
    /// Per-residue colors for this entity (used by Backbone-mode
    /// sidechain coloring during animation frames). Mirrors
    /// [`super::prepared::FullRebuildEntity::per_residue_colors`] at the
    /// last rebuild.
    pub per_residue_colors: Option<Vec<[f32; 3]>>,
    /// Resolved sidechain color mode for this entity (per-entity
    /// appearance overrides applied).
    pub sidechain_color_mode: SidechainColorMode,
}

/// Input data for [`process_animation_frame`].
pub(super) struct AnimationFrameInput<'a> {
    pub positions: &'a EntityPositions,
    pub cache: &'a AnimationFrameCache,
    pub geometry: &'a GeometryOptions,
    pub per_chain_lod: Option<&'a [ChainLod]>,
    pub include_sidechains: bool,
}

/// Generate backbone + optional sidechain mesh for an animation frame
/// using only derived state + interpolated positions.
pub(super) fn process_animation_frame(
    input: &AnimationFrameInput,
    generation: u64,
) -> PreparedAnimationFrame {
    let (protein_chains, na_chains, na_seeds, na_guides) =
        collect_cartoon_chains(input);

    let total_residues: usize =
        protein_chains.iter().map(|c| c.ca().len()).sum::<usize>()
            + na_chains.iter().map(|c| c.p().len()).sum::<usize>();
    let safe_geo = input.geometry.clamped_for_residues(total_residues);

    let backbone_mesh = BackboneRenderer::generate_mesh_colored(
        &protein_chains,
        &na_chains,
        input.cache.cartoon_ss_types.as_deref(),
        input.cache.cartoon_per_residue_colors.as_deref(),
        &safe_geo,
        input.per_chain_lod,
        input.cache.cartoon_na_base_colors.as_deref(),
        (!na_seeds.is_empty()).then_some(na_seeds.as_slice()),
        (!na_guides.is_empty()).then_some(na_guides.as_slice()),
    );
    let backbone_tube_index_count = backbone_mesh.tube_indices.len() as u32;
    let backbone_ribbon_index_count = backbone_mesh.ribbon_indices.len() as u32;
    let backbone_vertices =
        bytemuck::cast_slice(&backbone_mesh.vertices).to_vec();
    let backbone_tube_indices =
        bytemuck::cast_slice(&backbone_mesh.tube_indices).to_vec();
    let backbone_ribbon_indices =
        bytemuck::cast_slice(&backbone_mesh.ribbon_indices).to_vec();

    let (sidechain_instances, sidechain_instance_count) =
        if input.include_sidechains {
            generate_animation_sidechains(input, &backbone_mesh.sheet_offsets)
        } else {
            (None, 0)
        };

    PreparedAnimationFrame {
        backbone: BackboneMeshData {
            vertices: backbone_vertices,
            tube_indices: backbone_tube_indices,
            tube_index_count: backbone_tube_index_count,
            ribbon_indices: backbone_ribbon_indices,
            ribbon_index_count: backbone_ribbon_index_count,
            sheet_offsets: backbone_mesh.sheet_offsets,
            chain_ranges: backbone_mesh.chain_ranges,
        },
        sidechain_instances,
        sidechain_instance_count,
        generation,
    }
}

/// Walk cached Cartoon-mode entities in their last rebuild order and
/// resolve each entity's backbone positions from the current animator
/// frame.
fn collect_cartoon_chains(
    input: &AnimationFrameInput,
) -> (
    Vec<crate::renderer::entity_topology::ProteinBackboneChain>,
    Vec<crate::renderer::entity_topology::NaBackboneChain>,
    Vec<Option<Vec3>>,
    Vec<Vec3>,
) {
    let mut protein_chains = Vec::new();
    let mut na_chains: Vec<crate::renderer::entity_topology::NaBackboneChain> =
        Vec::new();
    // Per-NA-chain RMF roll seed (index-parallel with `na_chains`) and
    // per-residue C1'-P guide field (residue-parallel with the
    // concatenated P stream `process_na_chains` walks), both recomputed
    // from the *interpolated* positions each frame.
    let mut na_seeds: Vec<Option<Vec3>> = Vec::new();
    let mut na_guides: Vec<Vec3> = Vec::new();
    for id in &input.cache.entity_order {
        let Some(meta) = input.cache.entity_meta.get(id) else {
            continue;
        };
        if meta.drawing_mode != DrawingMode::Cartoon {
            continue;
        }
        let Some(topology) = input.cache.topologies.get(id) else {
            continue;
        };
        let Some(positions) = input.positions.get(*id) else {
            continue;
        };
        if topology.is_nucleic_acid() {
            na_chains.extend(topology.na_backbone_chain_positions(positions));
            na_seeds.extend(topology.na_chain_seed_normals(positions));
            na_guides.extend(topology.na_residue_guide_dirs(positions));
        } else {
            protein_chains.extend(topology.protein_backbone_chains(positions));
        }
    }
    (protein_chains, na_chains, na_seeds, na_guides)
}

/// Concatenate per-entity sidechain capsule bytes for the animation
/// frame.
fn generate_animation_sidechains(
    input: &AnimationFrameInput,
    sheet_offsets: &[SheetOffset],
) -> (Option<Vec<u8>>, u32) {
    let mut combined: Vec<u8> = Vec::new();
    let mut total_count: u32 = 0;

    for id in &input.cache.entity_order {
        let Some(meta) = input.cache.entity_meta.get(id) else {
            continue;
        };
        if meta.drawing_mode != DrawingMode::Cartoon {
            continue;
        }
        let Some(topology) = input.cache.topologies.get(id) else {
            continue;
        };
        if topology.sidechain_layout.atom_indices.is_empty() {
            continue;
        }
        let Some(positions) = input.positions.get(*id) else {
            continue;
        };
        let layout = &topology.sidechain_layout;
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
        let backbone_colors = (meta.sidechain_color_mode
            == SidechainColorMode::Backbone)
            .then_some(meta.per_residue_colors.as_deref())
            .flatten();
        let insts = SidechainRenderer::generate_instances(
            &view,
            None,
            Some(input.cache.sidechain_palette),
            backbone_colors,
        );
        total_count += insts.len() as u32;
        combined.extend_from_slice(bytemuck::cast_slice(&insts));
    }

    (Some(combined), total_count)
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
