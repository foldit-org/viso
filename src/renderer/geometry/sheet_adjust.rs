use std::collections::HashMap;

use glam::Vec3;

use crate::renderer::geometry::sidechain::{OwnedSidechainView, SidechainView};

/// Total residue count from SoA protein backbone chains.
pub(crate) fn backbone_residue_count(
    backbone_chains: &[crate::renderer::entity_topology::ProteinBackboneChain],
) -> usize {
    backbone_chains.iter().map(|c| c.ca.len()).sum()
}

/// Apply sheet-surface offsets to sidechain positions and backbone-sidechain
/// bonds, returning an [`OwnedSidechainView`] ready for the renderer.
pub(crate) fn sheet_adjusted_view(
    sidechain: &SidechainView<'_>,
    offset_map: &HashMap<u32, Vec3>,
) -> OwnedSidechainView {
    let positions = adjust_sidechains_for_sheet(
        sidechain.positions,
        sidechain.residue_indices,
        offset_map,
    );
    let backbone_bonds = adjust_bonds_for_sheet(
        sidechain.backbone_bonds,
        sidechain.residue_indices,
        offset_map,
    );
    OwnedSidechainView {
        positions,
        bonds: sidechain.bonds.to_vec(),
        backbone_bonds,
        hydrophobicity: sidechain.hydrophobicity.to_vec(),
        residue_indices: sidechain.residue_indices.to_vec(),
    }
}

/// Translate sidechain atom positions by sheet-flattening offsets.
pub(crate) fn adjust_sidechains_for_sheet(
    positions: &[Vec3],
    sidechain_residue_indices: &[u32],
    offset_map: &HashMap<u32, Vec3>,
) -> Vec<Vec3> {
    if offset_map.is_empty() {
        return positions.to_vec();
    }
    positions
        .iter()
        .enumerate()
        .map(|(i, &pos)| {
            let res_idx = sidechain_residue_indices
                .get(i)
                .copied()
                .unwrap_or(u32::MAX);
            if let Some(&offset) = offset_map.get(&res_idx) {
                pos + offset
            } else {
                pos
            }
        })
        .collect()
}

/// Translate CA-CB bond base positions by sheet-flattening offsets.
pub(crate) fn adjust_bonds_for_sheet(
    bonds: &[(Vec3, u32)],
    sidechain_residue_indices: &[u32],
    offset_map: &HashMap<u32, Vec3>,
) -> Vec<(Vec3, u32)> {
    if offset_map.is_empty() {
        return bonds.to_vec();
    }
    bonds
        .iter()
        .map(|(ca_pos, cb_idx)| {
            let res_idx = sidechain_residue_indices
                .get(*cb_idx as usize)
                .copied()
                .unwrap_or(u32::MAX);
            if let Some(&offset) = offset_map.get(&res_idx) {
                (*ca_pos + offset, *cb_idx)
            } else {
                (*ca_pos, *cb_idx)
            }
        })
        .collect()
}
