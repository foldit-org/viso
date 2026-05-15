use std::collections::HashMap;

use glam::Vec3;

use crate::renderer::geometry::sidechain::{OwnedSidechainView, SidechainView};

/// Total residue count from SoA protein backbone chains.
pub(crate) fn backbone_residue_count(
    backbone_chains: &[crate::renderer::entity_topology::ProteinBackboneChain],
) -> usize {
    backbone_chains.iter().map(|c| c.ca().len()).sum()
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
    // `positions` and `sidechain_residue_indices` are parallel sidechain-
    // atom arrays from the same `SidechainLayout`; zip rather than index
    // so there is no "missing residue index" case to paper over with a
    // sentinel. A length mismatch is layout corruption, asserted here.
    debug_assert_eq!(
        positions.len(),
        sidechain_residue_indices.len(),
        "sidechain positions / residue-index arrays desynced"
    );
    positions
        .iter()
        .zip(sidechain_residue_indices.iter())
        .map(|(&pos, &res_idx)| match offset_map.get(&res_idx) {
            Some(&offset) => pos + offset,
            None => pos,
        })
        .collect()
}

/// Translate CA-CB bond base positions by sheet-flattening offsets.
// `cb_idx` indexes the sidechain layout; an out-of-range value is layout
// corruption, not data, so fail loudly rather than masking it with a
// sentinel residue index that silently never matches an offset.
#[allow(clippy::panic)]
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
            let Some(&res_idx) =
                sidechain_residue_indices.get(*cb_idx as usize)
            else {
                panic!(
                    "backbone-bond layout index {cb_idx} out of range for {} \
                     sidechain residue indices: layout corruption",
                    sidechain_residue_indices.len(),
                )
            };
            if let Some(&offset) = offset_map.get(&res_idx) {
                (*ca_pos + offset, *cb_idx)
            } else {
                (*ca_pos, *cb_idx)
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// An out-of-range backbone-bond layout index is layout corruption;
    /// it must fail loudly rather than resolve to a `u32::MAX` sentinel
    /// residue that silently never matches an offset.
    #[test]
    #[should_panic(expected = "layout corruption")]
    fn adjust_bonds_panics_on_out_of_range_layout_index() {
        let bonds = vec![(Vec3::ZERO, 7u32)];
        let residue_indices = vec![0u32, 1, 2];
        let offset_map: HashMap<u32, Vec3> = HashMap::from([(0u32, Vec3::X)]);
        let _ = adjust_bonds_for_sheet(&bonds, &residue_indices, &offset_map);
    }

    /// Sidechain adjustment pairs each position with its residue index by
    /// position (parallel arrays), applying the matching offset.
    #[test]
    fn adjust_sidechains_applies_offsets_by_parallel_index() {
        let positions = vec![Vec3::ZERO, Vec3::ONE, Vec3::splat(2.0)];
        let residue_indices = vec![10u32, 11, 12];
        let offset_map: HashMap<u32, Vec3> = HashMap::from([(11u32, Vec3::X)]);
        let out = adjust_sidechains_for_sheet(
            &positions,
            &residue_indices,
            &offset_map,
        );
        assert_eq!(out[0], Vec3::ZERO);
        assert_eq!(out[1], Vec3::ONE + Vec3::X);
        assert_eq!(out[2], Vec3::splat(2.0));
    }
}
