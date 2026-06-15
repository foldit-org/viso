use glam::Vec3;

use crate::renderer::geometry::backbone::SheetOffset;

/// Total residue count from SoA protein backbone chains.
pub(crate) fn backbone_residue_count(
    backbone_chains: &[crate::renderer::entity_topology::ProteinBackboneChain],
) -> usize {
    backbone_chains.iter().map(|c| c.ca().len()).sum()
}

/// Look up a residue's sheet offset by binary search.
///
/// `offsets` is produced in ascending `residue_idx` order (per-segment,
/// per-chain, concatenated monotonically in `mesh_concat`), so a binary
/// search replaces the per-frame `HashMap` rebuild this used to do.
///
/// Caller must validate sortedness once via [`debug_assert_offsets_sorted`]
/// before looping: this is called once per sidechain atom, so the
/// O(n) sorted-check must not live here (that is O(atoms x offsets) per
/// camera-drag frame).
fn lookup_offset(offsets: &[SheetOffset], res_idx: u32) -> Option<Vec3> {
    offsets
        .binary_search_by_key(&res_idx, |so| so.residue_idx)
        .ok()
        .map(|i| offsets[i].offset)
}

/// Sheet-flattening offset for one residue, or `None` if the residue is
/// not on a flattened beta-strand.
///
/// Same binary search the mesh-build adjustment uses (the offsets are
/// ascending by `residue_idx`), exposed so the render-position resolver
/// can re-anchor a sidechain bond endpoint onto the drawn sheet-flattened
/// stick. Caller passes an entity-local residue index and the entity's own
/// offset slice (ascending, base 0).
#[must_use]
pub(crate) fn sheet_offset_at(
    offsets: &[SheetOffset],
    residue: u32,
) -> Option<Vec3> {
    lookup_offset(offsets, residue)
}

/// One-time precondition check for [`lookup_offset`]'s binary search.
/// Call once per `offsets` slice before the per-atom loop, never inside it.
fn debug_assert_offsets_sorted(offsets: &[SheetOffset]) {
    debug_assert!(
        offsets
            .windows(2)
            .all(|w| w[0].residue_idx <= w[1].residue_idx),
        "sheet offsets must be sorted by residue_idx for binary search"
    );
}

/// Translate sidechain atom positions by sheet-flattening offsets.
pub(crate) fn adjust_sidechains_for_sheet(
    positions: &[Vec3],
    sidechain_residue_indices: &[u32],
    offsets: &[SheetOffset],
) -> Vec<Vec3> {
    if offsets.is_empty() {
        return positions.to_vec();
    }
    debug_assert_offsets_sorted(offsets);
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
        .map(|(&pos, &res_idx)| {
            lookup_offset(offsets, res_idx).map_or(pos, |o| pos + o)
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
    offsets: &[SheetOffset],
) -> Vec<(Vec3, u32)> {
    if offsets.is_empty() {
        return bonds.to_vec();
    }
    debug_assert_offsets_sorted(offsets);
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
            lookup_offset(offsets, res_idx)
                .map_or((*ca_pos, *cb_idx), |o| (*ca_pos + o, *cb_idx))
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
        let offsets = [SheetOffset {
            residue_idx: 0,
            offset: Vec3::X,
        }];
        let _ = adjust_bonds_for_sheet(&bonds, &residue_indices, &offsets);
    }

    /// Sidechain adjustment pairs each position with its residue index by
    /// position (parallel arrays), applying the matching offset.
    #[test]
    fn adjust_sidechains_applies_offsets_by_parallel_index() {
        let positions = vec![Vec3::ZERO, Vec3::ONE, Vec3::splat(2.0)];
        let residue_indices = vec![10u32, 11, 12];
        let offsets = [SheetOffset {
            residue_idx: 11,
            offset: Vec3::X,
        }];
        let out =
            adjust_sidechains_for_sheet(&positions, &residue_indices, &offsets);
        assert_eq!(out[0], Vec3::ZERO);
        assert_eq!(out[1], Vec3::ONE + Vec3::X);
        assert_eq!(out[2], Vec3::splat(2.0));
    }
}
