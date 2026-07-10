//! Per-residue color derivation for the sync pipeline: turns each visible
//! protein entity's backbone chains, SS, scores, and B-factors into the
//! flat per-residue color buffer the full-rebuild request carries.

use molex::SSType;

use super::super::scene::Scene;
use crate::options::score_color::{
    compute_per_residue_colors_styled, SchemeInputs,
};
use crate::options::DisplayOptions;
use crate::renderer::entity_topology::{EntityTopology, ProteinBackboneChain};

/// Per-residue B-factor: the max over the residue's backbone atoms. The
/// backbone atom indices come from `protein_backbone_layout`, the single
/// owner of that derivation; segments are walked in order so the result is
/// residue-indexed in ascending residue order. An index past
/// `atom_b_factors` is skipped; a residue whose four indices are all out of
/// range contributes `0.0`.
pub(super) fn residue_max_backbone_b(topology: &EntityTopology) -> Vec<f32> {
    let b = &topology.atom_b_factors;
    let mut out = Vec::with_capacity(topology.residue_atom_ranges.len());
    for seg in &topology.protein_backbone_layout {
        for i in 0..seg.ca.len() {
            let max = [seg.n[i], seg.ca[i], seg.c[i], seg.o[i]]
                .into_iter()
                .filter_map(|idx| b.get(idx).copied())
                .reduce(f32::max)
                .unwrap_or(0.0);
            out.push(max);
        }
    }
    out
}

/// Assembly-global (min, max) B-factor across every entity's atoms.
/// Returns `(0.0, 0.0)` when no atom carries one.
pub(super) fn assembly_b_range(scene: &Scene) -> (f32, f32) {
    let mut lo = f32::INFINITY;
    let mut hi = f32::NEG_INFINITY;
    for state in scene.entity_state.values() {
        for &b in &state.topology.atom_b_factors {
            if b.is_nan() {
                continue;
            }
            lo = lo.min(b);
            hi = hi.max(b);
        }
    }
    if lo.is_finite() && hi.is_finite() {
        (lo, hi)
    } else {
        (0.0, 0.0)
    }
}

pub(super) fn per_entity_colors(
    entity_index: usize,
    backbone_chains: &[ProteinBackboneChain],
    ss_types: &[SSType],
    scores: Option<&[f64]>,
    b_factors: Option<&[f32]>,
    b_range: (f32, f32),
    display: &DisplayOptions,
) -> Option<Vec<[f32; 3]>> {
    if backbone_chains.is_empty() {
        return None;
    }
    let scores_slice = [scores];
    let inputs = SchemeInputs {
        scores: &scores_slice,
        b_factors,
        b_range,
    };
    let colors = compute_per_residue_colors_styled(
        backbone_chains,
        ss_types,
        &inputs,
        &display.backbone_color_scheme(),
        &display.backbone_palette(),
        entity_index,
        display.overrides.provisional.unwrap_or(false),
    );
    if colors.is_empty() {
        None
    } else {
        Some(colors)
    }
}

#[cfg(test)]
mod tests {
    use molex::MoleculeType;

    use super::residue_max_backbone_b;
    use crate::renderer::entity_topology::{
        EntityTopology, ProteinBackboneIndices, SidechainLayout,
    };

    fn protein_topology(
        segments: Vec<ProteinBackboneIndices>,
        atom_b_factors: Vec<f32>,
    ) -> EntityTopology {
        let residue_count: usize = segments.iter().map(|s| s.ca.len()).sum();
        EntityTopology {
            molecule_type: MoleculeType::Protein,
            protein_backbone_layout: segments,
            na_backbone_chain_layout: Vec::new(),
            sidechain_layout: SidechainLayout::empty(),
            ring_topology: Vec::new(),
            na_residue_base_colors: Vec::new(),
            na_guide_atom_indices: Vec::new(),
            ss_types: Vec::new(),
            atom_elements: Vec::new(),
            atom_b_factors,
            atom_residue_index: Vec::new(),
            residue_names: Vec::new(),
            residue_atom_ranges: vec![0..0; residue_count],
            bonds: Vec::new(),
        }
    }

    #[test]
    fn max_over_backbone_atoms_in_segment_order() {
        // Two segments, four atoms per residue: B rises with atom index so
        // the O atom (highest index) is the per-residue max.
        let b: Vec<f32> = (0..12).map(|i| i as f32).collect();
        let seg0 = ProteinBackboneIndices {
            n: vec![0, 4],
            ca: vec![1, 5],
            c: vec![2, 6],
            o: vec![3, 7],
        };
        let seg1 = ProteinBackboneIndices {
            n: vec![8],
            ca: vec![9],
            c: vec![10],
            o: vec![11],
        };
        let topo = protein_topology(vec![seg0, seg1], b);
        assert_eq!(residue_max_backbone_b(&topo), vec![3.0, 7.0, 11.0]);
    }

    #[test]
    fn out_of_range_indices_yield_zero_without_panic() {
        // One residue whose four backbone indices all exceed
        // `atom_b_factors.len()`: contributes 0.0 rather than panicking.
        let seg = ProteinBackboneIndices {
            n: vec![50],
            ca: vec![51],
            c: vec![52],
            o: vec![53],
        };
        let topo = protein_topology(vec![seg], vec![1.0, 2.0, 3.0, 4.0]);
        assert_eq!(residue_max_backbone_b(&topo), vec![0.0]);
    }

    #[test]
    fn partially_out_of_range_takes_max_of_in_range() {
        // n/ca in range, c/o past the end: max over the resolvable pair.
        let seg = ProteinBackboneIndices {
            n: vec![0],
            ca: vec![1],
            c: vec![99],
            o: vec![100],
        };
        let topo = protein_topology(vec![seg], vec![5.0, 8.0]);
        assert_eq!(residue_max_backbone_b(&topo), vec![8.0]);
    }
}
