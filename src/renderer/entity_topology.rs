//! Render-ready per-entity contract.
//!
//! [`EntityTopology`] is the self-sufficient snapshot the mesh worker
//! and main-thread render consumers read from. It's Arc-shared with
//! the background worker so mesh-gen is a pure function of
//! `(&EntityTopology, &[Vec3])` -- neither `&Assembly` nor
//! `&MoleculeEntity` appear in any render-path signature.
//!
//! [`SidechainLayout`] and [`NucleotideRingLayout`] are entity-internal
//! geometry descriptions consumed by the sidechain and NA renderers.
//!
//! The `derive_topology` factory that produces an `EntityTopology`
//! from a `MoleculeEntity` lives on the engine side in
//! [`crate::engine::entity_view`] because deriving it is engine-side
//! sync work; this module only defines the contract.

use std::ops::Range;

use glam::Vec3;
use molex::{CovalentBond, Element, MoleculeType, NucleotideRing, SSType};
use rustc_hash::FxHashMap;

// ---------------------------------------------------------------------------
// EntityTopology
// ---------------------------------------------------------------------------

/// Per-segment protein backbone atom indices, struct-of-arrays form.
///
/// Each inner vec is parallel and residue-stride: index `i` of every
/// field refers to the same residue. `n`/`ca`/`c`/`o` are
/// entity-local atom indices that can be resolved against the
/// entity's `positions` slice via [`ProteinBackboneIndices::resolve`].
///
/// Replaces the prior interleaved `[N, CA, C, ...]` `Vec<usize>` shape
/// so the carbonyl-O atom is a first-class member of the backbone
/// (load-bearing for the sheet peptide-plane normal: PyMOL / Mol* /
/// ChimeraX / rosetta-interactive all use O directly or indirectly).
#[derive(Clone, Default)]
pub(crate) struct ProteinBackboneIndices {
    pub(crate) n: Vec<usize>,
    pub(crate) ca: Vec<usize>,
    pub(crate) c: Vec<usize>,
    pub(crate) o: Vec<usize>,
}

impl ProteinBackboneIndices {
    /// Resolve every backbone atom index against `positions`, all-or-nothing.
    ///
    /// An out-of-range index means the topology layout and the position
    /// slice have desynced (a sync-state bug upstream), not a recoverable
    /// data condition. Resolving partially -- as a `filter_map` would --
    /// silently shortens one field and breaks the "all four parallel,
    /// equal length" SoA invariant every downstream consumer relies on,
    /// with zero signal. So fail loudly with the offending role/index.
    // Deliberate hard-fail: a desynced layout is unrecoverable corruption,
    // not data, so a panic with context is the correct boundary behavior.
    #[allow(clippy::panic)]
    pub(crate) fn resolve(&self, positions: &[Vec3]) -> ProteinBackboneChain {
        let resolve = |role: &str, slot: &[usize]| -> Vec<Vec3> {
            slot.iter()
                .map(|&i| match positions.get(i) {
                    Some(&p) => p,
                    None => panic!(
                        "protein backbone {role} atom index {i} out of range \
                         for {} positions ({} backbone residues in this \
                         segment): topology/position desync",
                        positions.len(),
                        slot.len(),
                    ),
                })
                .collect()
        };
        let chain = ProteinBackboneChain {
            n: resolve("N", &self.n),
            ca: resolve("CA", &self.ca),
            c: resolve("C", &self.c),
            o: resolve("O", &self.o),
        };
        debug_assert!(
            chain.n.len() == chain.ca.len()
                && chain.c.len() == chain.ca.len()
                && chain.o.len() == chain.ca.len(),
            "ProteinBackboneChain SoA invariant violated: n={}, ca={}, c={}, \
             o={} must be equal length",
            chain.n.len(),
            chain.ca.len(),
            chain.c.len(),
            chain.o.len(),
        );
        chain
    }
}

/// Resolved positions for one continuous protein backbone segment.
///
/// Parallel residue-stride vectors matching [`ProteinBackboneIndices`]:
/// index `i` of [`n`](Self::n), [`ca`](Self::ca), [`c`](Self::c), and
/// [`o`](Self::o) refer to the same residue, and all four are guaranteed
/// equal length (enforced in [`ProteinBackboneIndices::resolve`]). Fields
/// are private so the only construction path is that fallible resolve.
#[derive(Clone, Default)]
pub(crate) struct ProteinBackboneChain {
    n: Vec<Vec3>,
    ca: Vec<Vec3>,
    c: Vec<Vec3>,
    o: Vec<Vec3>,
}

impl ProteinBackboneChain {
    /// Backbone amide-N positions, residue-stride.
    pub(crate) fn n(&self) -> &[Vec3] {
        &self.n
    }
    /// Backbone alpha-carbon positions, residue-stride.
    pub(crate) fn ca(&self) -> &[Vec3] {
        &self.ca
    }
    /// Backbone carbonyl-C positions, residue-stride.
    pub(crate) fn c(&self) -> &[Vec3] {
        &self.c
    }
    /// Backbone carbonyl-O positions, residue-stride.
    pub(crate) fn o(&self) -> &[Vec3] {
        &self.o
    }

    /// Number of residues in this segment. All four vecs have the same
    /// length under the SoA invariant.
    pub(crate) fn residue_count(&self) -> usize {
        self.ca.len()
    }
}

/// Resolved P-atom positions for one continuous nucleic-acid backbone
/// chain (stride 1: `[P0, P1, ...]`).
///
/// The NA analogue of [`ProteinBackboneChain`]: a newtype so chain lists
/// can't be confused with arbitrary nested point vectors. The field is
/// private -- the only construction path is
/// [`EntityTopology::na_backbone_chain_positions`].
#[derive(Clone, Default)]
pub(crate) struct NaBackboneChain {
    p: Vec<Vec3>,
}

impl NaBackboneChain {
    /// Backbone phosphorus positions, residue-stride.
    pub(crate) fn p(&self) -> &[Vec3] {
        &self.p
    }
}

/// Self-sufficient render-ready view of a single entity.
///
/// Duplicates the structural metadata the renderer needs (atom elements,
/// bond list, residue ranges) at sync time so the render path is a pure
/// function of `(&EntityTopology, &[Vec3])` for per-entity mesh-gen.
#[derive(Clone)]
pub(crate) struct EntityTopology {
    /// Molecule type of the source entity, so the render dispatcher can
    /// pick the right mesh-gen path without looking at the `Assembly`.
    pub(crate) molecule_type: MoleculeType,

    /// Protein backbone atom indices, one entry per continuous backbone
    /// segment. Each `ProteinBackboneIndices` is a struct-of-arrays of
    /// `[N, CA, C, O]` indices (residue-stride; each inner vec parallel
    /// and equal-length). Empty for non-protein entities.
    pub(crate) protein_backbone_layout: Vec<ProteinBackboneIndices>,

    /// Nucleic acid P-atom indices, one inner vec per continuous chain
    /// segment (stride 1: `[P0, P1, ...]`). Empty for non-NA entities.
    pub(crate) na_backbone_chain_layout: Vec<Vec<usize>>,

    /// Sidechain atom indices and bond topology for ball-and-stick /
    /// sidechain-capsule rendering. Empty for non-protein entities.
    pub(crate) sidechain_layout: SidechainLayout,

    /// Nucleotide ring atom-index layout per residue, for DNA/RNA
    /// rendering. Empty for non-NA entities.
    pub(crate) ring_topology: Vec<NucleotideRingLayout>,

    /// Per-NA-residue backbone base color, one entry per residue in
    /// `na.residues` order (the same order
    /// [`na_backbone_chain_positions`](Self::na_backbone_chain_positions)
    /// walks). Recognized bases get their NDB color; unrecognized /
    /// modified bases get the default NA color sentinel so the slice
    /// stays residue-parallel with the P-atom stream. Empty for non-NA
    /// entities.
    pub(crate) na_residue_base_colors: Vec<[f32; 3]>,

    /// Per-NA-residue `(from_atom_index, to_atom_index)` for the Mol*
    /// ribbon direction vector `pos(to) - pos(from)` (DNA `C3'->C1'`,
    /// RNA `C4'->C3'`), residue-parallel with the P-atom stream. Empty
    /// for non-NA entities.
    pub(crate) na_guide_atom_indices: Vec<(Option<u32>, Option<u32>)>,

    /// Per-residue secondary structure from
    /// [`molex::Assembly::ss_types`]. Empty for non-protein entities.
    pub(crate) ss_types: Vec<SSType>,

    /// Element of each atom, in entity-local index order.
    pub(crate) atom_elements: Vec<Element>,
    /// Which residue each atom belongs to (index into
    /// [`residue_atom_ranges`](Self::residue_atom_ranges)).
    pub(crate) atom_residue_index: Vec<u32>,
    /// 3-byte residue name (`b"ALA"`, `b"GLY"`, ...) per residue.
    pub(crate) residue_names: Vec<[u8; 3]>,
    /// Atom-index range per residue, in entity-local indices.
    pub(crate) residue_atom_ranges: Vec<Range<u32>>,
    /// Every intra-entity covalent bond. Endpoints use
    /// [`AtomId`](molex::AtomId) so the renderer can map back to
    /// positions via the owning entity.
    pub(crate) bonds: Vec<CovalentBond>,
}

impl EntityTopology {
    /// Whether this entity renders through the protein backbone path.
    #[must_use]
    pub(crate) fn is_protein(&self) -> bool {
        self.molecule_type == MoleculeType::Protein
    }

    /// Whether this entity renders through the nucleic acid path.
    #[must_use]
    pub(crate) fn is_nucleic_acid(&self) -> bool {
        matches!(self.molecule_type, MoleculeType::DNA | MoleculeType::RNA)
    }

    /// Resolve [`protein_backbone_layout`](Self::protein_backbone_layout)
    /// into per-segment SoA backbone positions. Empty for non-protein
    /// entities.
    #[must_use]
    pub(crate) fn protein_backbone_chains(
        &self,
        positions: &[Vec3],
    ) -> Vec<ProteinBackboneChain> {
        self.protein_backbone_layout
            .iter()
            .map(|seg| seg.resolve(positions))
            .collect()
    }

    /// Resolve [`na_backbone_chain_layout`](Self::na_backbone_chain_layout)
    /// into per-chain P-atom positions (stride 1). Empty for non-NA
    /// entities.
    ///
    /// An out-of-range P index means the topology layout and the position
    /// slice have desynced (a sync-state bug upstream), not a recoverable
    /// data condition. Resolving partially -- as a `filter_map` would --
    /// silently shortens the chain, advancing `process_na_chains`'
    /// residue/offset bookkeeping by the short count and desyncing every
    /// later chain's base colors with zero signal. So fail loudly with
    /// the offending index, mirroring
    /// [`ProteinBackboneIndices::resolve`].
    // Deliberate hard-fail: a desynced layout is unrecoverable corruption,
    // not data, so a panic with context is the correct boundary behavior.
    #[allow(clippy::panic)]
    #[must_use]
    pub(crate) fn na_backbone_chain_positions(
        &self,
        positions: &[Vec3],
    ) -> Vec<NaBackboneChain> {
        self.na_backbone_chain_layout
            .iter()
            .map(|chain| NaBackboneChain {
                p: chain
                    .iter()
                    .map(|&idx| match positions.get(idx) {
                        Some(&p) => p,
                        None => panic!(
                            "NA backbone P atom index {idx} out of range for \
                             {} positions ({} P atoms in this chain): \
                             topology/position desync",
                            positions.len(),
                            chain.len(),
                        ),
                    })
                    .collect(),
            })
            .collect()
    }

    /// Per-NA-backbone-chain RMF seed normal: the first resolvable base
    /// ring's Newell plane normal for each chain, in
    /// [`na_backbone_chain_layout`](Self::na_backbone_chain_layout) order.
    ///
    /// This is the nucleic-acid analogue of the protein peptide-plane
    /// seed: it fixes the ribbon roll by base-pair geometry rather than
    /// by which world octant the structure happens to load in. `None`
    /// for a chain whose rings all fail to resolve -- the RMF solver
    /// then falls back to a world axis for that chain only.
    #[must_use]
    pub(crate) fn na_chain_seed_normals(
        &self,
        positions: &[Vec3],
    ) -> Vec<Option<Vec3>> {
        let n_chains = self.na_backbone_chain_layout.len();
        let mut seeds: Vec<Option<Vec3>> = vec![None; n_chains];
        for layout in &self.ring_topology {
            let ci = layout.chain_idx as usize;
            if ci >= n_chains || seeds[ci].is_some() {
                continue;
            }
            if let Some(hex) = resolve_ring(&layout.hex_ring, positions) {
                let normal = crate::util::geom::newell_normal(&hex);
                if normal != Vec3::ZERO {
                    seeds[ci] = Some(normal);
                }
            }
        }
        seeds
    }

    /// Per-NA-residue Mol* ribbon direction vector `pos(to) - pos(from)`
    /// (DNA `C3'->C1'`, RNA `C4'->C3'`), residue-parallel with the
    /// P-atom stream.
    ///
    /// Raw (not normalized, not tangent-projected): the mesh path does
    /// the Mol* neighbour smoothing, tangent-orthogonalization, the
    /// normal/binormal swap+negate, and RMF propagation. A zero entry
    /// means an endpoint atom was missing (the ribbon keeps its RMF
    /// normal there).
    #[must_use]
    pub(crate) fn na_residue_guide_dirs(
        &self,
        positions: &[Vec3],
    ) -> Vec<Vec3> {
        self.na_guide_atom_indices
            .iter()
            .map(|&(from_idx, to_idx)| {
                match (
                    from_idx.and_then(|i| positions.get(i as usize)),
                    to_idx.and_then(|i| positions.get(i as usize)),
                ) {
                    (Some(&from), Some(&to)) => to - from,
                    _ => Vec3::ZERO,
                }
            })
            .collect()
    }

    /// Resolve the nucleotide ring layouts into renderer-facing
    /// [`ResolvedRing`] instances, pulling atom positions from the
    /// provided slice.
    #[must_use]
    pub(crate) fn resolve_rings(
        &self,
        positions: &[Vec3],
    ) -> Vec<ResolvedRing> {
        self.ring_topology
            .iter()
            .filter_map(|layout| layout.resolve(positions))
            .collect()
    }
}

// ---------------------------------------------------------------------------
// SidechainLayout
// ---------------------------------------------------------------------------

/// Sidechain atom-index layout for a single entity.
///
/// All indices are entity-local (into the entity's atom positions slice
/// in [`EntityPositions`](crate::engine::positions::EntityPositions)).
#[derive(Clone)]
pub(crate) struct SidechainLayout {
    /// Atom index (entity-local) of each sidechain atom.
    pub(crate) atom_indices: Vec<u32>,
    /// Residue index (entity-local) of each sidechain atom. Parallel to
    /// [`atom_indices`](Self::atom_indices).
    pub(crate) residue_indices: Vec<u32>,
    /// Hydrophobicity flag per sidechain atom. Parallel to
    /// [`atom_indices`](Self::atom_indices).
    pub(crate) hydrophobicity: Vec<bool>,
    /// Intra-sidechain bonds as `(a, b)` indices into
    /// [`atom_indices`](Self::atom_indices).
    pub(crate) bonds: Vec<(u32, u32)>,
    /// Backbone -> sidechain bonds as `(ca_atom_idx, cb_layout_idx)` where
    /// `ca_atom_idx` is an entity-local atom index of CA and
    /// `cb_layout_idx` is the index into
    /// [`atom_indices`](Self::atom_indices) of the CB that CA connects
    /// to.
    pub(crate) backbone_bonds: Vec<(u32, u32)>,
    /// `(residue_idx, atom_name) -> entity-local atom index` for O(1)
    /// constraint-resolution lookup. Built at topology-derivation time
    /// from the parallel `atom_indices` + `residue_indices` vecs plus
    /// the source atom names.
    pub(crate) atom_lookup: FxHashMap<u32, FxHashMap<Box<str>, u32>>,
}

impl SidechainLayout {
    /// Empty layout (no sidechain atoms).
    #[must_use]
    pub(crate) fn empty() -> Self {
        Self {
            atom_indices: Vec::new(),
            residue_indices: Vec::new(),
            hydrophobicity: Vec::new(),
            bonds: Vec::new(),
            backbone_bonds: Vec::new(),
            atom_lookup: FxHashMap::default(),
        }
    }

    /// Look up an entity-local atom index by `(residue_idx, atom_name)`.
    /// O(1).
    #[must_use]
    pub(crate) fn atom_index(
        &self,
        residue: u32,
        atom_name: &str,
    ) -> Option<u32> {
        self.atom_lookup
            .get(&residue)
            .and_then(|m| m.get(atom_name).copied())
    }
}

// ---------------------------------------------------------------------------
// NucleotideRingLayout
// ---------------------------------------------------------------------------

/// Resolve a ring's entity-local atom indices to world positions,
/// all-or-nothing.
///
/// Returns `None` if *any* index is out of range, so a partially
/// resolvable ring is omitted cleanly rather than rendered as a
/// wrong-shape, wrong-normal polygon (the hex and pent rings shared
/// divergent failure policies before this was factored out -- hex was
/// all-or-nothing, pent was a `filter_map` that silently shortened a
/// purine into a 4-gon that passed the polygon vertex-count gate).
#[must_use]
pub(crate) fn resolve_ring(
    indices: &[u32],
    positions: &[Vec3],
) -> Option<Vec<Vec3>> {
    indices
        .iter()
        .map(|&idx| positions.get(idx as usize).copied())
        .collect()
}

/// A resolved nucleotide ring plus the geometry needed to anchor its
/// stem to the rendered backbone.
///
/// `owning_p` is this base's own residue's P-atom position, which lies
/// exactly on the rendered dual-Hermite ribbon (the ribbon passes
/// through every P control point). Anchoring the stem there fixes it to
/// the *drawn* backbone of the *owning* strand, replacing the prior
/// global nearest-sample search over a second, undrawn Catmull-Rom
/// spline flattened across every strand.
pub(crate) struct ResolvedRing {
    /// The base ring geometry (hex + optional pent + C1').
    pub(crate) ring: NucleotideRing,
    /// Owning residue's P-atom world position, or `None` if its index
    /// did not resolve.
    pub(crate) owning_p: Option<Vec3>,
    /// Which [`na_backbone_chain_layout`](EntityTopology::na_backbone_chain_layout)
    /// chain this base belongs to (for per-strand normal coherence).
    pub(crate) chain_idx: u32,
}

/// Atom-index layout for a single nucleotide's ring(s).
///
/// All indices are entity-local.
#[derive(Clone)]
pub(crate) struct NucleotideRingLayout {
    /// Six atom indices for the hexagonal ring (N1, C2, N3, C4, C5, C6).
    pub(crate) hex_ring: [u32; 6],
    /// Optional five atom indices for the pentagonal ring on purines
    /// (C4, C5, N7, C8, N9). `None` for pyrimidines.
    pub(crate) pent_ring: Option<[u32; 5]>,
    /// Atom index of C1' (sugar anchor; informational only since the
    /// stem now anchors to [`p_index`](Self::p_index)).
    pub(crate) c1_prime: Option<u32>,
    /// Entity-local atom index of this residue's P atom (canonical
    /// `residue.atom_range.start`), the stem's backbone anchor.
    pub(crate) p_index: u32,
    /// Which NA backbone chain (segment) this residue belongs to.
    pub(crate) chain_idx: u32,
    /// NDB base color.
    pub(crate) color: [f32; 3],
}

impl NucleotideRingLayout {
    /// Resolve atom indices to world positions using the provided slice.
    /// Returns `None` if any hex- or pent-ring atom is out of range.
    #[must_use]
    pub(crate) fn resolve(&self, positions: &[Vec3]) -> Option<ResolvedRing> {
        let hex_ring = resolve_ring(&self.hex_ring, positions)?;
        let pent_ring = match self.pent_ring {
            Some(pent) => resolve_ring(&pent, positions)?,
            None => Vec::new(),
        };
        let c1_prime = self
            .c1_prime
            .and_then(|idx| positions.get(idx as usize).copied());
        let owning_p = positions.get(self.p_index as usize).copied();
        Some(ResolvedRing {
            ring: NucleotideRing {
                hex_ring,
                pent_ring,
                c1_prime,
                color: self.color,
            },
            owning_p,
            chain_idx: self.chain_idx,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn indices(
        n: &[usize],
        ca: &[usize],
        c: &[usize],
        o: &[usize],
    ) -> ProteinBackboneIndices {
        ProteinBackboneIndices {
            n: n.to_vec(),
            ca: ca.to_vec(),
            c: c.to_vec(),
            o: o.to_vec(),
        }
    }

    #[test]
    fn resolve_yields_equal_length_parallel_vecs() {
        let positions: Vec<Vec3> =
            (0..8).map(|i| Vec3::splat(i as f32)).collect();
        let layout = indices(&[0, 4], &[1, 5], &[2, 6], &[3, 7]);
        let chain = layout.resolve(&positions);
        assert_eq!(chain.residue_count(), 2);
        assert_eq!(chain.n().len(), 2);
        assert_eq!(chain.ca().len(), 2);
        assert_eq!(chain.c().len(), 2);
        assert_eq!(chain.o().len(), 2);
    }

    /// An out-of-range index must abort resolve loudly, not silently
    /// produce a short vec that desyncs the SoA invariant.
    #[test]
    #[should_panic(expected = "topology/position desync")]
    fn resolve_panics_on_out_of_range_index() {
        let positions: Vec<Vec3> =
            (0..4).map(|i| Vec3::splat(i as f32)).collect();
        // `o` references atom index 99, far past the 4 positions.
        let layout = indices(&[0], &[1], &[2], &[99]);
        let _ = layout.resolve(&positions);
    }

    fn na_topology(
        na_backbone_chain_layout: Vec<Vec<usize>>,
        ring_topology: Vec<NucleotideRingLayout>,
    ) -> EntityTopology {
        EntityTopology {
            molecule_type: MoleculeType::RNA,
            protein_backbone_layout: Vec::new(),
            na_backbone_chain_layout,
            sidechain_layout: SidechainLayout::empty(),
            ring_topology,
            na_residue_base_colors: Vec::new(),
            na_guide_atom_indices: Vec::new(),
            ss_types: Vec::new(),
            atom_elements: Vec::new(),
            atom_residue_index: Vec::new(),
            residue_names: Vec::new(),
            residue_atom_ranges: Vec::new(),
            bonds: Vec::new(),
        }
    }

    /// T1-NA-A: a desynced NA P index must hard-fail with context, not
    /// silently shorten the chain (which would slide every later chain's
    /// base colors by one).
    #[test]
    #[should_panic(expected = "topology/position desync")]
    fn na_backbone_positions_panics_on_out_of_range() {
        let positions: Vec<Vec3> =
            (0..4).map(|i| Vec3::splat(i as f32)).collect();
        let topo = na_topology(vec![vec![0, 1, 99]], Vec::new());
        let _ = topo.na_backbone_chain_positions(&positions);
    }

    fn ring_layout(
        hex: [u32; 6],
        pent: Option<[u32; 5]>,
    ) -> NucleotideRingLayout {
        NucleotideRingLayout {
            hex_ring: hex,
            pent_ring: pent,
            c1_prime: None,
            p_index: 0,
            chain_idx: 0,
            color: [0.5, 0.5, 0.5],
        }
    }

    /// T0-NA-B / T4-NA-C: pent resolution is now all-or-nothing like
    /// hex. A purine missing one ring atom must drop the whole base, not
    /// render a wrong-shape 4-gon that passes the polygon vertex gate.
    #[test]
    #[allow(clippy::panic)]
    fn pent_resolution_is_all_or_nothing() {
        let positions: Vec<Vec3> =
            (0..8).map(|i| Vec3::splat(i as f32)).collect();

        // All hex + pent indices in range -> resolves with a 5-vert pent.
        let ok = ring_layout([0, 1, 2, 3, 4, 5], Some([0, 1, 2, 3, 4]));
        let Some(resolved) = ok.resolve(&positions) else {
            panic!("in-range hex+pent ring should resolve");
        };
        assert_eq!(resolved.ring.hex_ring.len(), 6);
        assert_eq!(resolved.ring.pent_ring.len(), 5);

        // One pent index out of range -> the whole ring is dropped,
        // not silently shortened to a 4-gon.
        let bad_pent = ring_layout([0, 1, 2, 3, 4, 5], Some([0, 1, 2, 3, 99]));
        assert!(bad_pent.resolve(&positions).is_none());

        // One hex index out of range -> dropped (unchanged behavior).
        let bad_hex = ring_layout([0, 1, 2, 3, 4, 99], None);
        assert!(bad_hex.resolve(&positions).is_none());
    }
}
