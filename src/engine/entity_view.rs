//! Engine-main-thread per-entity state.
//!
//! - [`EntityView`] holds the engine's per-entity render state (drawing mode,
//!   SS override, mesh-cache version) plus an `Arc` handle to the render-ready
//!   [`EntityTopology`] shared with the background mesh worker.
//! - [`derive_topology`] is the sync-time factory that derives the renderer
//!   contract from a `MoleculeEntity`. Defined here (not in the renderer
//!   module) because derivation is an engine-side concern; the renderer only
//!   defines the shape it wants.
//! - [`RibbonBackbone`] is a per-sync cache of spline-projected backbone anchor
//!   positions used by the bond resolver to attach H-bond capsules to the
//!   rendered ribbon in Cartoon mode.

use std::ops::Range;
use std::sync::Arc;

use glam::Vec3;
use molex::{Element, MoleculeEntity, MoleculeType, SSType};
use rustc_hash::FxHashMap;

use crate::options::DrawingMode;
use crate::renderer::entity_topology::{
    EntityTopology, NucleotideRingLayout, SidechainLayout,
};
use crate::renderer::geometry::backbone::curve::project_backbone_atoms;
use crate::renderer::geometry::backbone::SheetOffset;
use crate::renderer::geometry::nucleic_acid::NA_DEFAULT_COLOR;

// EntityView

/// Per-entity state held by [`VisoEngine`](super::VisoEngine).
///
/// Holds the drawing mode chosen for this entity, an optional SS
/// override, the Arc-shared (immutable) [`EntityTopology`], a cached
/// per-sync per-residue color vector used by Cartoon color uploads,
/// and a monotonically increasing `mesh_version`.
pub(crate) struct EntityView {
    /// Drawing mode for this entity (Cartoon / Stick / `BallAndStick`).
    pub(crate) drawing_mode: DrawingMode,
    /// Optional secondary-structure override. When present, takes
    /// priority over [`EntityTopology::ss_types`] at render time.
    pub(crate) ss_override: Option<Vec<SSType>>,
    /// Rederived render-ready view of this entity. Arc-wrapped so the
    /// background mesh worker can hold a handle without cloning the
    /// underlying buffers. Immutable after derive.
    pub(crate) topology: Arc<EntityTopology>,
    /// Per-residue Cartoon-mode vertex colors, rederived each sync.
    /// Cached here so main-thread color uploads can concatenate across
    /// entities without recomputing.
    pub(crate) per_residue_colors: Option<Vec<[f32; 3]>>,
    /// Per-residue sheet-flattening offsets for this entity's beta-strand
    /// residues, entity-local residue index, ascending. Lifted off the
    /// background mesh build (the same offsets that shift this entity's
    /// sidechain stick mesh in Cartoon mode) so structural-bond endpoint
    /// resolution can re-anchor sidechain atoms onto the drawn flattened
    /// sticks. Refreshed whenever the prepared mesh is applied; empty when
    /// no strand residues were flattened (or before the first mesh lands).
    pub(crate) sheet_offsets: Vec<SheetOffset>,
    /// Bumped whenever this entity's geometry needs to be regenerated.
    pub(crate) mesh_version: u64,
}

// RibbonBackbone -- per-sync cache for Cartoon-mode H-bond anchoring

/// Per-residue spline-projected backbone positions for Cartoon-mode
/// H-bond anchoring.
///
/// The cartoon ribbon is a smooth spline through a protein's `[N, CA, C]`
/// control points; the rendered curve doesn't pass exactly through the
/// raw atoms. [`project_backbone_atoms`] derives where N and C would sit
/// on the curve using standard peptide-bond fractions, producing the
/// anchor positions an H-bond capsule should attach to so it visually
/// connects to the ribbon rather than to atoms floating off it.
pub(crate) struct RibbonBackbone {
    /// Ribbon-projected N position per residue (donor anchoring).
    pub(crate) per_residue_n: Vec<Vec3>,
    /// Ribbon-projected C position per residue (acceptor anchoring).
    pub(crate) per_residue_c: Vec<Vec3>,
}

impl RibbonBackbone {
    /// Project per-residue N and C onto the rendered cartoon ribbon.
    ///
    /// Returns `None` for non-protein topologies and for inputs too
    /// short to support a spline projection -- callers fall back to raw
    /// atom positions in that case.
    #[must_use]
    pub(crate) fn project(
        topology: &EntityTopology,
        positions: &[Vec3],
    ) -> Option<Self> {
        if !topology.is_protein() {
            return None;
        }
        let chains = topology.protein_backbone_chains(positions);
        if chains.is_empty() {
            return None;
        }
        let (per_residue_n, per_residue_c) = project_backbone_atoms(&chains);
        if per_residue_n.is_empty() && per_residue_c.is_empty() {
            return None;
        }
        Some(Self {
            per_residue_n,
            per_residue_c,
        })
    }

    /// Residue-indexed ribbon-projected N position (donor anchor).
    #[must_use]
    pub(crate) fn n_at(&self, residue: u32) -> Option<Vec3> {
        self.per_residue_n.get(residue as usize).copied()
    }

    /// Residue-indexed ribbon-projected C position (acceptor anchor).
    #[must_use]
    pub(crate) fn c_at(&self, residue: u32) -> Option<Vec3> {
        self.per_residue_c.get(residue as usize).copied()
    }
}

// derive_topology -- engine-side derivation factory

/// Rederive the render-ready [`EntityTopology`] view of a single entity.
///
/// `ss` is the per-residue secondary structure for the entity, as
/// produced by [`Assembly::ss_types`](molex::Assembly::ss_types). For
/// non-protein entities it should be an empty slice.
#[must_use]
pub(crate) fn derive_topology(
    entity: &MoleculeEntity,
    ss: &[SSType],
) -> EntityTopology {
    let molecule_type = entity.molecule_type();
    match entity {
        MoleculeEntity::Protein(protein) => {
            let protein_backbone_layout = protein_backbone_indices(protein);
            let sidechain_layout = protein_sidechain_layout(protein);
            let (residue_names, residue_atom_ranges, atom_residue_index) =
                residue_tables(
                    protein
                        .residues
                        .iter()
                        .map(|r| (r.name, r.atom_range.clone())),
                    protein.atoms.len(),
                );
            EntityTopology {
                molecule_type,
                protein_backbone_layout,
                na_backbone_chain_layout: Vec::new(),
                sidechain_layout,
                ring_topology: Vec::new(),
                na_residue_base_colors: Vec::new(),
                na_guide_atom_indices: Vec::new(),
                ss_types: ss.to_vec(),
                atom_elements: atom_elements(&protein.atoms),
                atom_residue_index,
                residue_names,
                residue_atom_ranges,
                bonds: protein.bonds.clone(),
            }
        }
        MoleculeEntity::NucleicAcid(na) => {
            let (residue_names, residue_atom_ranges, atom_residue_index) =
                residue_tables(
                    na.residues.iter().map(|r| (r.name, r.atom_range.clone())),
                    na.atoms.len(),
                );
            EntityTopology {
                molecule_type,
                protein_backbone_layout: Vec::new(),
                na_backbone_chain_layout: na_backbone_chain_layout(na),
                sidechain_layout: SidechainLayout::empty(),
                ring_topology: na_ring_topology(na),
                na_residue_base_colors: na_residue_base_colors(na),
                na_guide_atom_indices: na_guide_atom_indices(na, molecule_type),
                ss_types: Vec::new(),
                atom_elements: atom_elements(&na.atoms),
                atom_residue_index,
                residue_names,
                residue_atom_ranges,
                bonds: na.bonds.clone(),
            }
        }
        MoleculeEntity::SmallMolecule(sm) => EntityTopology {
            molecule_type,
            protein_backbone_layout: Vec::new(),
            na_backbone_chain_layout: Vec::new(),
            sidechain_layout: SidechainLayout::empty(),
            ring_topology: Vec::new(),
            na_residue_base_colors: Vec::new(),
            na_guide_atom_indices: Vec::new(),
            ss_types: Vec::new(),
            atom_elements: atom_elements(&sm.atoms),
            atom_residue_index: vec![0; sm.atoms.len()],
            residue_names: vec![sm.residue_name],
            residue_atom_ranges: std::iter::once(0..sm.atoms.len() as u32)
                .collect(),
            bonds: sm.bonds.clone(),
        },
        MoleculeEntity::Bulk(bulk) => EntityTopology {
            molecule_type,
            protein_backbone_layout: Vec::new(),
            na_backbone_chain_layout: Vec::new(),
            sidechain_layout: SidechainLayout::empty(),
            ring_topology: Vec::new(),
            na_residue_base_colors: Vec::new(),
            na_guide_atom_indices: Vec::new(),
            ss_types: Vec::new(),
            atom_elements: atom_elements(&bulk.atoms),
            atom_residue_index: Vec::new(),
            residue_names: Vec::new(),
            residue_atom_ranges: Vec::new(),
            bonds: Vec::new(),
        },
    }
}

// Builder helpers -- private derivation used only by derive_topology

fn atom_elements(atoms: &[molex::Atom]) -> Vec<Element> {
    atoms.iter().map(|a| a.element).collect()
}

/// Build `(residue_names, residue_atom_ranges, atom_residue_index)` from
/// an iterator of `(name, atom_range)` residue metadata.
///
/// The atom-residue index is a flat `Vec<u32>` of length `atom_count`
/// where entry `i` is the residue index containing atom `i`. Atoms not
/// covered by any residue range (e.g. canonicalized-but-dropped residues
/// whose atoms remain in `atoms` but are unreferenced) default to `0`.
fn residue_tables<I>(
    residues: I,
    atom_count: usize,
) -> (Vec<[u8; 3]>, Vec<Range<u32>>, Vec<u32>)
where
    I: IntoIterator<Item = ([u8; 3], Range<usize>)>,
{
    let mut names = Vec::new();
    let mut ranges = Vec::new();
    let mut atom_residue_index = vec![0u32; atom_count];
    for (res_idx, (name, range)) in residues.into_iter().enumerate() {
        names.push(name);
        let start = range.start as u32;
        let end = range.end as u32;
        ranges.push(start..end);
        for slot in &mut atom_residue_index[range] {
            *slot = res_idx as u32;
        }
    }
    (names, ranges, atom_residue_index)
}

/// Build per-segment SoA backbone-atom indices for a protein entity.
/// `ProteinEntity::new` enforces canonical atom ordering -- N, CA, C, O
/// as the first four atoms of every kept residue -- so each role's
/// index is a fixed offset from the residue's `atom_range.start`.
fn protein_backbone_indices(
    protein: &molex::entity::molecule::protein::ProteinEntity,
) -> Vec<crate::renderer::entity_topology::ProteinBackboneIndices> {
    use molex::entity::molecule::traits::Polymer;

    use crate::renderer::entity_topology::ProteinBackboneIndices;
    let n_segments = protein.segment_count();
    (0..n_segments)
        .map(|seg_idx| {
            let range = protein.segment_range(seg_idx);
            let len = range.len();
            let mut indices = ProteinBackboneIndices {
                n: Vec::with_capacity(len),
                ca: Vec::with_capacity(len),
                c: Vec::with_capacity(len),
                o: Vec::with_capacity(len),
            };
            for residue in &protein.residues[range] {
                let base = residue.atom_range.start;
                indices.n.push(base);
                indices.ca.push(base + 1);
                indices.c.push(base + 2);
                indices.o.push(base + 3);
            }
            indices
        })
        .collect()
}

/// Flatten protein sidechain atoms with their metadata and topology.
///
/// Walks every kept residue's sidechain heavy atoms (canonical positions
/// `[4..]`, excluding hydrogens). Collects entity-local atom indices,
/// residue indices, names, and hydrophobicity; rebuilds intra-sidechain
/// and backbone->sidechain bond pairs from the entity's bond list.
fn protein_sidechain_layout(
    protein: &molex::entity::molecule::protein::ProteinEntity,
) -> SidechainLayout {
    use molex::chemistry::amino_acids::AminoAcid;

    let mut atom_indices: Vec<u32> = Vec::new();
    let mut residue_indices: Vec<u32> = Vec::new();
    let mut hydrophobicity: Vec<bool> = Vec::new();
    // Map entity-local atom index -> layout index, so we can resolve
    // bond endpoints back to positions within this layout.
    let mut atom_to_layout: FxHashMap<u32, u32> = FxHashMap::default();
    // (residue_idx, atom_name) -> entity-local atom index. Populated
    // inline with the layout walk so constraint resolution can do
    // O(1) atom-name lookups.
    let mut atom_lookup: FxHashMap<u32, FxHashMap<Box<str>, u32>> =
        FxHashMap::default();

    for (res_idx, residue) in protein.residues.iter().enumerate() {
        let start = residue.atom_range.start;
        let end = residue.atom_range.end;
        if end.saturating_sub(start) < 4 {
            continue;
        }
        let is_hydrophobic = AminoAcid::from_code(residue.name)
            .is_some_and(AminoAcid::is_hydrophobic);
        let res_idx_u32 = res_idx as u32;
        for atom_idx in (start + 4)..end {
            let atom = &protein.atoms[atom_idx];
            if atom.element == Element::H {
                continue;
            }
            let layout_idx = atom_indices.len() as u32;
            atom_indices.push(atom_idx as u32);
            residue_indices.push(res_idx_u32);
            hydrophobicity.push(is_hydrophobic);
            let _ = atom_to_layout.insert(atom_idx as u32, layout_idx);
            let name = atom_name_string(atom.name).into_boxed_str();
            let _ = atom_lookup
                .entry(res_idx_u32)
                .or_default()
                .insert(name, atom_idx as u32);
        }
    }

    let mut bonds: Vec<(u32, u32)> = Vec::new();
    let mut backbone_bonds: Vec<(u32, u32)> = Vec::new();
    for bond in &protein.bonds {
        let a_local = bond.a.index;
        let b_local = bond.b.index;
        let a_side = atom_to_layout.get(&a_local).copied();
        let b_side = atom_to_layout.get(&b_local).copied();
        match (a_side, b_side) {
            (Some(a), Some(b)) => bonds.push((a, b)),
            (Some(side), None) => {
                if let Some(ca) = backbone_anchor(protein, b_local) {
                    backbone_bonds.push((ca, side));
                }
            }
            (None, Some(side)) => {
                if let Some(ca) = backbone_anchor(protein, a_local) {
                    backbone_bonds.push((ca, side));
                }
            }
            (None, None) => {}
        }
    }

    SidechainLayout {
        atom_indices,
        residue_indices,
        hydrophobicity,
        bonds,
        backbone_bonds,
        atom_lookup,
    }
}

/// If `atom_local` is the CA atom of its residue (canonical offset 1),
/// return its entity-local atom index for wiring as a backbone anchor.
fn backbone_anchor(
    protein: &molex::entity::molecule::protein::ProteinEntity,
    atom_local: u32,
) -> Option<u32> {
    let idx = atom_local as usize;
    let residue = protein
        .residues
        .iter()
        .find(|r| r.atom_range.contains(&idx))?;
    if idx == residue.atom_range.start + 1 {
        Some(atom_local)
    } else {
        None
    }
}

/// Trim padding/null bytes from a 4-byte atom name and return an owned
/// `String`. Falls back to an empty string if the name is not valid
/// UTF-8.
fn atom_name_string(raw: [u8; 4]) -> String {
    std::str::from_utf8(&raw)
        .unwrap_or("")
        .trim_matches(|c: char| c == ' ' || c == '\0')
        .to_owned()
}

/// Per-segment P-atom indices for an NA entity, reusing the canonical
/// atom ordering (P is at `residue.atom_range.start` on every kept
/// residue) and `segment_breaks` (residue indices where a new segment
/// starts) populated at `NAEntity::new`.
fn na_backbone_chain_layout(
    na: &molex::entity::molecule::nucleic_acid::NAEntity,
) -> Vec<Vec<usize>> {
    if na.residues.is_empty() {
        return Vec::new();
    }
    let mut segments: Vec<Vec<usize>> = Vec::new();
    let mut current: Vec<usize> = Vec::new();
    for (res_idx, residue) in na.residues.iter().enumerate() {
        if na.segment_breaks.contains(&res_idx) && !current.is_empty() {
            segments.push(std::mem::take(&mut current));
        }
        current.push(residue.atom_range.start);
    }
    if !current.is_empty() {
        segments.push(current);
    }
    segments
}

/// Canonical hexagonal-ring atom names (all bases).
const HEX_RING_NAMES: &[&[u8]] = &[b"N1", b"C2", b"N3", b"C4", b"C5", b"C6"];
/// Canonical pentagonal-ring atom names (purines only).
const PENT_RING_NAMES: &[&[u8]] = &[b"C4", b"C5", b"N7", b"C8", b"N9"];

fn is_purine(res_name: [u8; 3]) -> bool {
    let name = std::str::from_utf8(&res_name).unwrap_or("").trim();
    matches!(
        name,
        "DA" | "DG" | "DI" | "A" | "G" | "ADE" | "GUA" | "I" | "RAD" | "RGU"
    )
}

fn ndb_base_color(res_name: [u8; 3]) -> Option<[f32; 3]> {
    let name = std::str::from_utf8(&res_name).unwrap_or("").trim();
    match name {
        "DA" | "A" | "ADE" | "RAD" => Some([0.85, 0.20, 0.20]),
        "DG" | "G" | "GUA" | "RGU" => Some([0.20, 0.80, 0.20]),
        "DC" | "C" | "CYT" | "RCY" => Some([0.90, 0.90, 0.20]),
        "DT" | "THY" => Some([0.20, 0.20, 0.85]),
        "DU" | "U" | "URA" => Some([0.20, 0.85, 0.85]),
        _ => None,
    }
}

fn trim_atom_name(raw: &[u8; 4]) -> &[u8] {
    let end = raw
        .iter()
        .position(|&b| b == 0 || b == b' ')
        .unwrap_or(raw.len());
    &raw[..end]
}

/// Per-residue chain (segment) index for an NA entity, parallel to
/// `na.residues`, using the exact same `segment_breaks` walk as
/// [`na_backbone_chain_layout`] so a ring's `chain_idx` matches the
/// chain index `process_na_chains` iterates.
fn na_residue_chain_indices(
    na: &molex::entity::molecule::nucleic_acid::NAEntity,
) -> Vec<u32> {
    let mut out = Vec::with_capacity(na.residues.len());
    let mut chain_idx: u32 = 0;
    let mut current_len: usize = 0;
    for res_idx in 0..na.residues.len() {
        if na.segment_breaks.contains(&res_idx) && current_len > 0 {
            chain_idx += 1;
            current_len = 0;
        }
        out.push(chain_idx);
        current_len += 1;
    }
    out
}

/// Residue-parallel base color for every NA residue (NDB color, or the
/// default sentinel for unrecognized/modified bases). Built per residue
/// rather than per resolvable ring so it stays aligned with the P-atom
/// stream `process_na_chains` indexes -- a `na_ring_topology`-derived
/// color slice silently shifts at any base it skips (T1-NA-C).
fn na_residue_base_colors(
    na: &molex::entity::molecule::nucleic_acid::NAEntity,
) -> Vec<[f32; 3]> {
    na.residues
        .iter()
        .map(|r| ndb_base_color(r.name).unwrap_or(NA_DEFAULT_COLOR))
        .collect()
}

/// Per-residue `(from_atom_index, to_atom_index)` for an NA entity's
/// ribbon direction vector, residue-parallel with `na.residues` (the
/// order `process_na_chains` walks).
///
/// This mirrors Mol*'s `setFromToVector`: the per-residue direction is
/// `pos(to) - pos(from)`, with the atom pair chosen by polymer type --
/// **DNA: `C3' -> C1'`**, **RNA: `C4' -> C3'`** (Mol*
/// `PolymerTypeAtomRoleId`, `directionFrom`/`directionTo`). Either slot
/// `None` means the atom is missing for that residue; the ribbon
/// solver keeps its RMF normal there.
fn na_guide_atom_indices(
    na: &molex::entity::molecule::nucleic_acid::NAEntity,
    mol_type: MoleculeType,
) -> Vec<(Option<u32>, Option<u32>)> {
    // (from_names, to_names), trying both `'` and `*` PDB conventions.
    let (from_names, to_names): (&[&[u8]], &[&[u8]]) =
        if mol_type == MoleculeType::RNA {
            (&[b"C4'", b"C4*"], &[b"C3'", b"C3*"])
        } else {
            (&[b"C3'", b"C3*"], &[b"C1'", b"C1*"])
        };
    let find = |range: Range<usize>, names: &[&[u8]]| -> Option<u32> {
        range
            .filter(|&idx| names.contains(&trim_atom_name(&na.atoms[idx].name)))
            .map(|idx| idx as u32)
            .next()
    };
    na.residues
        .iter()
        .map(|r| {
            (
                find(r.atom_range.clone(), from_names),
                find(r.atom_range.clone(), to_names),
            )
        })
        .collect()
}

/// Per-residue ring atom indices for an NA entity.
fn na_ring_topology(
    na: &molex::entity::molecule::nucleic_acid::NAEntity,
) -> Vec<NucleotideRingLayout> {
    let chain_indices = na_residue_chain_indices(na);
    let mut rings = Vec::new();
    for (res_idx, residue) in na.residues.iter().enumerate() {
        let Some(color) = ndb_base_color(residue.name) else {
            continue;
        };
        let mut name_to_idx: FxHashMap<Vec<u8>, u32> = FxHashMap::default();
        for idx in residue.atom_range.clone() {
            let trimmed = trim_atom_name(&na.atoms[idx].name);
            let _ = name_to_idx.insert(trimmed.to_vec(), idx as u32);
        }
        let hex_ring: Option<[u32; 6]> = {
            let mut out = [0u32; 6];
            let mut ok = true;
            for (slot, name) in out.iter_mut().zip(HEX_RING_NAMES) {
                if let Some(&idx) = name_to_idx.get(*name) {
                    *slot = idx;
                } else {
                    ok = false;
                    break;
                }
            }
            ok.then_some(out)
        };
        let Some(hex_ring) = hex_ring else {
            continue;
        };
        let pent_ring: Option<[u32; 5]> = if is_purine(residue.name) {
            let mut out = [0u32; 5];
            let mut ok = true;
            for (slot, name) in out.iter_mut().zip(PENT_RING_NAMES) {
                if let Some(&idx) = name_to_idx.get(*name) {
                    *slot = idx;
                } else {
                    ok = false;
                    break;
                }
            }
            ok.then_some(out)
        } else {
            None
        };
        let c1_prime = name_to_idx
            .get(b"C1'".as_slice())
            .or_else(|| name_to_idx.get(b"C1*".as_slice()))
            .copied();
        rings.push(NucleotideRingLayout {
            hex_ring,
            pent_ring,
            c1_prime,
            p_index: residue.atom_range.start as u32,
            chain_idx: chain_indices.get(res_idx).copied().unwrap_or(0),
            color,
        });
    }
    rings
}
