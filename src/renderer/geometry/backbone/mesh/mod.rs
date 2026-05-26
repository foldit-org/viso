//! Backbone mesh generation: ties spline, profile, and sheet modules together
//! into final vertex/index buffers for both protein and nucleic acid chains.

mod nucleic;
mod protein;

use glam::Vec3;
use molex::SSType;
use nucleic::generate_na_chain_mesh;
use protein::generate_protein_chain_mesh;

use super::arrows::apply_sheet_arrows;
use super::index::MeshParams;
use super::profile::{
    resolve_na_profile, resolve_profile, CrossSectionProfile,
};
use super::BackboneMeshOutput;
use crate::options::{ChainLod, GeometryOptions};
use crate::renderer::geometry::nucleic_acid::NA_DEFAULT_COLOR;

/// Per-chain index range and bounding sphere for frustum culling.
///
/// The tube/ribbon index spans are half-open `Range<u32>` into the
/// shared index buffer. The range fields are private so the only
/// construction path is [`ChainRange::new`], which asserts the two
/// invariants every consumer relies on: `start <= end` and triangle
/// alignment (`% 3 == 0`, since indices come in triples).
#[derive(Clone, Debug)]
pub(crate) struct ChainRange {
    tube: std::ops::Range<u32>,
    ribbon: std::ops::Range<u32>,
    pub(crate) bounding_center: Vec3,
    pub(crate) bounding_radius: f32,
}

impl ChainRange {
    /// Build a chain range, asserting the index-span invariants.
    pub(crate) fn new(
        tube: std::ops::Range<u32>,
        ribbon: std::ops::Range<u32>,
        bounding_center: Vec3,
        bounding_radius: f32,
    ) -> Self {
        debug_assert!(
            tube.start <= tube.end && ribbon.start <= ribbon.end,
            "ChainRange index span: start must not exceed end (tube {tube:?}, \
             ribbon {ribbon:?})",
        );
        debug_assert!(
            tube.start.is_multiple_of(3)
                && tube.end.is_multiple_of(3)
                && ribbon.start.is_multiple_of(3)
                && ribbon.end.is_multiple_of(3),
            "ChainRange index span must be triangle-aligned (multiple of 3): \
             tube {tube:?}, ribbon {ribbon:?}",
        );
        Self {
            tube,
            ribbon,
            bounding_center,
            bounding_radius,
        }
    }

    /// Half-open index span for the tube (round cross-section) pass.
    pub(crate) fn tube(&self) -> std::ops::Range<u32> {
        self.tube.clone()
    }

    /// Half-open index span for the ribbon (flat cross-section) pass.
    pub(crate) fn ribbon(&self) -> std::ops::Range<u32> {
        self.ribbon.clone()
    }
}

/// Generate unified backbone mesh from protein and nucleic acid chains.
pub(crate) fn generate_mesh_colored(
    protein: &[crate::renderer::entity_topology::ProteinBackboneChain],
    na: &[crate::renderer::entity_topology::NaBackboneChain],
    ss_override: Option<&[SSType]>,
    per_residue_colors: Option<&[[f32; 3]]>,
    geo: &GeometryOptions,
    per_chain_lod: Option<&[ChainLod]>,
    na_residue_colors: Option<&[[f32; 3]]>,
    na_seeds: Option<&[Option<Vec3>]>,
    na_guide_dirs: Option<&[Vec3]>,
) -> BackboneMeshOutput {
    let mut out = BackboneMeshOutput::default();

    // Protein block. The color slice is whole-assembly-indexed, so it
    // keys off `global_residue_idx`; `residue_offset` is unused here.
    let global_residue_idx = process_chains(
        protein,
        geo,
        per_chain_lod,
        &mut out,
        0,
        |atoms| atoms.ca().len(),
        |atoms, _chain_idx, global_residue_idx, _residue_offset, params| {
            let n_residues = atoms.ca().len();
            let chain_slice = ss_override.and_then(|o| {
                let start = global_residue_idx as usize;
                let end = (start + n_residues).min(o.len());
                (end.saturating_sub(start) == n_residues)
                    .then(|| &o[start..end])
            });
            // Engine sync always installs per-entity SS via
            // `Assembly::ss_types`, so every protein chain with >= 2 CA
            // atoms has a matching slice. If that invariant is ever
            // violated the chain renders as coil -- no DSSP recompute.
            let ss_types = chain_slice.map_or_else(
                || vec![SSType::Coil; n_residues],
                molex::analysis::merge_short_segments,
            );

            let mut profiles: Vec<CrossSectionProfile> = (0..n_residues)
                .map(|i| {
                    let color = per_residue_colors
                        .and_then(|c| {
                            c.get(global_residue_idx as usize + i).copied()
                        })
                        .unwrap_or_else(|| ss_types[i].color());
                    resolve_profile(
                        ss_types[i],
                        global_residue_idx + i as u32,
                        color,
                        geo,
                    )
                })
                .collect();

            if geo.sheet_arrows {
                apply_sheet_arrows(&ss_types, &mut profiles, geo);
            }

            // Widest the extruded ribbon/tube can sit off the CA spline:
            // the largest configured half-width/thickness, scaled by the
            // x1.5 sheet-arrow shoulder, plus Catmull-Rom overshoot.
            let max_extent = geo
                .sheet_width
                .max(geo.helix_width)
                .max(geo.coil_width)
                .max(geo.sheet_thickness)
                .max(geo.helix_thickness)
                .max(geo.coil_thickness)
                * 1.5;
            let (center, radius) = bounding_sphere(
                atoms.ca(),
                max_extent + SPLINE_OVERSHOOT_SLACK,
            );

            let chain_mesh = generate_protein_chain_mesh(
                atoms,
                &ss_types,
                &profiles,
                global_residue_idx,
                params,
            );
            (chain_mesh, center, radius)
        },
    );

    // Nucleic-acid block, continuing the threaded `global_residue_idx`.
    // Its color/seed/guide slices are NA-entity-local (0-based), so they
    // key off the call-local `residue_offset`, not `global_residue_idx`.
    let na_lod = per_chain_lod.and_then(|l| l.get(protein.len()..));
    // NA is the last block; the threaded counter past it has no consumer.
    let _final_residue_idx = process_chains(
        na,
        geo,
        na_lod,
        &mut out,
        global_residue_idx,
        |chain| chain.p().len(),
        |chain, chain_idx, global_residue_idx, residue_offset, params| {
            let points = chain.p();
            let n_residues = points.len();
            let profiles: Vec<CrossSectionProfile> = (0..n_residues)
                .map(|i| {
                    let color = na_residue_colors
                        .and_then(|c| c.get(residue_offset + i).copied())
                        .unwrap_or(NA_DEFAULT_COLOR);
                    resolve_na_profile(
                        global_residue_idx + i as u32,
                        color,
                        geo,
                    )
                })
                .collect();

            // The drawn NA geometry is not just the thin ribbon: base
            // paddles + stems extend well off the P backbone (rendered
            // by the separate NA renderer with no per-chain cull). Pad
            // the sphere by that reach so an edge-on duplex doesn't
            // frustum-cull its paddles while the backbone stays visible.
            let na_extent =
                geo.na_width.max(geo.na_thickness) + NA_PADDLE_REACH_SLACK;
            let (center, radius) =
                bounding_sphere(points, na_extent + SPLINE_OVERSHOOT_SLACK);
            let seed =
                na_seeds.and_then(|s| s.get(chain_idx).copied()).flatten();
            // Residue-parallel slice of the entity-wide C1'-P guide field.
            let chain_guides: &[Vec3] = na_guide_dirs
                .and_then(|g| {
                    g.get(residue_offset..residue_offset + n_residues)
                })
                .unwrap_or(&[]);
            let chain_mesh = generate_na_chain_mesh(
                points,
                &profiles,
                params,
                seed,
                chain_guides,
            );
            (chain_mesh, center, radius)
        },
    );

    out
}

/// Drive the shared per-chain backbone loop for one polymer block.
///
/// Owns the residue-counter bookkeeping in exactly one place -- the
/// running whole-assembly `global_residue_idx` (threaded across calls,
/// returned for the next block) and a call-local 0-based
/// `residue_offset` into this block's color/guide slices. Both advance
/// in lockstep, *including across the `< 2`-residue skip*, so the
/// color-slice desync class (T1-NA-C) cannot be reintroduced by a
/// hand-rolled counter inside a per-type body.
///
/// `n_residues_of` reports a chain's control-point count. `chain_mesh`
/// builds the type-specific mesh and its bounding sphere from the chain,
/// its index in `chains`, both residue indices, and the LOD-resolved
/// [`MeshParams`]; the driver pushes the result and advances the
/// counters.
fn process_chains<C>(
    chains: &[C],
    geo: &GeometryOptions,
    per_chain_lod: Option<&[ChainLod]>,
    out: &mut BackboneMeshOutput,
    mut global_residue_idx: u32,
    n_residues_of: impl Fn(&C) -> usize,
    mut chain_mesh: impl FnMut(
        &C,
        usize,
        u32,
        usize,
        &MeshParams,
    ) -> (BackboneMeshOutput, Vec3, f32),
) -> u32 {
    let spr = geo.segments_per_residue;
    let csv = geo.cross_section_verts;
    let mut residue_offset: usize = 0;

    for (chain_idx, chain) in chains.iter().enumerate() {
        let n_residues = n_residues_of(chain);
        if n_residues < 2 {
            global_residue_idx += n_residues as u32;
            residue_offset += n_residues;
            continue;
        }

        let lod = per_chain_lod
            .and_then(|l| l.get(chain_idx).copied())
            .unwrap_or(ChainLod {
                segments_per_residue: spr,
                cross_section_verts: csv,
            });
        let params = MeshParams {
            base_vertex: out.vertices.len() as u32,
            cross_section_verts: lod.cross_section_verts,
            segments_per_residue: lod.segments_per_residue,
        };

        let (chain_mesh_out, center, radius) = chain_mesh(
            chain,
            chain_idx,
            global_residue_idx,
            residue_offset,
            &params,
        );
        out.push_chain(chain_mesh_out, center, radius);

        global_residue_idx += n_residues as u32;
        residue_offset += n_residues;
    }

    global_residue_idx
}

/// Catmull-Rom interpolation can bow outside the CA control hull at
/// sharp turns; this bounds that overshoot for the culling sphere so a
/// chain isn't culled while its extruded curve is still on-screen.
const SPLINE_OVERSHOOT_SLACK: f32 = 1.0;

/// Worst-case excursion of a base paddle + stem off the P backbone
/// (stem P->ring centroid plus the ring half-extent). Conservatively
/// padded -- over-padding only makes the NA cull more lenient, never
/// false-negative.
const NA_PADDLE_REACH_SLACK: f32 = 12.0;

/// Compute bounding sphere (centroid + max distance) from a set of
/// positions, padded by `slack`.
///
/// The sphere is fit to the raw control points (CA / P atoms), but the
/// drawn mesh is an extruded tube/ribbon that extends past them by up to
/// the cross-section half-extent plus spline overshoot. `slack` widens
/// the radius to cover that, eliminating false-negative frustum culls at
/// sharp turns. Over-padding only makes culling more conservative.
fn bounding_sphere(positions: &[Vec3], slack: f32) -> (Vec3, f32) {
    if positions.is_empty() {
        return (Vec3::ZERO, 0.0);
    }
    let center =
        positions.iter().copied().sum::<Vec3>() / positions.len() as f32;
    let radius = positions
        .iter()
        .map(|p| (*p - center).length())
        .fold(0.0f32, f32::max);
    (center, radius + slack)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Pins the T4-NA-B unification's load-bearing invariant: the single
    /// `process_chains` driver advances `global_residue_idx` (threaded
    /// across blocks) and the call-local 0-based `residue_offset` in
    /// lockstep, including across the `< 2`-residue skip. This is the
    /// exact desync class (T1-NA-C) the old hand-rolled per-type
    /// counters were prone to. `C = usize` stands in for a chain's
    /// residue count so the bookkeeping is tested independent of the
    /// real protein/NA chain types.
    #[test]
    fn process_chains_threads_global_idx_and_resets_offset() {
        let geo = GeometryOptions::default();
        let mut out = BackboneMeshOutput::default();
        let mut seen: Vec<(u32, usize)> = Vec::new();

        // Protein-like block: chains of 3, 1 (skipped, < 2), 4 residues.
        let after_protein = process_chains(
            &[3usize, 1, 4],
            &geo,
            None,
            &mut out,
            0,
            |&n| n,
            |_c, _idx, gri, ro, _p| {
                seen.push((gri, ro));
                (BackboneMeshOutput::default(), Vec3::ZERO, 0.0)
            },
        );
        // chain0 -> (gri=0, ro=0); chain1 (n=1) skipped but still
        // advances both by 1; chain2 -> (gri=4, ro=4).
        assert_eq!(seen, vec![(0, 0), (4, 4)]);
        assert_eq!(after_protein, 8, "3 + 1(skip) + 4 residues threaded");

        // NA-like block: offset resets to 0, global idx continues at 8.
        seen.clear();
        let after_na = process_chains(
            &[2usize, 2],
            &geo,
            None,
            &mut out,
            after_protein,
            |&n| n,
            |_c, _idx, gri, ro, _p| {
                seen.push((gri, ro));
                (BackboneMeshOutput::default(), Vec3::ZERO, 0.0)
            },
        );
        assert_eq!(
            seen,
            vec![(8, 0), (10, 2)],
            "offset is call-local 0-based; global idx threads from 8",
        );
        assert_eq!(after_na, 12);
    }
}
