//! Sheet-specific backbone geometry: peptide-plane normals, iterative
//! flattening (PyMOL-style), and sidechain offset computation.

use std::ops::Range;

use glam::Vec3;
use molex::SSType;

use crate::renderer::entity_topology::ProteinBackboneChain;

/// Minimum squared length of the peptide-plane cross product `(CA-N)x(O-N)`
/// for it to be treated as a usable normal. Below this the three atoms are
/// effectively collinear.
const DEGENERATE_NORMAL_LENGTH_SQ: f32 = 1e-6;

/// Substitute normal used only when the peptide-plane cross product is
/// degenerate (collinear N/CA/O -- pathological backbone geometry).
const DEGENERATE_NORMAL: Vec3 = Vec3::Y;

/// One contiguous run of a single secondary-structure type.
///
/// `residues` is the half-open residue range `[start, end)`; carrying a
/// `Range` rather than two bare `usize`s makes `start <= end` structural
/// and removes the swap-bait of separate start/end fields.
#[derive(Debug)]
pub(crate) struct SSSegment {
    pub(crate) ss_type: SSType,
    pub(crate) residues: Range<usize>,
}

pub(crate) fn segment_by_ss(ss_types: &[SSType]) -> Vec<SSSegment> {
    if ss_types.is_empty() {
        return Vec::new();
    }

    let mut segments = Vec::new();
    let mut current = ss_types[0];
    let mut start = 0;

    for (i, &ss) in ss_types.iter().enumerate() {
        if ss != current {
            segments.push(SSSegment {
                ss_type: current,
                residues: start..i,
            });
            current = ss;
            start = i;
        }
    }
    segments.push(SSSegment {
        ss_type: current,
        residues: start..ss_types.len(),
    });

    segments
}

/// Position delta applied to one flattened beta-sheet residue.
///
/// Replaces the bare `(u32, Vec3)` pair that was threaded through ~15
/// signatures and fields -- `residue_idx` and `offset` were trivially
/// swappable as a positional tuple.
#[derive(Clone, Copy, Debug)]
pub(crate) struct SheetOffset {
    /// Global residue index this delta applies to.
    pub(crate) residue_idx: u32,
    /// Flattened-CA minus original-CA position delta.
    pub(crate) offset: Vec3,
}

/// Sheet-specific backbone geometry produced by [`compute_sheet_geometry`].
///
/// `flat_ca` and `normals` are residue-stride and equal length; `offsets`
/// holds only the residues that were flattened.
pub(crate) struct SheetGeometry {
    pub(crate) flat_ca: Vec<Vec3>,
    pub(crate) normals: Vec<Vec3>,
    pub(crate) offsets: Vec<SheetOffset>,
}

/// Compute sheet-specific geometry: flattened CA positions, peptide-plane
/// normals, and position offsets for sidechain adjustment.
pub(crate) fn compute_sheet_geometry(
    atoms: &ProteinBackboneChain,
    ss_types: &[SSType],
    global_residue_base: u32,
) -> SheetGeometry {
    let n = atoms.ca().len();
    let mut flat_ca = atoms.ca().to_vec();
    let mut normals = vec![Vec3::ZERO; n];
    let mut offsets = Vec::new();

    // Per-residue peptide-plane normal via PyMOL's convention:
    // `(CA - N) x (O - N)`. Uses three real atoms of the same residue
    // -- N, CA, and the carbonyl O -- so the resulting normal aligns
    // with the carbonyl direction (the broad ribbon face axis,
    // up to global sign which `propagate_segment_signs` resolves).
    // Matches `RepCartoon.c:2040-2058` in pymol-open-source.
    // `atoms` resolves all-or-nothing with n/ca/o equal length
    // (`ProteinBackboneIndices::resolve`), so `i` is always in range for
    // every backbone array here -- no index guard is needed, and a guard
    // would silently emit `Vec3::Y` for a real residue if it ever failed.
    debug_assert!(
        atoms.n().len() == n && atoms.o().len() == n,
        "backbone n/ca/o desynced: n={}, ca={}, o={}",
        atoms.n().len(),
        n,
        atoms.o().len(),
    );
    for (i, slot) in normals.iter_mut().enumerate() {
        let n_ca = (atoms.ca()[i] - atoms.n()[i]).normalize_or_zero();
        let n_o = (atoms.o()[i] - atoms.n()[i]).normalize_or_zero();
        let normal = n_ca.cross(n_o);
        // `DEGENERATE_NORMAL` only fires when N, CA and the carbonyl O of
        // this residue are collinear (a pathological/incomplete backbone),
        // making the peptide-plane cross product near-zero. For real
        // protein geometry the three atoms are never collinear.
        *slot = if normal.length_squared() > DEGENERATE_NORMAL_LENGTH_SQ {
            normal.normalize()
        } else {
            DEGENERATE_NORMAL
        };
    }

    // No beta-strand -> no flattening, no offsets. Skip the segment scan and
    // the per-segment loop entirely; `flat_ca` is already an exact copy
    // of the input CAs and `normals` is fully populated above.
    if !ss_types.contains(&SSType::Sheet) {
        return SheetGeometry {
            flat_ca,
            normals,
            offsets,
        };
    }

    let trace = super::sheet_trace::enabled();

    // Sign coherence is resolved per strand (below), not chain-globally:
    // a chain-wide pass threads each strand's broad-face sign back
    // through arbitrary loop/helix normals, so a strand could render
    // flipped depending on unrelated geometry. Reference renderers
    // process strands locally; downstream the per-spline-sample
    // hemisphere alignment in `compute_final_frames` reconciles each
    // strand's sign with the continuous RMF normal.

    // Find sheet segments and apply flattening
    let segments = segment_by_ss(ss_types);
    for seg in &segments {
        if seg.ss_type != SSType::Sheet {
            continue;
        }
        let start = seg.residues.start;
        let end = seg.residues.end.min(n);
        if end <= start + 1 {
            continue;
        }

        let mut seg_pos = flat_ca[start..end].to_vec();
        let mut seg_normals = normals[start..end].to_vec();

        let raw_snapshot = trace.then(|| seg_normals.clone());
        propagate_segment_signs(&seg_pos, &mut seg_normals);
        let signs_snapshot = trace.then(|| seg_normals.clone());
        flatten_sheet(&mut seg_pos, &mut seg_normals, 4);
        let flatten_snapshot = trace.then(|| seg_normals.clone());
        clamp_strand_end_normals(&seg_pos, &mut seg_normals);

        if let (Some(raw), Some(signs), Some(flat)) =
            (raw_snapshot, signs_snapshot, flatten_snapshot)
        {
            super::sheet_trace::trace_strand_stages(
                global_residue_base,
                start,
                &raw,
                &signs,
                &flat,
                &seg_normals,
            );
        }

        for (j, i) in (start..end).enumerate() {
            let offset = seg_pos[j] - atoms.ca()[i];
            flat_ca[i] = seg_pos[j];
            normals[i] = seg_normals[j];
            offsets.push(SheetOffset {
                residue_idx: global_residue_base + i as u32,
                offset,
            });
        }
    }

    SheetGeometry {
        flat_ca,
        normals,
        offsets,
    }
}

/// Local strand tangent at `i` (forward difference at the first point,
/// backward at the last, central in between).
fn strand_tangent(positions: &[Vec3], i: usize) -> Vec3 {
    let n = positions.len();
    if i == 0 {
        (positions[1] - positions[0]).normalize_or_zero()
    } else if i == n - 1 {
        (positions[n - 1] - positions[n - 2]).normalize_or_zero()
    } else {
        (positions[i + 1] - positions[i - 1]).normalize_or_zero()
    }
}

/// Minimum negative azimuthal cosine before a 180deg de-aliasing flip is
/// committed. A near-orthogonal junction (a genuine pleat break, not an
/// aliased sign) stays put so flattening can smooth it instead of being
/// coin-flipped into a hard crease.
const SIGN_FLIP_THRESHOLD: f32 = 0.3;

/// Make peptide-plane normals sign-coherent within a single strand.
///
/// `(CA-N)x(O-N)` alternates ~180deg every residue from beta-pleating. Each
/// normal is aligned to its predecessor's broad-face hemisphere, judged
/// on the components perpendicular to the local strand tangent (the
/// broad-face azimuth), and only when the two are clearly opposed.
fn propagate_segment_signs(positions: &[Vec3], normals: &mut [Vec3]) {
    for i in 1..normals.len() {
        let t = strand_tangent(positions, i);
        let prev =
            (normals[i - 1] - t * normals[i - 1].dot(t)).normalize_or_zero();
        let cur = (normals[i] - t * normals[i].dot(t)).normalize_or_zero();
        if prev.dot(cur) < -SIGN_FLIP_THRESHOLD {
            normals[i] = -normals[i];
        }
    }
}

/// Clamp each strand's terminal normals to the adjacent interior normal,
/// re-orthogonalized to the local tangent.
///
/// A strand's broad face must not twist at its end residues, whose raw
/// peptide plane is transitional and unreliable; flattening already
/// skips endpoints in its averaging, so without this an off-axis
/// terminal normal survives as a visible crease.
fn clamp_strand_end_normals(positions: &[Vec3], normals: &mut [Vec3]) {
    let n = normals.len();
    if n < 3 {
        return;
    }
    for &(end, inner) in &[(0usize, 1usize), (n - 1, n - 2)] {
        let t = strand_tangent(positions, end);
        let proj = normals[inner] - t * normals[inner].dot(t);
        if proj.length_squared() > 1e-6 {
            normals[end] = proj.normalize();
        }
    }
}

/// Iterative flattening of sheet positions and normals (PyMOL-style).
///
/// Each cycle averages each point/normal with its neighbors using a
/// weighted kernel (1,2,1)/4, then re-orthogonalizes the normal against
/// the backbone tangent.
fn flatten_sheet(positions: &mut [Vec3], normals: &mut [Vec3], cycles: usize) {
    let n = positions.len();
    if n < 3 {
        return;
    }

    // Scratch buffers hoisted out of the cycle loop: each cycle reseeds
    // them from the current state (so endpoints, which the averaging
    // skips, are preserved) instead of allocating two fresh Vecs.
    let mut scratch_pos = positions.to_vec();
    let mut scratch_normals = normals.to_vec();

    for _ in 0..cycles {
        // Average positions with neighbors (skip endpoints)
        scratch_pos.copy_from_slice(positions);
        for i in 1..n - 1 {
            scratch_pos[i] =
                (positions[i - 1] + positions[i] * 2.0 + positions[i + 1])
                    * 0.25;
        }
        positions.copy_from_slice(&scratch_pos);

        // Average normals with neighbors (skip endpoints)
        scratch_normals.copy_from_slice(normals);
        for i in 1..n - 1 {
            let avg = normals[i - 1] + normals[i] * 2.0 + normals[i + 1];
            scratch_normals[i] = if avg.length_squared() > 1e-6 {
                avg.normalize()
            } else {
                normals[i]
            };
        }
        normals.copy_from_slice(&scratch_normals);

        // Re-orthogonalize every normal against its local backbone
        // tangent, endpoints included (forward difference at the first
        // residue, backward at the last, central in between). Endpoints
        // are skipped by the averaging passes above, so without this they
        // would keep their raw, non-perpendicular peptide-plane normal.
        for i in 0..n {
            let tangent = if i == 0 {
                (positions[1] - positions[0]).normalize_or_zero()
            } else if i == n - 1 {
                (positions[n - 1] - positions[n - 2]).normalize_or_zero()
            } else {
                (positions[i + 1] - positions[i - 1]).normalize_or_zero()
            };
            let proj = normals[i] - tangent * normals[i].dot(tangent);
            normals[i] = if proj.length_squared() > 1e-6 {
                proj.normalize()
            } else {
                normals[i]
            };
        }
    }
}

/// Interpolate per-residue normals to spline resolution.
pub(crate) fn interpolate_per_residue_normals(
    normals: &[Vec3],
    total_spline: usize,
    n_residues: usize,
) -> Vec<Vec3> {
    (0..total_spline)
        .map(|i| {
            let (r0, r1, t) =
                super::spline::residue_bracket(i, total_spline, n_residues);
            // Flip the far endpoint into the near endpoint's hemisphere
            // before interpolating: a straight lerp between opposed
            // normals passes through zero at the midpoint, which
            // `normalize_or_zero` would turn into an undefined direction.
            let n1 = if normals[r0].dot(normals[r1]) < 0.0 {
                -normals[r1]
            } else {
                normals[r1]
            };
            normals[r0].lerp(n1, t).normalize_or_zero()
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    const TOL: f32 = 1e-4;

    fn local_tangent(positions: &[Vec3], i: usize) -> Vec3 {
        let n = positions.len();
        if i == 0 {
            (positions[1] - positions[0]).normalize_or_zero()
        } else if i == n - 1 {
            (positions[n - 1] - positions[n - 2]).normalize_or_zero()
        } else {
            (positions[i + 1] - positions[i - 1]).normalize_or_zero()
        }
    }

    /// Every flattened sheet normal -- endpoints included -- must be
    /// perpendicular to its local backbone tangent.
    #[test]
    fn flatten_sheet_endpoints_are_orthogonal() {
        let mut positions = vec![
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 1.0, 0.0),
            Vec3::new(2.0, 0.0, 0.0),
        ];
        // Endpoint normals carry a deliberate tangent-direction component.
        let oblique = Vec3::new(1.0, 0.0, 1.0).normalize();
        let mut normals = vec![oblique, Vec3::Z, oblique];

        flatten_sheet(&mut positions, &mut normals, 4);

        for (i, &nrm) in normals.iter().enumerate() {
            let t = local_tangent(&positions, i);
            let d = nrm.dot(t);
            assert!(
                d.abs() < TOL,
                "normal {i} not perpendicular to tangent (dot = {d}, normal = \
                 {:?}, tangent = {t:?})",
                normals[i],
            );
        }
    }

    /// Raw beta-pleat normals alternate ~180deg; after segment propagation
    /// every consecutive pair must share a hemisphere.
    #[test]
    fn segment_signs_dealias_pleat_alternation() {
        // Strand running along +X; broad face is +Z, alternating sign.
        let positions: Vec<Vec3> =
            (0..6).map(|i| Vec3::new(i as f32, 0.0, 0.0)).collect();
        let mut normals: Vec<Vec3> = (0..6)
            .map(|i| if i % 2 == 0 { Vec3::Z } else { -Vec3::Z })
            .collect();

        propagate_segment_signs(&positions, &mut normals);

        for i in 1..normals.len() {
            assert!(
                normals[i].dot(normals[i - 1]) > 0.0,
                "pair {i} still anti-parallel: {:?} vs {:?}",
                normals[i - 1],
                normals[i],
            );
        }
    }

    /// A near-orthogonal junction is a real pleat break, not an aliased
    /// sign: propagation must leave it rather than coin-flip it.
    #[test]
    fn segment_signs_do_not_flip_near_orthogonal() {
        let positions = vec![Vec3::ZERO, Vec3::X, Vec3::new(2.0, 0.0, 0.0)];
        // Azimuth ~ -0.16 vs the predecessor: slightly opposed but
        // within the hysteresis band. The old `< 0.0` rule would flip
        // this; the threshold rule must not.
        let before = Vec3::new(0.3, 0.95, -0.15).normalize();
        let mut normals = vec![Vec3::Z, before, Vec3::Z];

        propagate_segment_signs(&positions, &mut normals);

        assert!(
            normals[1].dot(before) > 0.999,
            "near-orthogonal normal was flipped: {:?}",
            normals[1],
        );
    }

    /// Interpolating across a sign-mismatched neighbor pair must not
    /// collapse to a zero-length (undefined) direction at the midpoint.
    #[test]
    fn interpolate_opposed_normals_stays_nonzero() {
        let normals = vec![Vec3::X, -Vec3::X];
        let result = interpolate_per_residue_normals(&normals, 3, 2);

        for (i, v) in result.iter().enumerate() {
            assert!(
                v.length() >= 1e-3,
                "output {i} collapsed to near-zero ({v:?}, len = {})",
                v.length(),
            );
        }
    }
}
