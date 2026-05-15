//! Sheet-specific backbone geometry: peptide-plane normals, iterative
//! flattening (PyMOL-style), and sidechain offset computation.

use glam::Vec3;
use molex::SSType;

use crate::renderer::entity_topology::ProteinBackboneChain;

/// Segment a residue SS-type array into contiguous runs.
#[derive(Debug)]
pub(crate) struct SSSegment {
    pub(crate) ss_type: SSType,
    pub(crate) start_residue: usize,
    pub(crate) end_residue: usize,
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
                start_residue: start,
                end_residue: i,
            });
            current = ss;
            start = i;
        }
    }
    segments.push(SSSegment {
        ss_type: current,
        start_residue: start,
        end_residue: ss_types.len(),
    });

    segments
}

/// Compute sheet-specific geometry: flattened CA positions, peptide-plane
/// normals, and position offsets for sidechain adjustment.
///
/// Returns `(flat_ca, normals, sheet_offsets)`.
pub(crate) fn compute_sheet_geometry(
    atoms: &ProteinBackboneChain,
    ss_types: &[SSType],
    global_residue_base: u32,
) -> (Vec<Vec3>, Vec<Vec3>, Vec<(u32, Vec3)>) {
    let n = atoms.ca.len();
    let mut flat_ca = atoms.ca.clone();
    let mut normals = vec![Vec3::ZERO; n];
    let mut offsets = Vec::new();

    // Per-residue peptide-plane normal via PyMOL's convention:
    // `(CA - N) × (O - N)`. Uses three real atoms of the same residue
    // -- N, CA, and the carbonyl O -- so the resulting normal aligns
    // with the carbonyl direction (the broad ribbon face axis,
    // up to global sign which `propagate_segment_signs` resolves).
    // Matches `RepCartoon.c:2040-2058` in pymol-open-source.
    for (i, slot) in normals.iter_mut().enumerate() {
        if i < atoms.n.len() && i < atoms.o.len() {
            let n_ca = (atoms.ca[i] - atoms.n[i]).normalize_or_zero();
            let n_o = (atoms.o[i] - atoms.n[i]).normalize_or_zero();
            let normal = n_ca.cross(n_o);
            *slot = if normal.length_squared() > 1e-6 {
                normal.normalize()
            } else {
                Vec3::Y
            };
        }
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
        let start = seg.start_residue;
        let end = seg.end_residue.min(n);
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
            let offset = seg_pos[j] - atoms.ca[i];
            flat_ca[i] = seg_pos[j];
            normals[i] = seg_normals[j];
            offsets.push((global_residue_base + i as u32, offset));
        }
    }

    (flat_ca, normals, offsets)
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

/// Minimum negative azimuthal cosine before a 180° de-aliasing flip is
/// committed. A near-orthogonal junction (a genuine pleat break, not an
/// aliased sign) stays put so flattening can smooth it instead of being
/// coin-flipped into a hard crease.
const SIGN_FLIP_THRESHOLD: f32 = 0.3;

/// Make peptide-plane normals sign-coherent within a single strand.
///
/// `(CA−N)×(O−N)` alternates ~180° every residue from β-pleating. Each
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

    for _ in 0..cycles {
        // Average positions with neighbors (skip endpoints)
        let mut new_pos = positions.to_vec();
        for i in 1..n - 1 {
            new_pos[i] =
                (positions[i - 1] + positions[i] * 2.0 + positions[i + 1])
                    * 0.25;
        }
        positions.copy_from_slice(&new_pos);

        // Average normals with neighbors (skip endpoints)
        let mut new_normals = normals.to_vec();
        for i in 1..n - 1 {
            let avg = normals[i - 1] + normals[i] * 2.0 + normals[i + 1];
            new_normals[i] = if avg.length_squared() > 1e-6 {
                avg.normalize()
            } else {
                normals[i]
            };
        }
        normals.copy_from_slice(&new_normals);

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
            let frac = i as f32 / (total_spline - 1).max(1) as f32;
            let rf = frac * (n_residues - 1) as f32;
            let r0 = (rf.floor() as usize).min(n_residues - 1);
            let r1 = (r0 + 1).min(n_residues - 1);
            let t = rf - r0 as f32;
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

    /// Every flattened sheet normal — endpoints included — must be
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

    /// Raw β-pleat normals alternate ~180°; after segment propagation
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
        // Azimuth ≈ -0.16 vs the predecessor: slightly opposed but
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
