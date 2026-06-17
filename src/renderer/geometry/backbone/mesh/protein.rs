//! Per-chain protein backbone mesh: SS-aware spline, sheet geometry,
//! and RMF/radial/sheet normal blending.

use glam::Vec3;

use super::super::curve::{cubic_bspline, sliding_window_centroids};
use super::super::index::{extrude_and_index, MeshParams};
use super::super::path::{
    compute_sheet_geometry, interpolate_per_residue_normals, RibbonAnchor,
    SheetGeometry,
};
use super::super::profile::{interpolate_profiles, CrossSectionProfile};
use super::super::spline::{
    build_traces, helix_aware_spline, rmf_frames, SplinePoint,
};
use super::super::BackboneMeshOutput;
use crate::util::geom::central_difference_tangents;

/// Generate mesh for a single protein chain (with SS detection, sheet
/// geometry, and RMF/radial/sheet normal blending). Takes the SoA
/// backbone-atom view directly from the topology -- no interleaved
/// stride shuffling.
pub(super) fn generate_protein_chain_mesh(
    atoms: &crate::renderer::entity_topology::ProteinBackboneChain,
    ss_types: &[molex::SSType],
    profiles: &[CrossSectionProfile],
    global_residue_base: u32,
    params: &MeshParams,
) -> BackboneMeshOutput {
    let n = atoms.ca().len();
    if n < 2 {
        return BackboneMeshOutput::default();
    }

    let SheetGeometry {
        flat_ca,
        normals: sheet_normals,
        offsets: sheet_offsets,
    } = compute_sheet_geometry(atoms, ss_types, global_residue_base);

    let spr = params.segments_per_residue;
    let spline_points = helix_aware_spline(&flat_ca, ss_types, spr);
    let total = spline_points.len();
    if total < 2 {
        return BackboneMeshOutput::default();
    }

    let tangents = central_difference_tangents(&spline_points);

    let helix_centers = sliding_window_centroids(atoms.ca());
    let spline_helix_centers = cubic_bspline(&helix_centers, spr);

    let traces = build_traces(&spline_points, &tangents);
    // Seed the RMF roll from the first residue's peptide-plane normal so
    // the whole chain's roll is fixed by backbone geometry rather than a
    // world axis. `compute_rmf` projects this perpendicular to the first
    // tangent and falls back to an axis only if it is zero/absent.
    let frames = rmf_frames(&traces, sheet_normals.first().copied());

    let spline_sheet_normals =
        interpolate_per_residue_normals(&sheet_normals, total, n);
    let spline_profiles = interpolate_profiles(profiles, total, n);

    let final_frames = compute_final_frames(
        &frames,
        &spline_helix_centers,
        &spline_sheet_normals,
        &spline_profiles,
    );

    if super::super::sheet_trace::enabled() {
        super::super::sheet_trace::trace_final_frames(
            global_residue_base,
            n,
            &tangents,
            &frames,
            &spline_sheet_normals,
            &final_frames,
            &spline_profiles,
        );
    }

    let (verts, tube_inds, ribbon_inds) =
        extrude_and_index(&final_frames, &spline_profiles, params);

    let ribbon_anchors = build_ribbon_anchors(
        &spline_points,
        &flat_ca,
        atoms,
        spr,
        global_residue_base,
    );

    BackboneMeshOutput {
        vertices: verts,
        tube_indices: tube_inds,
        ribbon_indices: ribbon_inds,
        sheet_offsets,
        ribbon_anchors,
        ..Default::default()
    }
}

/// Per-residue ribbon anchors on the drawn centerline.
///
/// `spline` is the SS-aware spline over the sheet-flattened CAs the mesh
/// extrudes; it carries `spr` samples per residue span plus one final
/// endpoint, so residue `i`'s centerline sits at sample `i * spr`. The N
/// and C anchors are sampled from that same curve at the fractional
/// residue offsets where the peptide N and C project (N two-thirds of the
/// way through the prior span, C about a third into the next), so a marker
/// endpoint lands on the rendered ribbon rather than on the raw atom. Edge
/// residues fall outside the spannable range, so their N/C use the raw
/// backbone atom.
fn build_ribbon_anchors(
    spline: &[Vec3],
    flat_ca: &[Vec3],
    atoms: &crate::renderer::entity_topology::ProteinBackboneChain,
    spr: usize,
    global_residue_base: u32,
) -> Vec<RibbonAnchor> {
    /// Fraction of the CA->CA span where C sits (CA->C / total).
    const C_FRAC: f32 = 0.35;
    /// Fraction of the CA->CA span where N sits (measured from the
    /// previous CA).
    const N_FRAC: f32 = 0.66;

    let n_res = flat_ca.len();
    let mut anchors = Vec::with_capacity(n_res);
    for (i, &ca) in flat_ca.iter().enumerate() {
        let n = if i == 0 {
            atoms.n()[i]
        } else {
            sample_spline(spline, (i - 1) as f32 + N_FRAC, spr)
                .unwrap_or_else(|| atoms.n()[i])
        };
        let c = if i + 1 >= n_res {
            atoms.c()[i]
        } else {
            sample_spline(spline, i as f32 + C_FRAC, spr)
                .unwrap_or_else(|| atoms.c()[i])
        };
        anchors.push(RibbonAnchor {
            residue_idx: global_residue_base + i as u32,
            n,
            ca,
            c,
        });
    }
    anchors
}

/// Sample the dense spline at fractional residue position `residue_pos`
/// (residue `i` lives at spline index `i * spr`), interpolating linearly
/// between the two bracketing dense samples. Returns `None` if the target
/// index falls outside the spline.
fn sample_spline(
    spline: &[Vec3],
    residue_pos: f32,
    spr: usize,
) -> Option<Vec3> {
    let idx = residue_pos * spr as f32;
    let lo = idx.floor() as usize;
    let hi = lo + 1;
    let frac = idx - lo as f32;
    let a = *spline.get(lo)?;
    let b = *spline.get(hi)?;
    Some(a.lerp(b, frac))
}

// NORMAL BLENDING (protein only)

fn compute_final_frames(
    rmf_frames: &[SplinePoint],
    helix_centers: &[Vec3],
    sheet_normals: &[Vec3],
    profiles: &[CrossSectionProfile],
) -> Vec<SplinePoint> {
    let total_spline = rmf_frames.len();
    let mut result: Vec<SplinePoint> = Vec::with_capacity(total_spline);

    for i in 0..total_spline {
        let frame = &rmf_frames[i];
        let profile = &profiles[i];

        let tangent = frame.tangent;
        let rmf_normal = frame.normal;

        // Radial candidate: outward from helix axis, projected perp to
        // tangent. Falls back to rmf_normal when degenerate.
        let radial_normal = if profile.radial_blend > 0.01 {
            let ci = i.min(helix_centers.len().saturating_sub(1));
            let to_surface = frame.pos - helix_centers[ci];
            let radial = (to_surface - tangent * tangent.dot(to_surface))
                .normalize_or_zero();
            if radial.length_squared() > 0.01 {
                radial
            } else {
                rmf_normal
            }
        } else {
            rmf_normal
        };

        // Non-sheet candidate: RMF blended toward radial by radial_blend.
        let non_sheet_candidate = {
            let blended = rmf_normal
                .lerp(radial_normal, profile.radial_blend)
                .normalize_or_zero();
            if blended.length_squared() > 0.01 {
                blended
            } else {
                rmf_normal
            }
        };

        // Sheet candidate: peptide-plane normal projected perp to
        // tangent. Falls back to the non-sheet candidate when
        // degenerate.
        let sheet_n = sheet_normals[i];
        let sheet_candidate = {
            let proj = sheet_n - tangent * sheet_n.dot(tangent);
            if proj.length_squared() > 1e-6 {
                proj.normalize()
            } else {
                non_sheet_candidate
            }
        };

        // Smooth blend between the two candidates via sheet_blend,
        // which `interpolate_profiles` already ramps 0->1 across sheet
        // boundaries. Replaces the old binary `has_sheet` switch that
        // caused one-sample ~90deg flips at every sheet<->non-sheet
        // transition.
        let normal = {
            // The broad-face normal has no geometrically meaningful sign
            // for a flat ribbon, but `propagate_segment_signs` aligns
            // peptide normals to their own strand neighbor, not to the
            // RMF chain. Flip the sheet candidate into the RMF hemisphere
            // so the blend can't pass through zero when the two are
            // opposed.
            let sheet_candidate =
                if non_sheet_candidate.dot(sheet_candidate) < 0.0 {
                    -sheet_candidate
                } else {
                    sheet_candidate
                };
            let blended = non_sheet_candidate
                .lerp(sheet_candidate, profile.sheet_blend)
                .normalize_or_zero();
            if blended.length_squared() > 0.01 {
                blended
            } else {
                non_sheet_candidate
            }
        };

        // The within-sample alignment above keeps the blend well-defined
        // but its branch can toggle on float noise when the RMF normal is
        // ~perpendicular to the sheet candidate, producing isolated
        // single-sample 180deg spikes. The broad-face normal sign is
        // geometrically free, so force each frame into the previous
        // frame's hemisphere: consecutive samples are densely spaced, so
        // a sign opposition between neighbors is always spurious.
        let normal = match result.last() {
            Some(prev) if normal.dot(prev.normal) < 0.0 => -normal,
            _ => normal,
        };

        let binormal = tangent.cross(normal).normalize_or_zero();

        result.push(SplinePoint {
            pos: frame.pos,
            tangent,
            normal,
            binormal,
        });
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    fn profile_with_sheet_blend(sheet_blend: f32) -> CrossSectionProfile {
        CrossSectionProfile {
            width: 1.0,
            thickness: 0.2,
            roundness: 0.0,
            radial_blend: 0.0,
            sheet_blend,
            color: [0.5, 0.5, 0.5],
            residue_idx: 0,
        }
    }

    /// At a strand entry the RMF normal and the peptide-plane sheet normal
    /// can be anti-parallel. Blending across the sheet ramp must not let the
    /// broad-face normal swing through zero into the opposite hemisphere:
    /// consecutive output normals must keep a positive dot product.
    #[test]
    fn sheet_blend_does_not_flip_hemisphere() {
        let n = 5;
        // Straight chain along +Z so every tangent is +Z.
        let rmf_frames: Vec<SplinePoint> = (0..n)
            .map(|i| SplinePoint {
                pos: Vec3::new(0.0, 0.0, i as f32),
                tangent: Vec3::Z,
                normal: Vec3::X,
                binormal: Vec3::Y,
            })
            .collect();
        let helix_centers = vec![Vec3::ZERO; n];
        // Peptide-plane normal anti-parallel to the RMF normal.
        let sheet_normals = vec![-Vec3::X; n];
        // sheet_blend ramps 0 -> 1 across the strand entry.
        let profiles: Vec<CrossSectionProfile> = (0..n)
            .map(|i| profile_with_sheet_blend(i as f32 / (n - 1) as f32))
            .collect();

        let result = compute_final_frames(
            &rmf_frames,
            &helix_centers,
            &sheet_normals,
            &profiles,
        );

        for i in 0..result.len() - 1 {
            let d = result[i].normal.dot(result[i + 1].normal);
            assert!(
                d > 0.0,
                "frame {i}->{}: normal flipped hemisphere (dot = {d}, {:?} -> \
                 {:?})",
                i + 1,
                result[i].normal,
                result[i + 1].normal,
            );
        }
    }

    /// Isolates the sequential hemisphere-coherence step (the
    /// `result.last()` alignment). Inputs are continuous, but the
    /// within-sample T0-A branch toggles its flip across samples (the
    /// RMF normal is ~perpendicular to the sheet candidate), so the
    /// pre-coherence normal sequence is +Y, -Y, +Y. Only the
    /// cross-sample step removes the single-sample 180deg spike; remove it
    /// and this test goes red while every other test stays green.
    #[test]
    fn seq_coherence_fixes_isolated_sign_toggle() {
        let rmf_frames: Vec<SplinePoint> = (0..3)
            .map(|i| SplinePoint {
                pos: Vec3::new(0.0, 0.0, i as f32),
                tangent: Vec3::Z,
                normal: Vec3::X,
                binormal: Vec3::Y,
            })
            .collect();
        let helix_centers = vec![Vec3::ZERO; 3];
        // Sheet normal ~ +Y with a tiny +/-x tilt: its sign vs the RMF
        // normal (X) alternates, toggling the T0-A flip sample to sample.
        let sheet_normals = vec![
            Vec3::new(0.02, 0.9998, 0.0),
            Vec3::new(-0.02, 0.9998, 0.0),
            Vec3::new(0.02, 0.9998, 0.0),
        ];
        let profiles: Vec<CrossSectionProfile> =
            (0..3).map(|_| profile_with_sheet_blend(1.0)).collect();

        let result = compute_final_frames(
            &rmf_frames,
            &helix_centers,
            &sheet_normals,
            &profiles,
        );

        for i in 1..result.len() {
            let d = result[i].normal.dot(result[i - 1].normal);
            assert!(
                d > 0.5,
                "sample {i}: isolated sign toggle not absorbed (dot = {d})",
            );
        }
    }

    /// Isolates T0-A's distinct job: keeping the within-sample blend
    /// pointing the right way. With near-opposed candidates at
    /// `sheet_blend = 0.5`, removing the T0-A flip makes the lerp a
    /// near-zero residual that `normalize_or_zero` rescues into a wild
    /// ~90deg-off direction (~ +Y here). The flip keeps the blended normal
    /// aligned with the intended broad face (~ the RMF/X hemisphere).
    /// Disable the flip and this test goes red while the others stay
    /// green.
    #[test]
    fn toa_blend_avoids_fallback_collapse() {
        let rmf_frames = vec![SplinePoint {
            pos: Vec3::ZERO,
            tangent: Vec3::Z,
            normal: Vec3::X,
            binormal: Vec3::Y,
        }];
        let helix_centers = vec![Vec3::ZERO];
        // ~179deg from the RMF normal: lerp at 0.5 collapses without the
        // flip, stays unit-length with it.
        let sheet_normals = vec![Vec3::new(-0.999_847_7, 0.017_452_4, 0.0)];
        let profiles = vec![profile_with_sheet_blend(0.5)];

        let result = compute_final_frames(
            &rmf_frames,
            &helix_centers,
            &sheet_normals,
            &profiles,
        );

        assert!(
            result[0].normal.dot(Vec3::X) > 0.9,
            "blend swung off the intended broad face (normal = {:?})",
            result[0].normal,
        );
    }
}
