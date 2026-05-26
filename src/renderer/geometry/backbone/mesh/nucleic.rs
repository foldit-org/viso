//! Per-chain nucleic-acid backbone mesh: P-atom spline, rotation-
//! minimizing frames, and Mol*-faithful sugar-guide orientation.

use glam::Vec3;

use super::super::index::{extrude_and_index, MeshParams};
use super::super::path::interpolate_per_residue_normals;
use super::super::profile::{interpolate_profiles, CrossSectionProfile};
use super::super::spline::{
    build_traces, dual_hermite_spline, rmf_frames, SplinePoint,
};
use super::super::BackboneMeshOutput;
use crate::util::geom::central_difference_tangents;

/// Generate mesh for a single NA chain (P-atom positions, rotation-
/// minimizing frames, no sheet geometry).
///
/// `seed` is the chain-roll seed -- the first base ring's plane normal,
/// the nucleic-acid analogue of the protein peptide-plane seed.
/// [`rmf_frames`] projects it perpendicular to the first tangent and
/// carries it along the chain with no per-sample axis reset and no
/// inflection flip (the two compounding instabilities of the prior raw-
/// Frenet path), giving a smooth, stable *fallback* roll.
///
/// The ribbon's broad face is then oriented along the per-residue
/// **backbone->sugar guide vector `C1' - P`**, projected perpendicular
/// to the tangent -- the orientation convention Mol*, ChimeraX and
/// PyMOL all converge on (ChimeraX uses `C1' - C5'`; viso's trace is
/// the canonical P). The guide is interpolated to spline resolution and
/// the RMF frame is rotated onto it with sequential sign coherence so
/// the flat ribbon faces the bases without per-nucleotide flipping. RMF
/// supplies the fallback roll only for residues with no resolvable C1'.
pub(super) fn generate_na_chain_mesh(
    positions: &[Vec3],
    profiles: &[CrossSectionProfile],
    params: &MeshParams,
    seed: Option<Vec3>,
    guide_dirs: &[Vec3],
) -> BackboneMeshOutput {
    let n = positions.len();
    if n < 2 {
        return BackboneMeshOutput::default();
    }

    let spr = params.segments_per_residue;
    let spline_points = dual_hermite_spline(positions, spr);
    let total = spline_points.len();
    if total < 2 {
        return BackboneMeshOutput::default();
    }

    let tangents = central_difference_tangents(&spline_points);

    let traces = build_traces(&spline_points, &tangents);
    let mut frames = rmf_frames(&traces, seed);

    // Mol*-faithful orientation: neighbour-smooth the per-residue
    // direction vectors (`setDirection`), interpolate to spline
    // resolution, then orient the ribbon's broad face along that
    // direction (Mol*'s pre-swap normal; the swap is a Mol*-builder
    // quirk -- see `orient_frames_to_guide`).
    if guide_dirs.len() == n {
        let smoothed = smooth_directions(guide_dirs);
        let spline_guides =
            interpolate_per_residue_normals(&smoothed, total, n);
        orient_frames_to_guide(&mut frames, &spline_guides);
    }

    let spline_profiles = interpolate_profiles(profiles, total, n);
    let (verts, tube_inds, ribbon_inds) =
        extrude_and_index(&frames, &spline_profiles, params);

    BackboneMeshOutput {
        vertices: verts,
        tube_indices: tube_inds,
        ribbon_indices: ribbon_inds,
        ..Default::default()
    }
}

/// Mol* `setDirection` neighbour smoothing: each per-residue direction
/// is replaced by `(matchDir(d_prev,d_cur) + matchDir(d_next,d_cur) +
/// 2*d_cur) / 4`, where `matchDir(v, ref)` flips `v` if it points
/// opposite `ref`. This is the sign-coherence + low-pass that keeps the
/// ribbon from flipping between consecutive nucleotides whose raw
/// `pos(to)-pos(from)` vectors differ in sign. Endpoints reuse the
/// current vector for the missing neighbour (Mol* iterator clamps).
fn smooth_directions(dirs: &[Vec3]) -> Vec<Vec3> {
    let n = dirs.len();
    let match_dir = |v: Vec3, r: Vec3| if v.dot(r) < 0.0 { -v } else { v };
    (0..n)
        .map(|i| {
            let cur = dirs[i];
            let prev = if i == 0 { cur } else { dirs[i - 1] };
            let next = if i + 1 == n { cur } else { dirs[i + 1] };
            (match_dir(prev, cur) + match_dir(next, cur) + 2.0 * cur) / 4.0
        })
        .collect()
}

/// Orient the ribbon's broad face **along** the per-sample direction
/// vector, projected perpendicular to the tangent -- Mol*'s pre-swap
/// `normalVec = orthogonalize(tangent, dir)`.
///
/// Mol* additionally swaps normal<->binormal and negates for NA, but
/// that compensates for *Mol*'s* `addSheet` builder assigning the broad
/// face to its binormal axis. viso's
/// [`extrude_cross_section`](super::super::profile::extrude_cross_section)
/// puts width along `binormal` and thickness along `normal`, so for the
/// flat NA ribbon the broad-face normal *is* `frame.normal` -- porting
/// Mol*'s swap on top would rotate the face 90deg twice. So we feed the
/// pre-swap direction-aligned normal straight in (this also equals
/// ChimeraX's `orthogonal_component(C1'-C5', tangent)`). Sequential
/// sign coherence keeps densely-spaced samples from flipping 180deg; a
/// sample with no usable direction keeps its smooth RMF normal.
fn orient_frames_to_guide(frames: &mut [SplinePoint], guides: &[Vec3]) {
    let mut prev_normal: Option<Vec3> = None;
    for (f, &g) in frames.iter_mut().zip(guides) {
        let t = f.tangent;
        let proj = g - t * t.dot(g);
        let mut normal = if proj.length_squared() > 1e-6 {
            proj.normalize()
        } else {
            f.normal
        };
        if let Some(p) = prev_normal {
            if normal.dot(p) < 0.0 {
                normal = -normal;
            }
        }
        prev_normal = Some(normal);
        f.normal = normal;
        f.binormal = t.cross(normal).normalize_or_zero();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Mol* `setDirection`: averaging a residue with sign-alternating
    /// neighbours must not collapse to ~zero -- `matchDirection` flips
    /// opposed neighbours before the (1,2,1)/4 blend, so magnitude is
    /// preserved and the per-residue sign is kept.
    #[test]
    fn smooth_directions_does_not_cancel_opposed_neighbours() {
        let dirs = vec![Vec3::X, -Vec3::X, Vec3::X, -Vec3::X];
        let out = smooth_directions(&dirs);
        for (i, v) in out.iter().enumerate() {
            assert!(
                (v.length() - 1.0).abs() < 1e-5,
                "dir {i}: collapsed under smoothing ({v:?})"
            );
            assert!(
                v.x.abs() > 0.98,
                "dir {i}: lost its axis under smoothing ({v:?})"
            );
        }
    }

    /// The ribbon broad face is oriented **along** the per-sample
    /// direction vector, tangent-projected (Mol*'s pre-swap normal /
    /// ChimeraX's `orthogonal_component`), so a +/-X direction along a
    /// +Z tangent yields a +/-X broad-face normal. Orthonormal and
    /// sign-coherent across samples even when the raw direction sign
    /// alternates; a zero direction keeps the RMF normal (seeded here on
    /// +Y, a distinct axis).
    #[test]
    fn na_ribbon_normal_follows_sugar_guide() {
        let mut frames: Vec<SplinePoint> = (0..6)
            .map(|i| SplinePoint {
                pos: Vec3::new(0.0, 0.0, i as f32),
                tangent: Vec3::Z,
                normal: Vec3::Y, // RMF fallback axis (distinct from +/-X)
                binormal: Vec3::X,
            })
            .collect();
        // Direction ~ +X with a tilt and alternating sign (raw C3'->C1'
        // is not chirality-stable); last is degenerate (no atom).
        let guides = vec![
            Vec3::new(0.99, 0.0, 0.14),
            Vec3::new(-0.99, 0.0, 0.14),
            Vec3::new(0.99, 0.0, -0.14),
            Vec3::new(-1.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::ZERO,
        ];
        orient_frames_to_guide(&mut frames, &guides);

        for (i, f) in frames.iter().enumerate() {
            assert!(
                f.tangent.dot(f.normal).abs() < 1e-4
                    && (f.normal.length() - 1.0).abs() < 1e-4
                    && f.normal.dot(f.binormal).abs() < 1e-4,
                "frame {i}: not orthonormal / perpendicular to tangent"
            );
        }
        for (i, f) in frames.iter().enumerate().take(5) {
            // Broad face along the +/-X direction.
            assert!(
                f.normal.x.abs() > 0.98,
                "frame {i}: ribbon face not along the direction vector ({:?})",
                f.normal,
            );
        }
        for i in 1..5 {
            assert!(
                frames[i].normal.dot(frames[i - 1].normal) > 0.0,
                "frame {i}: ribbon flipped hemisphere between samples"
            );
        }
        // Degenerate last sample fell back to the RMF normal (+Y axis).
        assert!(
            frames[5].normal.y.abs() > 0.98,
            "degenerate sample did not fall back to the RMF normal ({:?})",
            frames[5].normal,
        );
    }
}
