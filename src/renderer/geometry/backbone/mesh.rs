//! Backbone mesh generation: ties spline, profile, and sheet modules together
//! into final vertex/index buffers for both protein and nucleic acid chains.

use glam::Vec3;
use molex::SSType;

use super::path::{compute_sheet_geometry, interpolate_per_residue_normals};
use super::profile::{
    cap_offset, extrude_cross_section, interpolate_profiles,
    resolve_na_profile, resolve_profile, CrossSectionProfile,
};
use super::spline::{
    compute_frenet_frames, compute_helix_axis_points, compute_rmf,
    cubic_bspline, dual_hermite_spline, helix_aware_spline, SplinePoint,
};
use super::{BackboneMeshOutput, BackboneVertex};
use crate::options::GeometryOptions;

/// Per-chain index range and bounding sphere for frustum culling.
#[derive(Clone, Debug)]
pub(crate) struct ChainRange {
    pub(crate) tube_index_start: u32,
    pub(crate) tube_index_end: u32,
    pub(crate) ribbon_index_start: u32,
    pub(crate) ribbon_index_end: u32,
    pub(crate) bounding_center: Vec3,
    pub(crate) bounding_radius: f32,
}

/// Mesh generation parameters that always travel together.
struct MeshParams {
    base_vertex: u32,
    cross_section_verts: usize,
    segments_per_residue: usize,
}

/// Mutable mesh generation context: parameters + output buffers.
struct MeshWriter<'a> {
    params: &'a MeshParams,
    vertices: &'a mut Vec<BackboneVertex>,
    tube_indices: &'a mut Vec<u32>,
    ribbon_indices: &'a mut Vec<u32>,
}

/// Default nucleic acid backbone color (light blue-violet).
const NA_COLOR: [f32; 3] = [0.45, 0.55, 0.85];

/// Generate unified backbone mesh from protein and nucleic acid chains.
pub(crate) fn generate_mesh_colored(
    chains: &super::ChainPair,
    ss_override: Option<&[SSType]>,
    per_residue_colors: Option<&[[f32; 3]]>,
    geo: &GeometryOptions,
    per_chain_lod: Option<&[(usize, usize)]>,
    na_residue_colors: Option<&[[f32; 3]]>,
) -> BackboneMeshOutput {
    let (mut out, global_residue_idx) = process_protein_chains(
        chains.protein,
        ss_override,
        per_residue_colors,
        geo,
        per_chain_lod,
    );
    let na_lod = per_chain_lod.and_then(|l| l.get(chains.protein.len()..));
    process_na_chains(
        chains.na,
        geo,
        na_lod,
        &mut out,
        global_residue_idx,
        na_residue_colors,
    );

    out
}

fn process_protein_chains(
    chains: &[crate::renderer::entity_topology::ProteinBackboneChain],
    ss_override: Option<&[SSType]>,
    per_residue_colors: Option<&[[f32; 3]]>,
    geo: &GeometryOptions,
    per_chain_lod: Option<&[(usize, usize)]>,
) -> (BackboneMeshOutput, u32) {
    let mut out = BackboneMeshOutput::default();
    let mut global_residue_idx: u32 = 0;
    let spr = geo.segments_per_residue;
    let csv = geo.cross_section_verts;

    for (chain_idx, atoms) in chains.iter().enumerate() {
        if atoms.ca.len() < 2 {
            global_residue_idx += atoms.ca.len() as u32;
            continue;
        }

        let n_residues = atoms.ca.len();
        let chain_slice = ss_override.and_then(|o| {
            let start = global_residue_idx as usize;
            let end = (start + n_residues).min(o.len());
            (end.saturating_sub(start) == n_residues).then(|| &o[start..end])
        });
        // Engine sync always installs per-entity SS via
        // `Assembly::ss_types`, so every protein chain with ≥ 2 CA atoms
        // has a matching slice. If that invariant is ever violated the
        // chain renders as coil — it doesn't recompute DSSP here.
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

        let (chain_spr, chain_csv) = per_chain_lod
            .and_then(|l| l.get(chain_idx).copied())
            .unwrap_or((spr, csv));

        let params = MeshParams {
            base_vertex: out.vertices.len() as u32,
            cross_section_verts: chain_csv,
            segments_per_residue: chain_spr,
        };
        let (center, radius) = bounding_sphere(&atoms.ca);

        let chain_mesh = generate_protein_chain_mesh(
            atoms,
            &ss_types,
            &profiles,
            global_residue_idx,
            &params,
        );
        out.push_chain(chain_mesh, center, radius);

        global_residue_idx += n_residues as u32;
    }

    (out, global_residue_idx)
}

fn process_na_chains(
    chains: &[Vec<Vec3>],
    geo: &GeometryOptions,
    per_chain_lod: Option<&[(usize, usize)]>,
    out: &mut BackboneMeshOutput,
    mut global_residue_idx: u32,
    na_residue_colors: Option<&[[f32; 3]]>,
) {
    let spr = geo.segments_per_residue;
    let csv = geo.cross_section_verts;

    // Running index into the flat na_residue_colors slice.
    let mut na_residue_offset: usize = 0;

    for (na_idx, chain) in chains.iter().enumerate() {
        if chain.len() < 2 {
            global_residue_idx += chain.len() as u32;
            na_residue_offset += chain.len();
            continue;
        }

        let n_residues = chain.len();
        let profiles: Vec<CrossSectionProfile> = (0..n_residues)
            .map(|i| {
                let color = na_residue_colors
                    .and_then(|c| c.get(na_residue_offset + i).copied())
                    .unwrap_or(NA_COLOR);
                resolve_na_profile(global_residue_idx + i as u32, color, geo)
            })
            .collect();

        let (chain_spr, chain_csv) = per_chain_lod
            .and_then(|l| l.get(na_idx).copied())
            .unwrap_or((spr, csv));

        let params = MeshParams {
            base_vertex: out.vertices.len() as u32,
            cross_section_verts: chain_csv,
            segments_per_residue: chain_spr,
        };
        let (center, radius) = bounding_sphere(chain);
        let chain_mesh = generate_na_chain_mesh(chain, &profiles, &params);
        out.push_chain(chain_mesh, center, radius);

        global_residue_idx += n_residues as u32;
        na_residue_offset += n_residues;
    }
}

/// Compute bounding sphere (centroid + max distance) from a set of positions.
fn bounding_sphere(positions: &[Vec3]) -> (Vec3, f32) {
    if positions.is_empty() {
        return (Vec3::ZERO, 0.0);
    }
    let center =
        positions.iter().copied().sum::<Vec3>() / positions.len() as f32;
    let radius = positions
        .iter()
        .map(|p| (*p - center).length())
        .fold(0.0f32, f32::max);
    (center, radius)
}

// ==================== SHEET ARROW HEADS ====================

/// Maximum number of consecutive non-sheet residues that is still
/// treated as the interior of the same physical strand. Strands are
/// frequently split into several Sheet runs by short classification
/// breaks; an arrowhead belongs only at the strand's true C-terminus,
/// not at every internal break.
const MAX_INTERIOR_GAP: usize = 2;

/// Widen and narrow the C-terminal residues of each physical β-strand to
/// create an arrowhead at the strand→non-strand transition.
///
/// Short interior gaps (≤ [`MAX_INTERIOR_GAP`] non-sheet residues with
/// sheet resuming after) are bridged so the arrowhead is placed once, at
/// the last sheet residue of the strand:
/// - that residue (the arrow point): width → 0.05
/// - the preceding sheet residue (the arrow shoulder): width × 1.5
fn apply_sheet_arrows(
    ss_types: &[SSType],
    profiles: &mut [CrossSectionProfile],
    geo: &GeometryOptions,
) {
    let n = ss_types.len();
    if n == 0 {
        return;
    }

    let is_sheet = |k: usize| ss_types[k] == SSType::Sheet;

    let mut i = 0;
    while i < n {
        if !is_sheet(i) {
            i += 1;
            continue;
        }

        // Walk to the physical strand's C-terminus, stepping across
        // short interior gaps where sheet resumes within
        // MAX_INTERIOR_GAP.
        let strand_start = i;
        let mut arrow_point = i;
        i += 1;
        loop {
            while i < n && is_sheet(i) {
                arrow_point = i;
                i += 1;
            }
            let gap_end = (i + MAX_INTERIOR_GAP).min(n);
            match (i..gap_end).find(|&k| is_sheet(k)) {
                Some(k) => i = k,
                None => break,
            }
        }

        // Shoulder: the sheet residue immediately preceding the arrow
        // point (skipping any interior-gap residues between them).
        if arrow_point > strand_start {
            let mut shoulder = arrow_point - 1;
            while shoulder > strand_start && !is_sheet(shoulder) {
                shoulder -= 1;
            }
            if is_sheet(shoulder) {
                profiles[shoulder].width = geo.sheet_width * 1.5;
            }
        }
        profiles[arrow_point].width = 0.05;
    }
}

// ==================== PROTEIN CHAIN MESH ====================

/// Generate mesh for a single protein chain (with SS detection, sheet
/// geometry, and RMF/radial/sheet normal blending). Takes the SoA
/// backbone-atom view directly from the topology — no interleaved
/// stride shuffling.
fn generate_protein_chain_mesh(
    atoms: &crate::renderer::entity_topology::ProteinBackboneChain,
    ss_types: &[SSType],
    profiles: &[CrossSectionProfile],
    global_residue_base: u32,
    params: &MeshParams,
) -> BackboneMeshOutput {
    let n = atoms.ca.len();
    if n < 2 {
        return BackboneMeshOutput::default();
    }

    let (flat_ca, sheet_normals, sheet_offsets) =
        compute_sheet_geometry(atoms, ss_types, global_residue_base);

    let spr = params.segments_per_residue;
    let spline_points = helix_aware_spline(&flat_ca, ss_types, spr);
    let total = spline_points.len();
    if total < 2 {
        return BackboneMeshOutput::default();
    }

    let tangents = compute_tangents(&spline_points);

    let helix_centers = compute_helix_axis_points(&atoms.ca);
    let spline_helix_centers = cubic_bspline(&helix_centers, spr);

    let mut frames = build_frames(&spline_points, &tangents);
    // Seed the RMF roll from the first residue's peptide-plane normal so
    // the whole chain's roll is fixed by backbone geometry rather than a
    // world axis. `compute_rmf` projects this perpendicular to the first
    // tangent and falls back to an axis only if it is zero.
    if let Some(&seed) = sheet_normals.first() {
        frames[0].normal = seed;
    }
    compute_rmf(&mut frames);

    let spline_sheet_normals =
        interpolate_per_residue_normals(&sheet_normals, total, n);
    let spline_profiles = interpolate_profiles(profiles, total, n);

    let final_frames = compute_final_frames(
        &frames,
        &spline_helix_centers,
        &spline_sheet_normals,
        &spline_profiles,
    );

    if super::sheet_trace::enabled() {
        super::sheet_trace::trace_final_frames(
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

    BackboneMeshOutput {
        vertices: verts,
        tube_indices: tube_inds,
        ribbon_indices: ribbon_inds,
        sheet_offsets,
        ..Default::default()
    }
}

// ==================== NUCLEIC ACID CHAIN MESH ====================

/// Generate mesh for a single NA chain (P-atom positions, Frenet frames,
/// no sheet geometry).
fn generate_na_chain_mesh(
    positions: &[Vec3],
    profiles: &[CrossSectionProfile],
    params: &MeshParams,
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

    let tangents = compute_tangents(&spline_points);

    let mut frames = build_frames(&spline_points, &tangents);
    compute_frenet_frames(&mut frames);

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

// ==================== SHARED HELPERS ====================

/// Compute tangents from spline positions via central differences.
fn compute_tangents(spline: &[Vec3]) -> Vec<Vec3> {
    let n = spline.len();
    (0..n)
        .map(|i| {
            if i == 0 {
                (spline[1] - spline[0]).normalize_or_zero()
            } else if i == n - 1 {
                (spline[i] - spline[i - 1]).normalize_or_zero()
            } else {
                (spline[i + 1] - spline[i - 1]).normalize_or_zero()
            }
        })
        .collect()
}

/// Build SplinePoint shells (position + tangent, normals zeroed).
fn build_frames(spline: &[Vec3], tangents: &[Vec3]) -> Vec<SplinePoint> {
    spline
        .iter()
        .zip(tangents.iter())
        .map(|(&pos, &tangent)| SplinePoint {
            pos,
            tangent,
            normal: Vec3::ZERO,
            binormal: Vec3::ZERO,
        })
        .collect()
}

/// Extrude cross-sections and generate partitioned indices + end caps.
fn extrude_and_index(
    frames: &[SplinePoint],
    profiles: &[CrossSectionProfile],
    params: &MeshParams,
) -> (Vec<BackboneVertex>, Vec<u32>, Vec<u32>) {
    let csv = params.cross_section_verts;
    let total = frames.len();
    let mut vertices = Vec::with_capacity(total * csv);
    for (i, frame) in frames.iter().enumerate() {
        extrude_cross_section(frame, &profiles[i], csv, &mut vertices);
    }

    let mut tube_indices = Vec::new();
    let mut ribbon_indices = Vec::new();
    generate_partitioned_indices(
        frames,
        profiles,
        params,
        &mut tube_indices,
        &mut ribbon_indices,
    );
    let mut writer = MeshWriter {
        params,
        vertices: &mut vertices,
        tube_indices: &mut tube_indices,
        ribbon_indices: &mut ribbon_indices,
    };
    generate_end_caps(frames, profiles, &mut writer);

    (vertices, tube_indices, ribbon_indices)
}

// ==================== NORMAL BLENDING (protein only) ====================

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
        // which `interpolate_profiles` already ramps 0→1 across sheet
        // boundaries. Replaces the old binary `has_sheet` switch that
        // caused one-sample ~90° flips at every sheet↔non-sheet
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
        // single-sample 180° spikes. The broad-face normal sign is
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

// ==================== INDEX GENERATION ====================

fn generate_partitioned_indices(
    frames: &[SplinePoint],
    profiles: &[CrossSectionProfile],
    params: &MeshParams,
    tube_indices: &mut Vec<u32>,
    ribbon_indices: &mut Vec<u32>,
) {
    if frames.len() < 2 {
        return;
    }

    let base_vertex = params.base_vertex;
    let csv = params.cross_section_verts;

    for i in 0..frames.len() - 1 {
        let is_tube =
            profiles[i].roundness > 0.5 && profiles[i + 1].roundness > 0.5;

        let ring_a = base_vertex + (i * csv) as u32;
        let ring_b = base_vertex + ((i + 1) * csv) as u32;

        for k in 0..csv {
            let k_next = (k + 1) % csv;
            let v0 = ring_a + k as u32;
            let v1 = ring_a + k_next as u32;
            let v2 = ring_b + k as u32;
            let v3 = ring_b + k_next as u32;
            let target = if is_tube {
                &mut *tube_indices
            } else {
                &mut *ribbon_indices
            };
            target.extend_from_slice(&[v0, v2, v1]);
            target.extend_from_slice(&[v1, v2, v3]);
        }
    }
}

fn generate_end_caps(
    frames: &[SplinePoint],
    profiles: &[CrossSectionProfile],
    w: &mut MeshWriter,
) {
    if frames.len() < 2 {
        return;
    }

    emit_cap(&frames[0], &profiles[0], -frames[0].tangent, w, false);

    let last = frames.len() - 1;
    emit_cap(
        &frames[last],
        &profiles[last],
        frames[last].tangent,
        w,
        true,
    );
}

fn emit_cap(
    frame: &SplinePoint,
    profile: &CrossSectionProfile,
    cap_normal: Vec3,
    w: &mut MeshWriter,
    forward: bool,
) {
    let is_tube = profile.roundness > 0.5;
    let base_vertex = w.params.base_vertex;
    let csv = w.params.cross_section_verts;

    let center_idx = base_vertex + w.vertices.len() as u32;
    w.vertices.push(BackboneVertex {
        position: frame.pos.into(),
        normal: cap_normal.into(),
        color: profile.color,
        residue_idx: profile.residue_idx,
        center_pos: (frame.pos - cap_normal).into(),
    });

    let edge_base = base_vertex + w.vertices.len() as u32;
    for k in 0..csv {
        let offset = cap_offset(frame, profile, csv, k);
        let pos = frame.pos + offset;
        w.vertices.push(BackboneVertex {
            position: pos.into(),
            normal: cap_normal.into(),
            color: profile.color,
            residue_idx: profile.residue_idx,
            center_pos: (pos - cap_normal).into(),
        });
    }

    let target = if is_tube {
        &mut *w.tube_indices
    } else {
        &mut *w.ribbon_indices
    };
    for k in 0..csv {
        let k_next = (k + 1) % csv;
        if forward {
            target.extend_from_slice(&[
                center_idx,
                edge_base + k as u32,
                edge_base + k_next as u32,
            ]);
        } else {
            target.extend_from_slice(&[
                center_idx,
                edge_base + k_next as u32,
                edge_base + k as u32,
            ]);
        }
    }
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

    fn sheet_profiles(
        geo: &GeometryOptions,
        n: usize,
    ) -> Vec<CrossSectionProfile> {
        (0..n)
            .map(|_| CrossSectionProfile {
                width: geo.sheet_width,
                thickness: geo.sheet_thickness,
                roundness: geo.sheet_roundness,
                radial_blend: 0.0,
                sheet_blend: 1.0,
                color: [0.5, 0.5, 0.5],
                residue_idx: 0,
            })
            .collect()
    }

    /// A short interior break that splits one physical strand into two
    /// Sheet runs must not produce a mid-strand arrowhead — the narrowing
    /// belongs only at the strand's true C-terminus.
    #[test]
    fn sheet_arrow_bridges_interior_gap() {
        use molex::SSType::{Coil, Sheet};
        let geo = GeometryOptions::default();
        // strand spans residues 1..=7, interior gap (Coil) at residue 4.
        let ss = [
            Coil, Sheet, Sheet, Sheet, Coil, Sheet, Sheet, Sheet, Coil, Coil,
            Coil,
        ];
        let mut profiles = sheet_profiles(&geo, ss.len());
        apply_sheet_arrows(&ss, &mut profiles, &geo);

        // Narrow point at the true end, widened shoulder just before it.
        assert_eq!(profiles[7].width, 0.05, "arrow point at C-terminus");
        assert_eq!(
            profiles[6].width,
            geo.sheet_width * 1.5,
            "shoulder before the arrow point",
        );
        // The interior run-end (residue 3) must be untouched.
        assert_eq!(
            profiles[3].width, geo.sheet_width,
            "no mid-strand narrowing at the interior break",
        );
        assert_eq!(profiles[2].width, geo.sheet_width);
    }

    /// A genuine strand separation (long non-sheet stretch) still gets an
    /// arrowhead per strand.
    #[test]
    fn sheet_arrow_separates_distinct_strands() {
        use molex::SSType::{Coil, Sheet};
        let geo = GeometryOptions::default();
        let ss = [Sheet, Sheet, Sheet, Coil, Coil, Coil, Sheet, Sheet, Sheet];
        let mut profiles = sheet_profiles(&geo, ss.len());
        apply_sheet_arrows(&ss, &mut profiles, &geo);

        assert_eq!(profiles[2].width, 0.05);
        assert_eq!(profiles[1].width, geo.sheet_width * 1.5);
        assert_eq!(profiles[8].width, 0.05);
        assert_eq!(profiles[7].width, geo.sheet_width * 1.5);
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
    /// cross-sample step removes the single-sample 180° spike; remove it
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
        // Sheet normal ≈ +Y with a tiny ±x tilt: its sign vs the RMF
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
    /// ~90°-off direction (≈ +Y here). The flip keeps the blended normal
    /// aligned with the intended broad face (≈ the RMF/X hemisphere).
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
        // ~179° from the RMF normal: lerp at 0.5 collapses without the
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

    /// A non-sheet gap longer than `MAX_INTERIOR_GAP` is a real strand
    /// break: each side is its own strand and gets its own arrowhead.
    #[test]
    fn sheet_arrow_splits_on_wide_gap() {
        use molex::SSType::{Coil, Sheet};
        let geo = GeometryOptions::default();
        // Gap of MAX_INTERIOR_GAP + 1 Coil residues between two strands.
        let ss = [Sheet, Sheet, Sheet, Coil, Coil, Coil, Sheet, Sheet, Sheet];
        assert!(ss[3..6].len() > MAX_INTERIOR_GAP);
        let mut profiles = sheet_profiles(&geo, ss.len());
        apply_sheet_arrows(&ss, &mut profiles, &geo);

        assert_eq!(profiles[2].width, 0.05);
        assert_eq!(profiles[8].width, 0.05);
        assert_eq!(profiles[5].width, geo.sheet_width);
    }

    /// A single-residue strand has no room for a shoulder: only the
    /// arrow point is narrowed, and indexing must not underflow.
    #[test]
    fn sheet_arrow_single_residue_strand() {
        use molex::SSType::{Coil, Sheet};
        let geo = GeometryOptions::default();
        let ss = [Coil, Sheet, Coil];
        let mut profiles = sheet_profiles(&geo, ss.len());
        apply_sheet_arrows(&ss, &mut profiles, &geo);

        assert_eq!(profiles[1].width, 0.05);
        assert_eq!(profiles[0].width, geo.sheet_width);
        assert_eq!(profiles[2].width, geo.sheet_width);
    }
}
