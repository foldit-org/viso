//! Cross-section extrusion + index/end-cap generation, split out of
//! `mesh.rs`. Pure geometry: turns spline frames + profiles into the
//! shared vertex buffer and the tube/ribbon index buffers.

use glam::Vec3;

use super::profile::{cap_offset, extrude_cross_section, CrossSectionProfile};
use super::spline::SplinePoint;
use super::BackboneVertex;

/// Mesh generation parameters that always travel together.
pub(super) struct MeshParams {
    pub(super) base_vertex: u32,
    pub(super) cross_section_verts: usize,
    pub(super) segments_per_residue: usize,
}

/// Extrude cross-sections and generate partitioned indices + end caps.
pub(super) fn extrude_and_index(
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

    // Each of the (total-1) ring-to-ring segments emits 6 indices per
    // cross-section vertex; a segment goes entirely to one buffer, so the
    // per-buffer worst case is the full segment-index count.
    let seg_index_cap = 6 * total.saturating_sub(1) * csv;
    let mut tube_indices = Vec::with_capacity(seg_index_cap);
    let mut ribbon_indices = Vec::with_capacity(seg_index_cap);
    generate_partitioned_indices(
        frames,
        profiles,
        params,
        &mut tube_indices,
        &mut ribbon_indices,
    );
    generate_end_caps(
        frames,
        profiles,
        params,
        &mut vertices,
        &mut tube_indices,
        &mut ribbon_indices,
    );

    (vertices, tube_indices, ribbon_indices)
}

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
    params: &MeshParams,
    vertices: &mut Vec<BackboneVertex>,
    tube_indices: &mut Vec<u32>,
    ribbon_indices: &mut Vec<u32>,
) {
    if frames.len() < 2 {
        return;
    }

    emit_cap(
        &frames[0],
        &profiles[0],
        -frames[0].tangent,
        params,
        vertices,
        tube_indices,
        ribbon_indices,
        false,
    );

    let last = frames.len() - 1;
    emit_cap(
        &frames[last],
        &profiles[last],
        frames[last].tangent,
        params,
        vertices,
        tube_indices,
        ribbon_indices,
        true,
    );
}

#[allow(clippy::too_many_arguments)]
fn emit_cap(
    frame: &SplinePoint,
    profile: &CrossSectionProfile,
    cap_normal: Vec3,
    params: &MeshParams,
    vertices: &mut Vec<BackboneVertex>,
    tube_indices: &mut Vec<u32>,
    ribbon_indices: &mut Vec<u32>,
    forward: bool,
) {
    let is_tube = profile.roundness > 0.5;
    let base_vertex = params.base_vertex;
    let csv = params.cross_section_verts;

    let center_idx = base_vertex + vertices.len() as u32;
    vertices.push(BackboneVertex {
        position: frame.pos.into(),
        normal: cap_normal.into(),
        color: profile.color,
        residue_idx: profile.residue_idx,
        center_pos: (frame.pos - cap_normal).into(),
    });

    let edge_base = base_vertex + vertices.len() as u32;
    for k in 0..csv {
        let offset = cap_offset(frame, profile, csv, k);
        let pos = frame.pos + offset;
        vertices.push(BackboneVertex {
            position: pos.into(),
            normal: cap_normal.into(),
            color: profile.color,
            residue_idx: profile.residue_idx,
            center_pos: (pos - cap_normal).into(),
        });
    }

    let target = if is_tube {
        &mut *tube_indices
    } else {
        &mut *ribbon_indices
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
