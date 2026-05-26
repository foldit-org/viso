use super::prepared::{
    BackboneMeshData, BallAndStickInstances, CachedEntityMesh,
    NucleicAcidInstances, PreparedRebuild,
};
use crate::renderer::geometry::backbone::{ChainRange, SheetOffset};
use crate::renderer::picking::PickMap;

/// Offset the `residue_idx` field embedded in raw vertex bytes.
///
/// `BackboneVertex` and `NaVertex` share the same 52-byte
/// layout with a `u32 residue_idx` at byte offset 36. When concatenating
/// vertices from multiple entities, each entity's local residue indices
/// must be shifted by the global offset so the GPU's per-residue color
/// buffer is indexed correctly.
pub(crate) fn offset_vertex_residue_idx(
    dst: &mut Vec<u8>,
    src: &[u8],
    offset: u32,
) {
    const VERTEX_SIZE: usize = 52;
    const RESIDUE_IDX_OFFSET: usize = 36;

    if offset == 0 {
        dst.extend_from_slice(src);
        return;
    }

    let start = dst.len();
    dst.extend_from_slice(src);

    // Patch each vertex's residue_idx in-place
    let mut pos = start + RESIDUE_IDX_OFFSET;
    while pos + 4 <= dst.len() {
        let mut bytes = [0u8; 4];
        bytes.copy_from_slice(&dst[pos..pos + 4]);
        let patched = u32::from_ne_bytes(bytes) + offset;
        dst[pos..pos + 4].copy_from_slice(&patched.to_ne_bytes());
        pos += VERTEX_SIZE;
    }
}

/// Layout descriptor for patching pick IDs in BnS byte buffers.
struct BnsPickPatch {
    old_offset: u32,
    new_offset: u32,
    instance_stride: usize,
    pick_id_byte_offset: usize,
}

/// Patch pick IDs in BnS instance byte buffers.
///
/// Each instance has a pick ID stored as `f32` at `pick_id_byte_offset`
/// within each `instance_stride`-sized block. Adds `(new_offset - old_offset)`
/// to each pick ID so that 0-based local atom indices become globally unique.
fn offset_bns_pick_ids(dst: &mut Vec<u8>, src: &[u8], patch: &BnsPickPatch) {
    if patch.old_offset == patch.new_offset {
        dst.extend_from_slice(src);
        return;
    }

    let start = dst.len();
    dst.extend_from_slice(src);

    let delta = patch.new_offset as f32 - patch.old_offset as f32;
    let mut pos = start + patch.pick_id_byte_offset;
    while pos + 4 <= dst.len() {
        let mut bytes = [0u8; 4];
        bytes.copy_from_slice(&dst[pos..pos + 4]);
        let old_val = f32::from_ne_bytes(bytes);
        let patched = old_val + delta;
        dst[pos..pos + 4].copy_from_slice(&patched.to_ne_bytes());
        pos += patch.instance_stride;
    }
}

/// Accumulator that merges per-entity cached meshes into combined buffers.
#[derive(Default)]
struct MeshAccumulator {
    // Backbone
    backbone_verts: Vec<u8>,
    backbone_tube_inds: Vec<u32>,
    backbone_ribbon_inds: Vec<u32>,
    backbone_vert_offset: u32,
    sheet_offsets: Vec<SheetOffset>,
    chain_ranges: Vec<ChainRange>,
    // Sidechain instances
    sidechain_bytes: Vec<u8>,
    sidechain_count: u32,
    // Ball-and-stick
    bns_spheres: Vec<u8>,
    bns_sphere_count: u32,
    bns_capsules: Vec<u8>,
    bns_capsule_count: u32,
    bns_pick_offset: u32,
    // Nucleic acid instances
    na_stem_bytes: Vec<u8>,
    na_stem_count: u32,
    na_ring_bytes: Vec<u8>,
    na_ring_count: u32,
    /// Pick-map entries per entity that produced BnS content:
    /// `(entity_id, atom_count)`.
    bns_pick_entities: Vec<(u32, u32)>,
    // Residue tracking
    residue_offset: u32,
}

impl MeshAccumulator {
    fn push_entity(&mut self, mesh: &CachedEntityMesh) {
        self.push_backbone(mesh);

        // Sidechain instances (self-contained)
        self.sidechain_bytes
            .extend_from_slice(&mesh.sidechain_instances);
        self.sidechain_count += mesh.sidechain_instance_count;

        self.push_bns(mesh);

        // NA instances (self-contained)
        self.na_stem_bytes
            .extend_from_slice(&mesh.na.stem_instances);
        self.na_stem_count += mesh.na.stem_count;
        self.na_ring_bytes
            .extend_from_slice(&mesh.na.ring_instances);
        self.na_ring_count += mesh.na.ring_count;

        if mesh.bns_atom_count > 0 {
            self.bns_pick_entities
                .push((mesh.entity_id, mesh.bns_atom_count));
        }

        self.residue_offset += mesh.residue_count;
    }

    fn push_backbone(&mut self, mesh: &CachedEntityMesh) {
        offset_vertex_residue_idx(
            &mut self.backbone_verts,
            &mesh.backbone.verts,
            self.residue_offset,
        );
        for &idx in &mesh.backbone.tube_inds {
            self.backbone_tube_inds
                .push(idx + self.backbone_vert_offset);
        }
        for &idx in &mesh.backbone.ribbon_inds {
            self.backbone_ribbon_inds
                .push(idx + self.backbone_vert_offset);
        }
        for so in &mesh.backbone.sheet_offsets {
            self.sheet_offsets.push(SheetOffset {
                residue_idx: so.residue_idx + self.residue_offset,
                offset: so.offset,
            });
        }
        let tube_idx_offset = self.backbone_tube_inds.len() as u32
            - mesh.backbone.tube_inds.len() as u32;
        let ribbon_idx_offset = self.backbone_ribbon_inds.len() as u32
            - mesh.backbone.ribbon_inds.len() as u32;
        for r in &mesh.backbone.chain_ranges {
            let tube = r.tube();
            let ribbon = r.ribbon();
            self.chain_ranges.push(ChainRange::new(
                tube.start + tube_idx_offset..tube.end + tube_idx_offset,
                ribbon.start + ribbon_idx_offset
                    ..ribbon.end + ribbon_idx_offset,
                r.bounding_center,
                r.bounding_radius,
            ));
        }
        self.backbone_vert_offset += mesh.backbone.vert_count;
    }

    fn push_bns(&mut self, mesh: &CachedEntityMesh) {
        // SphereInstance: 32 bytes, pick_id at byte 28 (color.w)
        offset_bns_pick_ids(
            &mut self.bns_spheres,
            &mesh.bns.sphere_instances,
            &BnsPickPatch {
                old_offset: 0,
                new_offset: self.bns_pick_offset,
                instance_stride: 32,
                pick_id_byte_offset: 28,
            },
        );
        self.bns_sphere_count += mesh.bns.sphere_count;
        // CapsuleInstance: 64 bytes, pick_id at byte 28 (endpoint_b.w)
        offset_bns_pick_ids(
            &mut self.bns_capsules,
            &mesh.bns.capsule_instances,
            &BnsPickPatch {
                old_offset: 0,
                new_offset: self.bns_pick_offset,
                instance_stride: 64,
                pick_id_byte_offset: 28,
            },
        );
        self.bns_capsule_count += mesh.bns.capsule_count;
        self.bns_pick_offset += mesh.bns_atom_count;
    }

    /// Shift all BnS pick IDs by the global residue offset so they sit
    /// after backbone/sidechain residue IDs in the pick map.
    fn finalize_bns_pick_ids(&mut self) {
        if self.residue_offset == 0 {
            return;
        }
        let delta = self.residue_offset as f32;
        patch_pick_id_buffer(&mut self.bns_spheres, delta, 32, 28);
        patch_pick_id_buffer(&mut self.bns_capsules, delta, 64, 28);
    }

    fn build_pick_map(&self) -> PickMap {
        let mut atom_entries: Vec<(u32, u32)> = Vec::new();
        for &(entity_id, atom_count) in &self.bns_pick_entities {
            for atom_idx in 0..atom_count {
                atom_entries.push((entity_id, atom_idx));
            }
        }
        PickMap::new(self.residue_offset, atom_entries)
    }

    fn into_prepared_rebuild(mut self) -> PreparedRebuild {
        self.finalize_bns_pick_ids();
        let pick_map = self.build_pick_map();
        let tube_index_count = self.backbone_tube_inds.len() as u32;
        let ribbon_index_count = self.backbone_ribbon_inds.len() as u32;
        PreparedRebuild {
            generation: 0,
            backbone: BackboneMeshData {
                vertices: self.backbone_verts,
                tube_indices: bytemuck::cast_slice(&self.backbone_tube_inds)
                    .to_vec(),
                tube_index_count,
                ribbon_indices: bytemuck::cast_slice(
                    &self.backbone_ribbon_inds,
                )
                .to_vec(),
                ribbon_index_count,
                sheet_offsets: self.sheet_offsets,
                chain_ranges: self.chain_ranges,
            },
            sidechain_instances: self.sidechain_bytes,
            sidechain_instance_count: self.sidechain_count,
            bns: BallAndStickInstances {
                sphere_instances: self.bns_spheres,
                sphere_count: self.bns_sphere_count,
                capsule_instances: self.bns_capsules,
                capsule_count: self.bns_capsule_count,
            },
            na: NucleicAcidInstances {
                stem_instances: self.na_stem_bytes,
                stem_count: self.na_stem_count,
                ring_instances: self.na_ring_bytes,
                ring_count: self.na_ring_count,
            },
            pick_map,
        }
    }
}

/// Patch f32 pick IDs in a raw byte buffer by adding `delta`.
fn patch_pick_id_buffer(
    buf: &mut [u8],
    delta: f32,
    stride: usize,
    offset: usize,
) {
    let mut pos = offset;
    while pos + 4 <= buf.len() {
        let mut bytes = [0u8; 4];
        bytes.copy_from_slice(&buf[pos..pos + 4]);
        let patched = f32::from_ne_bytes(bytes) + delta;
        buf[pos..pos + 4].copy_from_slice(&patched.to_ne_bytes());
        pos += stride;
    }
}

/// Concatenate per-entity cached meshes into a single `PreparedRebuild`.
pub(crate) fn concatenate_meshes(
    entity_meshes: &[&CachedEntityMesh],
) -> PreparedRebuild {
    let mut acc = MeshAccumulator::default();
    for mesh in entity_meshes {
        acc.push_entity(mesh);
    }
    acc.into_prepared_rebuild()
}
