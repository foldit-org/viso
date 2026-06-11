//! GPU-based object picking and selection management.
//!
//! Renders residue indices to an offscreen buffer and reads back the pixel
//! under the cursor to determine which residue was clicked or hovered.

mod pick_map;
mod pipeline;
pub(crate) mod state;

use std::collections::{BTreeMap, BTreeSet};

use molex::entity::molecule::id::EntityId;
use molex::SSType;
pub(crate) use pick_map::PickMap;
pub use pick_map::PickTarget;
pub(crate) use pipeline::{Picking, PickingGeometry, SelectionBuffer};

use self::state::PickingState;
use super::Renderers;
use crate::gpu::residue_color::ResidueColorBuffer;
use crate::gpu::{RenderContext, ShaderComposer};
use crate::renderer::entity_topology::ProteinBackboneChain;

/// GPU picking, selection, and per-residue color buffers grouped together.
pub(crate) struct PickingSystem {
    /// The GPU picking pipeline.
    pub(crate) picking: Picking,
    /// Picking bind group state (capsule, ball-and-stick).
    pub(crate) groups: PickingState,
    /// Per-residue selection highlight buffer.
    pub(crate) selection: SelectionBuffer,
    /// Per-residue color buffer for shaders.
    pub(crate) residue_colors: ResidueColorBuffer,
    /// Current pick map (maps raw IDs to typed targets).
    pub(crate) pick_map: Option<PickMap>,
    /// Last resolved hover target.
    pub(crate) hovered_target: PickTarget,
    /// Each entity's first global residue index in the GPU selection /
    /// per-residue color space. Refreshed on every `upload_prepared`.
    pub(crate) entity_residue_offsets: BTreeMap<EntityId, u32>,
    /// Per-entity residue selection: the source of truth. The flat GPU
    /// bitset (`picking.selected_residues`) is a derived cache, re-derived
    /// from this map against the current `entity_residue_offsets` on every
    /// selection mutation AND on every mesh rebuild. Keeping the per-entity
    /// selection here (rather than a pre-flattened vector) makes staleness
    /// relative to a shifting residue space structurally impossible.
    pub(crate) selection_per_entity: BTreeMap<EntityId, BTreeSet<u32>>,
}

impl PickingSystem {
    /// Create the picking system from pre-built selection and residue-color
    /// buffers.
    pub(crate) fn new(
        context: &RenderContext,
        camera_layout: &wgpu::BindGroupLayout,
        selection: SelectionBuffer,
        residue_colors: ResidueColorBuffer,
        shader_composer: &mut ShaderComposer,
    ) -> Result<Self, crate::error::VisoError> {
        let picking = Picking::new(context, camera_layout, shader_composer)?;
        Ok(Self {
            picking,
            groups: PickingState::new(),
            selection,
            residue_colors,
            pick_map: None,
            hovered_target: PickTarget::None,
            entity_residue_offsets: BTreeMap::new(),
            selection_per_entity: BTreeMap::new(),
        })
    }

    /// Poll the GPU for a completed picking readback and resolve the raw ID
    /// to a typed [`PickTarget`].
    ///
    /// Non-blocking: returns immediately if no readback data is ready yet.
    /// When data is available, the internal `hovered_target` is updated.
    pub(crate) fn poll_and_resolve(&mut self, device: &wgpu::Device) {
        if let Some(raw_id) = self.picking.complete_readback(device) {
            self.hovered_target = self
                .pick_map
                .as_ref()
                .map_or(PickTarget::None, |pm| pm.resolve(raw_id));
        }
    }

    /// Clear the residue selection. Returns `true` if the selection was
    /// non-empty (i.e. it actually changed).
    pub(crate) fn clear_selection(&mut self) -> bool {
        if self.selection_per_entity.is_empty() {
            false
        } else {
            self.selection_per_entity.clear();
            self.picking.selected_residues.clear();
            true
        }
    }

    /// Replace the per-entity residue selection (the source of truth) and
    /// re-derive the flat GPU bitset cache against the current offsets.
    /// The derived cache is uploaded by the per-frame
    /// [`Self::update_selection_buffer`] call.
    pub(crate) fn set_selection(
        &mut self,
        selection: BTreeMap<EntityId, BTreeSet<u32>>,
    ) {
        self.selection_per_entity = selection;
        self.rederive_selection();
    }

    /// Re-derive the flat `picking.selected_residues` cache from the stored
    /// per-entity selection against the CURRENT `entity_residue_offsets`.
    /// Called on every selection mutation AND on every mesh rebuild (when
    /// the offsets table is overwritten), so the flat bitset can never go
    /// stale relative to a shifting residue space. Per-entity entries with
    /// no published offset (e.g. an entity not yet meshed) contribute
    /// nothing.
    pub(crate) fn rederive_selection(&mut self) {
        let mut flat: Vec<i32> = Vec::new();
        for (eid, residues) in &self.selection_per_entity {
            let Some(&base) = self.entity_residue_offsets.get(eid) else {
                continue;
            };
            for r in residues {
                flat.push((base + *r) as i32);
            }
        }
        self.picking.selected_residues = flat;
    }

    /// Walk backwards / forwards from `residue_idx` to find the
    /// contiguous run of residues with the same secondary structure.
    /// Returns the half-open range `[start, end+1)` of flat residue
    /// indices, or `None` if `residue_idx` is out of bounds.
    fn segment_range(
        residue_idx: i32,
        ss_types: &[SSType],
    ) -> Option<std::ops::Range<usize>> {
        if residue_idx < 0 || (residue_idx as usize) >= ss_types.len() {
            return None;
        }
        let idx = residue_idx as usize;
        let target_ss = ss_types[idx];

        let mut start = idx;
        while start > 0 && ss_types[start - 1] == target_ss {
            start -= 1;
        }

        let mut end = idx;
        while end + 1 < ss_types.len() && ss_types[end + 1] == target_ss {
            end += 1;
        }

        Some(start..end + 1)
    }

    /// Find the chain that contains `residue_idx` and return its
    /// half-open range of flat residue indices, or `None` if
    /// `residue_idx` falls outside every chain.
    fn chain_range(
        residue_idx: i32,
        backbone_chains: &[ProteinBackboneChain],
    ) -> Option<std::ops::Range<usize>> {
        if residue_idx < 0 {
            return None;
        }
        let target = residue_idx as usize;
        let mut global_start = 0usize;
        backbone_chains.iter().find_map(|chain| {
            let chain_residues = chain.ca().len();
            let global_end = global_start + chain_residues;
            let result = (target >= global_start && target < global_end)
                .then_some(global_start..global_end);
            global_start = global_end;
            result
        })
    }

    /// Map a flat residue index back to `(EntityId, residue_in_entity)`
    /// by walking the per-entity offsets table. Returns `None` when
    /// the offsets table is empty (before the first full rebuild) or
    /// when `flat` falls outside every entity's residue span.
    ///
    /// The offsets are stored in `BTreeMap<EntityId, u32>` keyed by
    /// `EntityId`; entity order in the GPU residue space is the
    /// insertion order produced by mesh concat (assembly order),
    /// which does not necessarily match `EntityId` order. We sort
    /// (offset, entity) pairs here and find the last entity whose
    /// offset is `<= flat`.
    pub(crate) fn flat_to_entity_residue(
        &self,
        flat: u32,
    ) -> Option<(EntityId, u32)> {
        let mut entries: Vec<(u32, EntityId)> = self
            .entity_residue_offsets
            .iter()
            .map(|(eid, off)| (*off, *eid))
            .collect();
        entries.sort_by_key(|(off, _)| *off);
        // Find the last entry whose offset is <= flat.
        let mut owner: Option<(u32, EntityId)> = None;
        for entry in entries {
            if entry.0 <= flat {
                owner = Some(entry);
            } else {
                break;
            }
        }
        owner.map(|(off, eid)| (eid, flat - off))
    }

    /// Map a half-open range of flat residue indices into the
    /// `(EntityId, residue_in_entity)` pairs each one resolves to.
    /// Entries that fall outside any entity's residue span are
    /// silently dropped (only happens before the first rebuild
    /// publishes offsets).
    fn flat_range_to_pairs(
        &self,
        range: std::ops::Range<usize>,
    ) -> Vec<(EntityId, u32)> {
        range
            .filter_map(|flat| self.flat_to_entity_residue(flat as u32))
            .collect()
    }

    /// Per-entity residue list for the SS segment that contains
    /// `residue_idx`. Empty if `residue_idx` is out of bounds or no
    /// entity owns the resulting flat residues.
    pub(crate) fn residues_in_segment(
        &self,
        residue_idx: i32,
        ss_types: &[SSType],
    ) -> Vec<(EntityId, u32)> {
        let Some(range) = Self::segment_range(residue_idx, ss_types) else {
            return Vec::new();
        };
        self.flat_range_to_pairs(range)
    }

    /// Per-entity residue list for the chain that contains
    /// `residue_idx`. Empty if `residue_idx` falls outside every
    /// chain.
    pub(crate) fn residues_in_chain(
        &self,
        residue_idx: i32,
        backbone_chains: &[ProteinBackboneChain],
    ) -> Vec<(EntityId, u32)> {
        let Some(range) = Self::chain_range(residue_idx, backbone_chains)
        else {
            return Vec::new();
        };
        self.flat_range_to_pairs(range)
    }

    /// Build the picking geometry descriptor from current renderer state.
    pub(crate) fn build_geometry<'a>(
        &'a self,
        renderers: &'a Renderers,
    ) -> PickingGeometry<'a> {
        PickingGeometry {
            backbone_vertex_buffer: renderers.backbone.vertex_buffer(),
            backbone_tube_index_buffer: renderers.backbone.tube_index_buffer(),
            backbone_tube_index_count: renderers.backbone.tube_index_count(),
            backbone_ribbon_index_buffer: renderers
                .backbone
                .ribbon_index_buffer(),
            backbone_ribbon_index_count: renderers
                .backbone
                .ribbon_index_count(),
            capsule_bind_group: self.groups.capsule.as_ref(),
            capsule_count: renderers.sidechain.instance_count(),
            bns_capsule_bind_group: self.groups.bns_bond.as_ref(),
            bns_capsule_count: renderers.ball_and_stick.bond_count(),
            bns_sphere_bind_group: self.groups.bns_sphere.as_ref(),
            bns_sphere_count: renderers.ball_and_stick.sphere_count(),
        }
    }

    /// Upload the current selection state to the GPU selection buffer.
    pub(crate) fn update_selection_buffer(&self, queue: &wgpu::Queue) {
        self.selection
            .update(queue, &self.picking.selected_residues);
    }
}
