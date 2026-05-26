//! GPU-based object picking and selection management.
//!
//! Renders residue indices to an offscreen buffer and reads back the pixel
//! under the cursor to determine which residue was clicked or hovered.

mod pick_map;
mod pipeline;
pub(crate) mod state;

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
        if self.picking.selected_residues.is_empty() {
            false
        } else {
            self.picking.selected_residues.clear();
            true
        }
    }

    /// Select all residues in the same secondary-structure segment as
    /// `residue_idx`. If `extend` is true the new residues are added to the
    /// existing selection; otherwise the selection is replaced.
    ///
    /// `ss_types` is the per-residue secondary-structure classification
    /// (indexed by flat residue index).
    ///
    /// Returns `true` if the selection changed.
    pub(crate) fn select_segment(
        &mut self,
        residue_idx: i32,
        ss_types: &[SSType],
        extend: bool,
    ) -> bool {
        if residue_idx < 0 || (residue_idx as usize) >= ss_types.len() {
            return false;
        }

        let idx = residue_idx as usize;
        let target_ss = ss_types[idx];

        // Walk backwards to the start of this SS segment
        let mut start = idx;
        while start > 0 && ss_types[start - 1] == target_ss {
            start -= 1;
        }

        // Walk forwards to the end of this SS segment
        let mut end = idx;
        while end + 1 < ss_types.len() && ss_types[end + 1] == target_ss {
            end += 1;
        }

        if !extend {
            self.picking.selected_residues.clear();
        }

        for i in start..=end {
            let residue = i as i32;
            if !self.picking.selected_residues.contains(&residue) {
                self.picking.selected_residues.push(residue);
            }
        }

        true
    }

    /// Select all residues in the same chain as `residue_idx`. Chains are
    /// described by `backbone_chains` (each entry is one chain's SoA
    /// backbone atoms; its residue count is
    /// [`ProteinBackboneChain::residue_count`]).
    ///
    /// If `extend` is true the new residues are added to the existing
    /// selection; otherwise the selection is replaced.
    ///
    /// Returns `true` if the selection changed.
    pub(crate) fn select_chain(
        &mut self,
        residue_idx: i32,
        backbone_chains: &[ProteinBackboneChain],
        extend: bool,
    ) -> bool {
        if residue_idx < 0 {
            return false;
        }
        let target = residue_idx as usize;

        let mut global_start = 0usize;
        let chain_range = backbone_chains.iter().find_map(|chain| {
            let chain_residues = chain.ca().len();
            let global_end = global_start + chain_residues;
            let result = (target >= global_start && target < global_end)
                .then_some(global_start..global_end);
            global_start = global_end;
            result
        });

        let Some(range) = chain_range else {
            return false;
        };

        if !extend {
            self.picking.selected_residues.clear();
        }
        for i in range {
            let residue = i as i32;
            if !self.picking.selected_residues.contains(&residue) {
                self.picking.selected_residues.push(residue);
            }
        }
        true
    }

    /// Build the picking geometry descriptor from current renderer state.
    pub(crate) fn build_geometry<'a>(
        &'a self,
        renderers: &'a Renderers,
        show_sidechains: bool,
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
            capsule_count: if show_sidechains {
                renderers.sidechain.instance_count()
            } else {
                0
            },
            bns_capsule_bind_group: self.groups.bns_bond.as_ref(),
            bns_capsule_count: renderers.ball_and_stick.bond_count(),
            bns_sphere_bind_group: self.groups.bns_sphere.as_ref(),
            bns_sphere_count: renderers.ball_and_stick.sphere_count(),
        }
    }

    /// Read-only access to the currently selected residue indices.
    #[allow(dead_code)] // API surface, not yet called by engine
    pub(crate) fn selected_residues(&self) -> &[i32] {
        &self.picking.selected_residues
    }

    /// Upload the current selection state to the GPU selection buffer.
    pub(crate) fn update_selection_buffer(&self, queue: &wgpu::Queue) {
        self.selection
            .update(queue, &self.picking.selected_residues);
    }
}
