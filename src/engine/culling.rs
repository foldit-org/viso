//! Per-frame frustum culling + LOD tier selection.
//!
//! Both decisions -- which sidechain instances to upload, and which
//! backbone LOD tier each chain should be remeshed at -- happen every
//! frame against the current camera. They live here rather than in
//! [`super::sync`] because they're not Assembly-sync logic; they're
//! rendering decisions driven by camera position + animator state.

use glam::Vec3;

/// Flat, assembly-ordered view of every visible entity's sidechain
/// layout + current positions. Produced once per cull update and
/// handed straight to the sidechain renderer via `SidechainView`.
#[derive(Default)]
struct FlatSidechainState {
    positions: Vec<Vec3>,
    bonds: Vec<(u32, u32)>,
    backbone_bonds: Vec<(Vec3, u32)>,
    hydrophobicity: Vec<bool>,
    residue_indices: Vec<u32>,
}

impl super::VisoEngine {
    /// Update sidechain instances with frustum culling when the camera
    /// moves, rebuilding the flat sidechain view from per-entity
    /// topology + positions.
    pub(crate) fn update_frustum_culling(&mut self) {
        if !self.has_any_sidechain_atoms() {
            return;
        }
        if !self.should_update_culling() {
            return;
        }

        self.gpu
            .set_last_cull_camera_eye(self.camera_controller.camera.eye);
        let frustum = self.camera_controller.frustum();

        let flat = self.flat_sidechain_state();
        let sheet_offsets = self.gpu.backbone_sheet_offsets();
        let raw_view = crate::renderer::geometry::SidechainView {
            positions: &flat.positions,
            bonds: &flat.bonds,
            backbone_bonds: &flat.backbone_bonds,
            hydrophobicity: &flat.hydrophobicity,
            residue_indices: &flat.residue_indices,
        };
        let adjusted =
            crate::renderer::geometry::sheet_adjust::sheet_adjusted_view(
                &raw_view,
                sheet_offsets,
            );
        let sc_colors = if self.options.display.sidechain_color_mode()
            == crate::options::SidechainColorMode::Backbone
        {
            let flat = self.scene.flat_cartoon_colors(&self.annotations);
            if flat.is_empty() {
                None
            } else {
                Some(flat)
            }
        } else {
            None
        };
        self.gpu.upload_frustum_culled_sidechains(
            &adjusted.as_view(),
            &frustum,
            sc_colors.as_deref(),
        );
    }

    fn has_any_sidechain_atoms(&self) -> bool {
        self.scene
            .entity_state
            .values()
            .any(|s| !s.topology.sidechain_layout.atom_indices.is_empty())
    }

    /// Flatten per-entity sidechain layout + positions into a single
    /// sidechain-view payload. Residue indices are offset per entity so
    /// each entity's sidechain atoms get a unique global residue index.
    fn flat_sidechain_state(&self) -> FlatSidechainState {
        let mut out = FlatSidechainState::default();
        let mut residue_offset: u32 = 0;

        for (_, eid, state) in self.scene.visible_entities(&self.annotations) {
            let layout = &state.topology.sidechain_layout;
            if layout.atom_indices.is_empty() {
                residue_offset +=
                    state.topology.residue_atom_ranges.len() as u32;
                continue;
            }
            let Some(entity_positions) = self.scene.positions.get(eid) else {
                continue;
            };
            let layout_offset = out.positions.len() as u32;
            for &atom_idx in &layout.atom_indices {
                let pos = entity_positions
                    .get(atom_idx as usize)
                    .copied()
                    .unwrap_or(Vec3::ZERO);
                out.positions.push(pos);
            }
            for &(a, b) in &layout.bonds {
                out.bonds.push((a + layout_offset, b + layout_offset));
            }
            for &(ca_atom_idx, layout_idx) in &layout.backbone_bonds {
                let ca = entity_positions
                    .get(ca_atom_idx as usize)
                    .copied()
                    .unwrap_or(Vec3::ZERO);
                out.backbone_bonds.push((ca, layout_idx + layout_offset));
            }
            out.hydrophobicity.extend_from_slice(&layout.hydrophobicity);
            for &ri in &layout.residue_indices {
                out.residue_indices.push(ri + residue_offset);
            }
            residue_offset += state.topology.residue_atom_ranges.len() as u32;
        }
        out
    }

    fn should_update_culling(&self) -> bool {
        const CULL_UPDATE_THRESHOLD: f32 = 5.0;
        if self.animation.animator.is_animating() {
            return true;
        }
        let camera_eye = self.camera_controller.camera.eye;
        let camera_delta =
            (camera_eye - self.gpu.last_cull_camera_eye).length();
        camera_delta >= CULL_UPDATE_THRESHOLD
    }

    /// Check per-chain LOD tiers and submit a background remesh if any
    /// chain's tier has changed.
    pub(crate) fn check_and_submit_lod(&mut self) {
        let camera_eye = self.camera_controller.camera.eye;
        let geo = self.options.resolved_geometry();
        self.gpu
            .check_and_submit_lod(camera_eye, &geo, &self.scene.positions);
    }

    /// Submit a backbone-only remesh with per-chain LOD.
    pub(crate) fn submit_per_chain_lod_remesh(&self, camera_eye: Vec3) {
        let geo = self.options.resolved_geometry();
        self.gpu
            .submit_lod_remesh(camera_eye, &geo, &self.scene.positions);
    }
}
