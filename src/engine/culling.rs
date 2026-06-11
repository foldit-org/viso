//! Per-frame frustum culling + LOD tier selection.
//!
//! Both decisions -- which sidechain instances to upload, and which
//! backbone LOD tier each chain should be remeshed at -- happen every
//! frame against the current camera. They live here rather than in
//! [`super::sync`] because they're not Assembly-sync logic; they're
//! rendering decisions driven by camera position + animator state.

use glam::Vec3;

impl super::VisoEngine {
    /// Re-filter the retained sidechain capsules against the camera
    /// frustum when the camera has moved enough to matter. The instances
    /// themselves (color, sheet adjustment, global pick ids) come from the
    /// last rebuild or animation frame; this only drops the ones now
    /// outside the view.
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
        self.gpu.upload_frustum_culled_sidechains(&frustum);
    }

    fn has_any_sidechain_atoms(&self) -> bool {
        self.scene
            .entity_state
            .values()
            .any(|s| !s.topology.sidechain_layout.atom_indices.is_empty())
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
