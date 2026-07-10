//! Host-pushed transient select-sphere overlay.

use super::command::SelectSphereInfo;
use super::VisoEngine;

impl VisoEngine {
    /// Set or clear the transient select-sphere overlay. `None` hides it.
    ///
    /// The sphere is given in world space (centre + radius), so it needs
    /// no per-frame reference resolution: the spec is stored and forwarded
    /// straight to the renderer, whose buffer persists until the next push.
    pub fn update_select_sphere(&mut self, sphere: Option<SelectSphereInfo>) {
        self.constraints.select_sphere_spec = sphere;
        self.resolve_and_render_constraints();
    }
}
