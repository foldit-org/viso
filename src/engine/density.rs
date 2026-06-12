//! Engine methods for loading and managing electron density maps.

use std::sync::Arc;

use molex::entity::surface::Density;

use super::annotations::EntityAnnotations;
use super::density_store::DensityStore;
use super::scene::Scene;
use super::surface_regen::{regenerate_surfaces, SurfaceRegen};
use super::{ExternalVoidField, VisoEngine};
use crate::camera::fit::combined_bounding_sphere;
use crate::options::VisoOptions;

/// Disjoint-borrow write view over the density store plus the scene
/// fields a regeneration needs to read.
///
/// Density mutations always trigger a surface/density mesh regeneration,
/// which reads the [`Scene`], [`EntityAnnotations`], [`VisoOptions`],
/// and the [`SurfaceRegen`] sender. Bundling those four shared borrows
/// alongside `&mut DensityStore` lets every mutator be expressed without
/// `&mut self` methods on `VisoEngine` fighting over the whole engine.
/// [`VisoEngine::density_mut`] is the constructor.
pub(crate) struct DensityScene<'a> {
    store: &'a mut DensityStore,
    scene: &'a Scene,
    annotations: &'a EntityAnnotations,
    external_void_field: Option<&'a ExternalVoidField>,
    options: &'a VisoOptions,
    regen: &'a SurfaceRegen,
}

impl DensityScene<'_> {
    /// Trigger a regeneration of all isosurface meshes (density maps,
    /// entity surfaces, cavities) on the background thread.
    fn regenerate(&self) {
        regenerate_surfaces(
            self.scene,
            self.annotations,
            &*self.store,
            self.external_void_field,
            self.options,
            self.regen,
        );
    }

    /// Load a density map. Returns the assigned map ID.
    ///
    /// The map is immediately visible at the default sigma level.
    /// Mesh generation runs on a background thread; the result appears
    /// on the next frame after completion.
    pub(crate) fn load(&mut self, map: Density) -> u32 {
        let id = self.store.add(map);
        log::info!("loaded density map id={id}");
        let visible = self
            .scene
            .current
            .entities()
            .iter()
            .filter(|e| self.annotations.is_visible(e.id()))
            .map(Arc::as_ref);
        if let Some((centroid, radius)) = combined_bounding_sphere(visible) {
            log::info!(
                "protein bounding sphere: center=[{:.1},{:.1},{:.1}], \
                 radius={:.1}, range=[{:.1},{:.1},{:.1}]→[{:.1},{:.1},{:.1}]",
                centroid.x,
                centroid.y,
                centroid.z,
                radius,
                centroid.x - radius,
                centroid.y - radius,
                centroid.z - radius,
                centroid.x + radius,
                centroid.y + radius,
                centroid.z + radius,
            );
        }
        self.regenerate();
        id
    }

    /// Remove a density map by ID.
    pub(crate) fn remove(&mut self, id: u32) {
        if self.store.remove(id) {
            log::info!("removed density map id={id}");
            self.regenerate();
        }
    }

    /// Set the raw density threshold for a density map.
    pub(crate) fn set_threshold(&mut self, id: u32, threshold: f32) {
        log::info!("set_density_threshold id={id} threshold={threshold:.4}");
        self.store.set_threshold(id, threshold);
        self.regenerate();
    }

    /// Set visibility for a density map.
    pub(crate) fn set_visible(&mut self, id: u32, visible: bool) {
        self.store.set_visible(id, visible);
        self.regenerate();
    }

    /// Set color for a density map.
    pub(crate) fn set_color(&mut self, id: u32, color: [f32; 3]) {
        self.store.set_color(id, color);
        self.regenerate();
    }

    /// Set opacity for a density map (0.0–1.0).
    pub(crate) fn set_opacity(&mut self, id: u32, opacity: f32) {
        self.store.set_opacity(id, opacity);
        self.regenerate();
    }
}

impl VisoEngine {
    /// Replace the host-supplied void distance field and re-mesh the
    /// isosurface pass.
    ///
    /// `phi` is a flat row-major scalar grid (`phi[x*ny*nz + y*nz + z]`)
    /// in marching-cubes polarity: HIGH at void centers, ~0 at atom walls
    /// / exterior. It is meshed at the positive `threshold` directly into
    /// a smooth blob carrying the same CAVITY kind / tint as a detected
    /// cavity, so it gets the cavity breathing + Beer-Lambert for free. An
    /// empty `phi` (or any zero dimension) clears the field. Meshing runs
    /// on the background thread; the result appears on the next frame
    /// after completion.
    pub fn set_external_void_field(
        &mut self,
        dims: [usize; 3],
        origin: [f32; 3],
        spacing: [f32; 3],
        phi: Vec<f32>,
        threshold: f32,
    ) {
        let [nx, ny, nz] = dims;
        self.external_void_field =
            if phi.is_empty() || nx == 0 || ny == 0 || nz == 0 {
                None
            } else {
                Some(ExternalVoidField {
                    dims,
                    origin,
                    spacing,
                    phi,
                    threshold,
                })
            };
        regenerate_surfaces(
            &self.scene,
            &self.annotations,
            &self.density,
            self.external_void_field.as_ref(),
            &self.options,
            &self.surface_regen,
        );
    }

    /// Disjoint-borrow write view over the density store plus the
    /// scene fields every density mutation has to read for regeneration.
    pub(crate) fn density_mut(&mut self) -> DensityScene<'_> {
        DensityScene {
            store: &mut self.density,
            scene: &self.scene,
            annotations: &self.annotations,
            external_void_field: self.external_void_field.as_ref(),
            options: &self.options,
            regen: &self.surface_regen,
        }
    }
}
