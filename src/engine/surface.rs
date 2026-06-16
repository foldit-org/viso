//! Per-entity molecular surface parameters and annotation mutators.
//!
//! Surfaces are generated on the shared scene-processor worker through
//! [`crate::engine::surface_regen::regenerate_surfaces`] and rendered
//! via the shared `IsosurfaceRenderer`. Multiple entities can each
//! have surfaces — their meshes are concatenated before upload.

use molex::entity::molecule::id::EntityId;

use super::annotations::EntityAnnotations;
use crate::options::SurfaceKindOption;

/// Which kind of molecular surface to generate.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) enum SurfaceKind {
    /// Smooth Gaussian blob surface.
    Gaussian,
    /// Solvent-excluded / Connolly surface.
    Ses,
}

/// Per-entity surface parameters.
#[derive(Debug, Clone)]
pub(crate) struct EntitySurface {
    /// Which surface type.
    pub(crate) kind: SurfaceKind,
    /// Grid resolution in Angstroms (lower = finer).
    pub(crate) resolution: f32,
    /// Probe radius for SES (Angstroms, default 1.4).
    pub(crate) probe_radius: f32,
    /// Gaussian isosurface level (only used for Gaussian kind).
    pub(crate) level: f32,
    /// Surface RGBA color.
    pub(crate) color: [f32; 4],
    /// Whether this surface is visible.
    pub(crate) visible: bool,
}

impl Default for EntitySurface {
    fn default() -> Self {
        Self {
            kind: SurfaceKind::Gaussian,
            resolution: 1.0,
            probe_radius: 1.4,
            level: 0.5,
            color: [0.7, 0.7, 0.7, 0.35],
            visible: true,
        }
    }
}

// EntityAnnotations: surface mutators

impl EntityAnnotations {
    /// Replace (or insert) the entity's surface and log it.
    pub(crate) fn set_entity_surface(
        &mut self,
        entity_id: EntityId,
        surface: EntitySurface,
    ) {
        log::info!(
            "set {:?} surface for entity {}",
            surface.kind,
            entity_id.raw()
        );
        let _ = self.surfaces.insert(entity_id, surface);
    }

    /// Update a single color channel on an existing entity surface.
    /// Returns `true` if the surface existed (and the channel was
    /// updated).
    pub(crate) fn set_surface_color_channel(
        &mut self,
        entity_id: EntityId,
        channel: usize,
        value: f32,
    ) -> bool {
        if let Some(surface) = self.surfaces.get_mut(&entity_id) {
            surface.color[channel] = value.clamp(0.0, 1.0);
            true
        } else {
            false
        }
    }

    /// Remove the entity's surface. When a global surface kind is
    /// active, leaves an invisible sentinel so this entity explicitly
    /// opts out of the global default. Returns `true` if any state
    /// changed (mesh regeneration is then required).
    pub(crate) fn remove_entity_surface(
        &mut self,
        entity_id: EntityId,
        global_kind: SurfaceKindOption,
    ) -> bool {
        let had = self.surfaces.remove(&entity_id).is_some();
        if global_kind != SurfaceKindOption::None {
            let _ = self.surfaces.insert(
                entity_id,
                EntitySurface {
                    visible: false,
                    ..Default::default()
                },
            );
        }
        had || global_kind != SurfaceKindOption::None
    }
}
