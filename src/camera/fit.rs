//! Fit-to-molecule-data helpers.
//!
//! Given one or more [`MoleculeEntity`]s,
//! compute a bounding sphere (centroid + radius) and drive the
//! [`CameraController`]
//! at it. Lets engine-side code avoid hand-rolling geometry every
//! time it wants to point the camera at something.

use glam::Vec3;
use molex::MoleculeEntity;

use super::controller::CameraController;

/// Centroid + bounding-sphere radius over an entity's atoms. `None` if
/// the entity has no atoms.
#[must_use]
pub(crate) fn bounding_sphere_of(
    entity: &MoleculeEntity,
) -> Option<(Vec3, f32)> {
    let atoms = entity.atom_set();
    if atoms.is_empty() {
        return None;
    }
    let n = atoms.len() as f32;
    let centroid = atoms.iter().fold(Vec3::ZERO, |acc, a| acc + a.position) / n;
    let radius = atoms
        .iter()
        .map(|a| (a.position - centroid).length())
        .fold(0.0f32, f32::max);
    Some((centroid, radius))
}

/// Combined weighted centroid + enclosing radius across a set of
/// entities. Weighting is by atom count so a small ion next to a
/// protein doesn't pull the centroid. `None` if every entity is empty.
#[must_use]
pub(crate) fn combined_bounding_sphere<'a>(
    entities: impl IntoIterator<Item = &'a MoleculeEntity>,
) -> Option<(Vec3, f32)> {
    let mut total_weight = 0.0f32;
    let mut weighted_sum = Vec3::ZERO;
    let mut radii: Vec<(Vec3, f32)> = Vec::new();
    for entity in entities {
        let Some((centroid, radius)) = bounding_sphere_of(entity) else {
            continue;
        };
        let w = entity.atom_count() as f32;
        if w > 0.0 {
            weighted_sum += centroid * w;
            total_weight += w;
            radii.push((centroid, radius));
        }
    }
    if total_weight == 0.0 {
        return None;
    }
    let centroid = weighted_sum / total_weight;
    let radius = radii
        .iter()
        .map(|(c, r)| (*c - centroid).length() + r)
        .fold(0.0f32, f32::max);
    Some((centroid, radius))
}

/// Fit the camera to a single entity's bounding sphere, animated.
/// No-op if the entity has no atoms.
pub(crate) fn fit_to_entity(
    controller: &mut CameraController,
    entity: &MoleculeEntity,
) {
    if let Some((centroid, radius)) = bounding_sphere_of(entity) {
        controller.fit_to_sphere_animated(centroid, radius);
    }
}

/// Fit the camera to the combined bounding sphere of an iterator of
/// entities (e.g. every visible entity). No-op if the set is empty.
pub(crate) fn fit_to_entities<'a>(
    controller: &mut CameraController,
    entities: impl IntoIterator<Item = &'a MoleculeEntity>,
) {
    if let Some((centroid, radius)) = combined_bounding_sphere(entities) {
        controller.fit_to_sphere_animated(centroid, radius);
    }
}
