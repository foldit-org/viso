//! Per-cavity mesh generation from molex's cavity detector.
//!
//! The analytical half of this module — voxelization, SES erosion,
//! cavity mask flood fill, connected-component labeling, per-cavity
//! sub-grid construction — lives in
//! [`molex::analysis::volumetric::cavity`]. This file only covers the
//! rendering-bound half: taking each `DetectedCavity` sub-grid mask,
//! converting it to an SDF, running marching cubes, smoothing the
//! triangles, and baking cavity-specific vertex attributes
//! (`CAVITY_RGBA`, `cavity_center`).

use molex::analysis::volumetric::{
    binary_to_sdf, detect_cavities, DetectedCavity,
};

use super::cpu_marching_cubes::extract_isosurface;
use super::mesh_smooth::taubin_smooth;
use super::{isosurface_kind, IsosurfaceVertex};

/// Number of Taubin smoothing iterations applied to each cavity mesh
/// after marching cubes. Each iteration is one λ pass + one μ pass.
/// Operates on triangles after extraction so it can never lose cavities
/// (unlike SDF-side smoothing, which blurs small features below the
/// iso-threshold and makes them disappear).
const CAVITY_SMOOTHING_ITERATIONS: usize = 8;

/// Unified RGBA tint baked into every cavity vertex. All cavities share
/// this color so the visual reads as a property of the negative space
/// itself, not of any particular chain. The alpha is a baseline that
/// gets modulated by Beer-Lambert thickness in the fragment shader.
const CAVITY_RGBA: [f32; 4] = [0.22, 0.30, 1.0, 0.90];

/// A single detected cavity with its extracted mesh.
#[derive(Clone)]
pub(crate) struct CavityMesh {
    /// Isosurface vertices for this cavity.
    pub(crate) vertices: Vec<IsosurfaceVertex>,
    /// Triangle indices into `vertices`.
    pub(crate) indices: Vec<u32>,
}

/// A collection of cavities detected in a single pose.
#[derive(Clone, Default)]
pub(crate) struct CavitySet {
    /// One entry per distinct cavity, in label order.
    pub(crate) meshes: Vec<CavityMesh>,
}

/// Generate cavity meshes from atom positions.
///
/// Delegates detection to [`molex::analysis::volumetric::detect_cavities`]
/// (voxelization + SES erosion + connected components) and wraps each
/// returned [`DetectedCavity`] with a per-cavity mesh extracted via
/// marching cubes + Taubin smoothing.
///
/// All cavities share the unified [`CAVITY_RGBA`] tint — the color is
/// not a parameter because cavities are meant to read as "negative
/// space", not as a per-entity visual.
///
/// - `positions`: atom world-space positions (Angstroms)
/// - `radii`: per-atom van der Waals radii (Angstroms)
/// - `probe_radius`: solvent probe radius; defaults to 1.4 Å
/// - `resolution`: grid spacing in Angstroms (lower = finer, typ. 0.5–1.0)
#[must_use]
pub(crate) fn generate_cavities(
    positions: &[glam::Vec3],
    radii: &[f32],
    probe_radius: Option<f32>,
    resolution: f32,
) -> CavitySet {
    let detected = detect_cavities(positions, radii, probe_radius, resolution);

    let meshes = detected.iter().filter_map(extract_cavity_mesh).collect();

    CavitySet { meshes }
}

/// Extract a single cavity's isosurface mesh on its sub-grid.
///
/// `binary_to_sdf` returns negative-inside / positive-outside. Negate
/// so inside-cavity is positive (matches the marching-cubes gradient
/// convention used elsewhere in this crate). The voxel-facet
/// appearance gets smoothed away on the triangle side after marching
/// cubes, not by blurring the field — blurring the field would shrink
/// small cavities below the iso-threshold and lose them entirely.
fn extract_cavity_mesh(cavity: &DetectedCavity) -> Option<CavityMesh> {
    let mut sub_sdf =
        binary_to_sdf(&cavity.sub_mask, cavity.sub_dims, &cavity.spacing);
    for v in &mut sub_sdf {
        *v = -*v;
    }

    let (mut vertices, indices) = extract_isosurface(
        &sub_sdf,
        cavity.sub_dims,
        0.0,
        [0, 0, 0],
        cavity.sub_dims,
        |gx, gy, gz| {
            [
                gx.mul_add(cavity.spacing[0], cavity.sub_origin[0]),
                gy.mul_add(cavity.spacing[1], cavity.sub_origin[1]),
                gz.mul_add(cavity.spacing[2], cavity.sub_origin[2]),
            ]
        },
        CAVITY_RGBA,
    );

    if vertices.is_empty() || indices.is_empty() {
        return None;
    }

    taubin_smooth(&mut vertices, &indices, CAVITY_SMOOTHING_ITERATIONS);

    // Tag every vertex as a cavity (so the isosurface shader can apply
    // cavity-specific effects without inspecting color) and bake the
    // centroid for radial-breath displacement.
    for v in &mut vertices {
        v.kind = isosurface_kind::CAVITY;
        v.cavity_center = cavity.centroid;
    }

    Some(CavityMesh { vertices, indices })
}

/// Mesh a host-supplied void distance field directly into a smooth blob.
///
/// Unlike [`extract_cavity_mesh`], which meshes a binary mask via
/// `binary_to_sdf` + negate, `phi` arrives already in marching-cubes
/// polarity: a positive distance that is HIGH at the void center and ~0
/// at atom walls / exterior. So it is fed straight into
/// [`extract_isosurface`] at the passed positive `threshold` (the
/// void-surface level) with no `binary_to_sdf` and no sign flip — the
/// extractor wraps the region where `value >= threshold` (inside = high),
/// which is exactly the void interior.
///
/// Every vertex is tagged [`isosurface_kind::CAVITY`] so the isosurface
/// shader applies the same breathing displacement + Beer-Lambert as a
/// detected cavity. The breathing anchor (`cavity_center`) is the
/// world-space centroid of the cells above `threshold` (the void region
/// center); a single centroid is shared by the whole field for now, so
/// all blobs breathe on one rhythm. Per-blob component labeling (a
/// distinct centroid per connected void) is a later refinement.
///
/// Returns an empty [`CavitySet`] when `phi` is empty, `dims` is
/// degenerate, no cell is above `threshold`, or the extraction produces
/// no triangles.
#[must_use]
pub(crate) fn mesh_void_field(
    phi: &[f32],
    dims: [usize; 3],
    origin: [f32; 3],
    spacing: [f32; 3],
    threshold: f32,
) -> CavitySet {
    let [nx, ny, nz] = dims;
    if phi.len() != nx * ny * nz || nx < 2 || ny < 2 || nz < 2 {
        return CavitySet::default();
    }

    // World-space centroid of the cells above threshold (the void region
    // center). Drives the radial-breath displacement; if nothing is above
    // threshold there is no void to mesh.
    let grid_to_world = |gx: f32, gy: f32, gz: f32| {
        [
            gx.mul_add(spacing[0], origin[0]),
            gy.mul_add(spacing[1], origin[1]),
            gz.mul_add(spacing[2], origin[2]),
        ]
    };
    let idx = |x: usize, y: usize, z: usize| x * ny * nz + y * nz + z;
    let mut sum = [0.0f32; 3];
    let mut count = 0usize;
    for x in 0..nx {
        for y in 0..ny {
            for z in 0..nz {
                if phi[idx(x, y, z)] >= threshold {
                    let w = grid_to_world(x as f32, y as f32, z as f32);
                    sum[0] += w[0];
                    sum[1] += w[1];
                    sum[2] += w[2];
                    count += 1;
                }
            }
        }
    }
    if count == 0 {
        return CavitySet::default();
    }
    let centroid = [
        sum[0] / count as f32,
        sum[1] / count as f32,
        sum[2] / count as f32,
    ];

    let (mut vertices, indices) = extract_isosurface(
        phi,
        dims,
        threshold,
        [0, 0, 0],
        dims,
        grid_to_world,
        CAVITY_RGBA,
    );

    if vertices.is_empty() || indices.is_empty() {
        return CavitySet::default();
    }

    taubin_smooth(&mut vertices, &indices, CAVITY_SMOOTHING_ITERATIONS);

    // Tag every vertex as a cavity (so the isosurface shader applies
    // cavity-specific effects without inspecting color) and bake the
    // shared centroid for radial-breath displacement.
    for v in &mut vertices {
        v.kind = isosurface_kind::CAVITY;
        v.cavity_center = centroid;
    }

    CavitySet {
        meshes: vec![CavityMesh { vertices, indices }],
    }
}

#[cfg(test)]
mod tests {
    use glam::Vec3;

    use super::*;

    #[test]
    fn generate_cavities_empty_atoms() {
        let set = generate_cavities(&[], &[], None, 1.0);
        assert!(set.meshes.is_empty());
    }

    #[test]
    fn generate_cavities_single_atom_has_none() {
        // A lone atom is a solid blob with no interior voids.
        let set = generate_cavities(&[Vec3::ZERO], &[1.5], Some(1.4), 0.5);
        assert!(set.meshes.is_empty());
    }

    #[test]
    fn mesh_void_field_high_center_bump_meshes_cavity() {
        // A field with a high central core (above threshold) ringed by
        // low (sub-threshold) border voxels marches into a closed blob,
        // and every vertex carries the CAVITY kind.
        let dims = [5usize, 5, 5];
        let idx =
            |x: usize, y: usize, z: usize| (x * dims[1] + y) * dims[2] + z;
        let mut phi = vec![0.0f32; dims[0] * dims[1] * dims[2]];
        for x in 1..4 {
            for y in 1..4 {
                for z in 1..4 {
                    phi[idx(x, y, z)] = 2.0;
                }
            }
        }

        let set =
            mesh_void_field(&phi, dims, [0.0, 0.0, 0.0], [1.0, 1.0, 1.0], 1.0);
        assert_eq!(set.meshes.len(), 1);
        let mesh = &set.meshes[0];
        assert!(!mesh.vertices.is_empty());
        assert!(!mesh.indices.is_empty());
        assert!(mesh
            .vertices
            .iter()
            .all(|v| v.kind == isosurface_kind::CAVITY));
    }

    #[test]
    fn mesh_void_field_all_below_threshold_is_empty() {
        let dims = [4usize, 4, 4];
        let phi = vec![0.0f32; dims[0] * dims[1] * dims[2]];
        let set =
            mesh_void_field(&phi, dims, [0.0, 0.0, 0.0], [1.0, 1.0, 1.0], 1.0);
        assert!(set.meshes.is_empty());
    }

    #[test]
    fn mesh_void_field_empty_input() {
        let set = mesh_void_field(&[], [0, 0, 0], [0.0; 3], [1.0; 3], 1.0);
        assert!(set.meshes.is_empty());
    }
}
