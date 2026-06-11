//! Bridge between molex `Density` and the marching cubes algorithm.
//!
//! Converts crystallographic density data into triangle mesh vertices
//! suitable for GPU rendering. Optionally crops to a world-space
//! bounding box so only density near the structure is meshed.

use molex::entity::surface::Density;

use super::cpu_marching_cubes::extract_isosurface;
use super::{isosurface_kind, IsosurfaceVertex};

/// Padding in Angstroms around the structure bounding box when cropping
/// the density map.
#[allow(dead_code)]
const CROP_PADDING: f32 = 5.0;

/// World-space axis-aligned bounding box for map cropping.
pub(crate) struct CropBox {
    /// Minimum corner (Angstroms).
    pub(crate) min: [f32; 3],
    /// Maximum corner (Angstroms).
    pub(crate) max: [f32; 3],
}

impl CropBox {
    /// Build a crop box from a bounding sphere (centroid + radius) with
    /// padding.
    #[must_use]
    #[allow(dead_code)]
    pub(crate) fn from_sphere(center: [f32; 3], radius: f32) -> Self {
        let r = radius + CROP_PADDING;
        Self {
            min: [center[0] - r, center[1] - r, center[2] - r],
            max: [center[0] + r, center[1] + r, center[2] + r],
        }
    }
}

/// Generate a triangle mesh from a density map at the given sigma level.
///
/// - `map`: parsed density map (CCP4/MRC format)
/// - `threshold`: raw density threshold for isosurface extraction
/// - `color`: uniform RGBA color for all mesh vertices
/// - `crop`: optional bounding box to restrict meshing (world-space)
///
/// Returns `(vertices, indices)` for an indexed triangle mesh.
pub(crate) fn generate_density_mesh(
    map: &Density,
    threshold: f32,
    color: [f32; 4],
    crop: Option<&CropBox>,
) -> (Vec<IsosurfaceVertex>, Vec<u32>) {
    let dims = [map.nx, map.ny, map.nz];

    let Some(data) = map.data.as_slice() else {
        log::warn!("density map data is not contiguous; skipping mesh");
        return (Vec::new(), Vec::new());
    };

    let vs = map.voxel_size();
    let corner_min = map.grid_to_cartesian_f32(0.0, 0.0, 0.0);
    let corner_max = map.grid_to_cartesian_f32(
        (map.nx - 1) as f32,
        (map.ny - 1) as f32,
        (map.nz - 1) as f32,
    );
    log::info!(
        "density map: dims=[{},{},{}], voxel=[{:.2},{:.2},{:.2}], \
         origin=[{:.1},{:.1},{:.1}], nstart=[{},{},{}], \
         cell_dims=[{:.1},{:.1},{:.1}], cell_angles=[{:.1},{:.1},{:.1}], \
         M=[{},{},{}], world range=[{:.1},{:.1},{:.1}]→[{:.1},{:.1},{:.1}]",
        dims[0],
        dims[1],
        dims[2],
        vs[0],
        vs[1],
        vs[2],
        map.origin[0],
        map.origin[1],
        map.origin[2],
        map.nxstart,
        map.nystart,
        map.nzstart,
        map.cell_dims[0],
        map.cell_dims[1],
        map.cell_dims[2],
        map.cell_angles[0],
        map.cell_angles[1],
        map.cell_angles[2],
        map.mx,
        map.my,
        map.mz,
        corner_min[0],
        corner_min[1],
        corner_min[2],
        corner_max[0],
        corner_max[1],
        corner_max[2],
    );

    // Determine grid-space iteration bounds (full grid or cropped).
    let (grid_min, grid_max) = crop.map_or(([0, 0, 0], dims), |bbox| {
        let gmin = map.cartesian_to_grid(bbox.min);
        let gmax = map.cartesian_to_grid(bbox.max);
        // Take min/max across both corners since the matrix transform
        // can swap axis ordering.
        let lo = [
            gmin[0].min(gmax[0]),
            gmin[1].min(gmax[1]),
            gmin[2].min(gmax[2]),
        ];
        let hi = [
            gmin[0].max(gmax[0]),
            gmin[1].max(gmax[1]),
            gmin[2].max(gmax[2]),
        ];
        (
            [
                (lo[0].floor() as isize).max(0) as usize,
                (lo[1].floor() as isize).max(0) as usize,
                (lo[2].floor() as isize).max(0) as usize,
            ],
            [
                ((hi[0].ceil() as usize) + 1).min(dims[0]),
                ((hi[1].ceil() as usize) + 1).min(dims[1]),
                ((hi[2].ceil() as usize) + 1).min(dims[2]),
            ],
        )
    });

    let (mut vertices, indices) = extract_isosurface(
        data,
        dims,
        threshold,
        grid_min,
        grid_max,
        |x, y, z| map.grid_to_cartesian_f32(x, y, z),
        color,
    );

    // Tag every vertex as a density mesh. The shared marching-cubes
    // emitter hard-codes SURFACE; re-tagging here lets the isosurface
    // shader exclude density from the molecular-surface opacity scale
    // (density opacity comes from the entry's own baked alpha).
    for v in &mut vertices {
        v.kind = isosurface_kind::DENSITY;
    }

    (vertices, indices)
}
