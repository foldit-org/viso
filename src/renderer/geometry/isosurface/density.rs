//! Bridge between molex `Density` and the marching cubes algorithm.
//!
//! Converts crystallographic density data into triangle mesh vertices
//! suitable for GPU rendering.

use molex::entity::surface::Density;

use super::cpu_marching_cubes::extract_isosurface;
use super::{isosurface_kind, IsosurfaceVertex};

/// Generate a triangle mesh from a density map at the given sigma level.
///
/// - `map`: parsed density map (CCP4/MRC format)
/// - `threshold`: raw density threshold for isosurface extraction
/// - `color`: uniform RGBA color for all mesh vertices
///
/// Returns `(vertices, indices)` for an indexed triangle mesh.
pub(crate) fn generate_density_mesh(
    map: &Density,
    threshold: f32,
    color: [f32; 4],
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

    let (grid_min, grid_max) = ([0, 0, 0], dims);

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
