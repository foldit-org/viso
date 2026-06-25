//! Marching cubes isosurface extraction.
//!
//! Pure algorithm — no GPU dependencies. Given a 3D scalar field,
//! extracts a triangle mesh at a given threshold using the classic
//! 256-entry lookup table approach.

use super::tables::{EDGE_TABLE, TRI_TABLE};
use super::IsosurfaceVertex;

/// Extract an isosurface from a 3D scalar field using marching cubes.
///
/// - `data`: flat row-major grid (`data[x*ny*nz + y*nz + z]`, ndarray C order)
/// - `dims`: full grid dimensions `[nx, ny, nz]`
/// - `threshold`: iso-level to extract
/// - `grid_min`: start of iteration sub-region (inclusive)
/// - `grid_max`: end of iteration sub-region (exclusive, clamped to dims)
/// - `grid_to_world`: maps fractional grid coords to world-space
/// - `color`: uniform color for all vertices
///
/// Returns `(vertices, indices)` for an indexed triangle mesh.
pub(crate) fn extract_isosurface(
    data: &[f32],
    dims: [usize; 3],
    threshold: f32,
    grid_min: [usize; 3],
    grid_max: [usize; 3],
    grid_to_world: impl Fn(f32, f32, f32) -> [f32; 3],
    color: [f32; 4],
) -> (Vec<IsosurfaceVertex>, Vec<u32>) {
    let [nx, ny, nz] = dims;
    if nx < 2 || ny < 2 || nz < 2 {
        return (Vec::new(), Vec::new());
    }

    // Clamp iteration bounds to valid marching-cubes range (need x+1 etc.)
    let x0 = grid_min[0];
    let y0 = grid_min[1];
    let z0 = grid_min[2];
    let x1 = grid_max[0].min(nx - 1);
    let y1 = grid_max[1].min(ny - 1);
    let z1 = grid_max[2].min(nz - 1);

    if x0 >= x1 || y0 >= y1 || z0 >= z1 {
        return (Vec::new(), Vec::new());
    }

    // Pre-estimate capacity (heuristic: ~2% of voxels produce triangles)
    let sub_volume = (x1 - x0) * (y1 - y0) * (z1 - z0);
    let estimate = sub_volume / 50;
    let mut vertices = Vec::with_capacity(estimate);
    let mut indices = Vec::with_capacity(estimate * 3);

    // ndarray row-major (C order): shape (nx, ny, nz) → last axis (z)
    // varies fastest in memory.
    let idx =
        |x: usize, y: usize, z: usize| -> usize { x * ny * nz + y * nz + z };

    for z in z0..z1 {
        for y in y0..y1 {
            for x in x0..x1 {
                let corners = [
                    data[idx(x, y, z)],
                    data[idx(x + 1, y, z)],
                    data[idx(x + 1, y + 1, z)],
                    data[idx(x, y + 1, z)],
                    data[idx(x, y, z + 1)],
                    data[idx(x + 1, y, z + 1)],
                    data[idx(x + 1, y + 1, z + 1)],
                    data[idx(x, y + 1, z + 1)],
                ];
                process_cube(
                    &corners,
                    [x as f32, y as f32, z as f32],
                    threshold,
                    data,
                    dims,
                    color,
                    &grid_to_world,
                    &mut vertices,
                    &mut indices,
                );
            }
        }
    }

    weld_vertices(&mut vertices, &mut indices);
    laplacian_smooth(&mut vertices, &indices, 3, 0.4);

    // Flip triangle winding so front faces point toward the camera
    // (marching cubes tables produce inside-out winding for our
    // threshold convention).
    for tri in indices.chunks_exact_mut(3) {
        tri.swap(1, 2);
    }

    (vertices, indices)
}

/// Merge coincident vertices and average their normals for smooth
/// shading across cube boundaries.
///
/// Uses spatial hashing to find vertices at the same position (within
/// a small epsilon). Merged vertices get averaged normals, giving
/// smooth interpolation across adjacent marching-cubes cells.
fn weld_vertices(vertices: &mut Vec<IsosurfaceVertex>, indices: &mut [u32]) {
    use std::collections::HashMap;

    if vertices.is_empty() {
        return;
    }

    // Quantize positions to a grid for hashing. Use a scale that
    // merges vertices within ~1e-4 of each other.
    const SCALE: f32 = 10000.0;
    let quantize = |v: [f32; 3]| -> (i32, i32, i32) {
        (
            (v[0] * SCALE).round() as i32,
            (v[1] * SCALE).round() as i32,
            (v[2] * SCALE).round() as i32,
        )
    };

    // Map: quantized position → new vertex index
    let mut pos_to_idx: HashMap<(i32, i32, i32), u32> =
        HashMap::with_capacity(vertices.len() / 2);
    // Accumulated normals + count for averaging
    let mut new_verts: Vec<IsosurfaceVertex> =
        Vec::with_capacity(vertices.len() / 2);
    let mut normal_counts: Vec<u32> = Vec::with_capacity(vertices.len() / 2);
    // Old index → new index
    let mut remap: Vec<u32> = Vec::with_capacity(vertices.len());

    for v in vertices.iter() {
        let key = quantize(v.position);
        let new_idx = *pos_to_idx.entry(key).or_insert_with(|| {
            let idx = new_verts.len() as u32;
            new_verts.push(*v);
            normal_counts.push(0);
            idx
        });
        // Accumulate normal for averaging
        let nv = &mut new_verts[new_idx as usize];
        nv.normal[0] += v.normal[0];
        nv.normal[1] += v.normal[1];
        nv.normal[2] += v.normal[2];
        normal_counts[new_idx as usize] += 1;
        remap.push(new_idx);
    }

    // Normalize accumulated normals
    for (v, &count) in new_verts.iter_mut().zip(&normal_counts) {
        if count > 1 {
            let len = (v.normal[0] * v.normal[0]
                + v.normal[1] * v.normal[1]
                + v.normal[2] * v.normal[2])
                .sqrt();
            if len > 1e-10 {
                v.normal[0] /= len;
                v.normal[1] /= len;
                v.normal[2] /= len;
            }
        }
    }

    // Remap indices
    for idx in indices.iter_mut() {
        *idx = remap[*idx as usize];
    }

    *vertices = new_verts;
}

/// Laplacian smoothing: iteratively move each vertex toward the average
/// of its edge-connected neighbors. Smooths geometric staircase
/// artifacts from marching cubes without changing topology.
fn laplacian_smooth(
    vertices: &mut [IsosurfaceVertex],
    indices: &[u32],
    iterations: u32,
    lambda: f32,
) {
    if vertices.is_empty() || indices.is_empty() {
        return;
    }

    // Build adjacency: for each vertex, which vertices are neighbors
    let n = vertices.len();
    let mut neighbors: Vec<Vec<u32>> = vec![Vec::new(); n];
    for tri in indices.chunks_exact(3) {
        let (a, b, c) = (tri[0], tri[1], tri[2]);
        neighbors[a as usize].push(b);
        neighbors[a as usize].push(c);
        neighbors[b as usize].push(a);
        neighbors[b as usize].push(c);
        neighbors[c as usize].push(a);
        neighbors[c as usize].push(b);
    }
    // Deduplicate neighbor lists
    for nbrs in &mut neighbors {
        nbrs.sort_unstable();
        nbrs.dedup();
    }

    let mut new_pos = vec![[0.0f32; 3]; n];

    for _ in 0..iterations {
        for (i, nbrs) in neighbors.iter().enumerate() {
            if nbrs.is_empty() {
                new_pos[i] = vertices[i].position;
                continue;
            }
            let mut avg = [0.0f32; 3];
            for &j in nbrs {
                let p = &vertices[j as usize].position;
                avg[0] += p[0];
                avg[1] += p[1];
                avg[2] += p[2];
            }
            let k = nbrs.len() as f32;
            avg[0] /= k;
            avg[1] /= k;
            avg[2] /= k;

            let cur = &vertices[i].position;
            new_pos[i] = [
                cur[0] + lambda * (avg[0] - cur[0]),
                cur[1] + lambda * (avg[1] - cur[1]),
                cur[2] + lambda * (avg[2] - cur[2]),
            ];
        }
        for (i, v) in vertices.iter_mut().enumerate() {
            v.position = new_pos[i];
        }
    }
}

/// Process a single marching-cubes cell, appending triangles to the
/// output buffers.
fn process_cube(
    corners: &[f32; 8],
    origin: [f32; 3],
    threshold: f32,
    data: &[f32],
    dims: [usize; 3],
    color: [f32; 4],
    grid_to_world: &impl Fn(f32, f32, f32) -> [f32; 3],
    vertices: &mut Vec<IsosurfaceVertex>,
    indices: &mut Vec<u32>,
) {
    let cube_index = classify_corners(corners, threshold);
    let edge_bits = EDGE_TABLE[cube_index as usize];
    if edge_bits == 0 {
        return;
    }

    let edge_verts = interpolate_edges(corners, edge_bits, threshold, origin);
    let base = vertices.len() as u32;
    let tri_row = &TRI_TABLE[cube_index as usize];

    let mut edge_emitted = [u32::MAX; 12];
    let mut local_count = 0u32;

    let mut i = 0;
    while tri_row[i] != -1 {
        let edge = tri_row[i] as usize;
        if edge_emitted[edge] == u32::MAX {
            let ev = edge_verts[edge];
            vertices.push(IsosurfaceVertex {
                position: grid_to_world(ev[0], ev[1], ev[2]),
                normal: gradient_normal(data, dims, ev),
                color,
                kind: super::isosurface_kind::SURFACE,
                cavity_center: [0.0; 3],
            });
            edge_emitted[edge] = base + local_count;
            local_count += 1;
        }
        indices.push(edge_emitted[edge]);
        i += 1;
    }
}

/// Build cube index from corner classification (above/below threshold).
fn classify_corners(corners: &[f32; 8], threshold: f32) -> u8 {
    let mut cube_index: u8 = 0;
    for (i, &val) in corners.iter().enumerate() {
        if val >= threshold {
            cube_index |= 1 << i;
        }
    }
    cube_index
}

/// Interpolate vertex positions along the 12 edges of a cube.
fn interpolate_edges(
    corners: &[f32; 8],
    edge_bits: u16,
    threshold: f32,
    origin: [f32; 3],
) -> [[f32; 3]; 12] {
    let mut verts = [[0.0f32; 3]; 12];
    let [x, y, z] = origin;

    // Edge endpoints in local cube coordinates
    const EDGE_ENDPOINTS: [([f32; 3], [f32; 3], usize, usize); 12] = [
        ([0.0, 0.0, 0.0], [1.0, 0.0, 0.0], 0, 1),
        ([1.0, 0.0, 0.0], [1.0, 1.0, 0.0], 1, 2),
        ([0.0, 1.0, 0.0], [1.0, 1.0, 0.0], 3, 2),
        ([0.0, 0.0, 0.0], [0.0, 1.0, 0.0], 0, 3),
        ([0.0, 0.0, 1.0], [1.0, 0.0, 1.0], 4, 5),
        ([1.0, 0.0, 1.0], [1.0, 1.0, 1.0], 5, 6),
        ([0.0, 1.0, 1.0], [1.0, 1.0, 1.0], 7, 6),
        ([0.0, 0.0, 1.0], [0.0, 1.0, 1.0], 4, 7),
        ([0.0, 0.0, 0.0], [0.0, 0.0, 1.0], 0, 4),
        ([1.0, 0.0, 0.0], [1.0, 0.0, 1.0], 1, 5),
        ([1.0, 1.0, 0.0], [1.0, 1.0, 1.0], 2, 6),
        ([0.0, 1.0, 0.0], [0.0, 1.0, 1.0], 3, 7),
    ];

    for (i, &(p0, p1, c0, c1)) in EDGE_ENDPOINTS.iter().enumerate() {
        if edge_bits & (1 << i) == 0 {
            continue;
        }
        let v0 = corners[c0];
        let v1 = corners[c1];
        let t = if (v1 - v0).abs() < 1e-10 {
            0.5
        } else {
            (threshold - v0) / (v1 - v0)
        };
        verts[i] = [
            x + p0[0] + t * (p1[0] - p0[0]),
            y + p0[1] + t * (p1[1] - p0[1]),
            z + p0[2] + t * (p1[2] - p0[2]),
        ];
    }

    verts
}

/// Sample the scalar field at a fractional position using trilinear
/// interpolation.
fn sample_trilinear(data: &[f32], dims: [usize; 3], pos: [f32; 3]) -> f32 {
    let [nx, ny, nz] = dims;
    let x0 = (pos[0].floor() as usize).min(nx.saturating_sub(2));
    let y0 = (pos[1].floor() as usize).min(ny.saturating_sub(2));
    let z0 = (pos[2].floor() as usize).min(nz.saturating_sub(2));
    let x1 = x0 + 1;
    let y1 = y0 + 1;
    let z1 = z0 + 1;

    let tx = (pos[0] - x0 as f32).clamp(0.0, 1.0);
    let ty = (pos[1] - y0 as f32).clamp(0.0, 1.0);
    let tz = (pos[2] - z0 as f32).clamp(0.0, 1.0);

    let idx = |x: usize, y: usize, z: usize| -> f32 {
        data[x * ny * nz + y * nz + z]
    };

    let c00 = idx(x0, y0, z0).mul_add(1.0 - tx, idx(x1, y0, z0) * tx);
    let c01 = idx(x0, y0, z1).mul_add(1.0 - tx, idx(x1, y0, z1) * tx);
    let c10 = idx(x0, y1, z0).mul_add(1.0 - tx, idx(x1, y1, z0) * tx);
    let c11 = idx(x0, y1, z1).mul_add(1.0 - tx, idx(x1, y1, z1) * tx);

    let c0 = c00.mul_add(1.0 - ty, c10 * ty);
    let c1 = c01.mul_add(1.0 - ty, c11 * ty);

    c0.mul_add(1.0 - tz, c1 * tz)
}

/// Compute surface normal via central-difference gradient with
/// trilinear interpolation at the exact fractional grid position.
///
/// Produces smoothly varying normals across voxel boundaries, avoiding
/// the blocky appearance of integer-truncated gradient sampling.
fn gradient_normal(data: &[f32], dims: [usize; 3], pos: [f32; 3]) -> [f32; 3] {
    // Use a half-voxel offset for central differences
    const H: f32 = 0.5;

    let dx = sample_trilinear(data, dims, [pos[0] + H, pos[1], pos[2]])
        - sample_trilinear(data, dims, [pos[0] - H, pos[1], pos[2]]);
    let dy = sample_trilinear(data, dims, [pos[0], pos[1] + H, pos[2]])
        - sample_trilinear(data, dims, [pos[0], pos[1] - H, pos[2]]);
    let dz = sample_trilinear(data, dims, [pos[0], pos[1], pos[2] + H])
        - sample_trilinear(data, dims, [pos[0], pos[1], pos[2] - H]);

    let len = (dx * dx + dy * dy + dz * dz).sqrt();
    if len < 1e-10 {
        [0.0, 1.0, 0.0]
    } else {
        [-dx / len, -dy / len, -dz / len]
    }
}

#[cfg(test)]
#[allow(clippy::panic, clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    /// Sphere SDF: positive inside, negative outside (inverted for
    /// marching cubes which extracts where value >= threshold).
    fn sphere_sdf(dims: [usize; 3], center: [f32; 3], radius: f32) -> Vec<f32> {
        let [nx, ny, nz] = dims;
        let mut data = vec![0.0f32; nx * ny * nz];
        for z in 0..nz {
            for y in 0..ny {
                for x in 0..nx {
                    let dx = x as f32 - center[0];
                    let dy = y as f32 - center[1];
                    let dz = z as f32 - center[2];
                    let dist = (dx * dx + dy * dy + dz * dz).sqrt();
                    // Positive inside the sphere
                    data[x * ny * nz + y * nz + z] = radius - dist;
                }
            }
        }
        data
    }

    #[test]
    fn sphere_produces_triangles() {
        let dims = [20, 20, 20];
        let data = sphere_sdf(dims, [10.0, 10.0, 10.0], 5.0);
        let (verts, indices) = extract_isosurface(
            &data,
            dims,
            0.0,
            [0, 0, 0],
            dims,
            |x, y, z| [x, y, z],
            [1.0, 1.0, 1.0, 1.0],
        );

        assert!(!verts.is_empty(), "should produce vertices");
        assert!(!indices.is_empty(), "should produce indices");
        assert_eq!(indices.len() % 3, 0, "indices must be triangle multiples");
    }

    #[test]
    fn normals_point_outward() {
        let dims = [30, 30, 30];
        let center = [15.0, 15.0, 15.0];
        let data = sphere_sdf(dims, center, 8.0);
        let (verts, _) = extract_isosurface(
            &data,
            dims,
            0.0,
            [0, 0, 0],
            dims,
            |x, y, z| [x, y, z],
            [1.0, 1.0, 1.0, 1.0],
        );

        let mut outward_count = 0;
        for v in &verts {
            let to_center = [
                center[0] - v.position[0],
                center[1] - v.position[1],
                center[2] - v.position[2],
            ];
            let dot = v.normal[0] * to_center[0]
                + v.normal[1] * to_center[1]
                + v.normal[2] * to_center[2];
            // Normal should point away from center (negative dot)
            if dot < 0.0 {
                outward_count += 1;
            }
        }
        let ratio = outward_count as f32 / verts.len() as f32;
        assert!(
            ratio > 0.9,
            "at least 90% of normals should point outward, got {ratio:.1}%"
        );
    }

    #[test]
    fn empty_field_produces_nothing() {
        let dims = [5, 5, 5];
        let data = vec![0.0f32; 125];
        let (verts, indices) = extract_isosurface(
            &data,
            dims,
            1.0,
            [0, 0, 0],
            dims,
            |x, y, z| [x, y, z],
            [1.0, 0.0, 0.0, 1.0],
        );
        assert!(verts.is_empty());
        assert!(indices.is_empty());
    }

    #[test]
    fn few_degenerate_triangles() {
        let dims = [20, 20, 20];
        let data = sphere_sdf(dims, [10.0, 10.0, 10.0], 5.0);
        let (verts, indices) = extract_isosurface(
            &data,
            dims,
            0.0,
            [0, 0, 0],
            dims,
            |x, y, z| [x, y, z],
            [1.0, 1.0, 1.0, 1.0],
        );

        let total = indices.len() / 3;
        let mut degenerate = 0;
        for tri in indices.chunks_exact(3) {
            let a = verts[tri[0] as usize].position;
            let b = verts[tri[1] as usize].position;
            let c = verts[tri[2] as usize].position;

            let ab = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
            let ac = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];
            let cross = [
                ab[1] * ac[2] - ab[2] * ac[1],
                ab[2] * ac[0] - ab[0] * ac[2],
                ab[0] * ac[1] - ab[1] * ac[0],
            ];
            let area = (cross[0] * cross[0]
                + cross[1] * cross[1]
                + cross[2] * cross[2])
                .sqrt();
            if area < 1e-10 {
                degenerate += 1;
            }
        }
        // Marching cubes produces degenerate triangles at axis-aligned
        // grid intersections and near-vertex iso-crossings. These are
        // harmless for rendering (zero-area fragments are GPU-discarded).
        // Tolerate up to 25%.
        let ratio = degenerate as f32 / total as f32;
        assert!(
            ratio < 0.25,
            "too many degenerate triangles: {degenerate}/{total} ({ratio:.1}%)"
        );
    }
}
