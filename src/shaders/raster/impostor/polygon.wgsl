// Procedurally extruded polygon for nucleic acid base rings.
//
// Each instance encodes up to 6 coplanar vertices, a face normal, and a
// half-thickness.  The vertex shader generates:
//   vid  0..17  top face fan  (center + 6 edges = 18 verts for 6 tris)
//   vid 18..35  bottom face fan (reversed winding)
//   vid 36..71  side quads (6 quads × 2 tris × 3 verts = 36 verts)
//
// Pentagon instances set vertex_count = 5 and v5 = centroid, so the last
// triangle of each fan collapses to zero area (degenerate).
//
// draw(0..72, 0..instance_count)

#import viso::camera::CameraUniform
#import viso::lighting::LightingUniform
#import viso::selection::check_selection
#import viso::highlight::apply_highlight
#import viso::shade::{shade_geometry, ShadingResult}
#import viso::constants::MAX_IBL_MIP

fn is_selected(residue_idx: u32) -> bool {
    return check_selection(residue_idx, arrayLength(&selection), selection[residue_idx / 32u]);
}

struct PolygonInstance {
    v0: vec4<f32>,  // xyz=position, w=vertex_count
    v1: vec4<f32>,  // xyz=position, w=half_thickness
    v2: vec4<f32>,  // xyz=position, w=centroid.x
    v3: vec4<f32>,  // xyz=position, w=centroid.y
    v4: vec4<f32>,  // xyz=position, w=centroid.z
    v5: vec4<f32>,
    normal: vec4<f32>,  // xyz=face normal, w=residue_idx
    color: vec4<f32>,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_pos: vec3<f32>,
    @location(1) world_normal: vec3<f32>,
    @location(2) base_color: vec3<f32>,
    @location(3) @interpolate(flat) residue_idx: u32,
};

struct FragOut {
    @location(0) color: vec4<f32>,
    @location(1) normal: vec4<f32>,
};

@group(0) @binding(0) var<uniform> camera: CameraUniform;
@group(1) @binding(0) var<uniform> lighting: LightingUniform;
@group(1) @binding(1) var irradiance_map: texture_cube<f32>;
@group(1) @binding(2) var env_sampler: sampler;
@group(1) @binding(3) var prefiltered_map: texture_cube<f32>;
@group(1) @binding(4) var brdf_lut: texture_2d<f32>;
@group(2) @binding(0) var<storage, read> selection: array<u32>;
@group(3) @binding(0) var<storage, read> polygons: array<PolygonInstance>;

// Access the i-th vertex (0..5) of the polygon instance
fn get_vertex(inst: PolygonInstance, i: u32) -> vec3<f32> {
    switch (i) {
        case 0u: { return inst.v0.xyz; }
        case 1u: { return inst.v1.xyz; }
        case 2u: { return inst.v2.xyz; }
        case 3u: { return inst.v3.xyz; }
        case 4u: { return inst.v4.xyz; }
        default: { return inst.v5.xyz; }
    }
}

@vertex
fn vs_main(
    @builtin(vertex_index) vid: u32,
    @builtin(instance_index) iid: u32
) -> VertexOutput {
    let inst = polygons[iid];
    let n = u32(inst.v0.w);    // 5 or 6
    let half_t = inst.v1.w;
    let face_n = inst.normal.xyz;
    let residue_idx = u32(inst.normal.w);
    let offset = face_n * half_t;

    // Centroid is precomputed CPU-side and packed into the v2/v3/v4 .w
    // slots, so it is read once here instead of re-summing all n verts
    // for each of the 72 verts in this instance.
    let centroid = vec3<f32>(inst.v2.w, inst.v3.w, inst.v4.w);

    var pos: vec3<f32>;
    var normal: vec3<f32>;

    if (vid < 18u) {
        // --- Top face fan: 6 triangles, 3 verts each ---
        let tri = vid / 3u;       // which triangle (0..5)
        let vert = vid % 3u;      // which vertex in triangle
        if (vert == 0u) {
            pos = centroid + offset;
        } else if (vert == 1u) {
            pos = get_vertex(inst, tri % n) + offset;
        } else {
            pos = get_vertex(inst, (tri + 1u) % n) + offset;
        }
        normal = face_n;
    } else if (vid < 36u) {
        // --- Bottom face fan (reversed winding): 6 triangles ---
        let local = vid - 18u;
        let tri = local / 3u;
        let vert = local % 3u;
        if (vert == 0u) {
            pos = centroid - offset;
        } else if (vert == 1u) {
            // Reversed winding: swap edge order
            pos = get_vertex(inst, (tri + 1u) % n) - offset;
        } else {
            pos = get_vertex(inst, tri % n) - offset;
        }
        normal = -face_n;
    } else {
        // --- Side walls: 6 quads = 12 triangles = 36 verts ---
        let local = vid - 36u;
        let quad = local / 6u;    // which edge (0..5)
        let qv = local % 6u;     // vertex within quad (2 tris)

        let i0 = quad % n;
        let i1 = (quad + 1u) % n;
        let p0 = get_vertex(inst, i0);
        let p1 = get_vertex(inst, i1);

        let t0 = p0 + offset;
        let t1 = p1 + offset;
        let b0 = p0 - offset;
        let b1 = p1 - offset;

        // Outward normal for this edge
        let edge = p1 - p0;
        let side_n = normalize(cross(edge, face_n));

        // Two triangles: (t0, t1, b0) and (b0, t1, b1)
        switch (qv) {
            case 0u: { pos = t0; }
            case 1u: { pos = t1; }
            case 2u: { pos = b0; }
            case 3u: { pos = b0; }
            case 4u: { pos = t1; }
            default: { pos = b1; }
        }
        normal = side_n;
    }

    // Expand selected residues outward
    if (is_selected(residue_idx)) {
        pos = pos + normal * 0.15;
    }

    var out: VertexOutput;
    out.clip_position = camera.view_proj * vec4<f32>(pos, 1.0);
    out.world_pos = pos;
    out.world_normal = normal;
    out.base_color = inst.color.xyz;
    out.residue_idx = residue_idx;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> FragOut {
    let normal = normalize(in.world_normal);
    let view_dir = normalize(camera.position - in.world_pos);

    // Highlight
    let hovered = camera.hovered_residue >= 0 && u32(camera.hovered_residue) == in.residue_idx;
    let highlighted = apply_highlight(in.base_color, hovered, is_selected(in.residue_idx));
    var base_color = highlighted.xyz;
    let outline_factor = highlighted.w;

    // Pre-sample IBL textures (modules cannot reference bindings)
    let NdotV = max(dot(normal, view_dir), 0.0);
    let R = reflect(-view_dir, normal);
    let irradiance = textureSample(irradiance_map, env_sampler, normal).rgb;
    let prefiltered = textureSampleLevel(prefiltered_map, env_sampler, R,
        lighting.roughness * MAX_IBL_MIP).rgb;
    let brdf = textureSample(brdf_lut, env_sampler, vec2<f32>(NdotV, lighting.roughness)).rg;

    let result = shade_geometry(normal, view_dir, base_color, outline_factor,
        lighting, irradiance, prefiltered, brdf);
    let final_color = result.color;
    let ambient_ratio = result.ambient_ratio;

    var out: FragOut;
    if (camera.debug_mode == 1u) {
        out.color = vec4<f32>(normal * 0.5 + 0.5, 1.0);
    } else {
        out.color = vec4<f32>(final_color, 1.0);
    }
    out.normal = vec4<f32>(normal, ambient_ratio);
    return out;
}
