// Transient select-sphere overlay: a flat, semi-transparent shell drawn
// while a select-sphere drag gesture is active. Each instance is a single
// camera-facing billboard quad with per-pixel ray-sphere intersection.
// Unlit and flat: the colour comes straight from the host-supplied
// instance (the Rust side is the single source of truth for the tint), so
// there is no PBR, lighting, selection, or pulse lookup here.

#import viso::camera::CameraUniform
#import viso::ray::intersect_sphere
#import viso::constants::BILLBOARD_SCALE

// Twin of the Rust SelectSphereInstance.
// center: xyz = world-space centre, w = radius. color: rgb + alpha.
struct SelectSphereInstance {
    center: vec4<f32>,
    color: vec4<f32>,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_pos: vec3<f32>,
    @location(1) sphere_center: vec3<f32>,
    @location(2) radius: f32,
    @location(3) color: vec4<f32>,
};

struct FragOut {
    @builtin(frag_depth) depth: f32,
    @location(0) color: vec4<f32>,
    @location(1) normal: vec4<f32>,
};

@group(0) @binding(0) var<uniform> camera: CameraUniform;
@group(3) @binding(0) var<storage, read> spheres: array<SelectSphereInstance>;

@vertex
fn vs_main(
    @builtin(vertex_index) vidx: u32,
    @builtin(instance_index) iidx: u32,
) -> VertexOutput {
    let quad = array<vec2<f32>, 6>(
        vec2(-1.0, -1.0), vec2(1.0, -1.0), vec2(-1.0, 1.0),
        vec2(-1.0, 1.0), vec2(1.0, -1.0), vec2(1.0, 1.0)
    );

    let sph = spheres[iidx];
    let center = sph.center.xyz;
    let radius = sph.center.w;

    let to_camera = normalize(camera.position - center);

    // Build billboard basis facing the camera.
    var right = cross(to_camera, vec3<f32>(0.0, 1.0, 0.0));
    if (length(right) < 0.001) {
        right = cross(to_camera, vec3<f32>(0.0, 0.0, 1.0));
    }
    right = normalize(right);
    let up = normalize(cross(right, to_camera));

    let half_size = radius * BILLBOARD_SCALE;

    let local_uv = quad[vidx];
    let world_offset = right * local_uv.x * half_size + up * local_uv.y * half_size;
    let world_pos = center + world_offset;

    var out: VertexOutput;
    out.clip_position = camera.view_proj * vec4<f32>(world_pos, 1.0);
    out.world_pos = world_pos;
    out.sphere_center = center;
    out.radius = radius;
    out.color = sph.color;

    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> FragOut {
    let ray_origin = camera.position;
    let ray_dir = normalize(in.world_pos - camera.position);

    let t = intersect_sphere(ray_origin, ray_dir, in.sphere_center, in.radius);
    if (t < 0.0) {
        discard;
    }

    let world_hit = ray_origin + ray_dir * t;
    let normal = normalize(world_hit - in.sphere_center);

    let clip_pos = camera.view_proj * vec4<f32>(world_hit, 1.0);
    let ndc_depth = clip_pos.z / clip_pos.w;

    var out: FragOut;
    out.depth = ndc_depth;
    out.color = vec4<f32>(in.color.rgb, in.color.a);
    // Real geometric normal for outline detection; w = 0 so the flat shell
    // is not darkened by SSAO in composite.
    out.normal = vec4<f32>(normal, 0.0);
    return out;
}
