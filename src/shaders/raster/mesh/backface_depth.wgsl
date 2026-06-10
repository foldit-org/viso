// Generic back-face depth pre-pass for isosurface meshes.
//
// Renders the back-faces of every isosurface (cavity, SES, Gaussian,
// density map) into an R32Float color attachment, writing the linear
// view-space distance from the camera plane along `camera.forward`.
// The main isosurface fragment shader samples this texture to compute
//
//     thickness = back_view_z - front_view_z
//
// which feeds Beer-Lambert absorption per-kind. No discard, no
// conditionals — every back-face writes its view_z. For non-overlapping
// isosurfaces (the common case) every pixel where a back-face exists
// has the correct thickness reference. For overlapping isosurfaces the
// depth test (Less) keeps the nearest back-face; the visual artifact
// for overlap regions is a thickness discontinuity which we accept.

#import viso::camera::CameraUniform
#import viso::cavity::{cavity_displacement, ISO_KIND_CAVITY}

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) color: vec4<f32>,
    @location(3) kind: u32,
    @location(4) cavity_center: vec3<f32>,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    // Linear view-space depth (perpendicular distance to the camera
    // plane along camera.forward). Positive in front of the camera.
    @location(0) view_z: f32,
};

@group(0) @binding(0) var<uniform> camera: CameraUniform;

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;

    // Apply the SAME cavity displacement as the isosurface front-face
    // pass. The Beer-Lambert thickness is (back_view_z - front_view_z),
    // so back-faces must be displaced identically or the thickness is
    // geometrically wrong. Non-cavity kinds are unaffected.
    var pos = in.position;
    if (in.kind == ISO_KIND_CAVITY) {
        pos = pos + cavity_displacement(
            in.position, in.normal, in.cavity_center, camera.time,
        );
    }

    out.clip_position = camera.view_proj * vec4<f32>(pos, 1.0);
    out.view_z = dot(pos - camera.position, camera.forward);
    return out;
}

// Single R32Float color attachment.
@fragment
fn fs_main(in: VertexOutput) -> @location(0) f32 {
    return in.view_z;
}
