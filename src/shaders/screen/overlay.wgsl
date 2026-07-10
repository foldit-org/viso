// Overlay pass - blends a host-supplied RGBA texture over the presented frame.
//
// Runs after FXAA against the swapchain view, so it samples 1:1 at window
// resolution and is never resampled by the render-scale (SSAA) chain.
//
// The source is premultiplied alpha in sRGB space, matching what every 2D
// rasterizer (cairo, Skia, Core Graphics) hands back. The blend unit, however,
// composites in whatever space the color target stores: raw for a `*Unorm`
// target, linear for a `*Srgb` one. Each entry point below prepares the source
// for one of those; `OverlayPass` picks by `TextureFormat::is_srgb`.

#import viso::fullscreen::{FullscreenVertexOutput, fullscreen_vertex}

@group(0) @binding(0) var overlay_texture: texture_2d<f32>;
@group(0) @binding(1) var tex_sampler: sampler;

@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> FullscreenVertexOutput {
    return fullscreen_vertex(vertex_index);
}

/// Non-sRGB target: the blend unit sees the stored bytes as-is, so compositing
/// happens in sRGB space and the source passes through untouched.
@fragment
fn fs_main(in: FullscreenVertexOutput) -> @location(0) vec4<f32> {
    return textureSample(overlay_texture, tex_sampler, in.uv);
}

/// sRGB target: the blend unit decodes the destination to linear before
/// compositing. Premultiplication does not commute with the transfer function,
/// so the source is un-premultiplied, decoded, and premultiplied again.
@fragment
fn fs_linear_target(in: FullscreenVertexOutput) -> @location(0) vec4<f32> {
    let source = textureSample(overlay_texture, tex_sampler, in.uv);
    if source.a <= 0.0 {
        return vec4<f32>(0.0);
    }
    return vec4<f32>(srgb_to_linear(source.rgb / source.a) * source.a, source.a);
}

fn srgb_to_linear(c: vec3<f32>) -> vec3<f32> {
    let low = c / 12.92;
    let high = pow((c + 0.055) / 1.055, vec3<f32>(2.4));
    return select(high, low, c <= vec3<f32>(0.04045));
}
