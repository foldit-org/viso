#import viso::camera::CameraUniform
#import viso::lighting::{LightingUniform, compute_rim}
#import viso::shade::{shade_geometry, ShadingResult}
#import viso::constants::MAX_IBL_MIP
#import viso::cavity::{cavity_displacement, ISO_KIND_CAVITY, ISO_KIND_SURFACE}

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) color: vec4<f32>,
    @location(3) kind: u32,
    @location(4) cavity_center: vec3<f32>,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_position: vec3<f32>,
    @location(1) world_normal: vec3<f32>,
    @location(2) vertex_color: vec4<f32>,
    @location(3) @interpolate(flat) kind: u32,
    @location(4) view_z: f32,
};

@group(0) @binding(0) var<uniform> camera: CameraUniform;
@group(1) @binding(0) var<uniform> lighting: LightingUniform;
@group(1) @binding(1) var irradiance_map: texture_cube<f32>;
@group(1) @binding(2) var env_sampler: sampler;
@group(1) @binding(3) var prefiltered_map: texture_cube<f32>;
@group(1) @binding(4) var brdf_lut: texture_2d<f32>;
@group(2) @binding(0) var backface_depth_tex: texture_2d<f32>;

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;

    // Lava-lamp displacement: bounded sinusoidal motion around the rest
    // position, gated on cavity kind. Surface meshes (SES / Gaussian /
    // density) are unaffected.
    var pos = in.position;
    if (in.kind == ISO_KIND_CAVITY) {
        pos = pos + cavity_displacement(
            in.position, in.normal, in.cavity_center, camera.time,
        );
    }

    out.clip_position = camera.view_proj * vec4<f32>(pos, 1.0);
    out.world_position = pos;
    out.world_normal = in.normal;
    out.vertex_color = in.color;
    out.kind = in.kind;
    out.view_z = dot(pos - camera.position, camera.forward);
    return out;
}

struct FragOutput {
    @location(0) color: vec4<f32>,
    @location(1) normal: vec4<f32>,
};

@fragment
fn fs_main(in: VertexOutput) -> FragOutput {
    let normal = normalize(in.world_normal);
    let view_dir = normalize(camera.position - in.world_position);

    // Pre-sample IBL textures
    let NdotV = max(dot(normal, view_dir), 0.0);
    let R = reflect(-view_dir, normal);
    let irradiance = textureSample(irradiance_map, env_sampler, normal).rgb;
    let prefiltered = textureSampleLevel(prefiltered_map, env_sampler, R,
        lighting.roughness * MAX_IBL_MIP).rgb;
    let brdf = textureSample(brdf_lut, env_sampler, vec2<f32>(NdotV, lighting.roughness)).rg;

    let result = shade_geometry(normal, view_dir, in.vertex_color.rgb, 0.0,
        lighting, irradiance, prefiltered, brdf);

    var final_color = result.color;
    var final_alpha = in.vertex_color.a;

    // Beer-Lambert thickness absorption. The back-face pre-pass wrote
    // linear view-space z for every isosurface back-face into
    // `backface_depth_tex`. We sample it at this fragment's screen
    // position and subtract this fragment's view_z to get the
    // thickness of the isosurface slab along the view ray. Per-kind
    // absorption coefficients produce the right look for each kind:
    //
    //   - CAVITY : strong blue-biased absorption → deep saturated blue
    //              centers, clear edges. Makes cavities read as dense
    //              pockets of glowing gel.
    //   - SURFACE : mild neutral absorption → center slightly
    //              more opaque than edges, gives the translucent shell
    //              a sense of depth without heavy tinting.
    let pixel_coord = vec2<i32>(in.clip_position.xy);
    let back_z = textureLoad(backface_depth_tex, pixel_coord, 0).r;
    let thickness = max(back_z - in.view_z, 0.0);

    var absorption: f32;
    if (in.kind == ISO_KIND_CAVITY) {
        absorption = 0.35;
    } else {
        absorption = 0.10;
    }
    let opacity = 1.0 - exp(-thickness * absorption);
    // For molecular surfaces, a sentinel (negative) baked alpha means "use the
    // global surface-opacity uniform"; a non-negative baked alpha is a per-entity
    // absolute opacity and is used as-is. Cavities and density keep their baked alpha.
    // The opacity factor scales the volumetric Beer-Lambert absorption at
    // low/mid values but reaches fully opaque at 1.0: the absorption factor is
    // mixed toward 1.0 as the factor rises, so a maxed slider renders solid.
    var a = in.vertex_color.a;
    if (in.kind == ISO_KIND_SURFACE) {
        a = select(in.vertex_color.a, lighting.surface_opacity, in.vertex_color.a < 0.0);
    }
    final_alpha = a * mix(opacity, 1.0, a);

    // Cavity-specific rim, layered over the PBR pass. At the silhouette
    // thickness ≈ 0 so the Beer-Lambert opacity also ≈ 0, which would
    // make the rim invisible after alpha blending. Use the rim
    // magnitude itself as a floor on final_alpha so the brightest part
    // of the rim always gets enough alpha to be visible.
    if (in.kind == ISO_KIND_CAVITY) {
        let cavity_rim = compute_rim(
            normal,
            view_dir,
            2.0,
            1.0,
            0.0,
            vec3<f32>(0.40, 0.60, 1.00),
            vec3<f32>(0.0, -1.0, 0.0),
        );
        final_color = final_color + cavity_rim;

        let rim_strength = (cavity_rim.r + cavity_rim.g + cavity_rim.b) / 3.0;
        final_alpha = max(final_alpha, rim_strength);
    }

    var out: FragOutput;
    if (camera.debug_mode == 1u) {
        out.color = vec4<f32>(normal * 0.5 + 0.5, final_alpha);
    } else {
        out.color = vec4<f32>(final_color, final_alpha);
    }
    out.normal = vec4<f32>(normal, result.ambient_ratio);
    return out;
}
