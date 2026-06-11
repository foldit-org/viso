#define_import_path viso::lighting

struct LightingUniform {
    light1_dir: vec3<f32>,
    light2_dir: vec3<f32>,
    light1_intensity: f32,
    light2_intensity: f32,
    ambient: f32,
    specular_intensity: f32,
    shininess: f32,
    rim_power: f32,
    rim_intensity: f32,
    rim_directionality: f32,
    rim_color: vec3<f32>,
    ibl_strength: f32,
    rim_dir: vec3<f32>,
    roughness: f32,
    metalness: f32,
    surface_opacity: f32,
};

// --- PBR functions ---

const PI: f32 = 3.14159265359;

// GGX/Trowbridge-Reitz normal distribution function
fn distribution_ggx(NdotH: f32, roughness: f32) -> f32 {
    let a = roughness * roughness;
    let a2 = a * a;
    let denom = NdotH * NdotH * (a2 - 1.0) + 1.0;
    return a2 / (PI * denom * denom);
}

// Schlick-GGX geometry function (single direction)
fn geometry_schlick_ggx(NdotV: f32, roughness: f32) -> f32 {
    let r = roughness + 1.0;
    let k = (r * r) / 8.0;
    return NdotV / (NdotV * (1.0 - k) + k);
}

// Smith's geometry function (both view and light)
fn geometry_smith(NdotV: f32, NdotL: f32, roughness: f32) -> f32 {
    return geometry_schlick_ggx(NdotV, roughness) * geometry_schlick_ggx(NdotL, roughness);
}

// Fresnel-Schlick approximation
fn fresnel_schlick(cos_theta: f32, F0: vec3<f32>) -> vec3<f32> {
    return F0 + (1.0 - F0) * pow(clamp(1.0 - cos_theta, 0.0, 1.0), 5.0);
}

// Compute rim lighting: view-dependent + optional directional back-light.
// Takes explicit parameters so this module remains self-contained.
fn compute_rim(
    normal: vec3<f32>,
    view_dir: vec3<f32>,
    rim_power: f32,
    rim_intensity: f32,
    rim_directionality: f32,
    rim_color: vec3<f32>,
    rim_dir: vec3<f32>,
) -> vec3<f32> {
    let NdotV = max(dot(normal, view_dir), 0.0);
    let rim = pow(1.0 - NdotV, rim_power);

    // Directional modulation: surfaces facing away from rim light get brighter rims
    let back_factor = max(dot(normal, -rim_dir), 0.0);
    let directional_rim = rim * mix(1.0, back_factor, rim_directionality);

    return directional_rim * rim_intensity * rim_color;
}
