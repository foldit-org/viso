use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, JsonSchema)]
#[schemars(title = "Lighting", inline)]
#[serde(default)]
/// Lighting parameters controlling the scene illumination model.
pub struct LightingOptions {
    /// Key (primary) directional light intensity.
    #[schemars(title = "Key Light", range(min = 0.0, max = 3.5), extend("step" = 0.05), extend("x-group" = "Lights"))]
    pub light1_intensity: f32,
    /// Fill (secondary) directional light intensity.
    #[schemars(title = "Fill Light", range(min = 0.0, max = 2.0), extend("step" = 0.05), extend("x-group" = "Lights"))]
    pub light2_intensity: f32,
    /// Ambient light intensity.
    #[schemars(title = "Ambient", range(min = 0.0, max = 0.7), extend("step" = 0.01), extend("x-group" = "Lights"))]
    pub ambient: f32,
    /// Specular highlight intensity.
    #[schemars(skip)]
    pub specular_intensity: f32,
    /// Specular shininess exponent.
    #[schemars(skip)]
    pub shininess: f32,
    /// Rim lighting falloff exponent.
    #[schemars(title = "Rim Power", range(min = 0.5, max = 10.0), extend("step" = 0.1), extend("x-group" = "Rim"))]
    pub rim_power: f32,
    /// Rim lighting intensity multiplier.
    #[schemars(title = "Rim Intensity", range(min = 0.0, max = 1.0), extend("step" = 0.01), extend("x-group" = "Rim"))]
    pub rim_intensity: f32,
    /// How much rim light follows the view direction.
    #[schemars(skip)]
    pub rim_directionality: f32,
    /// RGB color of the rim light.
    #[schemars(skip)]
    pub rim_color: [f32; 3],
    /// Image-based lighting strength.
    #[schemars(title = "IBL Strength", range(min = 0.0, max = 1.0), extend("step" = 0.05), extend("x-group" = "Material"))]
    pub ibl_strength: f32,
    /// Surface roughness for PBR shading.
    #[schemars(title = "Roughness", range(min = 0.05, max = 1.0), extend("step" = 0.01), extend("x-group" = "Material"))]
    pub roughness: f32,
    /// Surface metalness for PBR shading.
    #[schemars(title = "Metalness", range(min = 0.0, max = 1.0), extend("step" = 0.01), extend("x-group" = "Material"))]
    pub metalness: f32,
}

impl Default for LightingOptions {
    fn default() -> Self {
        Self {
            light1_intensity: 2.0,
            light2_intensity: 1.1,
            ambient: 0.45,
            specular_intensity: 0.35,
            shininess: 38.0,
            rim_power: 5.0,
            rim_intensity: 0.3,
            rim_directionality: 0.3,
            rim_color: [1.0, 0.85, 0.7],
            ibl_strength: 0.5,
            roughness: 0.5,
            metalness: 0.5,
        }
    }
}
