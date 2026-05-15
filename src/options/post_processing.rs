use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, JsonSchema)]
#[schemars(title = "Effects", inline)]
#[serde(default)]
/// Post-processing effect parameters for the rendering pipeline.
pub struct PostProcessingOptions {
    /// Thickness of depth-based outlines in pixels.
    #[schemars(title = "Outline Thickness", range(min = 0.0, max = 2.0), extend("step" = 0.1), extend("x-group" = "Outlines"))]
    pub outline_thickness: f32,
    /// Strength multiplier for depth-based outlines.
    #[schemars(title = "Outline Strength", range(min = 0.0, max = 1.0), extend("step" = 0.01), extend("x-group" = "Outlines"))]
    pub outline_strength: f32,
    /// Ambient occlusion intensity.
    #[schemars(title = "AO Strength", range(min = 0.0, max = 1.5), extend("step" = 0.05), extend("x-group" = "Ambient Occlusion"))]
    pub ao_strength: f32,
    /// Sampling radius for ambient occlusion.
    #[schemars(title = "AO Radius", range(min = 0.1, max = 2.0), extend("step" = 0.05), extend("x-group" = "Ambient Occlusion"))]
    pub ao_radius: f32,
    /// Bias offset to prevent AO self-shadowing.
    #[schemars(skip)]
    pub ao_bias: f32,
    /// Exponent controlling AO falloff curve.
    #[schemars(skip)]
    pub ao_power: f32,
    /// Distance at which distance fog begins.
    #[schemars(title = "Fog Start", range(min = 10.0, max = 500.0), extend("step" = 5.0), extend("x-group" = "Fog"))]
    pub fog_start: f32,
    /// Exponential fog density factor.
    #[schemars(title = "Fog Density", range(min = 0.0, max = 0.02), extend("step" = 0.001), extend("x-group" = "Fog"))]
    pub fog_density: f32,
    /// Tone-mapping exposure value.
    #[schemars(title = "Exposure", range(min = 0.5, max = 2.0), extend("step" = 0.05), extend("x-group" = "Tone Mapping"))]
    pub exposure: f32,
    /// Strength of normal-discontinuity outlines.
    #[schemars(title = "Normal Outline", range(min = 0.0, max = 1.0), extend("step" = 0.01), extend("x-group" = "Outlines"))]
    pub normal_outline_strength: f32,
    /// Bloom glow intensity multiplier.
    #[schemars(title = "Bloom Intensity", range(min = 0.0, max = 0.5), extend("step" = 0.01), extend("x-group" = "Bloom"))]
    pub bloom_intensity: f32,
    /// Luminance threshold for bloom extraction.
    #[schemars(title = "Bloom Threshold", range(min = 0.5, max = 2.0), extend("step" = 0.05), extend("x-group" = "Bloom"))]
    pub bloom_threshold: f32,
    /// Optional path to an Adobe ASCII `.cube` LUT file (native: loaded on apply).
    #[schemars(title = "Adobe LUT Path", extend("x-group" = "Color LUT"))]
    pub adobe_cube_lut_path: Option<String>,
}

impl Default for PostProcessingOptions {
    fn default() -> Self {
        Self {
            outline_thickness: 1.0,
            outline_strength: 0.7,
            ao_strength: 0.85,
            ao_radius: 0.5,
            ao_bias: 0.025,
            ao_power: 2.0,
            fog_start: 100.0,
            fog_density: 0.005,
            exposure: 1.0,
            normal_outline_strength: 0.5,
            bloom_intensity: 0.0,
            bloom_threshold: 1.0,
            adobe_cube_lut_path: None,
        }
    }
}
