//! Color palette system: preset palettes with configurable distribution modes.
//!
//! The palette separates *what colors* to use from *what property* maps to
//! color (the latter is [`ColorScheme`](crate::options::ColorScheme)). A
//! palette has a [`PalettePreset`] (which colors) and a [`PaletteMode`] (how
//! to distribute them).

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

/// How colors are distributed across data values.
#[derive(
    Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Default, JsonSchema,
)]
#[serde(rename_all = "snake_case")]
pub enum PaletteMode {
    /// Smooth continuous interpolation between gradient stops.
    Gradient,
    /// Gradient discretized into N uniform color bins.
    Stepped {
        /// Number of discrete color bins.
        steps: u32,
    },
    /// Discrete set of categorical colors (one per category).
    #[default]
    Categorical,
}

/// Named palette presets. Each resolves to concrete color stops.
#[derive(
    Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Default, JsonSchema,
)]
#[serde(rename_all = "snake_case")]
pub enum PalettePreset {
    // === Categorical (vivid) ===
    /// High-saturation 12-color palette (viso default).
    #[default]
    Vivid,
    /// Tableau 10 — professional, widely recognized.
    Tableau,
    /// Okabe-Ito — gold standard colorblind-safe (8 colors).
    OkabeIto,

    // === Categorical (muted) ===
    /// ColorBrewer Set2 — soft pastel tones (8 colors).
    Pastel,
    /// Paul Tol's muted palette — colorblind-safe, subdued (10 colors).
    Muted,

    // === Sequential gradients ===
    /// Viridis — perceptually uniform, colorblind-safe.
    Viridis,
    /// Plasma — warm purple-to-yellow.
    Plasma,
    /// Blues — single-hue blue gradient.
    Blues,
    /// Greens — single-hue green gradient.
    Greens,

    // === Diverging gradients ===
    /// Blue-White-Red (ColorBrewer RdBu). Colorblind-safe.
    BlueWhiteRed,
    /// CoolWarm (Moreland) — smooth luminance, ideal for 3D.
    CoolWarm,
    /// Green-Yellow-Red — Foldit legacy score colors. NOT colorblind-safe.
    GreenYellowRed,
    /// Brown-White-Teal (BrBG) — natural hydrophobicity mapping.
    BrownTeal,

    /// User-defined custom stops.
    Custom,
}

/// A resolved palette: concrete color data ready for use.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Palette {
    /// Which named preset to use.
    pub preset: PalettePreset,
    /// How to distribute the colors.
    pub mode: PaletteMode,
    /// Custom color stops. Used when preset is `Custom`, or to override any
    /// preset. For `Gradient`/`Stepped`: `(t, color)` pairs where `t` is in
    /// `[0, 1]`. For `Categorical`: colors in order, `t` values ignored.
    pub stops: Vec<(f32, [f32; 3])>,
}

impl JsonSchema for Palette {
    fn schema_name() -> std::borrow::Cow<'static, str> {
        std::borrow::Cow::Borrowed("Palette")
    }

    fn json_schema(
        generator: &mut schemars::SchemaGenerator,
    ) -> schemars::Schema {
        use schemars::json_schema;
        json_schema!({
            "type": "object",
            "properties": {
                "preset": generator.subschema_for::<PalettePreset>(),
                "mode": generator.subschema_for::<PaletteMode>(),
            },
        })
    }
}

impl Default for Palette {
    fn default() -> Self {
        Self {
            preset: PalettePreset::Vivid,
            mode: PaletteMode::Categorical,
            stops: Vec::new(),
        }
    }
}

impl PalettePreset {
    /// Returns the concrete RGB color stops for this preset.
    ///
    /// For gradient presets, stops are `(t, [r, g, b])` pairs where `t` is
    /// in `[0, 1]`. For categorical presets, `t` values are evenly spaced.
    #[must_use]
    pub fn default_stops(&self) -> Vec<(f32, [f32; 3])> {
        match self {
            // Viso default vivid palette (matches CHAIN_PALETTE)
            Self::Vivid => evenly_spaced(&[
                [0.05, 0.45, 1.00], // electric blue
                [1.00, 0.35, 0.00], // pure orange
                [0.00, 0.85, 0.20], // neon green
                [1.00, 0.10, 0.25], // fire red
                [0.55, 0.20, 1.00], // violet
                [1.00, 0.85, 0.00], // bright yellow
                [0.00, 0.85, 0.85], // electric cyan
                [1.00, 0.05, 0.60], // hot pink
                [0.40, 0.90, 0.00], // chartreuse
                [0.10, 0.30, 1.00], // deep blue
                [1.00, 0.50, 0.20], // tangerine
                [0.80, 0.00, 1.00], // purple
            ]),

            // Tableau 10 (sRGB linear approximations)
            Self::Tableau => evenly_spaced(&[
                [0.122, 0.467, 0.706], // blue
                [1.000, 0.498, 0.055], // orange
                [0.173, 0.627, 0.173], // green
                [0.839, 0.153, 0.157], // red
                [0.580, 0.404, 0.741], // purple
                [0.549, 0.337, 0.294], // brown
                [0.890, 0.467, 0.761], // pink
                [0.498, 0.498, 0.498], // gray
                [0.737, 0.741, 0.133], // olive
                [0.090, 0.745, 0.812], // cyan
            ]),

            // Okabe-Ito (8 colorblind-safe colors)
            Self::OkabeIto => evenly_spaced(&[
                [0.902, 0.624, 0.000], // orange
                [0.337, 0.706, 0.914], // sky blue
                [0.000, 0.620, 0.451], // bluish green
                [0.941, 0.894, 0.259], // yellow
                [0.000, 0.447, 0.698], // blue
                [0.835, 0.369, 0.000], // vermillion
                [0.800, 0.475, 0.655], // reddish purple
                [0.000, 0.000, 0.000], // black
            ]),

            // ColorBrewer Set2 (8 pastel)
            Self::Pastel => evenly_spaced(&[
                [0.400, 0.761, 0.647], // teal
                [0.988, 0.553, 0.384], // salmon
                [0.553, 0.627, 0.796], // blue-gray
                [0.906, 0.541, 0.765], // pink
                [0.651, 0.847, 0.329], // green
                [1.000, 0.851, 0.184], // yellow
                [0.898, 0.769, 0.580], // tan
                [0.702, 0.702, 0.702], // gray
            ]),

            // Paul Tol muted (10 colorblind-safe)
            Self::Muted => evenly_spaced(&[
                [0.467, 0.467, 0.780], // indigo
                [0.369, 0.651, 0.831], // cyan
                [0.529, 0.780, 0.631], // teal
                [0.624, 0.788, 0.365], // green
                [0.906, 0.831, 0.341], // olive
                [0.957, 0.667, 0.431], // sand
                [0.902, 0.506, 0.451], // rose
                [0.882, 0.369, 0.490], // wine
                [0.753, 0.412, 0.620], // purple
                [0.820, 0.820, 0.820], // pale gray
            ]),

            // Viridis 5-stop approximation
            Self::Viridis => vec![
                (0.00, [0.267, 0.004, 0.329]),
                (0.25, [0.282, 0.141, 0.458]),
                (0.50, [0.127, 0.567, 0.551]),
                (0.75, [0.455, 0.788, 0.255]),
                (1.00, [0.992, 0.906, 0.145]),
            ],

            // Plasma 5-stop approximation
            Self::Plasma => vec![
                (0.00, [0.050, 0.030, 0.528]),
                (0.25, [0.494, 0.012, 0.658]),
                (0.50, [0.798, 0.280, 0.470]),
                (0.75, [0.973, 0.585, 0.251]),
                (1.00, [0.940, 0.975, 0.131]),
            ],

            // Blues (light → dark)
            Self::Blues => vec![
                (0.00, [0.937, 0.953, 1.000]),
                (0.50, [0.416, 0.616, 0.812]),
                (1.00, [0.031, 0.188, 0.420]),
            ],

            // Greens (light → dark)
            Self::Greens => vec![
                (0.00, [0.937, 0.988, 0.937]),
                (0.50, [0.416, 0.776, 0.443]),
                (1.00, [0.000, 0.267, 0.106]),
            ],

            // Blue-White-Red (diverging)
            Self::BlueWhiteRed => vec![
                (0.00, [0.020, 0.188, 0.380]),
                (0.25, [0.263, 0.576, 0.765]),
                (0.50, [0.969, 0.969, 0.969]),
                (0.75, [0.827, 0.376, 0.302]),
                (1.00, [0.404, 0.000, 0.051]),
            ],

            // CoolWarm (Moreland)
            Self::CoolWarm => vec![
                (0.00, [0.230, 0.299, 0.754]),
                (0.50, [0.865, 0.865, 0.865]),
                (1.00, [0.706, 0.016, 0.150]),
            ],

            // Green-Yellow-Red (Foldit legacy score)
            Self::GreenYellowRed => vec![
                (0.00, [0.100, 0.800, 0.200]),
                (0.50, [1.000, 0.900, 0.100]),
                (1.00, [0.900, 0.150, 0.100]),
            ],

            // Brown-White-Teal (BrBG, diverging)
            Self::BrownTeal => vec![
                (0.00, [0.329, 0.188, 0.020]),
                (0.25, [0.749, 0.506, 0.176]),
                (0.50, [0.969, 0.969, 0.969]),
                (0.75, [0.357, 0.706, 0.667]),
                (1.00, [0.000, 0.235, 0.188]),
            ],

            // Custom: empty stops; user must provide their own.
            Self::Custom => Vec::new(),
        }
    }
}

impl Palette {
    /// Returns the effective color stops: custom stops if non-empty,
    /// otherwise the preset defaults.
    #[must_use]
    pub fn resolved_stops(&self) -> Vec<(f32, [f32; 3])> {
        if self.stops.is_empty() {
            self.preset.default_stops()
        } else {
            self.stops.clone()
        }
    }

    /// Sample the palette at position `t` in `[0, 1]`.
    ///
    /// Behavior depends on mode:
    /// - `Gradient`: smooth interpolation between stops.
    /// - `Stepped`: quantize `t` into N bins, then sample gradient.
    /// - `Categorical`: index by `floor(t * n)` with wraparound.
    #[must_use]
    pub fn sample(&self, t: f32) -> [f32; 3] {
        let stops = self.resolved_stops();
        if stops.is_empty() {
            return [0.5, 0.5, 0.5];
        }
        match &self.mode {
            PaletteMode::Gradient => lerp_stops(&stops, t.clamp(0.0, 1.0)),
            PaletteMode::Stepped { steps } => {
                let s = *steps.max(&1);
                let quantized =
                    (t.clamp(0.0, 1.0) * s as f32).floor() / s as f32;
                lerp_stops(&stops, quantized.clamp(0.0, 1.0))
            }
            PaletteMode::Categorical => {
                let n = stops.len();
                let idx = ((t.clamp(0.0, 1.0) * n as f32) as usize).min(n - 1);
                stops[idx].1
            }
        }
    }

    /// Sample with smooth gradient interpolation regardless of palette mode.
    ///
    /// Used by color schemes like `ResidueIndex` that always need a
    /// continuous N-to-C gradient.
    #[must_use]
    pub fn sample_gradient(&self, t: f32) -> [f32; 3] {
        let stops = self.resolved_stops();
        if stops.is_empty() {
            return [0.5, 0.5, 0.5];
        }
        lerp_stops(&stops, t.clamp(0.0, 1.0))
    }

    /// Direct index into categorical stops with wraparound.
    #[must_use]
    pub fn categorical_color(&self, index: usize) -> [f32; 3] {
        let stops = self.resolved_stops();
        if stops.is_empty() {
            return [0.5, 0.5, 0.5];
        }
        stops[index % stops.len()].1
    }
}

// Helpers

/// Evenly space an array of colors from `t = 0` to `t = 1`.
fn evenly_spaced(colors: &[[f32; 3]]) -> Vec<(f32, [f32; 3])> {
    let n = colors.len();
    if n == 1 {
        return vec![(0.0, colors[0])];
    }
    colors
        .iter()
        .enumerate()
        .map(|(i, &c)| (i as f32 / (n - 1) as f32, c))
        .collect()
}

/// Linearly interpolate between gradient stops at position `t`.
fn lerp_stops(stops: &[(f32, [f32; 3])], t: f32) -> [f32; 3] {
    if stops.len() == 1 {
        return stops[0].1;
    }
    // Find the two stops that bracket t
    let mut lo = 0;
    for (i, &(st, _)) in stops.iter().enumerate() {
        if st <= t {
            lo = i;
        }
    }
    let hi = (lo + 1).min(stops.len() - 1);
    if lo == hi {
        return stops[lo].1;
    }
    let range = stops[hi].0 - stops[lo].0;
    let frac = if range.abs() < 1e-6 {
        0.0
    } else {
        (t - stops[lo].0) / range
    };
    let a = &stops[lo].1;
    let b = &stops[hi].1;
    [
        a[0] + (b[0] - a[0]) * frac,
        a[1] + (b[1] - a[1]) * frac,
        a[2] + (b[2] - a[2]) * frac,
    ]
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn vivid_has_12_stops() {
        let stops = PalettePreset::Vivid.default_stops();
        assert_eq!(stops.len(), 12);
        assert!((stops[0].0 - 0.0).abs() < 1e-6);
        assert!((stops[11].0 - 1.0).abs() < 1e-6);
    }

    #[test]
    fn gradient_sample_endpoints() {
        let p = Palette {
            preset: PalettePreset::GreenYellowRed,
            mode: PaletteMode::Gradient,
            stops: Vec::new(),
        };
        let green = p.sample(0.0);
        let red = p.sample(1.0);
        assert!(green[1] > 0.7); // mostly green
        assert!(red[0] > 0.7); // mostly red
    }

    #[test]
    fn stepped_quantizes() {
        let p = Palette {
            preset: PalettePreset::Blues,
            mode: PaletteMode::Stepped { steps: 2 },
            stops: Vec::new(),
        };
        // t=0.1 and t=0.4 should both map to the first bin (t_q=0.0)
        let c1 = p.sample(0.1);
        let c2 = p.sample(0.4);
        assert!((c1[0] - c2[0]).abs() < 1e-6);
    }

    #[test]
    fn categorical_wraps() {
        let p = Palette::default(); // Vivid, 12 colors
        let c0 = p.categorical_color(0);
        let c12 = p.categorical_color(12);
        assert_eq!(c0, c12);
    }

    #[test]
    fn custom_stops_override_preset() {
        let p = Palette {
            preset: PalettePreset::Viridis,
            mode: PaletteMode::Gradient,
            stops: vec![(0.0, [1.0, 0.0, 0.0]), (1.0, [0.0, 0.0, 1.0])],
        };
        let mid = p.sample(0.5);
        assert!((mid[0] - 0.5).abs() < 1e-5);
        assert!((mid[2] - 0.5).abs() < 1e-5);
    }

    #[test]
    fn all_presets_produce_nonempty_stops() {
        let presets = [
            PalettePreset::Vivid,
            PalettePreset::Tableau,
            PalettePreset::OkabeIto,
            PalettePreset::Pastel,
            PalettePreset::Muted,
            PalettePreset::Viridis,
            PalettePreset::Plasma,
            PalettePreset::Blues,
            PalettePreset::Greens,
            PalettePreset::BlueWhiteRed,
            PalettePreset::CoolWarm,
            PalettePreset::GreenYellowRed,
            PalettePreset::BrownTeal,
        ];
        for preset in &presets {
            assert!(
                !preset.default_stops().is_empty(),
                "{preset:?} produced empty stops",
            );
        }
        // Custom should be empty
        assert!(PalettePreset::Custom.default_stops().is_empty());
    }

    #[test]
    fn palette_round_trips_through_toml() {
        let p = Palette {
            preset: PalettePreset::Viridis,
            mode: PaletteMode::Stepped { steps: 5 },
            stops: vec![(0.0, [1.0, 0.0, 0.0]), (1.0, [0.0, 0.0, 1.0])],
        };
        let toml_str = toml::to_string_pretty(&p).unwrap();
        let parsed: Palette = toml::from_str(&toml_str).unwrap();
        assert_eq!(p, parsed);
    }
}
