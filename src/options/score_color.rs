//! Per-residue color mapping: scheme + palette driven.
//!
//! Maps a [`ColorScheme`](crate::options::ColorScheme) (what property drives
//! color) and a [`Palette`](crate::options::Palette) (which colors to use) to
//! per-residue RGB arrays for backbone rendering.
//!
//! Score modes:
//! - **Absolute** (`score`): Fixed REU thresholds (-4 to +4).
//! - **Relative** (`score_relative`): 5th/95th percentile normalization.

/// Absolute energy thresholds in REU.
///
/// Calibrated against Foldit's per-residue distribution, not pure REU
/// "good vs. bad" intuition: even a healthy starting structure tends to
/// sit at a small positive mean per residue (e.g. ~3 REU), with isolated
/// clashes spiking into the tens. The previous `[-4, +4]` window
/// saturated every typical residue to red and erased the gradient on
/// puzzles where total game score had already crossed the pass line.
/// `[-2, +20]` keeps clearly-good residues green, places typical
/// residues in the yellow/orange band, and reserves saturated red for
/// genuine clashes.
const GOOD_THRESHOLD: f64 = -2.0;
const BAD_THRESHOLD: f64 = 20.0;

/// Flat RGB used for an entity rendered as a provisional preview, ignoring
/// its color scheme.
pub(crate) const PROVISIONAL_GRAY: [f32; 3] = [0.6, 0.6, 0.6];

/// Vertex alpha baked into a provisional entity's cartoon tube (committed
/// entities pack `1.0`). Matches the half-alpha preview ghost.
pub(crate) const PROVISIONAL_ALPHA: f32 = 0.5;

/// Absolute mode: map a per-residue energy (REU) to [0, 1] using fixed
/// thresholds.
fn score_to_t_absolute(score: f64) -> f32 {
    if score <= GOOD_THRESHOLD {
        0.0
    } else if score >= BAD_THRESHOLD {
        1.0
    } else if score <= 0.0 {
        (0.5 * (1.0 - score / GOOD_THRESHOLD)) as f32
    } else {
        (0.5 + 0.5 * score / BAD_THRESHOLD) as f32
    }
}

/// Compute per-residue colors using the scheme + palette system.
///
/// Supports all [`ColorScheme`](super::ColorScheme) variants. For schemes
/// that require data not available in the current pipeline (BFactor,
/// Hydrophobicity), falls back to neutral gray.
///
/// `entity_index` is the position of the entity within the assembly, used
/// by [`ColorScheme::Entity`](super::ColorScheme::Entity) so every entity
/// gets a distinct categorical color.
pub(crate) fn compute_per_residue_colors_styled(
    backbone_chains: &[crate::renderer::entity_topology::ProteinBackboneChain],
    ss_types: &[molex::SSType],
    per_residue_scores: &[Option<&[f64]>],
    scheme: &super::ColorScheme,
    palette: &super::palette::Palette,
    entity_index: usize,
    provisional: bool,
) -> Vec<[f32; 3]> {
    use molex::SSType;

    let residue_count = ss_types.len().max(1);
    if provisional {
        return vec![PROVISIONAL_GRAY; residue_count];
    }
    match scheme {
        super::ColorScheme::Entity => per_entity_color(
            entity_index,
            backbone_chains,
            residue_count,
            palette,
        ),
        super::ColorScheme::SecondaryStructure => {
            if ss_types.is_empty() {
                vec![[0.5, 0.5, 0.5]; residue_count]
            } else {
                ss_types
                    .iter()
                    .map(|ss| {
                        let idx = match ss {
                            SSType::Helix => 0,
                            SSType::Sheet => 1,
                            SSType::Coil => 2,
                        };
                        palette.categorical_color(idx)
                    })
                    .collect()
            }
        }
        super::ColorScheme::ResidueIndex => {
            per_chain_gradient(backbone_chains, residue_count, palette)
        }
        super::ColorScheme::Score | super::ColorScheme::ScoreRelative => {
            let mut all_scores: Vec<f64> = Vec::new();
            for &s in per_residue_scores.iter().flatten() {
                all_scores.extend_from_slice(s);
            }
            if all_scores.is_empty() {
                return vec![[0.5, 0.5, 0.5]; residue_count];
            }
            match scheme {
                super::ColorScheme::Score => {
                    per_residue_score_colors_with_palette(&all_scores, palette)
                }
                _ => per_residue_score_colors_relative_with_palette(
                    &all_scores,
                    palette,
                ),
            }
        }
        super::ColorScheme::Solid => {
            let color = palette
                .resolved_stops()
                .first()
                .map_or([0.5, 0.5, 0.5], |s| s.1);
            vec![color; residue_count]
        }
        super::ColorScheme::BFactor | super::ColorScheme::Hydrophobicity => {
            // These schemes require per-atom data not available in the
            // current pipeline. Fall back to gray.
            vec![[0.5, 0.5, 0.5]; residue_count]
        }
    }
}

/// Absolute score colors using a palette instead of the hardcoded ramp.
fn per_residue_score_colors_with_palette(
    scores: &[f64],
    palette: &super::palette::Palette,
) -> Vec<[f32; 3]> {
    scores
        .iter()
        .map(|&s| palette.sample(score_to_t_absolute(s)))
        .collect()
}

/// Relative score colors using a palette.
fn per_residue_score_colors_relative_with_palette(
    scores: &[f64],
    palette: &super::palette::Palette,
) -> Vec<[f32; 3]> {
    if scores.is_empty() {
        return Vec::new();
    }

    let mut sorted: Vec<f64> = scores.to_vec();
    sorted
        .sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let lo_idx = (sorted.len() as f64 * 0.05) as usize;
    let hi_idx = ((sorted.len() as f64 * 0.95) as usize).min(sorted.len() - 1);
    let min_score = sorted[lo_idx];
    let max_score = sorted[hi_idx];
    let range = max_score - min_score;

    scores
        .iter()
        .map(|&score| {
            let t = if range.abs() < 1e-6 {
                0.5
            } else {
                ((score - min_score) / range).clamp(0.0, 1.0) as f32
            };
            palette.sample(t)
        })
        .collect()
}

/// N→C gradient per chain using the palette.
fn per_chain_gradient(
    backbone_chains: &[crate::renderer::entity_topology::ProteinBackboneChain],
    residue_count: usize,
    palette: &super::palette::Palette,
) -> Vec<[f32; 3]> {
    if backbone_chains.is_empty() {
        return vec![[0.5, 0.5, 0.5]; residue_count];
    }
    let mut colors = Vec::with_capacity(residue_count);
    for chain in backbone_chains {
        let n_residues = chain.ca().len();
        if n_residues == 0 {
            continue;
        }
        for i in 0..n_residues {
            let t = if n_residues == 1 {
                0.0
            } else {
                i as f32 / (n_residues - 1) as f32
            };
            colors.push(palette.sample_gradient(t));
        }
    }
    colors
}

/// Solid color per entity: every residue of every chain of the entity
/// gets the same `palette.categorical_color(entity_index)`.
fn per_entity_color(
    entity_index: usize,
    backbone_chains: &[crate::renderer::entity_topology::ProteinBackboneChain],
    residue_count: usize,
    palette: &super::palette::Palette,
) -> Vec<[f32; 3]> {
    let color = palette.categorical_color(entity_index);
    if backbone_chains.is_empty() {
        return vec![color; residue_count];
    }
    backbone_chains
        .iter()
        .flat_map(|chain| std::iter::repeat_n(color, chain.ca().len()))
        .collect()
}
