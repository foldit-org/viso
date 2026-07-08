// Bit-packed pulse lookup + attention-blink compose — shared by all
// geometry shaders.
//
// Like the selection buffer, the pulse buffer binding lives in each
// consuming shader (at varying group indices), so this module provides
// pure functions that take the relevant word by value.
//
// A residue whose bit is set oscillates the brightness of its base color on
// a per-residue phase-offset cosine cycle, drawing the player's attention
// without hiding the underlying score / selection color.

#define_import_path viso::pulse

/// Check whether `residue_idx` is marked as pulsing.
///
/// `word_count` — `arrayLength(&pulse)` from the caller.
/// `word`       — `pulse[residue_idx / 32u]` from the caller.
///
/// The bounds check on `word_count` makes an out-of-range index return
/// `false` (not pulsing) even under robust buffer access.
fn check_pulsing(residue_idx: u32, word_count: u32, word: u32) -> bool {
    let word_idx = residue_idx / 32u;
    if (word_idx >= word_count) {
        return false;
    }
    let bit_idx = residue_idx % 32u;
    return (word & (1u << bit_idx)) != 0u;
}

/// Oscillate the brightness of `base` when the residue is pulsing. Call
/// after `apply_designability` so the pulse composes on top of the score
/// color, selection tint, and whiteout. Non-pulsing residues pass through
/// unchanged.
fn apply_pulse(
    base: vec3<f32>,
    is_pulsing: bool,
    residue_index: f32,
    time: f32,
) -> vec3<f32> {
    if (!is_pulsing) {
        return base;
    }
    // Hue-preserving brightness oscillation CENTERED on the residue's own color:
    // the multiplier is 1.0 at the phase midpoint, so the base color is the
    // centerpoint of the pulse and the swing is symmetric brighter/darker.
    let brightness = 1.0 + 0.3 * cos(7.0 * time + 0.5 * residue_index);   // [0.70, 1.30]
    return base * brightness;
}
