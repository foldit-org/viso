// Bit-packed non-designable lookup + desaturate-toward-white compose —
// shared by all geometry shaders.
//
// Like the selection buffer, the non-designable buffer binding lives in
// each consuming shader (at varying group indices), so this module provides
// pure functions that take the relevant word by value.
//
// A residue whose bit is set may not be designed (mutated) in the current
// puzzle. Such residues are desaturated toward white so the player can see
// at a glance which parts of the structure are locked, while the underlying
// score / selection color stays legible through the wash.

#define_import_path viso::designability

// Fraction of the way toward white a locked residue's color is pushed.
// Single fixed factor for v1 (no tiers / outline / stipple).
const DESIGN_WHITEOUT: f32 = 0.6;

/// Check whether `residue_idx` is marked non-designable.
///
/// `word_count` — `arrayLength(&non_designable)` from the caller.
/// `word`       — `non_designable[residue_idx / 32u]` from the caller.
///
/// With robust buffer access the caller's word read is always safe; this
/// function performs the bounds check on `word_count` so an out-of-range
/// index still returns `false` (designable).
fn check_non_designable(residue_idx: u32, word_count: u32, word: u32) -> bool {
    let word_idx = residue_idx / 32u;
    if (word_idx >= word_count) {
        return false;
    }
    let bit_idx = residue_idx % 32u;
    return (word & (1u << bit_idx)) != 0u;
}

/// Desaturate `base_color` toward white when the residue is non-designable.
///
/// Call after `apply_highlight` so the whiteout composes on top of the
/// score color and selection tint. Designable residues pass through
/// unchanged.
fn apply_designability(
    base_color: vec3<f32>,
    is_non_designable: bool,
) -> vec3<f32> {
    if (is_non_designable) {
        return mix(base_color, vec3<f32>(1.0, 1.0, 1.0), DESIGN_WHITEOUT);
    }
    return base_color;
}
