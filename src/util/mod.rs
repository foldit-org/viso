//! Shared utilities for the rendering engine.

/// Easing functions for smooth interpolation curves.
pub(crate) mod easing;
/// Pure geometric-math primitives (Newell normal, etc.).
pub(crate) mod geom;
/// Fast hashing helpers for change detection on Vec3 data.
pub(crate) mod hash;
/// Adobe / Resolve ASCII `.cube` LUT parsing (CPU-only).
pub(crate) mod lut_adobe_cube;
