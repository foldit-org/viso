//! Post-processing effect passes.
//!
//! Provides screen-space ambient occlusion (SSAO), bloom, depth fog,
//! tone-mapping composite, and FXAA anti-aliasing.

pub(crate) mod bloom;
pub(crate) mod composite;
pub(crate) mod fxaa;
pub(crate) mod overlay;
pub(crate) mod post_process;
pub(crate) mod screen_pass;
pub(crate) mod ssao;

pub(crate) use bloom::BloomPass;
pub(crate) use composite::{CompositeInputs, CompositePass};
pub(crate) use fxaa::FxaaPass;
pub(crate) use post_process::PostProcessStack;
pub(crate) use screen_pass::ScreenPass;
pub(crate) use ssao::SsaoRenderer;
