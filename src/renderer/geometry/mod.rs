//! Molecular geometry renderers.
//!
//! Each renderer produces GPU-ready vertex/instance data for a specific
//! molecular representation: unified backbone (tubes + ribbons), sidechain
//! capsules, ball-and-stick ligands, nucleic acid rings/stems, constraint
//! bands, and interactive pulls.

/// Unified backbone renderer (protein + nucleic acid).
pub(crate) mod backbone;
/// Ball-and-stick renderer for ligands, ions, and waters.
pub(crate) mod ball_and_stick;
/// Constraint band renderer (pulls, H-bonds, disulfides).
pub(crate) mod band;
/// Structural bond renderer (H-bonds, disulfide bonds).
pub(crate) mod bond;
/// Isosurface mesh renderer (electron density maps).
pub(crate) mod isosurface;
/// Nucleic acid ring + stem renderer.
pub(crate) mod nucleic_acid;
/// Interactive pull arrow renderer.
pub(crate) mod pull;
/// Sheet-surface sidechain position adjustment.
pub(crate) mod sheet_adjust;
/// Capsule sidechain renderer.
pub(crate) mod sidechain;

pub(crate) use backbone::BackboneRenderer;
pub(crate) use ball_and_stick::{
    BallAndStickRenderer, PreparedBallAndStickData,
};
pub(crate) use band::BandRenderer;
pub(crate) use bond::BondRenderer;
pub(crate) use nucleic_acid::NucleicAcidRenderer;
pub(crate) use pull::PullRenderer;
pub(crate) use sidechain::{SidechainRenderer, SidechainView};
