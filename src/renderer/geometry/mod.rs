//! Molecular geometry renderers.
//!
//! Each renderer produces GPU-ready vertex/instance data for a specific
//! molecular representation: unified backbone (tubes + ribbons), sidechain
//! capsules, ball-and-stick ligands, nucleic acid rings/stems, constraint
//! bands, and interactive pulls.

pub(crate) mod backbone;
pub(crate) mod ball_and_stick;
pub(crate) mod band;
pub(crate) mod bond;
pub(crate) mod clash;
pub(crate) mod exposed_hydrophobic;
pub(crate) mod isosurface;
pub(crate) mod nucleic_acid;
pub(crate) mod pull;
pub(crate) mod sheet_adjust;
pub(crate) mod sidechain;

pub(crate) use backbone::BackboneRenderer;
pub(crate) use ball_and_stick::{
    BallAndStickRenderer, PreparedBallAndStickData,
};
pub(crate) use band::BandRenderer;
pub(crate) use bond::BondRenderer;
pub(crate) use clash::ClashArcRenderer;
pub(crate) use exposed_hydrophobic::GreaseBeadRenderer;
pub(crate) use nucleic_acid::NucleicAcidRenderer;
pub(crate) use pull::PullRenderer;
pub(crate) use sidechain::{SidechainRenderer, SidechainView};
