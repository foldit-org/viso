use std::sync::Arc;

use glam::Vec3;
use molex::entity::molecule::id::EntityId;
use molex::SSType;
use rustc_hash::FxHashMap;

use crate::engine::positions::EntityPositions;
use crate::options::{
    ColorOptions, DisplayOptions, DrawingMode, GeometryOptions,
};
use crate::renderer::entity_topology::EntityTopology;
use crate::renderer::geometry::backbone::{ChainRange, SheetOffset};
use crate::renderer::picking::PickMap;

// ---------------------------------------------------------------------------
// Shared sub-structs
// ---------------------------------------------------------------------------

/// Backbone mesh data ready for GPU upload (byte buffers).
#[derive(Clone)]
pub(crate) struct BackboneMeshData {
    /// Backbone mesh vertex bytes (shared by tube and ribbon passes).
    pub(crate) vertices: Vec<u8>,
    /// Backbone tube index bytes.
    pub(crate) tube_indices: Vec<u8>,
    /// Number of backbone tube indices.
    pub(crate) tube_index_count: u32,
    /// Backbone ribbon index bytes.
    pub(crate) ribbon_indices: Vec<u8>,
    /// Number of backbone ribbon indices.
    pub(crate) ribbon_index_count: u32,
    /// Per-residue sheet normal offsets for sidechain adjustment.
    pub(crate) sheet_offsets: Vec<SheetOffset>,
    /// Per-chain index ranges and bounding spheres for frustum culling.
    pub(crate) chain_ranges: Vec<ChainRange>,
}

/// Ball-and-stick instance data (GPU-ready byte buffers).
#[derive(Clone)]
pub(crate) struct BallAndStickInstances {
    /// Sphere instance bytes.
    pub(crate) sphere_instances: Vec<u8>,
    /// Number of spheres.
    pub(crate) sphere_count: u32,
    /// Capsule (bond) instance bytes.
    pub(crate) capsule_instances: Vec<u8>,
    /// Number of capsules.
    pub(crate) capsule_count: u32,
}

/// Nucleic acid instance data (GPU-ready byte buffers).
#[derive(Clone)]
pub(crate) struct NucleicAcidInstances {
    /// Stem capsule instance bytes.
    pub(crate) stem_instances: Vec<u8>,
    /// Number of stem instances.
    pub(crate) stem_count: u32,
    /// Ring polygon instance bytes.
    pub(crate) ring_instances: Vec<u8>,
    /// Number of ring instances.
    pub(crate) ring_count: u32,
}

// ---------------------------------------------------------------------------
// CachedEntityMesh sub-struct
// ---------------------------------------------------------------------------

/// Cached backbone data for a single entity with typed indices for offsetting.
pub(super) struct CachedBackbone {
    pub verts: Vec<u8>,
    pub tube_inds: Vec<u32>,
    pub ribbon_inds: Vec<u32>,
    pub vert_count: u32,
    pub sheet_offsets: Vec<SheetOffset>,
    pub chain_ranges: Vec<ChainRange>,
}

// ---------------------------------------------------------------------------
// Per-entity input carried on a SceneRequest::FullRebuild
// ---------------------------------------------------------------------------

/// Per-entity snapshot used by the background mesh worker.
///
/// Topology is Arc-shared across requests (stable between `Assembly`
/// syncs); positions are cloned per request because the animator writes
/// them every frame on the main thread.
#[derive(Clone)]
pub(crate) struct FullRebuildEntity {
    /// Molex entity id.
    pub(crate) id: EntityId,
    /// Monotonic cache key. Bumped when this entity's topology was
    /// rederived or the engine otherwise wants a remesh.
    pub(crate) mesh_version: u64,
    /// Resolved drawing mode.
    pub(crate) drawing_mode: DrawingMode,
    /// Immutable render-ready view (atom elements, bond list,
    /// backbone/sidechain layout, ring topology, ...).
    pub(crate) topology: Arc<EntityTopology>,
    /// Interpolated atom positions at request-build time (entity-local,
    /// parallel to `topology.atom_elements`).
    pub(crate) positions: Vec<Vec3>,
    /// Optional SS override, taking priority over `topology.ss_types`.
    pub(crate) ss_override: Option<Vec<SSType>>,
    /// Per-residue vertex colors for Cartoon-mode protein entities.
    /// `None` when the current color scheme produces no per-residue colors.
    pub(crate) per_residue_colors: Option<Vec<[f32; 3]>>,
}

/// Body of a full scene rebuild request, boxed on the enum variant to
/// keep [`SceneRequest`] compact.
pub(crate) struct FullRebuildBody {
    /// Per-entity snapshots for mesh generation.
    pub(crate) entities: Vec<FullRebuildEntity>,
    /// Current display options for mesh generation.
    pub(crate) display: DisplayOptions,
    /// Current color options for mesh generation.
    pub(crate) colors: ColorOptions,
    /// Current geometry options for mesh generation.
    pub(crate) geometry: GeometryOptions,
    /// Per-entity resolved display+geometry overrides.
    pub(crate) entity_options:
        FxHashMap<u32, (DisplayOptions, GeometryOptions)>,
    /// Rebuild generation counter (monotonically increasing).
    pub(crate) generation: u64,
}

/// Body of an animation-frame request, boxed for variant-size balance.
pub(crate) struct AnimationFrameBody {
    /// Interpolated positions keyed on entity id. The animator
    /// writes these on the main thread; the worker reads.
    pub(crate) positions: EntityPositions,
    /// Geometry options for mesh generation.
    pub(crate) geometry: GeometryOptions,
    /// Per-chain LOD overrides. When `Some`, each chain uses its own
    /// detail level instead of the global geo settings.
    pub(crate) per_chain_lod: Option<Vec<crate::options::ChainLod>>,
    /// Whether to regenerate sidechain capsules this frame.
    pub(crate) include_sidechains: bool,
    /// Rebuild generation this frame belongs to.
    pub(crate) generation: u64,
}

/// Request sent from main thread to scene processor.
pub(crate) enum SceneRequest {
    /// Full scene rebuild with per-entity derived state.
    FullRebuild(Box<FullRebuildBody>),
    /// Per-frame animation mesh generation (backbone + optional sidechains).
    ///
    /// Carries interpolated positions directly. The background thread
    /// regenerates backbone / sidechain meshes only, reusing topology
    /// + scene-state snapshots from the last `FullRebuild`.
    AnimationFrame(Box<AnimationFrameBody>),
    /// Shut down the background thread.
    Shutdown,
}

/// All pre-computed CPU data, ready for GPU-only upload on the main thread.
#[derive(Clone)]
pub(crate) struct PreparedRebuild {
    /// Rebuild generation this prepared rebuild was produced for.
    pub(crate) generation: u64,
    /// Backbone mesh data.
    pub(crate) backbone: BackboneMeshData,
    /// Sidechain capsule instance bytes.
    pub(crate) sidechain_instances: Vec<u8>,
    /// Number of sidechain capsule instances.
    pub(crate) sidechain_instance_count: u32,
    /// Ball-and-stick instance data.
    pub(crate) bns: BallAndStickInstances,
    /// Nucleic acid instance data.
    pub(crate) na: NucleicAcidInstances,
    /// Mapping from raw GPU pick IDs to typed pick targets.
    pub(crate) pick_map: PickMap,
}

/// Pre-computed animation frame data, ready for GPU upload.
#[derive(Clone)]
pub(crate) struct PreparedAnimationFrame {
    /// Backbone mesh data.
    pub(crate) backbone: BackboneMeshData,
    /// Optional sidechain capsule instance bytes.
    pub(crate) sidechain_instances: Option<Vec<u8>>,
    /// Number of sidechain capsule instances.
    pub(crate) sidechain_instance_count: u32,
    /// Rebuild generation this frame was produced for.
    pub(crate) generation: u64,
}

// ---------------------------------------------------------------------------
// Per-entity cached mesh
// ---------------------------------------------------------------------------

/// Cached mesh data for a single entity. Stored as byte buffers ready for
/// concatenation, plus typed intermediates needed for index offsetting.
pub(super) struct CachedEntityMesh {
    /// Backbone data with typed indices for concatenation.
    pub backbone: CachedBackbone,
    /// Sidechain capsule instance bytes.
    pub sidechain_instances: Vec<u8>,
    /// Number of sidechain instances.
    pub sidechain_instance_count: u32,
    /// Ball-and-stick instance data.
    pub bns: BallAndStickInstances,
    /// Nucleic acid instance data.
    pub na: NucleicAcidInstances,
    /// Number of protein backbone residues contributed by this entity.
    pub residue_count: u32,
    /// Atom count contributed to the BnS pick map (0 when this entity
    /// did not produce any ball-and-stick instances).
    pub bns_atom_count: u32,
    /// Entity id, recorded per cached mesh for pick map reconstruction.
    pub entity_id: u32,
}
