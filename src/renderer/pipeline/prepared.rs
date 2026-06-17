use std::sync::Arc;

use glam::Vec3;
use molex::entity::molecule::id::EntityId;
use molex::entity::surface::Density;
use molex::SSType;
use rustc_hash::FxHashMap;

use crate::engine::positions::EntityPositions;
use crate::options::{
    ColorOptions, DisplayOptions, DrawingMode, GeometryOptions,
};
use crate::renderer::entity_topology::EntityTopology;
use crate::renderer::geometry::backbone::{
    ChainRange, RibbonAnchor, SheetOffset,
};
use crate::renderer::geometry::isosurface::IsosurfaceVertex;
use crate::renderer::picking::PickMap;

// Shared sub-structs

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
    /// Per-residue ribbon anchors sampled from the drawn centerline.
    /// CPU-only side channel (never uploaded to the GPU), used to attach
    /// structural-bond and clash markers to the rendered ribbon.
    pub(crate) ribbon_anchors: Vec<RibbonAnchor>,
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

// CachedEntityMesh sub-struct

/// Cached backbone data for a single entity with typed indices for offsetting.
pub(super) struct CachedBackbone {
    pub verts: Vec<u8>,
    pub tube_inds: Vec<u32>,
    pub ribbon_inds: Vec<u32>,
    pub vert_count: u32,
    pub sheet_offsets: Vec<SheetOffset>,
    pub ribbon_anchors: Vec<RibbonAnchor>,
    pub chain_ranges: Vec<ChainRange>,
}

// Per-entity input carried on a SceneRequest::FullRebuild

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
    /// Rebuild generation counter (monotonically increasing, bumped on
    /// every submit). Used for animation-frame staleness.
    pub(crate) generation: u64,
    /// Topology generation: advances only when the visible entity-id set
    /// changes (entity added/removed). Carried through so the consumer
    /// can keep a rebuild whose topology still matches the current scene
    /// even when newer same-topology submits have bumped `generation`.
    pub(crate) topology_generation: u64,
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
    /// Topology generation this frame belongs to. The consumer discards a
    /// frame only when this is behind the current topology generation (the
    /// entity-id set changed since it was built); a newer same-topology
    /// coordinate rebuild does not invalidate it.
    pub(crate) topology_generation: u64,
}

/// Which kind of molecular surface a [`SurfaceJob`] asks the worker to
/// extract. The scalar counterpart of the engine-side `SurfaceKind`,
/// flattened here so the worker never needs the engine's `EntitySurface`.
#[derive(Clone, Copy)]
pub(crate) enum SurfaceJobKind {
    /// Smooth Gaussian blob surface.
    Gaussian,
    /// Solvent-excluded / Connolly surface.
    Ses,
}

/// One entity-surface generation job, flattened to the scalar parameters
/// the isosurface generators consume so the worker stays free of the
/// engine-side `EntitySurface` type.
pub(crate) struct SurfaceJob {
    /// Atom world-space positions (Angstroms).
    pub(crate) positions: Vec<Vec3>,
    /// Per-atom van der Waals radii (Angstroms), parallel to `positions`.
    pub(crate) radii: Vec<f32>,
    /// Which surface to extract.
    pub(crate) kind: SurfaceJobKind,
    /// Grid resolution in Angstroms (lower = finer).
    pub(crate) resolution: f32,
    /// Probe radius for SES (Angstroms).
    pub(crate) probe_radius: f32,
    /// Gaussian isosurface level (only used for the Gaussian kind).
    pub(crate) level: f32,
    /// Surface RGBA color.
    pub(crate) color: [f32; 4],
}

/// A host-supplied void distance field to mesh as a smooth blob.
///
/// `phi` is a flat row-major scalar grid (`phi[x*ny*nz + y*nz + z]`) that
/// is HIGH at void centers and ~0 at atom walls / exterior; the worker
/// meshes its isosurface at `threshold` directly (see
/// [`cavity::mesh_void_field`](crate::renderer::geometry::isosurface::cavity::mesh_void_field)).
pub(crate) struct VoidFieldJob {
    /// Grid dimensions `[nx, ny, nz]`.
    pub(crate) dims: [usize; 3],
    /// World-space origin of grid cell `(0,0,0)` (Angstroms).
    pub(crate) origin: [f32; 3],
    /// Per-axis grid spacing (Angstroms).
    pub(crate) spacing: [f32; 3],
    /// Flat row-major distance field.
    pub(crate) phi: Vec<f32>,
    /// Positive iso-level: the void-surface level to wrap.
    pub(crate) threshold: f32,
}

/// Body of a surface-regeneration request, boxed on the enum variant to
/// keep [`SceneRequest`] compact.
///
/// Carries the already-gathered job lists (gathered on the main thread,
/// which alone can read the scene / annotations / density store); the
/// worker runs the generators and concatenates the meshes.
pub(crate) struct SurfaceRebuildBody {
    /// Density-map meshes: `(map, threshold, rgba)`, mirroring the shape
    /// the density generator consumes.
    pub(crate) density_jobs: Vec<(Density, f32, [f32; 4])>,
    /// Per-entity surface jobs.
    pub(crate) surface_jobs: Vec<SurfaceJob>,
    /// Cavity jobs: `(positions, radii)`. Color is the fixed cavity tint.
    pub(crate) cavity_jobs: Vec<(Vec<Vec3>, Vec<f32>)>,
    /// Host-supplied void distance field, meshed as a smooth blob in the
    /// cavity stream. `None` when no field is set.
    pub(crate) void_field_job: Option<VoidFieldJob>,
    /// Surface generation this request was minted at. The consumer
    /// discards a result whose generation is behind the latest submitted.
    pub(crate) surface_generation: u64,
}

impl SurfaceRebuildBody {
    /// Run all isosurface generators and concatenate the meshes into one
    /// GPU-ready buffer.
    ///
    /// Density maps, entity surfaces, then cavities, in that order; each
    /// mesh's indices are rebased onto the running vertex count before
    /// being appended. Runs on the scene-processor worker.
    pub(crate) fn generate(self) -> PreparedSurface {
        use crate::renderer::geometry::isosurface::{
            cavity, density, gaussian_surface, ses,
        };

        let Self {
            density_jobs,
            surface_jobs,
            cavity_jobs,
            void_field_job,
            surface_generation,
        } = self;

        let mut all_verts: Vec<IsosurfaceVertex> = Vec::new();
        let mut all_idxs: Vec<u32> = Vec::new();

        // Generate density map meshes first
        for (map, threshold, color) in &density_jobs {
            let (v, i) =
                density::generate_density_mesh(map, *threshold, *color, None);
            let base = all_verts.len() as u32;
            all_verts.extend(v);
            all_idxs.extend(i.iter().map(|&idx| idx + base));
        }

        // Generate entity surface meshes
        for job in &surface_jobs {
            let (v, i) = match job.kind {
                SurfaceJobKind::Gaussian => {
                    gaussian_surface::generate_gaussian_surface(
                        &job.positions,
                        &job.radii,
                        job.resolution,
                        job.level,
                        job.color,
                    )
                }
                SurfaceJobKind::Ses => ses::generate_ses(
                    &job.positions,
                    &job.radii,
                    Some(job.probe_radius),
                    job.resolution,
                    job.color,
                ),
            };
            let base = all_verts.len() as u32;
            all_verts.extend(v);
            all_idxs.extend(i.iter().map(|&idx| idx + base));
        }

        // Generate cavity meshes on a 0.6 Å grid — coarser than SES
        // because cavity detection is topological (flood fill from
        // grid boundary), so finer voxels can flip whether a thin
        // SES-wall separates a cavity from the exterior. 0.6 Å was
        // verified to detect the expected number of cavities on
        // benchmark structures (e.g. 1bbc has 3).
        let mut cavity_count = 0usize;
        for (positions, radii) in &cavity_jobs {
            let set =
                cavity::generate_cavities(positions, radii, Some(1.4), 0.6);
            for mesh in &set.meshes {
                let base = all_verts.len() as u32;
                all_verts.extend(mesh.vertices.iter().copied());
                all_idxs.extend(mesh.indices.iter().map(|&idx| idx + base));
            }
            cavity_count += set.meshes.len();
        }

        // Mesh the host-supplied void distance field directly into the
        // cavity stream. Same CAVITY kind / tint, so it concatenates and
        // renders alongside the detected + pre-built cavities.
        if let Some(job) = &void_field_job {
            let set = cavity::mesh_void_field(
                &job.phi,
                job.dims,
                job.origin,
                job.spacing,
                job.threshold,
            );
            for mesh in &set.meshes {
                let base = all_verts.len() as u32;
                all_verts.extend(mesh.vertices.iter().copied());
                all_idxs.extend(mesh.indices.iter().map(|&idx| idx + base));
            }
            cavity_count += set.meshes.len();
        }

        log::info!(
            "surface mesh: {} verts, {} triangles ({} cavities)",
            all_verts.len(),
            all_idxs.len() / 3,
            cavity_count,
        );

        PreparedSurface {
            surface_generation,
            vertices: all_verts,
            indices: all_idxs,
        }
    }
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
    /// Regenerate all isosurface meshes (density maps, entity surfaces,
    /// cavities) from pre-gathered job lists.
    SurfaceRebuild(Box<SurfaceRebuildBody>),
    /// Shut down the background thread.
    Shutdown,
}

/// Concatenated isosurface mesh, ready for GPU upload on the main thread.
#[derive(Clone)]
pub(crate) struct PreparedSurface {
    /// Surface generation this mesh was produced for. The consumer
    /// discards a result whose generation is behind the latest submitted.
    pub(crate) surface_generation: u64,
    /// Concatenated isosurface vertices.
    pub(crate) vertices: Vec<IsosurfaceVertex>,
    /// Triangle indices into `vertices`.
    pub(crate) indices: Vec<u32>,
}

/// All pre-computed CPU data, ready for GPU-only upload on the main thread.
#[derive(Clone)]
pub(crate) struct PreparedRebuild {
    /// Rebuild generation this prepared rebuild was produced for.
    pub(crate) generation: u64,
    /// Topology generation this rebuild was built for. The consumer
    /// discards a rebuild only when this is behind the current topology
    /// generation (the entity-id set changed since it was built).
    pub(crate) topology_generation: u64,
    /// Backbone mesh data.
    pub(crate) backbone: BackboneMeshData,
    /// Sidechain capsule instance bytes.
    pub(crate) sidechain_instances: Vec<u8>,
    /// Number of sidechain capsule instances.
    pub(crate) sidechain_instance_count: u32,
    /// True when this frame deliberately carries no sidechains because the
    /// producer omitted them (a backbone-only animation frame). Sidechain
    /// positions are unchanged by level-of-detail, so the apply side leaves
    /// the previously uploaded and retained sidechains in place rather than
    /// clobbering them with this empty set. A full rebuild always carries
    /// the resolved sidechain set (legitimately empty for an all-Stick
    /// scene), so it leaves this false.
    pub(crate) sidechains_omitted: bool,
    /// Ball-and-stick instance data.
    pub(crate) bns: BallAndStickInstances,
    /// Nucleic acid instance data.
    pub(crate) na: NucleicAcidInstances,
    /// Mapping from raw GPU pick IDs to typed pick targets.
    pub(crate) pick_map: PickMap,
    /// Each entity's first global residue index in the GPU selection /
    /// per-residue color space, in assembly-visible order.
    pub(crate) entity_residue_offsets: Vec<(EntityId, u32)>,
    /// The atom positions that produced this mesh, per entity, keyed
    /// parallel to `entity_residue_offsets`. Entity-local indexed (parallel
    /// to each entity's `topology.atom_elements`). Lifted onto each
    /// `EntityView` on apply so the overlay resolvers read the same frame
    /// the displayed mesh and its anchors were built from, rather than the
    /// live (worker-round-trip-ahead) positions.
    pub(crate) displayed_positions: Vec<(EntityId, Vec<Vec3>)>,
}

// Per-entity cached mesh

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
    pub entity_id: EntityId,
}
