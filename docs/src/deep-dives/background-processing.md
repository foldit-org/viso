# Background Scene Processing

Mesh generation for molecular structures is CPU-intensive; generating
backbone splines, ribbon surfaces, and sidechain capsule instances can
take 20 to 40ms for complex structures. Viso runs it on a background
worker thread so the main thread keeps rendering at full frame rate.

## Architecture

```
Main Thread                          Background Worker
  ├─ submit(SceneRequest)              ├─ blocks on mpsc::Receiver
  │   → mpsc::Sender                   ├─ processes the request
  │                                    ├─ generates / caches per-entity meshes
  ├─ try_recv_rebuild()                ├─ concatenates into PreparedRebuild
  │   ← triple_buffer::Output          └─ writes to the matching triple buffer
  ├─ GPU upload (<1ms)
  └─ Render
```

There is a single background worker (the `scene-processor` thread). Full
rebuilds, per-frame animation meshes, and isosurface regeneration all run
on it; each result kind returns over its own triple buffer.

### Communication Channels

| Channel | Type | Direction | Purpose |
|---------|------|-----------|---------|
| Request | `mpsc::Sender<SceneRequest>` | Main to worker | Submit work |
| Rebuild result | `triple_buffer` | Worker to main | `Option<PreparedRebuild>` |
| Animation result | `triple_buffer` | Worker to main | `Option<PreparedRebuild>` |
| Surface result | `triple_buffer` | Worker to main | `Option<PreparedSurface>` |

Triple buffers are lock-free: the writer always has a buffer to write to,
and the reader always gets the latest completed result. Neither side
blocks.

## SceneProcessor

```rust
let processor = SceneProcessor::new()?; // spawns the background worker

// Submit work (non-blocking).
processor.submit(SceneRequest::FullRebuild(Box::new(body)));

// Check for results (non-blocking).
if let Some(prepared) = processor.try_recv_rebuild() { /* upload */ }
if let Some(frame)    = processor.try_recv_animation() { /* upload */ }
if let Some(surface)  = processor.try_recv_surface() { /* upload */ }

// Shutdown: sends Shutdown, joins the thread.
processor.shutdown();
```

`request_sender()` hands out a clone of the request channel so other
subsystems (the surface-regen holder) can submit work without a
`&mut` borrow of the processor.

## Request Types

```rust
pub(crate) enum SceneRequest {
    FullRebuild(Box<FullRebuildBody>),
    AnimationFrame(Box<AnimationFrameBody>),
    SurfaceRebuild(Box<SurfaceRebuildBody>),
    Shutdown,
}
```

### FullRebuild

A complete scene rebuild with per-entity render-ready snapshots:

```rust
pub(crate) struct FullRebuildBody {
    pub entities: Vec<FullRebuildEntity>,
    pub display: DisplayOptions,
    pub colors: ColorOptions,
    pub geometry: GeometryOptions,
    pub entity_options:
        FxHashMap<u32, (DisplayOptions, GeometryOptions)>,
    pub generation: u64,
    pub topology_generation: u64,
}

pub(crate) struct FullRebuildEntity {
    pub id: EntityId,
    pub mesh_version: u64,
    pub drawing_mode: DrawingMode,
    pub topology: Arc<EntityTopology>,
    pub positions: Vec<Vec3>,
    pub ss_override: Option<Vec<SSType>>,
    pub per_residue_colors: Option<Vec<[f32; 3]>>,
}
```

`FullRebuild` is submitted when a new `Assembly` snapshot is consumed,
when display/color/geometry options change, or when a scoped reset clears
local state. Per-entity `mesh_version` is the cache key: an entity whose
version is unchanged since the previous rebuild reuses its cached mesh.

`generation` bumps on every submit. `topology_generation` advances only
when the visible entity-id set changes (an entity added or removed); it
lets the consumer keep a rebuild whose topology still matches the current
scene even after newer same-topology submits bumped `generation`.

### AnimationFrame

Per-frame mesh regeneration during animation:

```rust
pub(crate) struct AnimationFrameBody {
    pub positions: EntityPositions,            // interpolated
    pub geometry: GeometryOptions,
    pub per_chain_lod: Option<Vec<ChainLod>>,  // per-chain detail override
    pub include_sidechains: bool,
    pub generation: u64,
    pub topology_generation: u64,
}
```

Submitted while animation is in progress. It regenerates backbone meshes
(and optionally sidechains) from interpolated positions, reusing topology
and other state cached from the last `FullRebuild`. The result is a
`PreparedRebuild` delivered over the animation triple buffer.

### SurfaceRebuild

Regenerates all isosurface meshes (density maps, entity surfaces,
cavities, and a host-supplied void field) from job lists gathered on the
main thread:

```rust
pub(crate) struct SurfaceRebuildBody {
    pub density_jobs: Vec<(Density, f32, [f32; 4])>,
    pub surface_jobs: Vec<SurfaceJob>,
    pub cavity_jobs: Vec<(Vec<Vec3>, Vec<f32>)>,
    pub void_field_job: Option<VoidFieldJob>,
    pub surface_generation: u64,
}
```

A `SurfaceJob` carries the flattened scalar parameters the isosurface
generators consume (atom positions and radii, surface kind, grid
resolution, probe radius, level, color), so the worker never needs the
engine-side surface type. A `VoidFieldJob` carries a host-supplied
distance field (grid dims, origin, spacing, the flat `phi` grid, and an
iso-`threshold`) meshed as a smooth blob into the cavity stream. The
worker runs the generators, concatenates the meshes, and returns a
`PreparedSurface { surface_generation, vertices, indices }`. The main
thread polls it with `try_recv_surface`.

### Shutdown

Terminates the background worker.

## Per-Entity Mesh Caching

The worker keeps a per-entity mesh cache keyed on `EntityId`:

```
FxHashMap<EntityId, CachedEntityMesh>
```

`CachedEntityMesh` stores GPU-ready byte buffers (backbone vertices and
indices, sidechain instances, ball-and-stick spheres and capsules,
nucleic-acid stems and rings) plus typed intermediates needed for index
concatenation.

### Cache Invalidation

On a `FullRebuild`, the worker checks each entity's `mesh_version`
against the cached version:

1. **Same version**: reuse the cached mesh (skip generation).
2. **Different version**: regenerate and update the cache.
3. **Entity removed**: evict from the cache.

Version-based invalidation is a `u64` comparison. For a 3-entity scene
where only 1 changed, this skips regenerating the two unchanged entities.

### Global vs Per-Entity Settings

A bumped `mesh_version` is the universal "regenerate me" signal.
Option-change paths in the engine bump the affected entities' versions
before submitting the rebuild; color-only changes update color buffers
without forcing a full geometry regenerate.

## Mesh Generation

For each entity, the worker generates whichever of these apply to its
`drawing_mode`:

1. **Backbone mesh**: cubic Hermite splines with rotation-minimizing
   frames, with separate index ranges for the tube and ribbon passes.
2. **Sidechain capsule instances**: packed capsule structs for the
   storage buffer.
3. **Ball-and-stick instances**: sphere and capsule instances for
   non-protein entities (and proteins drawn in BallAndStick mode).
4. **Nucleic-acid instances**: stem capsules and ring polygons.

### Mesh Concatenation

After generating (or retrieving from cache) all entity meshes, they are
concatenated into one `PreparedRebuild`: vertex buffers appended, index
buffers appended with per-entity offset adjustment, instance buffers
concatenated, and a single `PickMap` built from raw GPU pick IDs to typed
targets.

## PreparedRebuild

The output of a `FullRebuild` (and of an `AnimationFrame`), ready for GPU
upload:

```rust
pub(crate) struct PreparedRebuild {
    pub generation: u64,
    pub topology_generation: u64,
    pub backbone: BackboneMeshData,           // verts + tube/ribbon idx
    pub sidechain_instances: Vec<u8>,
    pub sidechain_instance_count: u32,
    pub sidechains_omitted: bool,             // backbone-only anim frame
    pub bns: BallAndStickInstances,           // sphere + capsule instances
    pub na: NucleicAcidInstances,             // stem + ring instances
    pub pick_map: PickMap,
    pub entity_residue_offsets: Vec<(EntityId, u32)>,
    pub displayed_positions: Vec<(EntityId, Vec<Vec3>)>,
}
```

All byte arrays are raw GPU buffer data (`bytemuck::cast_slice`), ready
for `queue.write_buffer()` with no further processing.

`sidechains_omitted` marks a backbone-only animation frame: because
level-of-detail does not change sidechain positions, the apply side
leaves the previously uploaded sidechains in place rather than clobbering
them with an empty set. `entity_residue_offsets` records each entity's
first global residue index in the GPU selection / color space.
`displayed_positions` records the atom positions the mesh was built from,
so overlay resolvers read the same frame the displayed mesh and its
anchors came from.

## Stale Frame Discarding

When a scene is replaced, in-flight animation frames from the old scene
become stale. Two counters guard against applying them:

1. **`generation`** bumps on every `FullRebuild`. The worker skips a
   queued animation frame whose `generation` is behind the latest
   rebuild before it spends time generating; the main thread discards a
   stale result before GPU upload.
2. **`topology_generation`** advances only when the entity-id set
   changes. A coordinate-only rebuild that bumped `generation` does not
   invalidate a frame whose topology still matches, so same-topology
   frames are not dropped needlessly.

## Surface Rest-Detection

Marching-cubes isosurfaces are expensive, so they re-mesh only when the
conformation comes to rest rather than on every animation frame.

Each consumed publish restarts a quiet-window clock (`last_publish_at`).
Once per `update` tick, `maybe_settle_surface` asks
`should_settle_surface` whether to regenerate. It regenerates only when:

- a surface could currently be shown (`surface_active`),
- the displayed generation differs from the generation the surface was
  last built against (`surface_built_for_generation`), and
- the publish stream has stayed quiet for at least `SURFACE_SETTLE_WINDOW`
  (180ms).

A continuous edit (wiggle, drag) keeps pushing the quiet window out, so
the re-mesh runs at most once per rest, never per motion frame. At rest
the scene's reference coordinates equal the displayed coordinates, so the
re-meshed surface matches what is on screen.

## Threading Model Summary

| Thread | Owns | Does |
|--------|------|------|
| **Main thread** | GPU resources, engine, scene | Input, render, GPU upload |
| **Scene-processor worker** | Per-entity mesh cache | CPU mesh, animation, and isosurface generation |
| **Bridge** | Triple buffers + mpsc channel | Lock-free data transfer |

The main thread never blocks on the worker. If meshes aren't ready, the
previous frame's meshes keep rendering, so frame rates stay consistent
even during expensive regeneration.
