# Architecture Overview

How viso's subsystems fit together: the engine's components, the path
data takes from file to screen, and the threading model.

## System Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                      Application Layer                          │
│  (your application, e.g. foldit)                                │
│                                                                 │
│  Owns the authoritative `molex::Assembly`. All structural       │
│  mutations push a new Arc<Assembly> via engine.set_assembly.    │
│                                                                 │
│  winit events ──► engine.feed_* / KeyBindings::dispatch         │
└──────────────────────────────┬──────────────────────────────────┘
                               │ engine.set_assembly(Arc<Assembly>)
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                         VisoEngine                              │
│                                                                 │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐  ┌────────────┐ │
│  │ Scene      │  │ Animation  │  │ Camera     │  │ GpuPipeline│ │
│  │ + Annot.   │  │ State      │  │ Controller │  │            │ │
│  │            │  │            │  │            │  │ Renderers  │ │
│  │ Per-entity │  │ Animator   │  │ Arcball    │  │ Picking    │ │
│  │ derived    │  │ Trajectory │  │ Animation  │  │ Post-proc  │ │
│  │ state +    │  │ Pending    │  │ Frustum    │  │ Lighting   │ │
│  │ overrides  │  │ trans.     │  │            │  │ Density    │ │
│  └─────┬──────┘  └─────┬──────┘  └─────┬──────┘  └─────┬──────┘ │
│        │               │               │               │        │
│        ▼               ▼               │               │        │
│  ┌───────────────────────────┐         │               │        │
│  │ Background scene          │         │               │        │
│  │ processor (worker thread) │         │               │        │
│  │                           │         │               │        │
│  │ Per-entity mesh cache     │         │               │        │
│  │ Backbone / sidechain /    │         │               │        │
│  │ ball-and-stick / NA       │         │               │        │
│  └─────────────┬─────────────┘         │               │        │
│                │ triple buffer         │               │        │
│                ▼                       ▼               ▼        │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │ Renderers                                                 │  │
│  │                                                           │  │
│  │  Molecular:                  Post-processing:             │  │
│  │  ├─ BackboneRenderer         ├─ SSAO                      │  │
│  │  │  (tubes + ribbons)        ├─ Bloom                     │  │
│  │  ├─ SidechainRenderer        ├─ Composite                 │  │
│  │  ├─ BondRenderer             └─ FXAA                      │  │
│  │  ├─ BandRenderer                                          │  │
│  │  ├─ ClashArcRenderer         ShaderComposer:              │  │
│  │  ├─ GreaseBeadRenderer       └─ naga_oil composition      │  │
│  │  ├─ PullRenderer                                          │  │
│  │  ├─ BallAndStickRenderer                                  │  │
│  │  ├─ NucleicAcidRenderer                                   │  │
│  │  └─ IsosurfaceRenderer                                    │  │
│  └───────────────────────────────────────────────────────────┘  │
│                            │                                    │
│                            ▼                                    │
│                    ┌──────────────┐                             │
│                    │ RenderContext│                             │
│                    │ wgpu device  │                             │
│                    │ queue        │                             │
│                    │ surface      │                             │
│                    └──────────────┘                             │
└─────────────────────────────────────────────────────────────────┘
```

## High-Level Data Flow

```
┌───────────────────────────────────────────────────────────┐
│                     INITIALIZATION                        │
│                                                           │
│  File path (.cif/.pdb/.bcif)  ──or──  Vec<MoleculeEntity> │
│         │                                    │            │
│         ▼                                    │            │
│  molex::adapters parse ──► Vec<MoleculeEntity> ◄┘         │
│                                │                          │
│                                ▼                          │
│                     molex::Assembly (owned by host)       │
│                              │                            │
│                              ▼ engine.set_assembly(...)   │
│                     pending: Option<Arc<Assembly>>        │
│                              │  (engine-internal slot)    │
│                              ▼                            │
│                     Scene (in VisoEngine)                 │
└────────────────────────────┬──────────────────────────────┘
                             │  engine.update() drains,
                             │  rederives on generation bump
                             ▼
┌───────────────────────────────────────────────────────────┐
│                         SCENE                             │
│                                                           │
│  Scene + EntityAnnotations: per-entity derived render     │
│  state + user-authored overrides (focus, visibility,      │
│  appearance, behaviors, scores, SS overrides, surfaces).  │
│                                                           │
│  Driven by `mesh_version` per-entity for cache            │
│  invalidation. During animation, `EntityPositions` holds  │
│  interpolated atom positions read by the renderers.       │
└────────────────────────────┬──────────────────────────────┘
                             │
                             ▼
┌───────────────────────────────────────────────────────────┐
│                       RENDERER                            │
│                                                           │
│  Consumes Scene + annotations read-only.                  │
│                                                           │
│  ┌─────────────────────────────────────────────────────┐  │
│  │ Background mesh processor                           │  │
│  │   per-entity FullRebuildEntity → cached meshes →    │  │
│  │   PreparedRebuild (raw byte buffers)                │  │
│  └──────────────────────┬──────────────────────────────┘  │
│                         │ triple buffer                   │
│                         ▼                                 │
│  ┌─────────────────────────────────────────────────────┐  │
│  │ GPU passes                                          │  │
│  │                                                     │  │
│  │ 1. Geometry pass (color + normals + depth)          │  │
│  │ 2. Picking pass (residue ID readback, async)        │  │
│  │ 3. Post-process (SSAO, bloom, fog, FXAA)            │  │
│  │           │                                         │  │
│  │           ▼                                         │  │
│  │ Final 2D screen-space texture                       │  │
│  └─────────────────────────────────────────────────────┘  │
└────────────────────────────┬──────────────────────────────┘
                             │
                             ▼
┌───────────────────────────────────────────────────────────┐
│                   OUTPUT / EMBEDDING                      │
│                                                           │
│  The final texture is consumed by the host:               │
│    • winit window (standalone viewer)                     │
│    • HTML canvas (wasm / web embed)                       │
│    • PNG snapshot (headless)                              │
│    • dioxus / egui / any framework with a texture slot    │
│                                                           │
│  Use `engine.render()` for swapchain present, or          │
│  `engine.render_to_texture(view)` to render into a        │
│  caller-owned texture view.                               │
└───────────────────────────────────────────────────────────┘
```

## Data Flow: File to Screen

### 1. Parsing

```
PDB / CIF / BCIF file → molex::adapters → Vec<MoleculeEntity>
```

`molex` parses structure files into `MoleculeEntity` values (atomic
coordinates, names, chains, residue info, molecule type, computed
H-bonds, DSSP-classified SS).

### 2. Assembly Construction

```
Vec<MoleculeEntity> → molex::Assembly  (owned by your application)
```

The `Assembly` is the authoritative structural state. Your
application owns it and pushes the latest snapshot to the engine via
`engine.set_assembly(Arc::new(assembly.clone()))` whenever it
changes.

### 3. Scene Rederivation

Each `engine.update(dt)` drains the engine's pending Assembly slot;
if a new snapshot is ready, the engine rebuilds its derived per-entity
state (chains, sidechain topology, SS arrays, color metadata) from the
new assembly.

### 4. Background Mesh Generation

```
Scene → per-entity FullRebuildEntity → SceneProcessor → PreparedRebuild
```

The sync layer collects per-entity render data and submits a
`SceneRequest::FullRebuild` to the background thread. The processor
generates (or retrieves cached) meshes per entity, concatenates them
into a single `PreparedRebuild`, and writes it to the result triple
buffer.

### 5. GPU Upload

```
PreparedRebuild → queue.write_buffer() → GPU buffers
```

The main thread picks up the prepared rebuild and writes raw byte
arrays directly to GPU buffers. This is a memcpy-level operation,
typically under 1ms.

### 6. Rendering

```
GPU buffers → Geometry Pass → Post-Processing → Swapchain
```

All molecular renderers draw to HDR render targets. Post-processing
applies SSAO, bloom, compositing (outlines, fog, tone mapping), and
FXAA before presenting to the swapchain (or writing into the
caller-owned texture view).

## Threading Model

Viso uses one background worker alongside the main thread, with lock-free
communication:

### Main Thread

Owns all GPU resources and runs the render loop:

- Processing input events (mouse, keyboard, IPC)
- Draining the pending `Assembly` slot for new snapshots
- Running animation per frame
- Submitting scene requests to the worker (non-blocking)
- Picking up completed meshes from the triple buffers (non-blocking)
- Uploading data to the GPU
- Executing the render pipeline
- Initiating GPU picking readback and resolving completed reads

The main thread **never blocks**. If meshes aren't ready, it renders
the previous frame's data.

### Background Scene-Processor Worker

A single worker thread owns the per-entity mesh cache and performs all
CPU-intensive generation:

- Receiving scene requests via `mpsc::Receiver` (blocks when idle)
- Generating backbone, sidechain, ball-and-stick, and nucleic-acid
  meshes
- Regenerating isosurface meshes (Gaussian / SES / cavity / density /
  void field) on the same worker via `SceneRequest::SurfaceRebuild`
- Maintaining a per-entity cache keyed on `mesh_version`
- Writing each result kind to its own triple buffer (non-blocking)

There is no separate surface thread; surface regeneration shares this
worker and returns a `PreparedSurface` over a triple buffer.

### Lock-Free Bridges

| Mechanism                  | Direction        | Semantics                                       |
| -------------------------- | ---------------- | ----------------------------------------------- |
| `mpsc::channel`            | Main to worker   | Submit scene requests (non-blocking send)       |
| `triple_buffer` (rebuild)  | Worker to main   | Latest `PreparedRebuild` (non-blocking read)    |
| `triple_buffer` (anim)     | Worker to main   | Latest `PreparedRebuild` (non-blocking read)    |
| `triple_buffer` (surface)  | Worker to main   | Latest `PreparedSurface` (non-blocking read)    |

Triple buffers guarantee:

- The writer always has a buffer to write to (never blocks)
- The reader always gets the latest completed result
- No data races or mutex contention

## Module Structure

```
viso/src/
├── lib.rs              # Public API (flat re-exports only)
├── main.rs             # Standalone CLI entry point (binary feature)
├── animation/          # Structural animation
│   ├── animator.rs     # StructureAnimator + per-entity runners
│   ├── runner.rs       # AnimationRunner phase evaluation
│   ├── state.rs        # AnimationState (animator + trajectory + pending)
│   └── transition.rs   # AnimationPhase, Transition presets (public API)
├── app/                # Standalone-app layer (feature-gated)
│   ├── viewer.rs       # winit Viewer + ViewerBuilder (feature = "viewer")
│   ├── gui/            # wry-webview options panel (feature = "gui")
│   ├── web/            # WASM entry (feature = "web")
│   └── mod.rs          # VisoApp (host of Assembly in standalone),
│                       # publish helper that calls engine.set_assembly
├── bridge/             # GUI / IPC action types (feature = "gui")
├── camera/             # Orbital camera controller, animation, frustum
├── engine/             # Core engine struct + frame loop
│   ├── mod.rs          # VisoEngine (thin dispatcher)
│   ├── annotations.rs  # EntityAnnotations: focus, visibility, behaviors,
│   │                   # appearance, scores, SS, surfaces
│   ├── bootstrap.rs    # GPU init + VisoEngine::new + FrameTiming
│   ├── command.rs      # Constraint payload types (AtomRef, BandInfo,
│   │                   # ClashInfo, ExposedHydrophobicInfo, PullInfo)
│   ├── constraint.rs   # Band/pull resolution
│   ├── culling.rs      # Frustum culling
│   ├── density.rs      # Density map loading + isosurface integration
│   ├── density_store.rs# DensityStore (loaded electron density maps)
│   ├── entity_view.rs  # Per-entity render-ready derived data
│   ├── focus.rs        # Focus enum
│   ├── options_apply.rs# set_options / set_surface_scale / etc.
│   ├── positions.rs    # EntityPositions: interpolated atom positions
│   ├── scene.rs        # Scene: pending Assembly + last_seen_generation + state
│   ├── scene_state.rs  # SceneRenderState: per-entity render aggregations
│   ├── surface.rs      # Surface options resolution
│   ├── surface_regen.rs# Background isosurface regeneration
│   ├── sync/           # Scene → renderer pipeline
│   └── trajectory.rs   # TrajectoryPlayer (DCD frame sequencer)
├── error.rs            # VisoError
├── gpu/                # wgpu device init, dynamic buffers, lighting,
│                       # shader composition, residue color buffer
├── input/              # Pointer/key intake types: ClickEvent, KeyBindings
├── options/            # TOML-serializable runtime options + score color
├── renderer/           # GPU rendering pipeline
│   ├── mod.rs          # PipelineLayouts, Renderers, GeometryPassInput
│   ├── gpu_pipeline.rs # GpuPipeline (rendering entry point)
│   ├── draw_context.rs # DrawBindGroups
│   ├── entity_topology.rs # Per-entity topology metadata for renderers
│   ├── geometry/       # Mesh + impostor generation (backbone, sidechain,
│   │                   # ball-and-stick, NA, isosurface, band, pull, bond)
│   ├── impostor/       # Impostor primitives (sphere, capsule, cone, polygon)
│   ├── mesh.rs         # Generic mesh helpers
│   ├── picking/        # GPU picking + PickingSystem + PickTarget + PickMap
│   ├── pipeline/       # Background mesh-gen pipeline
│   │   ├── prepared.rs # SceneRequest, PreparedRebuild,
│   │   │               # SurfaceRebuildBody, PreparedSurface
│   │   ├── mesh_gen.rs # Per-entity / per-frame mesh generation
│   │   ├── mesh_concat.rs # Merge per-entity meshes
│   │   └── processor.rs   # Background thread + cache
│   ├── pipeline_util.rs# Helper utilities
│   └── postprocess/    # SSAO, bloom, composite, FXAA, screen passes
├── shaders/            # WGSL sources organized by role
│   ├── modules/        # Shared modules: camera, lighting, ray, sdf, ...
│   ├── raster/         # Mesh + impostor rasterization shaders
│   ├── screen/         # Full-screen passes (composite, FXAA, SSAO, bloom)
│   └── utility/        # Picking shaders
└── util/               # Helpers (easing.rs, hash.rs)
```

## Key Design Decisions

### Why Background Mesh Generation?

Mesh generation for complex proteins (>1000 residues) can take 20-40ms.
At 60fps, that's most of the frame budget. By offloading to a
background thread:

- The main thread maintains smooth rendering
- GPU upload is <1ms (raw buffer writes)
- The background thread can take as long as it needs without dropping
  frames

### Why Triple Buffers?

Triple buffers provide lock-free communication:

- The writer always has a buffer to write to
- The reader always reads the latest result
- No mutexes, no contention, no blocking on either side

The cost is memory (3× the buffer size), but mesh data is typically
1–10MB, so this is negligible.

### Why Per-Entity Mesh Caching?

Molecular scenes often have multiple entities where only some change
at a time (e.g. Rosetta updates one entity while others stay static).
Per-entity caching with `mesh_version`-based invalidation means only
changed entities are regenerated. For a 3-entity scene where one
changes, this saves 60–80% of generation time.

### Why Capsule Impostors?

Sidechains and ball-and-stick atoms use ray-marched impostor rendering
instead of mesh-based spheres and cylinders:

- **Memory**: a capsule is 48 bytes vs hundreds of bytes for a mesh
  sphere
- **Quality**: impostors are pixel-perfect at any zoom level
- **Performance**: GPU ray-marching is efficient for the simple SDF
  shapes (spheres, capsules, cones)

### Why a Host-Owned Assembly?

`molex::Assembly` belongs to molex; viso just renders it. The host
application (typically `foldit`) owns the authoritative
`Assembly` because it also drives Rosetta and ML backends and needs
to mutate the assembly in response to their results. Viso never
mutates the structural state itself; the host pushes the latest
`Arc<Assembly>` snapshot via `engine.set_assembly`, and the engine
drains it on the next sync tick. The library API stays narrow: a
single setter and a generation check inside `update`. No viso-flavored
channels or publishers leak out of the engine.

When viso runs as a standalone application (`cargo run -p viso`,
`feature = "viewer" / "gui" / "web"`), the in-tree helper `VisoApp`
plays the host role for viso itself. `VisoApp` is purely an internal
standalone-deployment helper: it is feature-gated and is **not** part
of the library's public surface. Library consumers own their own
`Assembly` and call `engine.set_assembly` directly.
