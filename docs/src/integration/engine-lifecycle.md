# Engine Lifecycle

`VisoEngine` is the central rendering, animation, and picking coordinator.
It is **read-only with respect to structural state**: your application
owns a `molex::Assembly` and pushes the latest snapshot via
[`VisoEngine::set_assembly`]. This chapter covers how to create the
engine, what happens during initialization, and how to manage its
lifetime.

## Construction

You own your `molex::Assembly` and hand viso the latest snapshot. There
is no viso-defined channel, publisher, or consumer in the public API; the
structural ingest contract is one setter on the engine.

```rust
use std::sync::Arc;
use viso::{RenderContext, VisoEngine};
use viso::options::VisoOptions;
use molex::Assembly;

// 1. Build a wgpu RenderContext (async; use pollster or your runtime).
let context = pollster::block_on(
    RenderContext::new(window.clone(), (width, height))
)?;

// 2. Build the engine.
let mut engine = VisoEngine::new(context, VisoOptions::default())?;

// 3. Push your Assembly to the engine.
let assembly: Assembly = /* your owned Assembly */;
engine.set_assembly(Arc::new(assembly.clone()));
```

After every Assembly mutation, re-publish by calling `set_assembly`
again. The engine stages the snapshot in its pending slot and drains it
on the next `update(dt)` tick; a generation check skips work if nothing
changed.

For an embedded host that owns a wgpu device but no window surface, build
the context with `RenderContext::from_device(device, queue, format,
width, height)` instead of `RenderContext::new`, and present with
`render_to_texture` (see [The Render Loop](./render-loop.md)).

> **Note for standalone deployments only.** When viso is built as its own
> standalone app via `cargo run -p viso` (features `viewer` / `gui` /
> `web`), it uses an internal helper called `VisoApp` to play the host
> role for itself. Library users never go through `VisoApp`: own your
> `Assembly` and call `set_assembly` directly. `VisoApp` is not part of
> the library's public surface with `default-features = false`.

### What Happens During Init

1. **GPU setup**: `RenderContext` is configured with a surface (or an
   externally-owned device), adapter, device, and queue.
2. **Shader compilation**: `ShaderComposer` loads and composes all WGSL
   modules using `naga_oil`.
3. **Camera**: `CameraController` is created with default orbital
   parameters (FOV 45 degrees, fit to origin).
4. **Renderers**: backbone, sidechain, bond, band, clash arc, grease
   bead, pull, ball-and-stick, nucleic-acid, and isosurface.
5. **Post-processing**: SSAO, bloom, composite, and FXAA passes.
6. **Picking**: GPU picking system with an offscreen `R32Uint` target and
   staging buffer.
7. **Scene processor**: the background worker thread is spawned for mesh
   generation.
8. **Assembly slot**: `Scene` starts with an empty `current` Assembly and
   `pending: None`. The first `set_assembly` fills `pending`; the next
   `update(dt)` consumes it.

## Initial Scene Sync

The first `set_assembly` after construction pushes your initial snapshot.
The next `update(dt)` drains it, rederives the scene, and submits a full
mesh rebuild to the background thread. On the following frame, the
prepared meshes are uploaded to the GPU.

## Reloading or Swapping Topology

`set_assembly` plus `update` is the steady-state path. For a puzzle or
file reload, where the entire topology changes, use `replace_assembly`:

```rust
engine.replace_assembly(Arc::new(new_assembly));
```

`replace_assembly` tears down scene-local state (animation, surfaces,
derived per-entity views, annotations), stages the new snapshot, and
forces a synchronous sync so that follow-up calls (camera pose, SS
overrides, and so on) operate against synced state. `set_assembly` alone
leaves stale state from the previous topology around until the next
`update`, which is why reloads should go through `replace_assembly`.

## Resize and Scale Factor

Forward window resize events to the engine:

```rust
engine.resize(new_width, new_height);
```

This resizes the wgpu surface, all post-processing textures, the picking
render target, and the camera projection. For DPI changes:

```rust
engine.set_surface_scale(scale_factor);
let inner = window.inner_size();
engine.resize(inner.width, inner.height);
```

## Shutdown

The background scene processor is joined automatically on drop. To force
shutdown earlier:

```rust
engine.shutdown();
```

This sends a `Shutdown` request to the processor thread.

## Ownership Model

`VisoEngine` owns these subsystems (plus a few scalar state fields):

| Field | Type | Purpose |
|-------|------|---------|
| `gpu` | `GpuPipeline` | wgpu context, all renderers, picking, post-process, lighting, culling state |
| `camera_controller` | `CameraController` | Camera matrices, animation, frustum |
| `constraints` | `ConstraintSpecs` | Stored band / pull / clash / exposed-hydrophobic specs |
| `animation` | `AnimationState` | Structural animator, trajectory player, pending transitions |
| `options` | `VisoOptions` | Display, lighting, post-processing, geometry, etc. |
| `active_preset` | `Option<String>` | Name of the currently-applied options preset |
| `frame_timing` | `FrameTiming` | FPS smoothing, frame pacing |
| `density` | `DensityStore` | Loaded electron density maps |
| `external_void_field` | `Option<ExternalVoidField>` | Host-supplied void distance field, meshed as a blob |
| `scene` | `Scene` | Pending/current Assembly + derived per-entity state |
| `annotations` | `EntityAnnotations` | Per-entity overrides: focus, visibility, behaviors, appearance, scores, SS, surfaces |
| `surface_regen` | `SurfaceRegen` | Submit handle for background isosurface regeneration |

Alongside these it holds scalar state: `surface_built_for_generation` and
`last_publish_at` (surface rest-detection; see
[Background Scene Processing](../deep-dives/background-processing.md)),
and `input_state` / `mouse_pressed` / `shift_pressed` (pointer intake).

The engine is **not thread-safe** (`!Send`, `!Sync`) because it holds
wgpu GPU resources. All engine access happens on the main thread. The
background scene processor communicates via channels and triple buffers.
