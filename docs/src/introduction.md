# Introduction

Viso is a GPU-accelerated 3D protein visualization engine built in Rust on
top of [wgpu](https://wgpu.rs/). It powers the molecular graphics in
Foldit, rendering proteins, ligands, nucleic acids, and constraint visualizations at interactive frame
rates.

Viso is an **embeddable library**: you give it a window or surface, feed
it structure data, and it produces a 2D texture. The host
decides what to do with that texture: display it in a winit window, paint it
onto an HTML canvas, write it to a PNG, or drop it into a dioxus/egui
texture slot.

```rust
use viso::Viewer;

Viewer::builder()
    .with_path("1ubq")           // PDB code or local .cif/.pdb/.bcif path
    .with_title("My Viewer")
    .build()
    .run()?;
```

## Features

**Rendering**

- Ray-marched impostors for pixel-perfect spheres and capsules at any zoom
- Post-processing pipeline: SSAO, bloom, FXAA, depth-based outlines, fog, tone mapping

**Interaction**

- Arcball camera with animated transitions, panning, zoom, and auto-rotate
- GPU picking: click to select residues, double-click for SS segments, triple-click for chains, shift-click for multi-select

**Animation**

- Snap, smooth (cubic-hermite), and cascade transitions
- Per-entity targeted animation with behavior overrides

**Performance**

- Background mesh generation on a worker thread with triple-buffered results
- Per-entity mesh caching: only changed entities are regenerated
- Lock-free communication between main and background threads

**Configuration**

- TOML-serializable options for display, lighting, color, geometry, and camera
- Load/save presets, per-section diffing on update

## How It Works

```
File (.cif/.pdb/.bcif)  ──or──  Vec<MoleculeEntity>
        │                                │
        ▼                                │
 molex::adapters ──▶ Vec<MoleculeEntity>◄─┘
        │
        ▼
 molex::Assembly  (owned by your application, e.g. foldit)
        │  engine.set_assembly(Arc::new(assembly.clone()))
        ▼
 Scene + EntityAnnotations  (engine-side derived state,
        │                     rederived on generation change)
        │
        ├───▶ SceneProcessor (background thread)
        │       per-entity mesh cache, triple-buffered output
        │
        ▼
 Renderer (geometry → picking → post-process)
        │
        ▼
 2D texture ──▶ winit / canvas / PNG / embed
```

For the full architecture, see [Architecture Overview](./architecture/overview.md).

## Where to Start

**Embed viso in your application:**

- [Quick Start](./getting-started/quick-start.md): standalone viewer walkthrough
- [Engine Lifecycle](./integration/engine-lifecycle.md): creation, initialization, shutdown
- [The Render Loop](./integration/render-loop.md): per-frame sequence
- [Handling Input](./integration/handling-input.md): mouse and keyboard wiring

**Understand how Foldit uses viso:**

- [Scene Management](./integration/scene-management.md): Assembly, entities, focus
- [Dynamic Structure Updates](./integration/dynamic-updates.md): Rosetta and ML integration
- [Options and Presets](./configuration/options-and-presets.md): TOML configuration

**Dig into viso internals:**

- [Architecture Overview](./architecture/overview.md): system diagram and data flow
- [Rendering Pipeline](./deep-dives/rendering-pipeline.md): geometry pass and post-processing
- [Background Scene Processing](./deep-dives/background-processing.md): threading model
- [Animation System](./deep-dives/animation-system.md): transitions, behaviors, interpolation
