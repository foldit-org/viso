# Rendering Pipeline

Viso's rendering pipeline has two main stages: a geometry pass that
renders molecular structures to HDR render targets, and a
post-processing stack that applies screen-space effects.

## Overview

```
Geometry Pass (10 molecular renderers)
    ↓ Color (Rgba16Float) + Normals (Rgba16Float) + Depth (Depth32Float)
    ↓
Post-Processing Stack:
    1. SSAO: depth + normals → ambient occlusion texture
    2. Bloom: color → threshold → blur → half-res bloom texture
    3. Composite: color + SSAO + depth + normals + bloom → tone-mapped result
    4. FXAA: anti-aliased final output → swapchain
```

## Geometry Pass

### Render Targets

All molecular renderers write to two HDR render targets plus a depth
buffer:

| Target  | Format       | Contents |
|---------|--------------|----------|
| Color   | `Rgba16Float`| Scene color with alpha blending |
| Normal  | `Rgba16Float`| View-space normals / metadata (no blending) |
| Depth   | `Depth32Float`| Depth buffer (Less compare, writes enabled) |

`Rgba16Float` enables HDR lighting and bloom without banding
artifacts.

### Molecular Renderers

The `Renderers` struct holds ten renderers, drawn in the geometry pass:

#### 1. BackboneRenderer

Renders protein backbones as a single mesh with two index ranges: tube
indices (drawn first for coil segments and fully in tube mode) and ribbon
indices (drawn for helices and sheets in ribbon mode).

- **Geometry**: cubic Hermite splines with rotation-minimizing frames.
- **Per-SS appearance**: helix / sheet / coil width, thickness, and
  roundness from `GeometryOptions` (driven by `cartoon_style` preset
  unless `Custom`).
- **Detail**: `segments_per_residue` × `cross_section_verts` (defaults
  32 × 16, scalable per LOD tier).
- **Vertex data**: position, normal, color, residue idx, center pos.

#### 2. SidechainRenderer

Renders sidechain atoms as ray-marched capsule impostors.

- **Technique**: storage buffer of capsule instances rendered as
  ray-marched impostors.
- **Capsule radius**: 0.3 Å.
- **Color**: from the active sidechain color mode (Backbone or
  Hydrophobicity).
- **Frustum culling**: instances outside the view frustum are skipped
  on upload.

#### 3. BondRenderer

Renders structural bonds (H-bonds, disulfides) as configurable
capsules.

- **Style**: `Solid`, `Dashed`, or `Stippled` per bond type
  (`BondOptions`).
- **Source**: `Auto` (geometry-detected), `Manual` (caller-provided),
  or `Both`.

#### 4. BandRenderer

Renders constraint bands (e.g. for Rosetta minimization).

- **Visual**: capsule impostors with variable radius (0.1–0.4 Å,
  scaled by `strength`).
- **Colors by type**: default (purple), backbone (yellow-orange),
  disulfide (yellow-green), H-bond (cyan), disabled (gray).
- **Anchor spheres**: small spheres at band endpoints.

#### 5. ClashArcRenderer

Renders steric clashes as glowing deep-red electric bolts, one
camera-facing billboard ribbon per clashing atom pair.

- **Geometry**: one `LightningInstance` per clash, spanning the two
  clashing atoms; the jagged bolt is drawn procedurally in the fragment
  shader.
- **Animation**: the centerline jag scrolls with `camera.time` and a fast
  brightness flicker reads as electric energy; severity drives the jag
  amplitude.
- **Source**: clash specs supplied via `engine.update_clashes`, resolved
  per-entity to world-space each frame.

#### 6. GreaseBeadRenderer

Renders flagged exposed-hydrophobic residues as ray-cast SDF "grease
bead" impostors, one billboard quad per residue.

- **Geometry**: a central sphere smooth-min'd with a few slowly-boiling
  satellites, anchored at the residue's sidechain (CB if present, else
  the sidechain centroid, else CA).
- **Shading**: writes real depth and an SDF-gradient normal, lit with a
  warm-yellow greasy PBR and an animated highlight.
- **Source**: bead specs supplied via
  `engine.update_exposed_hydrophobics`, resolved to a world-space anchor
  each frame.

#### 7. PullRenderer

Renders the active drag constraint.

- **Cylinder**: capsule from atom to mouse target (purple).
- **Arrow**: cone impostor at the target end pointing toward the
  drag direction.

#### 8. BallAndStickRenderer

Renders ligands, ions, waters, and (in BallAndStick drawing mode)
proteins.

- **Atoms**: ray-cast sphere impostors with vdW-scaled radii.
- **Bonds**: capsule impostors (cylinders with hemispherical caps).
- **Lipid modes**: `Coarse` (P-only spheres + thin tail bonds) or
  `BallAndStick` (full detail).

#### 9. NucleicAcidRenderer

Renders DNA/RNA backbones and base rings.

- **Stems**: capsule instances tracing the phosphate backbone.
- **Rings**: polygon instances for the nucleobase rings.
- **Color**: per-base (default) or uniform.

#### 10. IsosurfaceRenderer

Renders molecular surfaces (Gaussian, SES, or cavity) and
electron-density isosurfaces, generated on the background scene-processor
worker via `SceneRequest::SurfaceRebuild`.

- **Backface depth pre-pass** is rendered separately so the composite
  pass can apply correct depth-aware blending for translucent
  surfaces.

### Shared Bind Groups

All renderers receive common bind groups via `DrawBindGroups`:

```rust
pub(crate) struct DrawBindGroups<'a> {
    pub camera: &'a wgpu::BindGroup,         // Projection / view matrices
    pub lighting: &'a wgpu::BindGroup,       // Light directions, intensities
    pub selection: &'a wgpu::BindGroup,      // Selection bit-array
    pub color: Option<&'a wgpu::BindGroup>,  // Per-residue color override
}
```

## Level of Detail

Backbone tessellation scales with camera distance to keep distant chains
cheap. Each frame `check_and_submit_lod` computes a per-chain LOD tier
from the chain's bounding center and the camera eye
(`select_chain_lod_tier`). When the per-chain tiers change, it submits an
animation-frame remesh whose `per_chain_lod: Option<Vec<ChainLod>>`
overrides the global `segments_per_residue` and `cross_section_verts` for
each chain. Tier 0 is full detail; higher tiers scale the segment and
cross-section counts down (`lod_scaled`). LOD is skipped while a full
rebuild is pending, since the backbone's cached chains are stale until it
applies.

## Post-Processing Stack

### 1. SSAO (Screen-Space Ambient Occlusion)

Computes local ambient occlusion from the depth and normal buffers.

- **Kernel**: hemisphere samples in view-space.
- **Noise**: 4×4 rotation noise texture to reduce banding.
- **Parameters**: `ao_radius` (0.5), `ao_bias` (0.025), `ao_power`
  (2.0).
- **Output**: single-channel AO texture.
- **Blur pass**: separable blur to smooth noise patterns.

### 2. Bloom

Extracts and blurs bright areas of the image.

- **Threshold**: extracts pixels above `bloom_threshold` (1.0) to a
  half-resolution texture.
- **Blur**: separable Gaussian blur (horizontal then vertical,
  ping-pong textures).
- **Mip chain**: progressive downsampling.
- **Upsample**: additive accumulation back to half-resolution.
- **Output**: half-resolution bloom texture.
- **Default `bloom_intensity`**: `0.0` (disabled).

### 3. Composite

Combines all post-processing inputs into the final image.

**Inputs**:
- Scene color texture
- SSAO texture
- Depth texture
- Normal G-buffer
- Bloom texture
- Composite params uniform

**Effects applied**:
- SSAO as a darkening multiplier on base color.
- Depth-based fog (configurable `fog_start` and `fog_density`).
- Depth-based outlines (edge detection on depth discontinuities).
- Normal-based outlines (edge detection on normal discontinuities).
- Bloom additive blend.
- HDR tone mapping with `exposure`.
- Gamma correction.

### 4. FXAA

Fast Approximate Anti-Aliasing as the final pass.

- Smooths jagged edges on mesh-based geometry that supersampling
  alone doesn't fully resolve.
- Reads from the composite output, writes to the swapchain surface
  (or to the caller-owned texture view in `render_to_texture`).

## ShaderComposer

Viso uses `naga_oil` for shader composition, enabling modular WGSL
with imports:

```wgsl
#import viso::camera
#import viso::lighting

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let light = calculate_lighting(in.normal, in.position);
    // …
}
```

Shaders live under `src/shaders/`:

- `shaders/modules/`: shared modules (`camera.wgsl`, `lighting.wgsl`,
  `pbr.wgsl`, `ray.wgsl`, `volume.wgsl`, `selection.wgsl`,
  `highlight.wgsl`, `shade.wgsl`, `depth.wgsl`, `constants.wgsl`,
  `fullscreen.wgsl`, `impostor_types.wgsl`).
- `shaders/raster/mesh/`: mesh rasterization (backbone, NA).
- `shaders/raster/impostor/`: impostor shaders (sphere, capsule, cone,
  polygon).
- `shaders/screen/`: full-screen passes (`composite.wgsl`, `fxaa.wgsl`,
  `ssao.wgsl`, `ssao_blur.wgsl`, `bloom_*.wgsl`).
- `shaders/utility/`: picking shaders (`picking_mesh.wgsl`,
  `picking_capsule.wgsl`, `picking_sphere.wgsl`).

The composer produces `naga::Module` IR directly (skipping WGSL
re-parse at runtime for performance).

## Render-Scale Supersampling

The rendering resolution can differ from the display resolution via
`engine.set_surface_scale(scale)`. All internal textures (color,
depth, normal, SSAO, bloom) are sized to the render resolution. FXAA
downsamples to the display resolution as the final step.
