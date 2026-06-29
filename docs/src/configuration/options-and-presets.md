# Options and Presets

Viso's visual appearance is controlled by the `VisoOptions` struct,
which can be loaded from and saved to TOML files. This enables
presets for different visualization styles.

## Options Structure

```rust
pub struct VisoOptions {
    pub display: DisplayOptions,
    pub lighting: LightingOptions,
    pub post_processing: PostProcessingOptions,
    pub camera: CameraOptions,
    pub colors: ColorOptions,
    pub geometry: GeometryOptions,
    pub debug: DebugOptions,
}
```

All sub-structs use `#[serde(default)]`, so TOML files can be partial:
only the fields you want to override need to be specified. Key bindings
are not part of `VisoOptions`. They live in the standalone `KeyBindings`
table (`src/input/key_bindings.rs`), which holds closures and is not
serializable; build it with `KeyBindings::default()` and edit it in code,
not via TOML.

## Display Options

```rust
pub struct DisplayOptions {
    // Ambient visibility (type-level toggles)
    pub show_waters: bool,
    pub show_ions: bool,
    pub show_solvent: bool,

    // Surface presentation mode (VSync, immediate, mailbox)
    pub present_mode: PresentMode,

    // Structural bond display (H-bonds, disulfides)
    pub bonds: BondOptions,

    // Legacy (prefer `overrides.color_scheme`)
    pub backbone_color_mode: BackboneColorMode,

    // Per-entity overridable fields (flattened for TOML compat).
    // These are also used at per-entity scope via `EntityAnnotations`;
    // `None` at either scope falls through to the next layer
    // (entity → global → built-in defaults).
    #[serde(flatten)]
    pub overrides: DisplayOverrides,
}
```

`DisplayOverrides` carries 14 per-entity overridable fields:
`drawing_mode`, `color_scheme`, `helix_style`, `sheet_style`,
`show_sidechains`, `show_hydrogens`, `surface_kind`, `surface_opacity`,
`show_cavities`, `sidechain_color_mode`, `na_color_mode`, `lipid_mode`,
`palette_preset`, `palette_mode`. Any field set to `Some(...)` at the
global scope acts as the default for entities that don't override it.

Resolved getters (`display.drawing_mode()`, `display.show_sidechains()`,
etc.) walk the override chain to produce a final value.

## Lighting Options

```rust
pub struct LightingOptions {
    pub light1_intensity: f32,    // default: 2.0   (key light)
    pub light2_intensity: f32,    // default: 1.1   (fill light)
    pub ambient: f32,             // default: 0.45
    pub specular_intensity: f32,  // default: 0.35
    pub shininess: f32,           // default: 38.0
    pub rim_power: f32,           // default: 5.0
    pub rim_intensity: f32,       // default: 0.3
    pub rim_directionality: f32,  // default: 0.3
    pub rim_color: [f32; 3],      // default: [1.0, 0.85, 0.7]
    pub ibl_strength: f32,        // default: 0.6
    pub roughness: f32,           // default: 0.35
    pub metalness: f32,           // default: 0.15
}
```

Light directions are derived per-frame from the camera ("headlamp"
lighting) rather than configured statically.

## Post-Processing Options

```rust
pub struct PostProcessingOptions {
    pub outline_thickness: f32,         // default: 1.0
    pub outline_strength: f32,          // default: 0.7
    pub ao_strength: f32,               // default: 0.85
    pub ao_radius: f32,                 // default: 0.5
    pub ao_bias: f32,                   // default: 0.025
    pub ao_power: f32,                  // default: 2.0
    pub fog_start: f32,                 // default: 100.0
    pub fog_density: f32,               // default: 0.005
    pub exposure: f32,                  // default: 1.0
    pub normal_outline_strength: f32,   // default: 0.5
    pub bloom_intensity: f32,           // default: 0.0   (disabled)
    pub bloom_threshold: f32,           // default: 1.0
}
```

## Camera Options

```rust
pub struct CameraOptions {
    pub fovy: f32,          // Field of view in degrees, default: 45.0
    pub znear: f32,         // Near clip plane, default: 5.0
    pub zfar: f32,          // Far clip plane, default: 2000.0
    pub rotate_speed: f32,  // Mouse rotation sensitivity, default: 0.5
    pub pan_speed: f32,     // Mouse pan sensitivity, default: 0.5
    pub zoom_speed: f32,    // Scroll zoom sensitivity, default: 0.1
}
```

## Color Options

```rust
pub struct ColorOptions {
    pub lipid_carbon_tint: [f32; 3],       // Warm beige/tan for lipid carbons
    pub hydrophobic_sidechain: [f32; 3],   // Blue for hydrophobic sidechains
    pub hydrophilic_sidechain: [f32; 3],   // Orange for hydrophilic sidechains
    pub nucleic_acid: [f32; 3],            // Light blue-violet for DNA/RNA
    pub band_default: [f32; 3],            // Purple
    pub band_backbone: [f32; 3],           // Yellow-orange
    pub band_disulfide: [f32; 3],          // Yellow-green
    pub band_hbond: [f32; 3],              // Cyan
    pub solvent_color: [f32; 3],
    pub cofactor_tints: HashMap<String, [f32; 3]>,
}
```

The default `cofactor_tints` includes greens for chlorophylls (CLA,
CHL), oranges for carotenoids (BCR, BCB), reds for hemes (HEM, HEC,
HEA, HEB), and others.

## Geometry Options

Geometry options control cartoon rendering detail. Per-SS parameters
(width, thickness, roundness) can be set directly or driven by a
`cartoon_style` preset:

```rust
pub struct GeometryOptions {
    pub cartoon_style: CartoonStyle,    // Ribbon | Tube | Cylindrical | Custom
    pub sheet_arrows: bool,             // default: true

    // Per-SS appearance (in Ångström)
    pub helix_width: f32,               // default: 1.4
    pub helix_thickness: f32,           // default: 0.25
    pub helix_roundness: f32,           // default: 0.0
    pub sheet_width: f32,               // default: 1.6
    pub sheet_thickness: f32,           // default: 0.25
    pub sheet_roundness: f32,           // default: 0.0
    pub coil_width: f32,                // default: 0.4
    pub coil_thickness: f32,            // default: 0.4
    pub coil_roundness: f32,            // default: 1.0

    // Nucleic acid backbone
    pub na_width: f32,                  // default: 1.2
    pub na_thickness: f32,              // default: 0.25
    pub na_roundness: f32,              // default: 0.0

    // Mesh detail
    pub segments_per_residue: usize,    // default: 32
    pub cross_section_verts: usize,     // default: 16

    // Small-molecule rendering
    pub solvent_radius: f32,            // default: 0.15
    pub ligand_sphere_radius: f32,      // default: 0.3
    pub ligand_bond_radius: f32,        // default: 0.12
}
```

`CartoonStyle::Custom` keeps the per-SS fields as-is; the other
presets overwrite them at resolve time.

## Debug Options

`DebugOptions` controls debug-only visualizations (frustum overlays,
LOD heatmaps, etc.). See `options/debug.rs` for the current field set.

## Loading and Saving

```rust
// Load from TOML file (partial files supported)
let options = VisoOptions::load(Path::new("presets/dark.toml"))?;

// Save to TOML file
options.save(Path::new("presets/my_preset.toml"))?;

// List available presets in a directory
let presets = VisoOptions::list_presets(Path::new("presets/"));
// Returns: ["dark", "publication", "presentation", ...]
```

`VisoOptions::json_schema()` returns a Schemars schema describing the
UI-exposed subset of options (used by the embedded webview panel).

## Example TOML Preset

```toml
[lighting]
light1_intensity = 2.5
ambient = 0.5
specular_intensity = 0.4
shininess = 50.0

[post_processing]
outline_thickness = 1.5
outline_strength = 0.8
ao_strength = 1.0
bloom_intensity = 0.15
bloom_threshold = 0.8

[camera]
fovy = 35.0
rotate_speed = 0.4

[colors]
hydrophobic_sidechain = [0.2, 0.4, 0.85]
hydrophilic_sidechain = [0.9, 0.55, 0.15]
```

## Applying Options at Runtime

`engine.set_options(new_options)` is the canonical entry point. It
diffs the new options against the current ones and dispatches the
right invalidations:

- Display/color/geometry changes that affect mesh content trigger a
  full scene resync via the background processor.
- Lighting changes are pushed directly to GPU lighting uniforms.
- Post-processing changes update GPU uniforms and SSAO/bloom render
  targets without touching geometry.
- Camera changes (FOV, znear/zfar, sensitivity) are applied to the
  controller in place.
