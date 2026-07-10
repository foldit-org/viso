# Color Modes

Viso supports several coloring schemes for backbones, sidechains, and
nucleic acids. Colors are computed during background scene processing
and uploaded to GPU buffers for zero-cost rendering.

The active color scheme is driven by the `ColorScheme` enum on
`DisplayOverrides`, which decouples *what data drives color* from
*which palette is used*.

## ColorScheme

```rust
pub enum ColorScheme {
    Entity,             // Each entity gets a distinct palette color
    SecondaryStructure, // Helix / sheet / coil
    ResidueIndex,       // N-to-C gradient per chain
    BFactor,            // Crystallographic B-factor gradient
    Hydrophobicity,     // Kyte-Doolittle hydrophobicity gradient
    Score,              // Absolute Rosetta energy score
    ScoreRelative,      // Score normalized to the 5th/95th percentiles
    Solid,              // Single uniform color (first palette stop)
}
```

`ColorScheme` chooses *what data* maps to color. The companion
`Palette` (selected via `palette_preset` and `palette_mode` on
`DisplayOverrides`) chooses *which colors*. Any scheme can be combined
with any palette.

## Color modes in detail

### Chain / Entity (Default)

Each entity gets a distinct color from the active palette. Single-chain
proteins use a gradient along the chain. This is the most common mode
for general visualization.

### Secondary Structure

Colors residues by their computed secondary structure type:
- **Alpha helix**: distinct helix color
- **Beta sheet**: distinct sheet color
- **Coil/Loop**: neutral color

Secondary structure is computed by `molex` (DSSP) by default. Per-entity
overrides via `engine.set_ss_override(id, ss_types)`.

### Score / ScoreRelative

Colors residues by per-residue energy values (e.g. from Rosetta).
`Score` uses absolute values; `ScoreRelative` normalizes to the
5th/95th percentiles within the structure. Scores are set via
`engine.set_per_residue_scores(id, Some(scores))`.

### ResidueIndex

N-to-C gradient per chain; useful for sequence-position visualization.

### BFactor / Hydrophobicity

Gradient by crystallographic B-factor or Kyte-Doolittle hydrophobicity.

### Solid

Single uniform color drawn from the first stop of the active palette.

## Sidechain Color Modes

```rust
pub enum SidechainColorMode {
    Hydrophobicity,
    Backbone,    // default: match the backbone color of the residue
}
```

### Backbone (Default)

Sidechain atoms inherit the backbone color of their residue. This is
the default because it makes sidechains read as part of their residue
visually rather than as an independent layer.

### Hydrophobicity

Hydrophobic / hydrophilic dichotomy:
- **Hydrophobic**: blue (default: `[0.3, 0.5, 0.9]`)
- **Hydrophilic**: orange (default: `[0.95, 0.6, 0.2]`)

Configurable via `ColorOptions::hydrophobic_sidechain` /
`hydrophilic_sidechain`.

## Nucleic Acid Color Modes

```rust
pub enum NaColorMode {
    Uniform,
    BaseColor,   // default: color each backbone segment by its base
}
```

### BaseColor (Default)

Each residue's backbone segment is colored to match its nucleobase
(A/T/G/C/U).

### Uniform

All nucleic acid backbone uses a single color (default light
blue-violet `[0.45, 0.55, 0.85]`), configurable via
`ColorOptions::nucleic_acid`.

## Non-Protein Coloring

Ligands, ions, and waters use element-based CPK coloring in the
ball-and-stick renderer:

- Standard CPK colors for common elements (C, N, O, S, P, etc.)
- **Lipid carbons** use a warm beige/tan tint (configurable via
  `ColorOptions::lipid_carbon_tint`)
- **Cofactors** can have per-residue-name carbon tints via
  `ColorOptions::cofactor_tints`

## Color Transitions During Animation

When backbone colors change between poses (e.g. score coloring updates
after minimization), the background processor caches per-residue
colors in the prepared scene. During animation, the renderers
interpolate between the old and new colors using the same easing
function as the backbone position interpolation.

Color changes interpolate over the animation duration rather than
snapping to the new colors in a single frame.

## How Colors Flow Through the Pipeline

1. **Scene sync**: `DisplayOptions`, `DisplayOverrides`, and
   `ColorOptions` are sent to the background processor as part of the
   `FullRebuild` request.
2. **Background thread**: during mesh generation, colors are computed
   per-residue from the resolved color scheme and palette and baked into
   vertex / instance buffers.
3. **GPU upload**: color buffers are uploaded to the GPU as part of the
   prepared rebuild.
4. **Rendering**: shaders read per-residue colors directly, with
   selection highlighting applied as an overlay in the fragment shader
   via the `SelectionBuffer` bit-array.
