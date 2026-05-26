# Viso

Viso is a GPU-accelerated 3D protein visualization engine written in Rust. It renders interactive, real-time views of macromolecular structures from PDB and mmCIF files using modern WebGPU graphics via [wgpu](https://wgpu.rs/).

Built with a full post-processing pipeline (bloom, SSAO, FXAA, tone mapping), background mesh generation, and an animation system for smooth structural transitions.

For architecture details, integration guides, and deep dives into individual subsystems, see the [full documentation](https://petridecus.github.io/viso/).

<p align="center">
  <img src="gallery/spike-protein.png" width="32%" />
  <img src="gallery/fatty-acid-synthase.png" width="32%" />
  <img src="gallery/hemolysin.png" width="32%" />
</p>
<p align="center">
  <img src="gallery/mitochondrial-complex.png" width="32%" />
  <img src="gallery/tetranucleosome.png" width="32%" />
  <img src="gallery/tri-snrnp-spliceosome.png" width="32%" />
</p>

## Features

- **Post-processing pipeline**: bloom, screen-space ambient occlusion, FXAA anti-aliasing, edge outlines, tone mapping
- **Ray-marched impostors**: pixel-perfect sidechains and ball-and-stick atoms at any zoom level
- **Interactive camera**: arcball rotation, panning, zoom, auto-rotation
- **GPU picking**: click to select individual residues, double-click to select segments, triple-click to select chains, shift-click for multi-select
- **Animation system**: multiple behaviors including smooth interpolation, cascading reveals, and collapse/expand transitions
- **RCSB integration**: pass a 4-character PDB ID and Viso downloads the structure automatically
- **TOML-based presets**: configure display, lighting, coloring, geometry, and post-processing via preset files
- **Background scene processing**: mesh generation runs on a dedicated CPU thread to keep the render loop responsive

## Prerequisites

Viso requires a Rust toolchain. Install one via [rustup](https://rustup.rs/):

```sh
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

Then add the **nightly** toolchain (used for formatting):

```sh
rustup toolchain install nightly
rustup component add rustfmt --toolchain nightly
```

Verify both are available:

```sh
cargo --version
cargo +nightly fmt --version
```

### GUI panel (viso-ui)

The default build embeds a WASM-based options panel. On first `cargo build`, the
build script runs [Trunk](https://trunkrs.dev/) automatically to compile it. Two
extra tools are required:

```sh
# WASM compilation target
rustup target add wasm32-unknown-unknown

# Trunk (WASM bundler) — install once
cargo install trunk
```

If Trunk or the WASM target is missing, the build still succeeds but the panel
will be non-functional. To skip the GUI entirely, build with
`--no-default-features --features viewer`.

### Platform-specific dependencies

**Linux** — the default build includes a GUI webview panel, which requires GTK3 and WebKit2GTK:

```sh
# Debian/Ubuntu
sudo apt-get install libgtk-3-dev libwebkit2gtk-4.1-dev

# Fedora
sudo dnf install gtk3-devel webkit2gtk4.1-devel

# Arch
sudo pacman -S gtk3 webkit2gtk-4.1
```

**macOS** and **Windows** require no additional system dependencies beyond those
listed in [GUI panel](#gui-panel-viso-ui) above.

## Building

```bash
# Debug build
cargo build

# Optimized build (recommended for interactive use)
cargo build --release
```

## Running

Viso takes a single argument: either a 4-character PDB ID or a path to a local `.cif`/`.pdb` file.

```bash
# Download and visualize a structure from RCSB PDB
cargo run --release -- 1ubq

# Visualize a local file
cargo run --release -- ./my_structure.cif
```

When given a PDB ID, Viso downloads the corresponding mmCIF file from RCSB and caches it in `assets/models/`.

### Controls

| Input | Action |
|-------|--------|
| Left-click drag | Rotate |
| Shift + drag | Pan |
| Scroll wheel | Zoom |
| Click | Select residue |
| Double-click | Select secondary structure segment |
| Triple-click | Select chain |
| Shift + click | Multi-select |
| `Tab` | Cycle focus (Session → Structure → Entity) |
| `` ` `` (backtick) | Reset focus to full session |
| `W` | Toggle water visibility |
| `Escape` | Clear selection |

Key bindings are configurable via TOML preset files.

### Logging

Viso uses `env_logger`. Set the `RUST_LOG` environment variable to control log output:

```bash
RUST_LOG=info cargo run --release -- 1ubq
```

## License

All rights reserved. See [LICENSE.md](LICENSE.md) for details.

Contributions are welcome under the terms of the [Contributor License Agreement](CLA.md).
