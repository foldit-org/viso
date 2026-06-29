# Building and Running

## Prerequisites

- **Rust** (stable, 1.80+)
- **A GPU with WebGPU support**: Metal (macOS), Vulkan (Linux/Windows), or DX12 (Windows)
- **Internet access** (optional, for RCSB downloads)

### GUI panel (viso-ui)

The default build embeds a WASM-based options panel. On first `cargo build`, the
build script runs [Trunk](https://trunkrs.dev/) automatically to compile it. Two
extra tools are required:

```sh
# WASM compilation target
rustup target add wasm32-unknown-unknown

# Trunk (WASM bundler)
cargo install trunk
```

If Trunk or the WASM target is missing, the build still succeeds but the panel
will be non-functional. To skip the GUI entirely, build with
`--no-default-features --features viewer`.

## Building

From the repository root:

```sh
# Build the standalone viewer
cargo build -p viso

# Build with optimizations (recommended for real use)
cargo build -p viso --release
```

## Running

### With a PDB ID

Pass a 4-character PDB code to auto-download from RCSB:

```sh
cargo run -p viso --release -- 1ubq
```

The file is downloaded as mmCIF and cached in `assets/models/1ubq.cif`. Subsequent runs with the same ID load from cache.

### With a Local File

```sh
cargo run -p viso --release -- path/to/structure.cif
```

Viso supports mmCIF (`.cif`), PDB (`.pdb`), and BinaryCIF (`.bcif`) files.

## Logging

Viso uses `env_logger`. Control verbosity with `RUST_LOG`:

```sh
# Errors only (default)
cargo run -p viso -- 1ubq

# Info-level (see download progress, frame counts, etc.)
RUST_LOG=info cargo run -p viso -- 1ubq

# Debug-level (animation frames, picking results, mesh timing)
RUST_LOG=debug cargo run -p viso -- 1ubq

# Module-specific filtering
RUST_LOG=viso::renderer::pipeline::processor=debug cargo run -p viso -- 1ubq
```

## Platform Notes

### macOS (Metal)

Metal is the default backend. No extra setup needed. Ensure your macOS version is 10.15+ (Catalina) or later.

### Linux (Vulkan)

Requires Vulkan drivers. Install:

```sh
# Ubuntu/Debian
sudo apt install libvulkan-dev vulkan-tools

# Fedora
sudo dnf install vulkan-loader-devel vulkan-tools
```

### Windows (DX12 / Vulkan)

DX12 is the default backend on Windows 10+. Vulkan is also supported if drivers are installed.

## Controls

| Input | Action |
|-------|--------|
| Left drag | Rotate camera |
| Shift + left drag | Pan camera |
| Scroll wheel | Zoom |
| Click residue | Select residue |
| Shift + click | Add/remove from selection |
| Double-click | Select secondary structure segment |
| Triple-click | Select entire chain |
| Click background | Clear selection |
| Q | Recenter camera on focus |
| Tab | Cycle focus through entities |
| R | Toggle turntable auto-rotation |
| T | Toggle trajectory playback |
| I | Toggle ion visibility |
| U | Toggle water visibility |
| O | Toggle solvent visibility |
| L | Cycle lipid display mode |
| \` | Reset focus to session |
| Escape | Clear selection |
| \\ | Toggle the GUI options panel (when built with `gui`) |
