# GPU Picking and Selection

Viso uses GPU-based picking to determine what is under the mouse
cursor. This is faster and more accurate than CPU ray-casting,
especially with complex molecular geometry.

## How Picking Works

### Offscreen Render Pass

The picking system renders all molecular geometry to an offscreen
texture with format `R32Uint`. Instead of colors, each fragment
writes a **pick ID** (an entity-and-element-specific 1-based index;
0 means "no hit").

```
Main render: geometry → HDR color + normals + depth
Picking render: same geometry → R32Uint pick IDs + depth
```

The picking pass uses depth testing (`Less` compare with depth
writes) so only the closest geometry's pick ID survives.

### Geometry Types in Picking

The picking pass renders the following geometry, each with its own
shader:

1. **Backbone tube + ribbon** (`picking_mesh.wgsl`). In Cartoon mode the
   renderer issues separate index ranges for tube (coil) segments and
   ribbon (helix/sheet) segments; both write their residue's pick ID.
2. **Sidechain capsules** (`picking_capsule.wgsl`) with a storage buffer
   of capsule instances.
3. **Ball-and-stick spheres** (`picking_sphere.wgsl`). Atom indices are
   mapped through the per-rebuild `PickMap`.
4. **Ball-and-stick capsules** (`picking_capsule.wgsl`) for bond capsules
   in BallAndStick mode.

### PickTarget and PickMap

A typed pick target:

```rust
pub enum PickTarget {
    None,
    Residue(u32),                            // residue index
    Atom { entity_id: u32, atom_idx: u32 },  // small-molecule atom
}
```

A `PickMap` (built per-rebuild, embedded in `PreparedRebuild`) maps
raw GPU pick IDs to typed targets:

- `0` → `None`
- `1..=residue_count` → `Residue(idx)`
- `residue_count+1..=residue_count+atom_count` → `Atom { entity, atom }`

### Non-Blocking Readback

Reading data back from the GPU is expensive if done synchronously.
Viso uses a two-frame pipeline:

**Frame N:**
1. The picking pass renders to the offscreen texture.
2. A single pixel at the mouse position is copied to a staging
   buffer (256 bytes minimum, aligned for wgpu).
3. `start_readback()` initiates an async buffer map without blocking.

**Frame N+1:**
1. `poll_and_resolve` polls the wgpu device without blocking.
2. If the map callback has fired (signaled via `AtomicBool`), the
   mapped data is read: 4 bytes as `u32`, resolved through the active
   `PickMap` to a `PickTarget`.
3. The staging buffer is unmapped.
4. The result is cached in `hovered_target` on the picking system.

If the readback isn't ready yet, the previous frame's cached value is
used, so hover is at most one frame behind the cursor.

The flow is wired up inside `engine.render()`:

```rust
self.gpu.pick.picking.start_readback();      // after queue.submit()
self.gpu.pick.poll_and_resolve(&device);     // wraps complete_readback()
```

`poll_and_resolve` (`src/renderer/picking/mod.rs`) is the call the engine
makes; it internally calls `complete_readback` on the picking pipeline
and maps the raw id through the `PickMap`.

## Public Hover API

Consumers query the resolved hover target through the engine:

```rust
let target: PickTarget = engine.hovered_target();

match target {
    PickTarget::None => { /* mouse on background */ }
    PickTarget::Residue(idx) => { /* hovering residue */ }
    PickTarget::Atom { entity_id, atom_idx } => { /* hovering ligand atom */ }
}
```

`hovered_target` is read from the cached pick resolved on the previous
frame, so it is current as of the last completed readback.

## Selection Buffer

The `SelectionBuffer` is a GPU storage buffer containing a bit-array
of selected residues. It's bound to all molecular renderers so
shaders can highlight selected residues.

### Bit Packing

Selection is stored as u32 words with one bit per residue:

```
Word 0: residues 0-31   (bit 0 = residue 0, bit 1 = residue 1, …)
Word 1: residues 32-63
Word 2: residues 64-95
…
```

### Updating Selection

The engine pushes the latest selection to the GPU each frame inside
`pre_render`. Consumers don't need to call this directly.

### Dynamic Capacity

The buffer grows as needed when entity counts change. The engine's
`ensure_residue_capacity` rebuilds the buffer and bind group when
the total residue count exceeds the current capacity.

## Click Handling

The engine classifies clicks; the host applies them. `feed_pointer_button`
returns a `ClickEvent` whose `expansion` field already lists the residues
the click selects (computed against the current scene). The host runs
`classify_click_for_selection` to get a `ClickSelectionAction`, applies
it to its own selection store, then pushes the result with
`engine.set_selection`. See
[Handling Input](../integration/handling-input.md) for the full loop.

| Pattern | Expansion |
|---------|-----------|
| `Single` | the clicked residue |
| `Double` | every residue in the clicked residue's SS segment |
| `Triple` | every residue in the clicked residue's chain |
| `Empty` | empty (background click; classifies as `Clear`) |

A plain click maps to `Replace(expansion)`, a shift-held click to
`Toggle(expansion)`, and an `Empty` click to `Clear`.

### Double Click (Secondary Structure Segment)

The double-click expansion walks the engine's concatenated cartoon SS
array backward and forward from the clicked residue until the SS type
changes, then returns every residue in the resulting range
(`residues_in_segment`).

### Triple Click (Chain)

The triple-click expansion finds the chain containing the clicked
residue and returns every residue in that chain (`residues_in_chain`).

### Click Type Detection

The engine's multi-click state machine (`src/input/click_state.rs`)
tracks timing between presses. Clicks within a threshold on the same
target increment the click counter (single, double, triple). If the
pointer moved between press and release, the gesture is classified as a
drag (which drives the camera) and `feed_pointer_button` returns `None`.

## Selection in Shaders

All molecular renderers receive the selection bind group. In the
fragment shader:

```wgsl
let word_idx = residue_idx / 32u;
let bit_idx = residue_idx % 32u;
let is_selected = (selection_data[word_idx] >> bit_idx) & 1u;

if is_selected == 1u {
    // Apply selection highlight (e.g. brighten color)
}
```

The hover effect uses the camera uniform's `hovered_residue` field: the
shader checks whether the fragment's residue index matches the hovered
residue and applies a highlight.

## Querying and Mutating Selection State

Selection is host-owned. Push the authoritative per-entity selection to
the engine and read back the hover target:

```rust
// Replace the selection. The engine stores this per-entity map as the
// source of truth and re-derives the flat GPU bitset from it.
engine.set_selection(&selection); // &BTreeMap<EntityId, BTreeSet<u32>>

// Clear the selection.
engine.clear_selection();

// Currently hovered target (resolved from the previous frame's pick).
let hovered: PickTarget = engine.hovered_target();
```

The engine keeps no public list of selected residues; the host's store
is authoritative, and the engine derives its GPU bitset from whatever
`set_selection` was last given.
