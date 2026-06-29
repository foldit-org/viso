# Scene Management

Viso's structural state lives in `molex::Assembly`, owned by **your**
application, not by viso. Viso is a pure consumer: you push the latest
`Arc<Assembly>` to the engine via [`VisoEngine::set_assembly`], and the
engine drains the snapshot on the next sync tick and rederives its render
state.

There is no "group" abstraction in viso. Every entity lives directly in
the `Assembly` and is identified by an opaque `EntityId`.

## Pushing Assembly Snapshots

```
┌──────────────────────┐                          ┌─────────────────┐
│   your application   │   Arc<Assembly>          │   VisoEngine    │
│                      │ ─engine.set_assembly──►  │                 │
│  molex::Assembly     │                          │  pending slot   │
│  (mutated freely)    │                          │  → Scene+derived│
└──────────────────────┘                          └─────────────────┘
```

- Your application mutates its `molex::Assembly` (using molex's APIs)
  and calls `engine.set_assembly(Arc::new(self.assembly.clone()))`.
- `VisoEngine::update(dt)` drains the pending snapshot; if its
  generation differs from the last applied one, the engine rederives
  its per-entity state and submits a full-rebuild request to the
  background mesh processor.

That's the entire structural ingest contract for library users. There
is no viso-defined channel, publisher, or consumer in the public API.

## Mutating the Scene

You mutate `molex::Assembly` through molex's own APIs and re-publish
to viso after each batch of changes:

```rust
use std::sync::Arc;
use molex::Assembly;
use viso::{Transition, VisoEngine};

// 1. Mutate the assembly however you like.
let mut assembly = /* your owned Assembly */;
assembly.add_entity(new_entity);
assembly.update_positions(eid, &new_coords);
// ... add/remove/update as needed ...

// 2. Push the new snapshot. Cheap: Arc<Assembly> is shared
//    by reference.
engine.set_assembly(Arc::new(assembly.clone()));

// 3. (Optional) For entities whose positions changed, queue a
//    per-entity transition so the next sync animates instead of
//    snapping. Without this, the engine snaps to the new state.
engine.set_entity_behavior(entity_id, Transition::smooth());
```

The next `engine.update(dt)` drains the pending snapshot, rederives
the scene, and submits a full-rebuild to the background mesh
processor. Mesh generation happens off-thread, so the main thread is
not blocked.

> **Note.** If you're embedding viso as a library, ignore `VisoApp`
> entirely. `VisoApp` is the standalone-app helper that viso uses to
> be its own host when run via `cargo run -p viso` (or the `viewer`
> / `gui` / `web` features). Library consumers own their own
> `Assembly` and don't need or want the convenience wrapper.

## Engine-Side Annotations

Some per-entity state is purely a viso concern (it doesn't belong on
the molecular structure itself). Those live on
`EntityAnnotations`, mutated through engine methods:

```rust
// Animation behavior overrides (keyed by EntityId, not raw u32).
let eid = engine.entity_id(raw_id).expect("known entity");
engine.set_entity_behavior(eid, Transition::smooth());
engine.clear_entity_behavior(eid);

// Per-entity appearance overrides (drawing mode, color scheme,
// helix/sheet style, surface kind, palette, etc.).
let mut overrides = DisplayOverrides::default();
overrides.color_scheme = Some(ColorScheme::SecondaryStructure);
engine.set_entity_appearance(eid, overrides);
engine.clear_entity_appearance(eid);
```

`set_entity_appearance` diffs against the previous overrides and
dispatches only the invalidations that matter: a `surface_kind` change
triggers surface regeneration, a `color_scheme` change triggers color
recomputation, and so on.

## Looking Up Entities

The engine exposes a small read-only surface for looking entities up:

```rust
// Translate a raw u32 (from IPC, TOML, CLI) to an opaque EntityId.
let eid: Option<EntityId> = engine.entity_id(raw_id);

// Walk the current Assembly snapshot directly.
for entity in engine.assembly().entities() {
    println!("{:?}: {}", entity.id(), entity.molecule_type());
}

// Total entity count.
let n = engine.entity_count();
```

`entity_id` is the canonical "boundary translator": wire formats carry
raw `u32` ids; viso-internal APIs use `EntityId`. Translate once at the
boundary and pass `EntityId` through.

## Focus

Focus determines what the camera follows and what the user is "working
on". It cycles through visible, focusable entities with `Tab`:

```rust
pub enum Focus {
    All,                 // All entities (default)
    Entity(EntityId),    // A specific entity
}
```

```rust
// Cycle: All -> entity 1 -> ... -> entity N -> All
engine.cycle_focus();

// Focus a specific entity (by EntityId).
engine.focus_entity(eid);

// Reset to the all-entities view.
engine.reset_focus();

// Set focus without reframing the camera (host owns the reframe cadence).
engine.set_focus(Focus::Entity(eid));

// Read current focus state.
let focus: Focus = engine.focus();
let focused_entity: Option<EntityId> = engine.focused_entity();

// The entities the Tab cycle steps through, without mutating focus.
let focusable: Vec<EntityId> = engine.focusable_entities();
```

`cycle_focus`, `focus_entity`, and `reset_focus` refit the camera;
`set_focus` only records the value, leaving the reframe to the host.

## Host Query API

Foldit drives selection, pulls, and a residue-info panel through a set of
read-only query methods on the engine. They translate between the GPU's
flat residue space, structural references, and screen coordinates:

```rust
// Pick resolution: owning entity + entity-local residue + closest atom.
let pick: Option<PickedResidueAtom> =
    engine.picked_residue_atom(flat_residue, (x, y));

// The heavy atom in a residue projecting closest to a screen point.
let atom: Option<String> = engine.closest_atom_in_residue(residue, (x, y));

// Flat residue index -> (raw entity id, entity-local residue).
let owner: Option<(u32, u32)> = engine.flat_to_entity_residue(flat);

// Resolve a structural atom reference to its current (interpolated) world
// position.
let pos: Option<Vec3> = engine.resolve_atom_position(residue, "CA");

// Screen/world conversions (also see the Camera System chapter).
let screen: Option<Vec2> = engine.world_to_screen(world);
let world: Vec3 = engine.screen_to_world_at_depth(screen_pos, world_point);

// Camera framing.
let centroid: Option<Vec3> = engine.focus_centroid();
engine.snap_camera_to_focus();
engine.set_camera_pose(center, eye, up);
```

`resolve_atom_position`, `picked_residue_atom`, and `closest_atom_in_residue`
read interpolated visual positions during animation, so they match what
is on screen rather than the live target coordinates.

## What Happens During Sync

When a new `Assembly` snapshot arrives:

1. **Rederive per-entity state.** For each entity, the engine builds
   render-ready derived data (backbone chains, sidechain topology, SS
   types, residue color metadata).
2. **Submit a `FullRebuild`.** The background processor receives a
   `Vec<FullRebuildEntity>` plus the active display, color, and
   geometry options. Per-entity mesh caching means only entities whose
   `mesh_version` changed are regenerated.
3. **Triple-buffer the result.** When the processor finishes, the
   resulting `PreparedRebuild` is written to a triple buffer.
4. **Apply on the next frame.** `engine.update(dt)` calls
   `apply_pending_scene`, which uploads the GPU buffers in a memcpy
   and rebuilds picking bind groups.

The main thread never blocks. If the new meshes aren't ready by the
next frame, the previous frame's data continues to render until they
are.
