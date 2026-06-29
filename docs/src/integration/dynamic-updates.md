# Dynamic Structure Updates

Viso supports live manipulation: structures can be updated mid-session
by computational backends (Rosetta energy minimization, ML structure
prediction) or user actions (mutations, drag operations).

All structural mutations happen on **your** `molex::Assembly`. After
each batch of changes, push the new snapshot to the engine via
`engine.set_assembly(Arc::new(assembly.clone()))`. The engine itself
is read-only with respect to structural state.

## Per-Entity Coordinate Updates

To stream new atom positions for an existing entity, mutate the
relevant entity's coordinates on your `Assembly` (using molex's
`update_protein_entities` codec helper, or
`assembly.update_positions(eid, &coords)` for direct position updates),
then re-publish:

```rust
use std::sync::Arc;
use molex::ops::codec::update_protein_entities;

// Apply caller-provided Coords through molex's shared codec so the
// path matches the byte-format pipeline.
let mut entities = vec![assembly.entity(eid).unwrap().clone()];
update_protein_entities(&mut entities, &new_coords);
if let Some(updated) = entities.into_iter().next() {
    assembly.remove_entity(eid);
    assembly.add_entity(updated);
}

engine.set_assembly(Arc::new(assembly.clone()));
```

To make the next sync animate (instead of snapping), queue a per-entity
behavior override before re-publishing:

```rust
engine.set_entity_behavior(entity_id, Transition::smooth());
engine.set_assembly(Arc::new(assembly.clone()));
```

The engine queues the transition for the affected entity on its next
sync, regardless of whether the override was set before or after the
`set_assembly` call (transitions are picked up in `update`).

### Per-Entity Behavior Overrides

Override the default transition for a specific entity. Once set, every
subsequent sync involving that entity uses the override (until
cleared):

```rust
let eid = engine.entity_id(raw_id).expect("known entity");

engine.set_entity_behavior(eid, Transition::smooth());

// Subsequent re-publishes that touch this entity will use the
// override instead of the default transition.
engine.set_assembly(Arc::new(assembly.clone()));

// Revert to default:
engine.clear_entity_behavior(eid);
```

## Transitions

Every update can specify a `Transition` controlling the visual
animation:

```rust
// Instant snap (no animation; used internally for initial loads
// and trajectory frames).
Transition::snap()

// Standard smooth interpolation (300ms cubic-hermite ease-out).
Transition::smooth()

// Staggered per-residue wave (quadratic-out).
Transition::cascade(
    Duration::from_millis(500),
    Duration::from_millis(5),
)

// Allow backbone size changes (residue mutations).
Transition::smooth().allowing_size_change()
```

See [Animation System](../deep-dives/animation-system.md) for details
on the data-driven phase model.

### Preemption

If a new update arrives while an animation is playing, the current
visual position becomes the new animation's start state and the timer
resets. This provides responsive feedback during rapid update cycles
(e.g. Rosetta wiggle).

## Constraint Visualization (Bands and Pulls)

Bands and pulls are not commands; they are stored constraint specs that
the engine resolves to world-space positions every frame so they
auto-track animated atoms. Steric clash arcs and exposed-hydrophobic
markers work the same way.

### Bands

A `BandInfo` references atoms structurally rather than by world-space
position:

```rust
use viso::{AtomRef, BandInfo, BandTarget, BandType};

let band = BandInfo {
    anchor_a: AtomRef { residue: 42, atom_name: "CA".into() },
    anchor_b: BandTarget::Atom(AtomRef {
        residue: 87,
        atom_name: "CA".into(),
    }),
    strength: 1.0,
    target_length: 3.5,
    band_type: Some(BandType::Disulfide),
    is_pull: false,
    is_push: false,
    is_disabled: false,
    from_script: false,
};
```

`BandTarget::Position(Vec3)` anchors one end to a fixed world-space
point (used for "space pulls"). `band_type` set to `None` lets the
engine auto-detect the type from `target_length`.

Visual properties:
- **Radius** scales with `strength` (0.1 to 0.4 Å)
- **Color** depends on `band_type`: default (purple), backbone
  (yellow-orange), disulfide (yellow-green), H-bond (cyan)
- **Disabled bands** are gray
- **Script-authored bands** (`from_script: true`) render dimmer

### Pulls

A `PullInfo` is a single active drag constraint. The atom is
referenced structurally; the target is given in screen-space (physical
pixels) and unprojected at the atom's depth each frame so the drag
stays parallel to the camera plane:

```rust
use viso::{AtomRef, PullInfo};

let pull = PullInfo {
    atom: AtomRef { residue: 42, atom_name: "CA".into() },
    screen_target: (mouse_x, mouse_y),
};
```

Pulls render as a purple cylinder from the atom to the target with a
cone arrow head at the target end.

Update band and pull specs through the engine. Both methods replace
the previous specs and re-resolve immediately:

```rust
engine.update_bands(vec![band1, band2]);
engine.update_pull(Some(pull));
engine.update_pull(None); // clear when drag ends
```

The engine resolves stored specs to world-space positions every frame,
so bands and pulls track animated atoms automatically.

### Clash Arcs

Steric clashes render as an electric arc between two atoms. A `ClashInfo`
names each atom per-entity (an entity id plus an entity-local residue and
PDB atom name) rather than by flat residue index:

```rust
use viso::{ClashEndpoint, ClashInfo};

let clash = ClashInfo {
    a: ClashEndpoint { entity: eid_a, residue: 12, atom_name: "CB".into() },
    b: ClashEndpoint { entity: eid_b, residue: 40, atom_name: "CG".into() },
    severity: 0.8, // drives emissive intensity and pulse brightness
};

engine.update_clashes(vec![clash]); // replaces the previous clash set
```

### Exposed Hydrophobics

Flagged exposed-hydrophobic residues render as a "grease bead" at the
sidechain anchor (CB if present, else the sidechain centroid, else CA).
An `ExposedHydrophobicInfo` names the residue per-entity:

```rust
use viso::ExposedHydrophobicInfo;

engine.update_exposed_hydrophobics(vec![ExposedHydrophobicInfo {
    entity: eid,
    residue: 23,
}]);
```

Both `update_clashes` and `update_exposed_hydrophobics` replace the
previous set and re-resolve immediately; the engine re-resolves their
anchors every frame so the markers track animated atoms.

## Host-Supplied Void Field

A host can push a precomputed void distance field to be meshed as a
smooth blob alongside detected cavities:

```rust
engine.set_external_void_field(dims, origin, spacing, phi, threshold);
```

`phi` is a flat row-major scalar grid (`phi[x*ny*nz + y*nz + z]`) in
marching-cubes polarity: high at void centers, near zero at atom walls
and exterior. It is meshed at the positive `threshold` into a blob that
carries the same cavity tint and breathing as a detected cavity. An empty
`phi` (or any zero dimension) clears the field. Meshing runs on the
background worker, so the result appears on the next frame after it
completes.
