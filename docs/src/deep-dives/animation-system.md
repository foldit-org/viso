# Animation System

Viso's animation system handles smooth visual transitions when protein
structures change. It is data-driven: a `Transition` describes the
animation as a sequence of phases, and an `AnimationRunner` evaluates
those phases each frame.

## Data-Driven Architecture

```
Transition              ->    AnimationRunner
  (phases + flags)            (evaluates phases per frame)
```

1. **Transition**: a struct holding a `Vec<AnimationPhase>` plus an
   `allows_size_change` flag. Each phase carries an easing function,
   duration, lerp range, and a sidechain-visibility flag.
2. **AnimationRunner**: evaluates a single animation from start to target,
   advancing through phases sequentially.

There are no trait objects or behavior types. The consumer constructs a
`Transition` from a preset constructor, and the runner evaluates the
phase sequence directly.

## Transition

`Transition` is the only animation type in the public API. Construct it
from a preset and tune it with the builder method:

```rust
pub struct Transition {
    pub allows_size_change: bool,
    // phases + name are crate-internal
}

// Preset constructors
Transition::snap()     // Instant; allows resize (also used for trajectory frames)
Transition::smooth()   // 300ms cubic-hermite ease-out (also Default)
Transition::cascade(base_dur, delay_per_residue)

// Total duration across phases
let d: Duration = transition.total_duration();

// Builder method
Transition::smooth().allowing_size_change()
```

## AnimationPhase (internal)

Each phase defines a segment of the animation:

```rust
pub(crate) struct AnimationPhase {
    pub(crate) easing: EasingFunction,
    pub(crate) duration: Duration,
    pub(crate) lerp_start: f32,    // e.g. 0.0
    pub(crate) lerp_end: f32,      // e.g. 1.0
    pub(crate) include_sidechains: bool,
}
```

`AnimationPhase` is `pub(crate)`: consumers don't construct it; the
preset constructors build it. The runner maps raw progress (0 to 1 over
the total duration) through the phase sequence, and each phase applies
its own easing within its lerp range.

## Presets

### Snap

Instant transition, zero duration. Used for initial loads (where
animation would delay the first meaningful frame) and for trajectory
frames fed through the animation pipeline.

### Smooth (Default)

Standard eased lerp from start to target: 300ms with cubic-hermite
ease-out (`CubicHermite { c1: 0.33, c2: 1.0 }`). Good for incremental
changes where start and target are close.

### Cascade

Single-phase quadratic-out lerp intended for a staggered per-residue
wave. Per-residue staggering is not yet integrated into the runner, so it
currently animates all residues with the same timing.

## Per-Entity Animation

The engine drives one in-flight `AnimationPlayer` (built by
`build_animation` from the current and pending snapshots) and a
`StructureAnimator` that holds per-entity runner state keyed on
`EntityId`. The animator writes interpolated atom positions into the
engine's `EntityPositions` each frame.

The mutation surface lives on `VisoApp` (`update_entities`,
`update_entity`, `sync_entities`). Each call sets new target coordinates
and queues a per-entity `Transition` for the engine's next sync.
Per-entity behavior overrides (`engine.set_entity_behavior`) take
precedence over the supplied default transition.

### How It Works

1. The host mutates its `Assembly` and pushes the new snapshot via
   `engine.set_assembly`; pending per-entity transitions are stored on
   the engine's `AnimationState`.
2. On the next `engine.update()`, the engine builds an animation from the
   new snapshot. A same-topology change adopts the target up front and
   eases the kept positions toward it; a topology-changing mutation
   defers adoption to a waypoint and stays on the old conformation until
   then.
3. Each frame the runner advances and interpolated positions are written
   into `EntityPositions`. Sidechain positions are interpolated with the
   same eased `t` as the backbone.
4. When a runner completes (progress reaches 1.0), the entity snaps to
   target and the runner is removed.

### Preemption

When a newer snapshot arrives mid-animation, the engine coalesces to the
latest target: it re-aims while still easing (or drops the in-flight
player to rebuild toward the latest), letting an in-flight expand finish
first. The result is responsive feedback during rapid update cycles such
as a Rosetta wiggle.

## Sidechain Animation

Sidechain positions are stored alongside the backbone start/target arrays
and lerped with the same eased `t`, so renderers and constraint
resolution read interpolated sidechains without recomputing them. When
`allows_size_change` is set (a residue mutation), the start sidechain
positions are written as CA coordinates at setup so the lerp grows them
outward into their target positions. A phase with
`include_sidechains: false` hides sidechains for that segment of the
animation.

## Trajectory Playback

DCD trajectory frames feed through the same animation pipeline.
`TrajectoryPlayer` (`src/engine/trajectory.rs`) is a frame sequencer with
no animation dependencies. Each frame it produces is applied through the
same path as `Transition::snap()`, so trajectory and structural animation
share one code path in the engine's `tick_animation`.

```rust
engine.load_trajectory(Path::new("path/to/traj.dcd"));
engine.toggle_trajectory();   // play / pause
let has = engine.has_trajectory();
```

## Easing Functions

Defined in `src/util/easing.rs`:

| Function | Description |
|----------|-------------|
| `Linear` | No easing |
| `QuadraticIn` | Slow start, fast end |
| `QuadraticOut` | Fast start, slow end |
| `CubicHermite { c1, c2 }` | Configurable control points (default: ease-out) |

All functions clamp input to `[0, 1]` and evaluate in well under 100ns.

## Disabling Animation

Use `Transition::snap()` per update, or set a `snap` behavior on an
entity so every subsequent update is instantaneous:

```rust
let eid = engine.entity_id(raw_id).expect("known entity");
engine.set_entity_behavior(eid, Transition::snap());
```
