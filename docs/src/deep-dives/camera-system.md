# Camera System

Viso uses an arcball camera that orbits a focus point. It supports
animated transitions between viewpoints, turntable auto-rotation, frustum
culling for sidechains, and screen/world coordinate conversion.

## Arcball Model

The camera is defined by four parameters:

- **Focus point**: the world-space point the camera orbits.
- **Distance**: how far the camera sits from the focus point.
- **Orientation**: a quaternion for the camera's rotation.
- **Bounding radius**: the radius of the structure being viewed, used to
  drive fog and culling.

All manipulation (rotate, pan, zoom) operates on these parameters, not on
a view matrix directly.

## Camera Controller

`CameraController` (`src/camera/controller.rs`) wraps the camera and owns
input, GPU uniforms, and animation. It is `pub(crate)`: consumers drive
the camera through `VisoEngine` methods, not through the controller.

The tunables come from `CameraOptions`:

- `rotate_speed` (default 0.5)
- `pan_speed` (default 0.5)
- `zoom_speed` (default 0.1)
- `fovy` (default 45.0 degrees)
- `znear` (default 5.0)
- `zfar` (default 2000.0)

## Pointer-Driven Manipulation

Rotate, pan, and zoom are applied by the pointer intake, not by a command
type. While the primary button is held over a non-pickable area,
`feed_pointer_motion` rotates the camera (or pans when shift is held);
`feed_scroll` zooms.

- **Rotate**: horizontal pointer movement rotates around the up vector,
  vertical movement around the right vector, scaled by `rotate_speed`.
- **Pan**: translates the focus point along the camera's right and up
  vectors, cancelling any in-progress focus animation, scaled by
  `pan_speed`.
- **Zoom**: adjusts the orbital distance (clamped to a sensible range),
  scaled by `zoom_speed`.

See [Handling Input](../integration/handling-input.md) for the intake
wiring. The underlying `rotate` / `pan` / `zoom` methods on the
controller are crate-internal.

## Camera Animation

The camera animates between states for smooth viewpoint changes when
loading structures or switching focus.

### Fitting to a Bounding Sphere

The engine computes a bounding sphere over the relevant entities and
calls one of the controller's fit methods:

- `fit_to_sphere(centroid, radius)`: instant fit (initial load).
- `fit_to_sphere_animated(centroid, radius)`: animated fit (focus cycle,
  scene replacement).

The fit accounts for both horizontal and vertical FOV so the structure
fits the viewport. Public entry points on the engine:

```rust
engine.fit_camera_to_focus();   // animated fit to the current focus
engine.recenter_camera();       // alias used by the default Q binding
engine.snap_camera_to_focus();  // instant fit (no animation)
```

`focus_centroid()` returns the atom-count-weighted centroid of the
visible scene, and `set_camera_pose(center, eye, up)` positions the
camera explicitly from a saved viewpoint.

### Per-Frame Update

`engine.update(dt)` ticks the controller's `update_animation`,
interpolating focus, distance, and bounding radius toward their targets.

## Auto-Rotation

```rust
engine.toggle_auto_rotate();
engine.set_auto_rotate(true);
```

When enabled, the camera spins around the up vector at a fixed turntable
speed (~29 degrees/sec). The spin axis is captured from the current up
vector at the moment auto-rotation is enabled.

## Frustum Culling

The camera produces a frustum used to cull sidechains: sidechains outside
the view frustum (with a small angstrom margin) are skipped during
rendering. The engine reuploads the frustum-filtered sidechain instance
buffer when the camera moves enough to invalidate the previous cull.

## Coordinate Conversion

Two conversions are exposed on the engine for input and constraint math:

- `world_to_screen(world) -> Option<Vec2>`: project a world point to
  pixels (origin top-left); `None` if the point is at or behind the
  camera.
- `screen_to_world_at_depth(screen_pos, world_point) -> Vec3`: unproject
  a screen pixel onto a plane parallel to the camera at the depth of
  `world_point`. Used for drag operations so the drag stays at the
  atom's depth.

## Fog Derivation

Fog parameters are derived from the camera each frame in `pre_render`:

- **Fog start**: the current orbital distance.
- **Fog density**: `2.0 / max(bounding_radius, 10.0)`.

The composite post-pass applies depth-based fog, fading distant geometry
to the background color.

## GPU Uniform

The camera uniform is uploaded each frame during the render pass. It
carries the projection and view matrices, inverse projection, camera
position, hovered residue id, screen dimensions, and an elapsed-time
field. All renderers bind it for vertex transformation and view-dependent
effects.
