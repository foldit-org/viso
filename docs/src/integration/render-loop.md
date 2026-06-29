# The Render Loop

Every frame follows a specific sequence, and the order matters. Polling
the assembly snapshot before applying pending mesh data ensures newly
generated meshes appear on the same frame their owning rebuild finishes,
and updating the camera before rendering keeps animation smooth.

## Minimal Render Loop (Standalone)

From viso's standalone viewer:

```rust
WindowEvent::RedrawRequested => {
    let now = Instant::now();
    let dt = now.duration_since(last_frame_time).as_secs_f32();
    last_frame_time = now;

    engine.update(dt);

    match engine.render() {
        Ok(()) => {}
        Err(wgpu::SurfaceError::Outdated | wgpu::SurfaceError::Lost) => {
            engine.resize(width, height);
        }
        Err(e) => log::error!("render error: {e:?}"),
    }
    window.request_redraw();
}
```

## What `engine.update(dt)` Does

`engine.update(dt)` handles all per-frame coordination work:

1. **Camera animation tick**: interpolates animated focus, distance, and
   bounding radius; advances turntable auto-rotation.
2. **Drain the pending Assembly snapshot.** If a new `Assembly`
   snapshot is waiting (because the host or `VisoApp` called
   `engine.set_assembly`), the engine rederives its per-entity state
   and submits a `FullRebuild` request to the background mesh
   processor.
3. **Apply any pending scene.** If the background processor has a
   completed `PreparedRebuild` ready, it is uploaded to the GPU
   (vertex/index/instance buffers, picking bind groups). GPU upload is
   typically <1ms.

The main thread never blocks on the background thread. If meshes
aren't ready, the previous frame's data continues to render until they
are.

## What `engine.render()` Does

The render method executes the full pipeline:

1. **Apply pending animation frame.** If an interpolated animation
   frame is ready from the background thread, upload it to the GPU.
2. **Tick animation.** Advance trajectory and structural animation;
   submit a new animation-frame request to the background thread if
   anything changed.
3. **Update camera and lighting uniforms.** Write camera matrices,
   hovered-residue id, derived fog parameters, and headlamp lighting.
4. **Frustum cull sidechains.**
5. **Resolve constraints.** Translate stored band/pull specs into
   world-space using current interpolated atom positions.
6. **Geometry pass.** Render all molecular geometry to HDR render
   targets (`Rgba16Float` color + normals, `Depth32Float` depth).
7. **Picking pass.** Render to the offscreen `R32Uint` target and copy
   the pixel under the cursor to a staging buffer.
8. **Post-processing.** SSAO, bloom, composite (outlines, fog, tone
   mapping), FXAA.
9. **Present** to the swapchain surface.
10. **Initiate non-blocking picking readback** for next frame.

Frame timing is throttled to a 300 fps target by default
(`FrameTiming::should_render` short-circuits if the previous frame's
elapsed time hasn't met the minimum frame duration).

## Error Handling

Surface errors are expected during resize or focus changes:

- `SurfaceError::Outdated` / `SurfaceError::Lost`: the surface needs
  reconfiguration. Call `resize()` with the current window dimensions.
- Other errors are logged but non-fatal; the next frame retries.

## Rendering to a Texture (Embedding)

If you're embedding viso in a host that gives you a target texture
view (e.g. a dioxus or egui texture slot), use `render_to_texture`
instead of `render`:

```rust
engine.render_to_texture(&texture_view);
```

This runs the same pipeline but writes the final composite to the
provided texture view instead of acquiring a swapchain frame, so the
caller owns presentation.

## Non-Blocking Picking Readback

GPU picking uses a two-frame pipeline to avoid stalling:

1. **Frame N**: The picking pass renders to an offscreen texture and
   copies the pixel under the mouse to a staging buffer.
   `start_readback()` initiates an async buffer map.
2. **Frame N+1**: `poll_and_resolve` polls the device without blocking
   (it wraps `complete_readback`). If the map is complete, it reads the
   residue ID and resolves it to a `PickTarget` via the engine's
   `PickMap`. Otherwise it keeps the cached value from the previous
   successful read.

Hover is one frame behind the cursor, which avoids GPU pipeline stalls.
