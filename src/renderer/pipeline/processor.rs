//! Background scene processor for non-blocking geometry generation.
//!
//! Moves all CPU-heavy mesh/instance generation off the main thread.
//! The main thread only does GPU uploads (<1ms) and render passes.
//!
//! Supports **per-entity mesh caching**: when an entity's `mesh_version`
//! hasn't changed between frames, its cached mesh is reused instead of
//! being regenerated. Global settings changes (view mode, display,
//! colors) clear the entire cache.

use std::sync::{mpsc, Arc};

use molex::entity::molecule::id::EntityId;
use molex::SSType;
use rustc_hash::{FxHashMap, FxHashSet};

use super::mesh_gen::{AnimationFrameCache, EntityMetaSnapshot};
use super::prepared::{
    AnimationFrameBody, CachedEntityMesh, FullRebuildBody, FullRebuildEntity,
    PreparedAnimationFrame, PreparedRebuild, SceneRequest,
};
use crate::options::{
    ColorOptions, DisplayOptions, DrawingMode, GeometryOptions, NaColorMode,
};

// ---------------------------------------------------------------------------
// Platform-abstracted background thread spawn
// ---------------------------------------------------------------------------

/// Handle to a background worker. On native this is a joinable OS thread;
/// on WASM it is a no-op because the worker runs on a rayon pool thread
/// (backed by web workers via `wasm-bindgen-rayon` + `SharedArrayBuffer`)
/// and exits when the channel disconnects.
#[cfg(not(target_arch = "wasm32"))]
type WorkerHandle = Option<std::thread::JoinHandle<()>>;
#[cfg(target_arch = "wasm32")]
type WorkerHandle = ();

/// Spawn a long-lived closure on a background thread.
///
/// - **Native:** dedicated OS thread via `std::thread::Builder`.
/// - **WASM:** `rayon::spawn` onto the `wasm-bindgen-rayon` pool.
fn spawn_background(
    f: impl FnOnce() + Send + 'static,
) -> Result<WorkerHandle, std::io::Error> {
    #[cfg(not(target_arch = "wasm32"))]
    {
        std::thread::Builder::new()
            .name("scene-processor".into())
            .spawn(f)
            .map(Some)
    }
    #[cfg(target_arch = "wasm32")]
    {
        rayon::spawn(f);
        Ok(())
    }
}

/// Join a background worker, blocking until it finishes.
fn join_background(handle: &mut WorkerHandle) {
    #[cfg(not(target_arch = "wasm32"))]
    if let Some(h) = handle.take() {
        let _ = h.join();
    }
    #[cfg(target_arch = "wasm32")]
    let _ = handle;
}

// ---------------------------------------------------------------------------
// SceneProcessor
// ---------------------------------------------------------------------------

/// Background thread that generates CPU-side geometry from scene data.
pub(crate) struct SceneProcessor {
    request_tx: mpsc::Sender<SceneRequest>,
    rebuild_result: triple_buffer::Output<Option<PreparedRebuild>>,
    anim_result: triple_buffer::Output<Option<PreparedAnimationFrame>>,
    worker: WorkerHandle,
    /// Monotonically increasing rebuild generation counter. Bumped
    /// each time a `FullRebuild` is submitted. Animation frame results
    /// with a lower generation are discarded as stale.
    rebuild_generation: u64,
    /// True between `FullRebuild` submission and `PreparedRebuild`
    /// consumption. While set, the backbone renderer's cached chains are
    /// stale — LOD must not read them.
    rebuild_pending: bool,
}

impl SceneProcessor {
    /// Spawn the background scene processing thread.
    ///
    /// # Errors
    ///
    /// Returns [`std::io::Error`] if the background thread fails to spawn.
    pub(crate) fn new() -> Result<Self, std::io::Error> {
        let (request_tx, request_rx) = mpsc::channel::<SceneRequest>();
        let (rebuild_input, rebuild_output) =
            triple_buffer::triple_buffer(&None);
        let (anim_input, anim_output) = triple_buffer::triple_buffer(&None);

        let worker = spawn_background(move || {
            Self::thread_loop(request_rx, rebuild_input, anim_input);
        })?;

        Ok(Self {
            request_tx,
            rebuild_result: rebuild_output,
            anim_result: anim_output,
            worker,
            rebuild_generation: 0,
            rebuild_pending: false,
        })
    }

    /// Increment and return the next rebuild generation counter.
    ///
    /// Also sets `rebuild_pending`, which prevents LOD from reading the
    /// backbone renderer's stale cached chains until the corresponding
    /// `PreparedRebuild` is consumed.
    pub(crate) fn next_generation(&mut self) -> u64 {
        self.rebuild_generation += 1;
        self.rebuild_pending = true;
        self.rebuild_generation
    }

    /// Current rebuild generation counter.
    pub(crate) fn generation(&self) -> u64 {
        self.rebuild_generation
    }

    /// Submit a scene request (non-blocking send).
    pub(crate) fn submit(&self, request: SceneRequest) {
        let _ = self.request_tx.send(request);
    }

    /// Non-blocking check for a completed full rebuild.
    ///
    /// Discards results whose generation is older than the current
    /// rebuild generation, preventing stale geometry from a previous
    /// structure from being uploaded after `replace_scene()`.
    ///
    /// Clears `rebuild_pending` on successful consumption so that LOD
    /// submission (gated by [`Self::is_rebuild_pending`]) resumes with
    /// the now-correct backbone renderer cache.
    pub(crate) fn try_recv_rebuild(&mut self) -> Option<PreparedRebuild> {
        let _ = self.rebuild_result.update();
        let prepared = self.rebuild_result.output_buffer_mut().take()?;
        if prepared.generation < self.rebuild_generation {
            log::debug!(
                "try_recv_rebuild: DISCARDING stale rebuild (gen {} < current \
                 {})",
                prepared.generation,
                self.rebuild_generation,
            );
            return None;
        }
        log::debug!(
            "try_recv_rebuild: ACCEPTED rebuild gen={} (current={})",
            prepared.generation,
            self.rebuild_generation,
        );
        self.rebuild_pending = false;
        Some(prepared)
    }

    /// Whether a `FullRebuild` has been submitted but its
    /// `PreparedRebuild` has not yet been consumed.
    ///
    /// While true, the backbone renderer's cached chains are stale —
    /// callers that read the cache to build `AnimationFrame` requests
    /// (notably LOD) must skip submission.
    pub(crate) fn is_rebuild_pending(&self) -> bool {
        self.rebuild_pending
    }

    /// Non-blocking check for completed animation frame.
    ///
    /// Discards frames whose generation is older than the current
    /// rebuild generation. As soon as a new `FullRebuild` is submitted
    /// (bumping `rebuild_generation`), all prior animation frames
    /// become stale — even before the rebuild result arrives on the
    /// main thread.
    pub(crate) fn try_recv_animation(
        &mut self,
    ) -> Option<PreparedAnimationFrame> {
        let _ = self.anim_result.update();
        let prepared = self.anim_result.output_buffer_mut().take()?;
        if prepared.generation < self.rebuild_generation {
            log::debug!(
                "Discarding stale animation frame (gen {} < current {})",
                prepared.generation,
                self.rebuild_generation,
            );
            return None;
        }
        Some(prepared)
    }

    /// Shut down the background thread and wait for it to finish.
    pub(crate) fn shutdown(&mut self) {
        let _ = self.request_tx.send(SceneRequest::Shutdown);
        join_background(&mut self.worker);
    }

    /// Background thread main loop with per-entity mesh caching.
    #[allow(clippy::needless_pass_by_value)]
    fn thread_loop(
        request_rx: mpsc::Receiver<SceneRequest>,
        mut rebuild_input: triple_buffer::Input<Option<PreparedRebuild>>,
        mut anim_input: triple_buffer::Input<Option<PreparedAnimationFrame>>,
    ) {
        let mut cache = MeshCache::new();
        // Generation of the last FullRebuild processed on this thread.
        let mut last_rebuild_generation: u64 = 0;

        while let Ok(request) = request_rx.recv() {
            let latest = drain_latest(request, &request_rx);

            match latest {
                SceneRequest::Shutdown => break,
                SceneRequest::FullRebuild(body) => {
                    let FullRebuildBody {
                        entities,
                        display,
                        colors,
                        geometry,
                        entity_options,
                        generation,
                    } = *body;
                    last_rebuild_generation = generation;
                    cache.cache_stable_data(
                        &entities,
                        &display,
                        &colors,
                        &entity_options,
                    );
                    let entity_meshes = cache.update(
                        &entities,
                        &display,
                        &colors,
                        &geometry,
                        &entity_options,
                    );
                    let mut prepared =
                        super::mesh_concat::concatenate_meshes(&entity_meshes);
                    prepared.generation = generation;
                    rebuild_input.write(Some(prepared));
                }
                SceneRequest::AnimationFrame(body) => {
                    let AnimationFrameBody {
                        positions,
                        geometry,
                        per_chain_lod,
                        include_sidechains,
                        generation,
                    } = *body;
                    if generation < last_rebuild_generation {
                        continue;
                    }
                    let prepared = super::mesh_gen::process_animation_frame(
                        &super::mesh_gen::AnimationFrameInput {
                            positions: &positions,
                            cache: &cache.anim_cache,
                            geometry: &geometry,
                            per_chain_lod: per_chain_lod.as_deref(),
                            include_sidechains,
                        },
                        generation,
                    );
                    anim_input.write(Some(prepared));
                }
            }
        }
    }
}

impl Drop for SceneProcessor {
    fn drop(&mut self) {
        self.shutdown();
    }
}

/// Per-entity mesh cache with settings-based invalidation.
///
/// Caches per-entity geometry keyed on [`EntityId`], plus an
/// [`AnimationFrameCache`] snapshot so `AnimationFrame` requests can be
/// regenerated using only derived state and interpolated positions.
struct MeshCache {
    meshes: FxHashMap<EntityId, (u64, CachedEntityMesh)>,
    last_display: Option<DisplayOptions>,
    last_colors: Option<ColorOptions>,
    last_geometry: Option<GeometryOptions>,
    anim_cache: AnimationFrameCache,
}

impl MeshCache {
    fn new() -> Self {
        Self {
            meshes: FxHashMap::default(),
            last_display: None,
            last_colors: None,
            last_geometry: None,
            anim_cache: AnimationFrameCache {
                topologies: FxHashMap::default(),
                entity_meta: FxHashMap::default(),
                cartoon_ss_types: None,
                cartoon_per_residue_colors: None,
                cartoon_na_base_colors: None,
                sidechain_palette: ([1.0, 1.0, 1.0], [0.5, 0.5, 0.5]),
                entity_order: Vec::new(),
            },
        }
    }

    /// Cache stable per-scene data so animation frames can regenerate
    /// backbone / sidechain meshes without re-sending topology.
    fn cache_stable_data(
        &mut self,
        entities: &[FullRebuildEntity],
        display: &DisplayOptions,
        colors: &ColorOptions,
        entity_options: &FxHashMap<u32, (DisplayOptions, GeometryOptions)>,
    ) {
        self.anim_cache.topologies.clear();
        self.anim_cache.entity_meta.clear();
        self.anim_cache.entity_order.clear();
        self.anim_cache.sidechain_palette =
            (colors.hydrophobic_sidechain, colors.hydrophilic_sidechain);
        for e in entities {
            let _ = self
                .anim_cache
                .topologies
                .insert(e.id, Arc::clone(&e.topology));
            let entity_display =
                entity_options.get(&e.id.raw()).map_or(display, |(d, _)| d);
            let _ = self.anim_cache.entity_meta.insert(
                e.id,
                EntityMetaSnapshot {
                    drawing_mode: e.drawing_mode,
                    per_residue_colors: e.per_residue_colors.clone(),
                    sidechain_color_mode: entity_display.sidechain_color_mode(),
                },
            );
            self.anim_cache.entity_order.push(e.id);
        }

        // SS types: only Cartoon-mode entities contribute, in entity
        // order. Uses ss_override when present, falls back to
        // topology.ss_types.
        let mut ss: Vec<SSType> = Vec::new();
        for e in entities {
            if e.drawing_mode != DrawingMode::Cartoon {
                continue;
            }
            if let Some(ovr) = e.ss_override.as_deref() {
                ss.extend_from_slice(ovr);
            } else {
                ss.extend_from_slice(&e.topology.ss_types);
            }
        }
        self.anim_cache.cartoon_ss_types =
            if ss.is_empty() { None } else { Some(ss) };

        // Per-residue colors: only Cartoon-mode entities contribute.
        let mut colors: Vec<[f32; 3]> = Vec::new();
        for e in entities {
            if e.drawing_mode != DrawingMode::Cartoon {
                continue;
            }
            if let Some(c) = e.per_residue_colors.as_deref() {
                colors.extend_from_slice(c);
            }
        }
        self.anim_cache.cartoon_per_residue_colors = if colors.is_empty() {
            None
        } else {
            Some(colors)
        };

        // NA base colors: only Cartoon-mode NA entities contribute when
        // BaseColor mode is active.
        let mut na_colors: Vec<[f32; 3]> = Vec::new();
        if display.na_color_mode() == NaColorMode::BaseColor {
            for e in entities {
                if e.drawing_mode != DrawingMode::Cartoon
                    || !e.topology.is_nucleic_acid()
                {
                    continue;
                }
                na_colors
                    .extend(e.topology.ring_topology.iter().map(|r| r.color));
            }
        }
        self.anim_cache.cartoon_na_base_colors = if na_colors.is_empty() {
            None
        } else {
            Some(na_colors)
        };
    }

    /// Update cached meshes and return entity-ordered references for
    /// concatenation.
    fn update(
        &mut self,
        entities: &[FullRebuildEntity],
        display: &DisplayOptions,
        colors: &ColorOptions,
        geometry: &GeometryOptions,
        entity_options: &FxHashMap<u32, (DisplayOptions, GeometryOptions)>,
    ) -> Vec<&CachedEntityMesh> {
        // Clamp geometry detail so the concatenated vertex buffer stays
        // under the wgpu 256 MB max.
        let total_residues: usize = entities
            .iter()
            .map(|e| {
                let protein = e
                    .topology
                    .protein_backbone_layout
                    .iter()
                    .map(|s| s.ca.len())
                    .sum::<usize>();
                let na = e
                    .topology
                    .na_backbone_chain_layout
                    .iter()
                    .map(Vec::len)
                    .sum::<usize>();
                protein + na
            })
            .sum();
        let geometry = geometry.clamped_for_residues(total_residues);

        // Any settings change (geometry, display, or colors) clears the
        // entire cache because backbone colors are baked into vertex data.
        let settings_changed = self.last_geometry.as_ref() != Some(&geometry)
            || self.last_display.as_ref() != Some(display)
            || self.last_colors.as_ref() != Some(colors);

        if settings_changed {
            self.meshes.clear();
        }
        self.last_display = Some(display.clone());
        self.last_colors = Some(colors.clone());
        self.last_geometry = Some(geometry.clone());

        // Generate or reuse per-entity meshes.
        for e in entities {
            let entity_u32 = *e.id;
            let needs_regen = self
                .meshes
                .get(&e.id)
                .is_none_or(|(v, _)| *v != e.mesh_version);
            if needs_regen {
                let (e_display, e_geometry) =
                    if let Some((d, g)) = entity_options.get(&entity_u32) {
                        (d, g.clamped_for_residues(total_residues))
                    } else {
                        (display, geometry.clone())
                    };
                let mesh = super::mesh_gen::generate_entity_mesh(
                    e,
                    e_display,
                    colors,
                    &e_geometry,
                );
                drop(self.meshes.insert(e.id, (e.mesh_version, mesh)));
            }
        }

        // Evict removed entities.
        let active_ids: FxHashSet<EntityId> =
            entities.iter().map(|e| e.id).collect();
        self.meshes.retain(|id, _| active_ids.contains(id));

        // Collect references in entity order.
        entities
            .iter()
            .filter_map(|e| self.meshes.get(&e.id).map(|(_, mesh)| mesh))
            .collect()
    }
}

/// Drain queued requests, keeping only the latest.
///
/// Special case: a queued `AnimationFrame` does NOT replace a pending
/// `FullRebuild` — the rebuild must still run so the mesh cache is
/// populated before animation frames can reference it.
fn drain_latest(
    initial: SceneRequest,
    rx: &mpsc::Receiver<SceneRequest>,
) -> SceneRequest {
    let mut latest = initial;
    while let Ok(newer) = rx.try_recv() {
        match (&latest, &newer) {
            (SceneRequest::FullRebuild(_), SceneRequest::AnimationFrame(_)) => {
            }
            _ => {
                latest = newer;
            }
        }
    }
    latest
}
