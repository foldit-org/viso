//! Background regeneration of all isosurface meshes (density maps,
//! entity surfaces, cavities).
//!
//! Isosurface meshing is kicked off from several surface-option,
//! annotation, and density mutation paths. The atom positions / radii /
//! per-entity surface parameters are gathered here on the main thread
//! (the only thread that can read the scene, annotations, and density
//! store), then handed to the shared
//! [`SceneProcessor`](crate::renderer::pipeline::SceneProcessor) worker
//! as a [`SceneRequest::SurfaceRebuild`]. The worker runs the generators and
//! concatenates the meshes; the main thread polls the result each frame
//! and uploads it.
//!
//! This module owns a thin [`SurfaceRegen`] holder carrying a clone of
//! the processor's request sender plus the shared surface-generation
//! counter. The holder is `Send` + `Clone`, so the `&`-borrow write
//! views can submit a regen without a `&mut GpuPipeline`.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{mpsc, Arc};

use super::annotations::EntityAnnotations;
use super::density_store::DensityStore;
use super::scene::Scene;
use super::surface::{EntitySurface, SurfaceKind};
use super::ExternalVoidField;
use crate::options::{SurfaceKindOption, VisoOptions};
use crate::renderer::pipeline::prepared::{
    SceneRequest, SurfaceJob, SurfaceJobKind, SurfaceRebuildBody, VoidFieldJob,
};

/// Decide whether the surface should be regenerated now: it must be
/// showable, the displayed geometry must differ from what it was last
/// built against, and the publish stream must have stayed quiet for at
/// least `window`. Pure so the logic is exercisable without a GPU.
pub(crate) fn should_settle_surface(
    surface_active: bool,
    displayed_generation: u64,
    built_generation: u64,
    quiet_elapsed: std::time::Duration,
    window: std::time::Duration,
) -> bool {
    surface_active
        && displayed_generation != built_generation
        && quiet_elapsed >= window
}

/// Baked into a surface vertex's alpha to signal "no per-entity opacity
/// override; scale by the global surface-opacity uniform at draw time."
/// A real opacity is in `[0,1]`, so a negative value is an unambiguous
/// sentinel.
const SURFACE_OPACITY_FROM_UNIFORM: f32 = -1.0;

/// Holder for the surface-regen submit channel.
///
/// Carries a clone of the
/// [`SceneProcessor`](crate::renderer::pipeline::SceneProcessor) request sender
/// plus the shared latest-surface-generation counter. [`regenerate_surfaces`]
/// mints a generation with `fetch_add` and
/// submits a [`SceneRequest::SurfaceRebuild`]; the matching result is
/// polled on the main thread via `SceneProcessor::try_recv_surface`. The
/// holder is constructed in [`crate::engine::VisoEngine::new`].
pub(crate) struct SurfaceRegen {
    /// Sender used by [`regenerate_surfaces`] to submit regen requests.
    tx: mpsc::Sender<SceneRequest>,
    /// Shared latest-surface-generation counter. Minted here, read by the
    /// processor's `try_recv_surface` to discard superseded results.
    surface_generation: Arc<AtomicU64>,
}

impl SurfaceRegen {
    /// Wrap the processor's request sender + shared generation counter.
    pub(crate) fn new(
        tx: mpsc::Sender<SceneRequest>,
        surface_generation: Arc<AtomicU64>,
    ) -> Self {
        Self {
            tx,
            surface_generation,
        }
    }
}

/// Regenerate all isosurface meshes (density + entity surfaces +
/// cavities) on the shared scene-processor worker.
///
/// Collects atom positions + radii from each entity that has a surface
/// or cavity rendering enabled, flattens each entity's surface parameters
/// into a worker-local [`SurfaceJob`], mints the next surface generation,
/// and submits a [`SceneRequest::SurfaceRebuild`]. The worker runs the
/// generators and concatenates the meshes (shared with density map
/// rendering).
pub(crate) fn regenerate_surfaces(
    scene: &Scene,
    annotations: &EntityAnnotations,
    density: &DensityStore,
    external_void_field: Option<&ExternalVoidField>,
    options: &VisoOptions,
    regen: &SurfaceRegen,
) {
    let all_entities = scene.current.entities();
    let palette = options.display.backbone_palette();

    // Collect jobs: (positions, radii, surface params with color)
    let mut jobs: Vec<(Vec<glam::Vec3>, Vec<f32>, EntitySurface)> = Vec::new();
    // Cavity jobs: (positions, radii). Color is the fixed CAVITY_RGBA
    // constant — cavities don't pick up per-entity coloring.
    let mut cavity_jobs: Vec<(Vec<glam::Vec3>, Vec<f32>)> = Vec::new();

    for (entity_idx, se) in all_entities.iter().enumerate() {
        let eid = se.id();
        if !annotations.is_visible(eid) {
            continue;
        }

        // Resolve this entity's surface settings through its appearance
        // overlay (the same overlay the mesh path applies), falling back
        // to the global display options when the entity has no override.
        let resolved = annotations.appearance.get(&eid).map_or_else(
            || options.display.clone(),
            |ovr| ovr.to_display_options(&options.display),
        );
        let kind = resolved.surface_kind();
        // Surface opacity is baked into vertex alpha as one of two things:
        // a per-entity override bakes its ABSOLUTE value (rendered exactly,
        // independent of the global slider); no override bakes the
        // [`SURFACE_OPACITY_FROM_UNIFORM`] sentinel, signalling the shader
        // to scale by the global surface-opacity uniform at draw time.
        let opacity = annotations
            .appearance
            .get(&eid)
            .and_then(|o| o.surface_opacity)
            .unwrap_or(SURFACE_OPACITY_FROM_UNIFORM);
        let show_cavities = resolved.show_cavities();

        // An explicit per-entity surface takes priority; otherwise fall
        // back to the per-entity resolved appearance (the global value
        // when the entity has no override).
        let base_surface = annotations.surfaces.get(&eid).map_or_else(
            || match kind {
                SurfaceKindOption::Gaussian => Some(EntitySurface {
                    kind: SurfaceKind::Gaussian,
                    color: [0.7, 0.7, 0.7, opacity],
                    ..Default::default()
                }),
                SurfaceKindOption::Ses => Some(EntitySurface {
                    kind: SurfaceKind::Ses,
                    color: [0.7, 0.7, 0.7, opacity],
                    ..Default::default()
                }),
                SurfaceKindOption::None => None,
            },
            |s| if s.visible { Some(s.clone()) } else { None },
        );

        // Skip atoms gathering only when neither the surface nor cavity
        // path wants this entity — cavities want it whenever this
        // entity's resolved toggle is on.
        if base_surface.is_none() && !show_cavities {
            continue;
        }

        let positions = se.positions().to_vec();
        if positions.is_empty() {
            continue;
        }
        let radii: Vec<f32> = se
            .elements()
            .iter()
            .map(molex::Element::vdw_radius)
            .collect();

        // Use the backbone palette so surface/cavity colors match the
        // backbone.
        let [r, g, b] = palette.categorical_color(entity_idx);

        if let Some(mut surface) = base_surface {
            surface.color = [r, g, b, surface.color[3]];
            // SES needs a finer grid than Gaussian to resolve atom-level
            // detail (ChimeraX default is 0.5 Å).
            if surface.kind == SurfaceKind::Ses {
                surface.resolution = 0.5;
            }
            jobs.push((positions.clone(), radii.clone(), surface));
        }

        if show_cavities {
            cavity_jobs.push((positions, radii));
        }
    }

    // Also include any visible density maps
    let density_jobs: Vec<_> = density
        .visible_entries()
        .map(|(_id, entry)| {
            let [r, g, b] = entry.color;
            (entry.map.clone(), entry.threshold, [r, g, b, entry.opacity])
        })
        .collect();

    // Host-supplied void distance field, meshed as a smooth blob in the
    // same cavity stream. An empty field clears to no job.
    let void_field_job = external_void_field.and_then(|f| {
        let [nx, ny, nz] = f.dims;
        if f.phi.is_empty() || nx == 0 || ny == 0 || nz == 0 {
            return None;
        }
        Some(VoidFieldJob {
            dims: f.dims,
            origin: f.origin,
            spacing: f.spacing,
            phi: f.phi.clone(),
            threshold: f.threshold,
        })
    });

    // Flatten each gathered EntitySurface into a worker-local SurfaceJob
    // so the worker never sees the engine-side surface type.
    let surface_jobs: Vec<SurfaceJob> = jobs
        .into_iter()
        .map(|(positions, radii, surface)| SurfaceJob {
            positions,
            radii,
            kind: match surface.kind {
                SurfaceKind::Gaussian => SurfaceJobKind::Gaussian,
                SurfaceKind::Ses => SurfaceJobKind::Ses,
            },
            resolution: surface.resolution,
            probe_radius: surface.probe_radius,
            level: surface.level,
            color: surface.color,
        })
        .collect();

    let surface_generation =
        regen.surface_generation.fetch_add(1, Ordering::Relaxed) + 1;
    let body = SurfaceRebuildBody {
        density_jobs,
        surface_jobs,
        cavity_jobs,
        void_field_job,
        surface_generation,
    };
    let _ = regen.tx.send(SceneRequest::SurfaceRebuild(Box::new(body)));
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use super::should_settle_surface;

    const WINDOW: Duration = Duration::from_millis(180);
    const QUIET: Duration = Duration::from_millis(200);
    const MOVING: Duration = Duration::from_millis(50);

    #[test]
    fn settles_when_active_stale_and_quiet() {
        assert!(should_settle_surface(true, 5, 4, QUIET, WINDOW));
    }

    #[test]
    fn waits_while_still_moving() {
        assert!(!should_settle_surface(true, 5, 4, MOVING, WINDOW));
    }

    #[test]
    fn skips_when_surface_already_current() {
        assert!(!should_settle_surface(true, 5, 5, QUIET, WINDOW));
    }

    #[test]
    fn skips_when_no_surface_shown() {
        assert!(!should_settle_surface(false, 5, 4, QUIET, WINDOW));
    }
}
