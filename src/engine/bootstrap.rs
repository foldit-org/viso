//! GPU pipeline initialization and engine assembly.
//!
//! The engine never creates or mutates an [`Assembly`](molex::Assembly):
//! the host application owns the assembly and pushes the latest snapshot
//! via [`crate::VisoEngine::set_assembly`]. In standalone deployments
//! [`crate::VisoApp`] plays the host role.

use std::time::Duration;

use glam::Vec3;
use web_time::Instant;

use super::annotations::EntityAnnotations;
use super::density_store::DensityStore;
use super::scene::Scene;
use super::surface_regen::SurfaceRegen;
use super::{ConstraintSpecs, VisoEngine};
use crate::animation::AnimationState;
use crate::camera::controller::CameraController;
use crate::error::VisoError;
use crate::gpu::lighting::Lighting;
use crate::gpu::residue_color::ResidueColorBuffer;
use crate::gpu::{RenderContext, ShaderComposer};
use crate::options::VisoOptions;
use crate::renderer::picking::{PickingSystem, SelectionBuffer};
use crate::renderer::pipeline::SceneProcessor;
use crate::renderer::postprocess::PostProcessStack;
use crate::renderer::{GpuPipeline, PipelineLayouts, Renderers};

/// Target FPS limit.
const TARGET_FPS: u32 = 300;

// ---------------------------------------------------------------------------
// FrameTiming
// ---------------------------------------------------------------------------

/// Frame timing with FPS calculation and optional frame limiting.
pub(crate) struct FrameTiming {
    target_fps: u32,
    min_frame_duration: Duration,
    last_frame: Instant,
    smoothed_fps: f32,
    smoothing: f32,
    start: Instant,
}

impl FrameTiming {
    pub(super) fn new(target_fps: u32) -> Self {
        let min_frame_duration = if target_fps > 0 {
            Duration::from_secs_f64(1.0 / f64::from(target_fps))
        } else {
            Duration::ZERO
        };
        Self {
            target_fps,
            min_frame_duration,
            last_frame: Instant::now(),
            smoothed_fps: 60.0,
            smoothing: 0.05,
            start: Instant::now(),
        }
    }

    pub(crate) fn should_render(&self) -> bool {
        if self.target_fps == 0 {
            return true;
        }
        self.last_frame.elapsed() >= self.min_frame_duration
    }

    pub(crate) fn end_frame(&mut self) {
        let now = Instant::now();
        let elapsed = now.duration_since(self.last_frame);
        self.last_frame = now;

        let frame_time = elapsed.as_secs_f32();
        if frame_time > 0.0 {
            let instant_fps = 1.0 / frame_time;
            self.smoothed_fps = self
                .smoothed_fps
                .mul_add(1.0 - self.smoothing, instant_fps * self.smoothing);
        }
    }

    pub(crate) fn fps(&self) -> f32 {
        self.smoothed_fps
    }

    pub(crate) fn elapsed_secs(&self) -> f32 {
        self.start.elapsed().as_secs_f32()
    }
}

// ---------------------------------------------------------------------------
// GPU pipeline initialization
// ---------------------------------------------------------------------------

struct GpuBootstrap {
    shader_composer: ShaderComposer,
    camera_controller: CameraController,
    lighting: Lighting,
    renderers: Renderers,
    pick: PickingSystem,
    post_process: PostProcessStack,
}

fn init_gpu_pipeline(
    context: &RenderContext,
) -> Result<GpuBootstrap, VisoError> {
    let mut shader_composer = ShaderComposer::new()?;
    let mut camera_controller = CameraController::new(context);
    let lighting = Lighting::new(context);

    // Residue color buffer size — 1 here; grown on demand during sync.
    let n = 1;
    let selection = SelectionBuffer::new(&context.device, n);
    let residue_colors = ResidueColorBuffer::new(&context.device, n);
    let layouts = PipelineLayouts {
        camera: camera_controller.layout.clone(),
        lighting: lighting.layout.clone(),
        selection: selection.layout.clone(),
        color: residue_colors.layout.clone(),
    };
    let post_process = PostProcessStack::new(context, &mut shader_composer)?;
    let renderers = Renderers::new(
        context,
        &layouts,
        &mut shader_composer,
        &post_process.backface_depth_view,
    )?;
    let pick = PickingSystem::new(
        context,
        &camera_controller.layout,
        selection,
        residue_colors,
        &mut shader_composer,
    )?;
    camera_controller.fit_to_sphere(Vec3::ZERO, 0.0);

    Ok(GpuBootstrap {
        shader_composer,
        camera_controller,
        lighting,
        renderers,
        pick,
        post_process,
    })
}

// ---------------------------------------------------------------------------
// VisoEngine construction
// ---------------------------------------------------------------------------

impl VisoEngine {
    /// Build an engine over a prepared [`RenderContext`].
    ///
    /// The engine starts with an empty internal [`Assembly`](molex::Assembly).
    /// The host pushes its first snapshot — and every subsequent
    /// snapshot — via [`Self::set_assembly`]; viso never mutates the
    /// assembly itself.
    ///
    /// # Errors
    ///
    /// Returns [`VisoError`] if GPU pipeline initialization fails.
    pub fn new(
        context: RenderContext,
        options: VisoOptions,
    ) -> Result<Self, VisoError> {
        let bootstrap = init_gpu_pipeline(&context)?;
        let (density_tx, density_rx) = std::sync::mpsc::channel();
        Ok(Self {
            gpu: GpuPipeline {
                context,
                renderers: bootstrap.renderers,
                pick: bootstrap.pick,
                scene_processor: SceneProcessor::new()
                    .map_err(VisoError::ThreadSpawn)?,
                post_process: bootstrap.post_process,
                lighting: bootstrap.lighting,
                cursor_pos: (0.0, 0.0),
                last_cull_camera_eye: Vec3::ZERO,
                shader_composer: bootstrap.shader_composer,
                density_rx,
            },
            camera_controller: bootstrap.camera_controller,
            constraints: ConstraintSpecs {
                band_specs: Vec::new(),
                pull_spec: None,
            },
            animation: AnimationState::new(),
            options,
            active_preset: None,
            frame_timing: FrameTiming::new(TARGET_FPS),
            density: DensityStore::new(),
            scene: Scene::new(),
            annotations: EntityAnnotations::default(),
            surface_regen: SurfaceRegen::new(density_tx),
        })
    }
}
