use glam::{Quat, UVec2, Vec2, Vec3};
use wgpu::util::DeviceExt;

use crate::camera::core::{Camera, CameraUniform};
use crate::camera::frustum::Frustum;
use crate::gpu::RenderContext;

/// Speed of camera animation (higher = faster, 1.0 = instant)
const CAMERA_ANIMATION_SPEED: f32 = 3.0;

/// Turntable auto-rotation speed in radians/sec (~29 deg/sec)
const TURNTABLE_SPEED: f32 = 0.5;

/// Orbital camera controller managing rotation, panning, zoom, and GPU
/// resources.
pub(crate) struct CameraController {
    orientation: Quat,
    distance: f32,
    focus_point: Vec3,
    bounding_radius: f32, // Protein bounding sphere radius for fog computation

    // Animation targets (None = no animation in progress)
    target_focus_point: Option<Vec3>,
    target_distance: Option<f32>,
    target_bounding_radius: Option<f32>,

    /// The underlying perspective camera.
    pub(crate) camera: Camera,
    /// GPU uniform data for the camera.
    pub(crate) uniform: CameraUniform,
    /// GPU buffer holding the camera uniform.
    pub(crate) buffer: wgpu::Buffer,
    /// Bind group layout for the camera uniform.
    pub(crate) layout: wgpu::BindGroupLayout,
    /// Bind group for the camera uniform.
    pub(crate) bind_group: wgpu::BindGroup,

    /// When Some, the camera spins around the captured axis (camera up at
    /// toggle time).
    auto_rotate_axis: Option<Vec3>,
    /// Rotation sensitivity multiplier.
    rotate_speed: f32,
    /// Pan sensitivity multiplier.
    pan_speed: f32,
    /// Zoom sensitivity multiplier.
    zoom_speed: f32,
}

impl CameraController {
    /// Create the GPU buffer, bind group layout, and bind group for the camera
    /// uniform.
    fn create_gpu_resources(
        context: &RenderContext,
        uniform: &CameraUniform,
    ) -> (wgpu::Buffer, wgpu::BindGroupLayout, wgpu::BindGroup) {
        let mut buf = encase::UniformBuffer::new(Vec::new());
        let _ = buf.write(uniform);
        let buffer = context.device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("Camera Buffer"),
                contents: &buf.into_inner(),
                usage: wgpu::BufferUsages::UNIFORM
                    | wgpu::BufferUsages::COPY_DST,
            },
        );

        let layout = context.device.create_bind_group_layout(
            &wgpu::BindGroupLayoutDescriptor {
                label: Some("Camera Bind Group Layout"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX
                        | wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
            },
        );

        let bind_group =
            context
                .device
                .create_bind_group(&wgpu::BindGroupDescriptor {
                    layout: &layout,
                    entries: &[wgpu::BindGroupEntry {
                        binding: 0,
                        resource: buffer.as_entire_binding(),
                    }],
                    label: Some("Camera Bind Group"),
                });

        (buffer, layout, bind_group)
    }

    /// Create a new camera controller with default orbital parameters and GPU
    /// resources.
    pub(crate) fn new(context: &RenderContext) -> Self {
        let focus_point = Vec3::new(50.0, 50.0, 50.0);
        let distance = 150.0;
        let orientation = Quat::IDENTITY;

        let camera = Camera {
            eye: focus_point + Vec3::new(0.0, 0.0, distance),
            target: focus_point,
            up: Vec3::Y,
            aspect: context.config.width as f32 / context.config.height as f32,
            fovy: 45.0,
            znear: 5.0,
            zfar: 2000.0,
        };

        let mut uniform = CameraUniform::new();
        uniform.update_view_proj(&camera);

        let (buffer, layout, bind_group) =
            Self::create_gpu_resources(context, &uniform);

        Self {
            orientation,
            distance,
            focus_point,
            bounding_radius: 50.0, /* Default, will be updated by
                                    * fit_to_positions */
            target_focus_point: None,
            target_distance: None,
            target_bounding_radius: None,
            camera,
            uniform,
            buffer,
            layout,
            bind_group,
            auto_rotate_axis: None,
            rotate_speed: 0.01,
            pan_speed: 0.1,
            zoom_speed: 0.05,
        }
    }

    /// Update camera animation. Call this every frame.
    /// Returns true if animation is still in progress.
    pub(crate) fn update_animation(&mut self, dt: f32) -> bool {
        let mut animating = false;
        let t = (CAMERA_ANIMATION_SPEED * dt).min(1.0);

        // Animate focus point
        if let Some(target) = self.target_focus_point {
            let diff = target - self.focus_point;
            if diff.length() < 0.01 {
                self.focus_point = target;
                self.target_focus_point = None;
            } else {
                self.focus_point = self.focus_point.lerp(target, t);
                animating = true;
            }
        }

        // Animate distance
        if let Some(target) = self.target_distance {
            let diff = (target - self.distance).abs();
            if diff < 0.01 {
                self.distance = target;
                self.target_distance = None;
            } else {
                self.distance = self.distance + (target - self.distance) * t;
                animating = true;
            }
        }

        // Animate bounding radius (for fog)
        if let Some(target) = self.target_bounding_radius {
            let diff = (target - self.bounding_radius).abs();
            if diff < 0.01 {
                self.bounding_radius = target;
                self.target_bounding_radius = None;
            } else {
                self.bounding_radius =
                    self.bounding_radius + (target - self.bounding_radius) * t;
                animating = true;
            }
        }

        if animating {
            self.update_camera_pos();
        }

        // Turntable auto-rotation (independent of animation lerps)
        if let Some(axis) = self.auto_rotate_axis {
            let rot = Quat::from_axis_angle(axis, dt * TURNTABLE_SPEED);
            self.orientation = rot * self.orientation;
            self.update_camera_pos();
            return true;
        }

        animating
    }

    /// Toggle turntable auto-rotation. Captures the camera's current up vector
    /// as the spin axis when enabling.
    pub(crate) fn toggle_auto_rotate(&mut self) -> bool {
        if self.auto_rotate_axis.is_some() {
            self.auto_rotate_axis = None;
            false
        } else {
            self.auto_rotate_axis =
                Some((self.orientation * Vec3::Y).normalize());
            true
        }
    }

    /// Whether auto-rotation is currently active.
    pub(crate) fn is_auto_rotating(&self) -> bool {
        self.auto_rotate_axis.is_some()
    }

    /// Get the orbital distance from focus point.
    #[inline]
    pub(crate) fn distance(&self) -> f32 {
        self.distance
    }

    /// Get the bounding radius of the current protein.
    #[inline]
    pub(crate) fn bounding_radius(&self) -> f32 {
        self.bounding_radius
    }

    /// Get the camera's right vector from the orientation quaternion.
    #[inline]
    pub(crate) fn right(&self) -> Vec3 {
        self.orientation * Vec3::X
    }

    /// Get the camera's up vector from the orientation quaternion.
    #[inline]
    pub(crate) fn up(&self) -> Vec3 {
        self.orientation * Vec3::Y
    }

    /// Get the camera's forward vector from the orientation quaternion.
    #[inline]
    pub(crate) fn forward(&self) -> Vec3 {
        -(self.orientation * Vec3::Z)
    }

    fn update_camera_pos(&mut self) {
        // Renormalize quaternion to prevent accumulated floating-point drift
        // after many rotation operations
        self.orientation = self.orientation.normalize();

        let dir = self.orientation * Vec3::Z;

        self.camera.eye = self.focus_point + (dir * self.distance);
        self.camera.target = self.focus_point;
        self.camera.up = self.orientation * Vec3::Y;
    }

    /// Write the current camera uniform to the GPU buffer.
    pub(crate) fn update_gpu(&mut self, queue: &wgpu::Queue) {
        self.uniform.update_view_proj(&self.camera);
        let mut buf = encase::UniformBuffer::new(Vec::new());
        let _ = buf.write(&self.uniform);
        queue.write_buffer(&self.buffer, 0, &buf.into_inner());
    }

    /// Apply camera options from the configuration.
    pub(crate) fn apply_camera_options(
        &mut self,
        opts: &crate::options::CameraOptions,
    ) {
        self.camera.fovy = opts.fovy;
        self.camera.znear = opts.znear;
        self.camera.zfar = opts.zfar;
        self.rotate_speed = opts.rotate_speed * 0.02;
        self.pan_speed = opts.pan_speed * 0.2;
        self.zoom_speed = opts.zoom_speed * 0.5;
    }

    /// Apply debug options (normals visualization mode).
    pub(crate) fn apply_debug_options(
        &mut self,
        debug: &crate::options::DebugOptions,
        queue: &wgpu::Queue,
    ) {
        self.uniform.debug_mode = u32::from(debug.show_normals);
        self.update_gpu(queue);
    }

    /// Update the camera aspect ratio after a window resize.
    pub(crate) fn resize(&mut self, width: u32, height: u32) {
        self.camera.aspect = width as f32 / height as f32;
    }

    /// Apply an orbital rotation from a screen-space mouse delta.
    pub(crate) fn rotate(&mut self, delta: Vec2) {
        // Horizontal rotation around camera's up vector
        let up = self.orientation * Vec3::Y;
        let horizontal_rotation =
            Quat::from_axis_angle(up, -delta.x * self.rotate_speed);

        // Apply horizontal rotation
        self.orientation = horizontal_rotation * self.orientation;

        // Vertical rotation around camera's right vector (after horizontal
        // rotation)
        let right = self.orientation * Vec3::X;
        let vertical_rotation =
            Quat::from_axis_angle(right, -delta.y * self.rotate_speed);

        // Apply vertical rotation
        self.orientation = vertical_rotation * self.orientation;

        self.update_camera_pos();
    }

    /// Pan the camera focus point along the view plane.
    pub(crate) fn pan(&mut self, delta: Vec2) {
        // Cancel any animated focus point — user input takes priority
        self.target_focus_point = None;

        let right = self.orientation * Vec3::X;
        let up = self.orientation * Vec3::Y;

        let translation = right * (-delta.x * self.pan_speed)
            + up * (delta.y * self.pan_speed);

        self.focus_point += translation;
        self.update_camera_pos();
    }

    /// Zoom the camera by adjusting the orbital distance.
    pub(crate) fn zoom(&mut self, delta: f32) {
        // Cancel any animated zoom — user input takes priority
        self.target_distance = None;
        self.distance *= 1.0 - delta * self.zoom_speed;
        self.distance = self.distance.clamp(1.0, 1000.0);
        self.update_camera_pos();
    }

    /// Calculate the camera distance needed to frame a bounding sphere.
    fn fit_distance_for_radius(&self, radius: f32) -> f32 {
        let effective_radius = radius.max(5.0);
        let fovy_rad = self.camera.fovy.to_radians();
        let fovx_rad = fovy_rad * self.camera.aspect;
        let fit_y = effective_radius / (fovy_rad / 2.0).tan();
        let fit_x = effective_radius / (fovx_rad / 2.0).tan();
        fit_y.max(fit_x) * 1.05
    }

    /// Adjust camera to frame a pre-computed bounding sphere instantly.
    pub(crate) fn fit_to_sphere(&mut self, centroid: Vec3, radius: f32) {
        self.focus_point = centroid;
        self.bounding_radius = radius;
        self.distance = self.fit_distance_for_radius(radius);

        self.target_focus_point = None;
        self.target_distance = None;
        self.target_bounding_radius = None;

        self.update_camera_pos();
    }

    /// Adjust camera to frame a pre-computed bounding sphere with animation.
    pub(crate) fn fit_to_sphere_animated(
        &mut self,
        centroid: Vec3,
        radius: f32,
    ) {
        self.target_focus_point = Some(centroid);
        self.target_bounding_radius = Some(radius);
        self.target_distance = Some(self.fit_distance_for_radius(radius));
    }

    /// Set the camera pose explicitly from world-space center / eye / up.
    /// Leaves `bounding_radius` untouched so fog stays tied to molecule extent.
    pub(crate) fn set_pose(&mut self, center: Vec3, eye: Vec3, up: Vec3) {
        let offset = eye - center;
        let dist = offset.length();
        // Degenerate eye==center: keep current orientation/distance.
        if dist < 1e-5 {
            return;
        }
        let back = offset / dist;
        let mut up = up.normalize_or_zero();
        if up.length_squared() < 1e-10
            || up.cross(back).length_squared() < 1e-10
        {
            // Fallback when up is zero or parallel to back.
            up = if back.cross(Vec3::Y).length_squared() > 1e-6 {
                Vec3::Y
            } else {
                Vec3::Z
            };
        }
        // Gram-Schmidt: build right and re-derive up so basis is orthonormal.
        let right = up.cross(back).normalize();
        let up_corrected = back.cross(right).normalize();

        self.focus_point = center;
        self.distance = dist;
        self.orientation =
            Quat::from_mat3(&glam::Mat3::from_cols(right, up_corrected, back));

        self.target_focus_point = None;
        self.target_distance = None;
        self.target_bounding_radius = None;

        self.update_camera_pos();
    }

    /// Unproject screen coordinates to a world-space point on a plane at the
    /// given depth.
    ///
    /// # Arguments
    /// * `screen_x`, `screen_y` - Screen coordinates in pixels (origin
    ///   top-left)
    /// * `screen_width`, `screen_height` - Screen dimensions
    /// * `world_point` - A point in world space; the result will be on a plane
    ///   parallel to the camera at this point's depth
    pub(crate) fn screen_to_world_at_depth(
        &self,
        screen_pos: Vec2,
        viewport_size: UVec2,
        world_point: Vec3,
    ) -> Vec3 {
        // Convert screen coords to NDC [-1, 1]
        let ndc_x = (2.0 * screen_pos.x / viewport_size.x as f32) - 1.0;
        let ndc_y = 1.0 - (2.0 * screen_pos.y / viewport_size.y as f32); // flip Y

        // Calculate the depth of the world_point from camera
        let to_point = world_point - self.camera.eye;
        let depth = to_point.dot(self.forward());

        // Use FOV to calculate the half-extents of the view plane at that depth
        let fovy_rad = self.camera.fovy.to_radians();
        let half_height = depth * (fovy_rad / 2.0).tan();
        let half_width = half_height * self.camera.aspect;

        // Calculate world position on the plane at that depth
        let right = self.right();
        let up = self.up();
        let forward = self.forward();

        let center = self.camera.eye + forward * depth;
        center + right * (ndc_x * half_width) + up * (ndc_y * half_height)
    }

    /// Project a world-space point to screen coordinates (pixels, origin
    /// top-left). Returns `None` if the point is at or behind the camera.
    pub(crate) fn world_to_screen(
        &self,
        world: Vec3,
        viewport: UVec2,
    ) -> Option<Vec2> {
        let view_proj = self.camera.build_matrix();
        let clip = view_proj * glam::Vec4::new(world.x, world.y, world.z, 1.0);
        if clip.w <= 0.0 {
            return None;
        }
        let ndc_x = clip.x / clip.w;
        let ndc_y = clip.y / clip.w;
        Some(Vec2::new(
            (ndc_x + 1.0) * 0.5 * viewport.x as f32,
            (1.0 - ndc_y) * 0.5 * viewport.y as f32,
        ))
    }

    /// Get the current view frustum for culling
    pub(crate) fn frustum(&self) -> Frustum {
        Frustum::from_view_projection(self.camera.build_matrix())
    }
}
