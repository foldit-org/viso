use encase::ShaderType;
use glam::Vec3;
use wgpu::util::DeviceExt;

use super::RenderContext;
use crate::gpu::pipeline_helpers::{
    cube_texture, filtering_sampler, texture_2d, uniform_buffer,
};

/// Lighting configuration shared across all shaders.
///
/// Layout matches the WGSL `LightingUniform` struct (std140).
/// Padding is handled automatically by encase.
#[derive(Debug, Copy, Clone, ShaderType)]
pub(crate) struct LightingUniform {
    /// Primary light direction (normalized)
    pub(crate) light1_dir: Vec3,
    /// Secondary light direction (normalized)
    pub(crate) light2_dir: Vec3,
    /// Primary light intensity
    pub(crate) light1_intensity: f32,
    /// Secondary light intensity
    pub(crate) light2_intensity: f32,
    /// Ambient light intensity (used as fallback; IBL replaces this when
    /// ibl_strength > 0)
    pub(crate) ambient: f32,
    /// Specular intensity
    pub(crate) specular_intensity: f32,
    /// Specular shininess exponent
    pub(crate) shininess: f32,
    /// Rim edge falloff power (higher = tighter edge glow)
    pub(crate) rim_power: f32,
    /// Rim edge brightness
    pub(crate) rim_intensity: f32,
    /// Rim directionality: 0.0 = pure view-dependent, 1.0 = pure directional
    /// back-light
    pub(crate) rim_directionality: f32,
    /// Rim light tint color
    pub(crate) rim_color: Vec3,
    /// IBL diffuse strength (0.0 = use flat ambient, 1.0 = full IBL)
    pub(crate) ibl_strength: f32,
    /// Rim back-light direction (normalized, points toward the rim light
    /// source)
    pub(crate) rim_dir: Vec3,
    /// Surface roughness (0.05 = mirror-like, 1.0 = completely matte)
    pub(crate) roughness: f32,
    /// Surface metalness (0.0 = dielectric, 1.0 = metal)
    pub(crate) metalness: f32,
    /// Global molecular-surface opacity (1.0 = no scaling). Scales only
    /// SURFACE-kind isosurface fragments; cavities and density maps are
    /// unaffected. Held here so the global surface slider is a cheap
    /// uniform write instead of a per-vertex re-bake.
    pub(crate) surface_opacity: f32,
}

impl Default for LightingUniform {
    fn default() -> Self {
        Self {
            // Primary light: upper-left for strong directional contrast
            light1_dir: Vec3::new(-0.3, 0.9, -0.3).normalize(),
            // Secondary light: upper-right-front for fill
            light2_dir: Vec3::new(0.3, 0.6, -0.4).normalize(),
            // Values match LightingOptions::default()
            light1_intensity: 2.0,
            light2_intensity: 1.1,
            ambient: 0.45,
            specular_intensity: 0.35,
            shininess: 38.0,
            rim_power: 5.0,
            rim_intensity: 0.3,
            rim_directionality: 0.3,
            rim_color: Vec3::new(1.0, 0.85, 0.7),
            ibl_strength: 0.6,
            // Rim back-light: below-behind relative to camera
            rim_dir: Vec3::new(0.0, -0.7, 0.5).normalize(),
            roughness: 0.35,
            metalness: 0.15,
            // 1.0 = no scaling; the engine seeds the real global value at
            // construction and updates it on every surface-opacity change.
            surface_opacity: 1.0,
        }
    }
}

/// GPU lighting uniform, buffer, and bind group.
pub(crate) struct Lighting {
    /// Current lighting uniform data.
    pub(crate) uniform: LightingUniform,
    /// GPU buffer holding the lighting uniform.
    pub(crate) buffer: wgpu::Buffer,
    /// Bind group layout for the lighting uniform.
    pub(crate) layout: wgpu::BindGroupLayout,
    /// Bind group referencing the lighting buffer.
    pub(crate) bind_group: wgpu::BindGroup,
}

impl Lighting {
    /// Create a new lighting instance with default uniform and GPU resources.
    #[allow(clippy::too_many_lines)] // GPU resource setup is inherently verbose
    pub(crate) fn new(context: &RenderContext) -> Self {
        let uniform = LightingUniform::default();

        let mut buf = encase::UniformBuffer::new(Vec::new());
        let _ = buf.write(&uniform);
        let buffer = context.device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("Lighting Buffer"),
                contents: &buf.into_inner(),
                usage: wgpu::BufferUsages::UNIFORM
                    | wgpu::BufferUsages::COPY_DST,
            },
        );

        // Generate procedural irradiance cubemap
        let (_irradiance_texture, irradiance_view) =
            generate_irradiance_cubemap(&context.device, &context.queue);

        // Generate prefiltered environment cubemap for specular IBL
        let (_prefiltered_texture, prefiltered_view) =
            generate_prefiltered_cubemap(&context.device, &context.queue);

        // Generate BRDF integration LUT for split-sum approximation
        let (_brdf_lut_texture, brdf_lut_view) =
            generate_brdf_lut(&context.device, &context.queue);

        let irradiance_sampler =
            context.device.create_sampler(&wgpu::SamplerDescriptor {
                label: Some("Irradiance Sampler"),
                address_mode_u: wgpu::AddressMode::ClampToEdge,
                address_mode_v: wgpu::AddressMode::ClampToEdge,
                address_mode_w: wgpu::AddressMode::ClampToEdge,
                mag_filter: wgpu::FilterMode::Linear,
                min_filter: wgpu::FilterMode::Linear,
                mipmap_filter: wgpu::FilterMode::Linear,
                ..Default::default()
            });

        let layout = context.device.create_bind_group_layout(
            &wgpu::BindGroupLayoutDescriptor {
                label: Some("Lighting Bind Group Layout"),
                entries: &[
                    uniform_buffer(0),    // Lighting uniform
                    cube_texture(1),      // Irradiance cubemap
                    filtering_sampler(2), // IBL sampler
                    cube_texture(3),      // Prefiltered cubemap
                    texture_2d(4),        // BRDF LUT
                ],
            },
        );

        let bind_group =
            context
                .device
                .create_bind_group(&wgpu::BindGroupDescriptor {
                    layout: &layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: buffer.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: wgpu::BindingResource::TextureView(
                                &irradiance_view,
                            ),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: wgpu::BindingResource::Sampler(
                                &irradiance_sampler,
                            ),
                        },
                        wgpu::BindGroupEntry {
                            binding: 3,
                            resource: wgpu::BindingResource::TextureView(
                                &prefiltered_view,
                            ),
                        },
                        wgpu::BindGroupEntry {
                            binding: 4,
                            resource: wgpu::BindingResource::TextureView(
                                &brdf_lut_view,
                            ),
                        },
                    ],
                    label: Some("Lighting Bind Group"),
                });

        Self {
            uniform,
            buffer,
            layout,
            bind_group,
        }
    }

    /// Apply lighting options from the configuration to the GPU uniform.
    pub(crate) fn apply_options(
        &mut self,
        opts: &crate::options::LightingOptions,
        queue: &wgpu::Queue,
    ) {
        self.uniform.light1_intensity = opts.light1_intensity;
        self.uniform.light2_intensity = opts.light2_intensity;
        self.uniform.ambient = opts.ambient;
        self.uniform.specular_intensity = opts.specular_intensity;
        self.uniform.shininess = opts.shininess;
        self.uniform.rim_power = opts.rim_power;
        self.uniform.rim_intensity = opts.rim_intensity;
        self.uniform.rim_directionality = opts.rim_directionality;
        self.uniform.rim_color = Vec3::from(opts.rim_color);
        self.uniform.ibl_strength = opts.ibl_strength;
        self.uniform.roughness = opts.roughness;
        self.uniform.metalness = opts.metalness;
        self.update_gpu(queue);
    }

    /// Write the current lighting uniform to the GPU buffer.
    pub(crate) fn update_gpu(&self, queue: &wgpu::Queue) {
        let mut buf = encase::UniformBuffer::new(Vec::new());
        let _ = buf.write(&self.uniform);
        queue.write_buffer(&self.buffer, 0, &buf.into_inner());
    }

    /// Compute camera basis vectors and update headlamp directions.
    ///
    /// Convenience wrapper around [`update_headlamp`](Self::update_headlamp)
    /// that extracts the basis from a [`Camera`](crate::camera::core::Camera).
    pub(crate) fn update_headlamp_from_camera(
        &mut self,
        camera: &crate::camera::core::Camera,
    ) {
        let forward = (camera.target - camera.eye).normalize();
        let right = camera.up.cross(forward).normalize();
        let up = forward.cross(right);
        self.update_headlamp(right, up, forward);
    }

    /// Update light directions to follow camera (headlamp mode)
    /// Call this each frame after camera updates
    pub(crate) fn update_headlamp(
        &mut self,
        camera_right: Vec3,
        camera_up: Vec3,
        camera_forward: Vec3,
    ) {
        // Primary light: upper-left relative to camera view
        // In camera space: (-0.3, 0.9, -0.3) = left, up, toward viewer
        // Negative z ensures surfaces facing camera receive light
        let light1_camera = Vec3::new(-0.3, 0.9, -0.3).normalize();

        // Transform from camera space to world space
        let light1_world = camera_right * light1_camera.x
            + camera_up * light1_camera.y
            + camera_forward * light1_camera.z;

        self.uniform.light1_dir = light1_world.normalize();

        // Secondary fill light: upper-right relative to camera
        let light2_camera = Vec3::new(0.3, 0.6, -0.4).normalize();
        let light2_world = camera_right * light2_camera.x
            + camera_up * light2_camera.y
            + camera_forward * light2_camera.z;

        self.uniform.light2_dir = light2_world.normalize();

        // Rim back-light: below-behind relative to camera
        let rim_camera = Vec3::new(0.0, -0.7, 0.5).normalize();
        let rim_world = camera_right * rim_camera.x
            + camera_up * rim_camera.y
            + camera_forward * rim_camera.z;

        self.uniform.rim_dir = rim_world.normalize();
    }
}

/// Configuration for procedural cubemap generation.
struct CubemapDesc<'a> {
    /// Debug label for the GPU texture.
    label: &'a str,
    /// Texel resolution per face (e.g. 32, 64).
    base_size: u32,
    /// Number of mip levels.
    mip_count: u32,
}

/// Generate a cubemap where each texel is evaluated by a closure.
///
/// Creates an `Rgba8Unorm` cubemap per the descriptor.  For every texel the
/// closure receives the normalized world-space direction and the mip level,
/// and returns an RGB color in `[0, 1]`.
fn generate_cubemap(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    desc: &CubemapDesc,
    evaluate: impl Fn([f32; 3], u32) -> [f32; 3],
) -> (wgpu::Texture, wgpu::TextureView) {
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some(desc.label),
        size: wgpu::Extent3d {
            width: desc.base_size,
            height: desc.base_size,
            depth_or_array_layers: 6,
        },
        mip_level_count: desc.mip_count,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8Unorm,
        usage: wgpu::TextureUsages::TEXTURE_BINDING
            | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });

    for mip in 0..desc.mip_count {
        let size = desc.base_size >> mip;
        for face in 0..6u32 {
            let data = fill_cubemap_face(size, face, mip, &evaluate);
            queue.write_texture(
                wgpu::TexelCopyTextureInfo {
                    texture: &texture,
                    mip_level: mip,
                    origin: wgpu::Origin3d {
                        x: 0,
                        y: 0,
                        z: face,
                    },
                    aspect: wgpu::TextureAspect::All,
                },
                &data,
                wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(size * 4),
                    rows_per_image: Some(size),
                },
                wgpu::Extent3d {
                    width: size,
                    height: size,
                    depth_or_array_layers: 1,
                },
            );
        }
    }

    let view = texture.create_view(&wgpu::TextureViewDescriptor {
        label: Some(&format!("{} View", desc.label)),
        dimension: Some(wgpu::TextureViewDimension::Cube),
        ..Default::default()
    });

    (texture, view)
}

/// Evaluate the closure for every texel of a single cubemap face, returning
/// the RGBA8 pixel data.
#[allow(clippy::cast_precision_loss)]
#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
fn fill_cubemap_face(
    size: u32,
    face: u32,
    mip: u32,
    evaluate: &impl Fn([f32; 3], u32) -> [f32; 3],
) -> Vec<u8> {
    let mut data = vec![0u8; (size * size * 4) as usize];
    for y in 0..size {
        for x in 0..size {
            let u = (x as f32 + 0.5) / size as f32 * 2.0 - 1.0;
            let v = (y as f32 + 0.5) / size as f32 * 2.0 - 1.0;

            let dir = cubemap_face_direction(face, u, v);
            let len =
                (dir[0] * dir[0] + dir[1] * dir[1] + dir[2] * dir[2]).sqrt();
            let dir = [dir[0] / len, dir[1] / len, dir[2] / len];

            let color = evaluate(dir, mip);

            let offset = ((y * size + x) * 4) as usize;
            data[offset] = (color[0].clamp(0.0, 1.0) * 255.0) as u8;
            data[offset + 1] = (color[1].clamp(0.0, 1.0) * 255.0) as u8;
            data[offset + 2] = (color[2].clamp(0.0, 1.0) * 255.0) as u8;
            data[offset + 3] = 255;
        }
    }
    data
}

/// Generate a procedural gradient irradiance cubemap for studio-style ambient
/// lighting.
///
/// 32x32 per face, single mip. Warm white from above, cool blue from below,
/// neutral gray from sides.
fn generate_irradiance_cubemap(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
) -> (wgpu::Texture, wgpu::TextureView) {
    generate_cubemap(
        device,
        queue,
        &CubemapDesc {
            label: "Irradiance Cubemap",
            base_size: 32,
            mip_count: 1,
        },
        |dir, _mip| evaluate_irradiance(dir),
    )
}

/// Generate a prefiltered environment cubemap for specular IBL.
///
/// 64x64 base, 6 mip levels. Each mip represents a different roughness:
/// mip 0 = sharp reflections, higher mips = progressively blurred.
fn generate_prefiltered_cubemap(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
) -> (wgpu::Texture, wgpu::TextureView) {
    let mip_count = 6u32;
    #[allow(clippy::cast_precision_loss)]
    generate_cubemap(
        device,
        queue,
        &CubemapDesc {
            label: "Prefiltered Cubemap",
            base_size: 64,
            mip_count,
        },
        |dir, mip| {
            let roughness = mip as f32 / (mip_count - 1) as f32;
            evaluate_prefiltered(dir, roughness)
        },
    )
}

/// Convert cubemap face index + UV coordinates to a world-space direction.
/// Face order: +X, -X, +Y, -Y, +Z, -Z
fn cubemap_face_direction(face: u32, u: f32, v: f32) -> [f32; 3] {
    match face {
        0 => [1.0, -v, -u],  // +X
        1 => [-1.0, -v, u],  // -X
        2 => [u, 1.0, v],    // +Y
        3 => [u, -1.0, -v],  // -Y
        4 => [u, -v, 1.0],   // +Z
        5 => [-u, -v, -1.0], // -Z
        _ => [0.0, 0.0, 1.0],
    }
}

/// Evaluate studio-style irradiance for a given world-space direction.
///
/// Smooth vertical gradient: warm white from above, cool blue from below.
/// Values roughly match the energy of the old flat ambient (0.12).
fn evaluate_irradiance(dir: [f32; 3]) -> [f32; 3] {
    let y = dir[1];
    let t = y * 0.5 + 0.5;
    // Smooth-step for softer transitions
    let t = t * t * (3.0 - 2.0 * t);

    let top = [0.18f32, 0.16, 0.13];
    let bottom = [0.05f32, 0.06, 0.10];

    [
        bottom[0] + (top[0] - bottom[0]) * t,
        bottom[1] + (top[1] - bottom[1]) * t,
        bottom[2] + (top[2] - bottom[2]) * t,
    ]
}

/// Evaluate prefiltered environment color for a given direction and roughness.
///
/// At low roughness, returns a sharper version of the environment.
/// At high roughness, returns a heavily blurred/averaged version.
/// Uses the same studio gradient as the irradiance map but with a specular
/// highlight boost.
fn evaluate_prefiltered(dir: [f32; 3], roughness: f32) -> [f32; 3] {
    // Base irradiance gradient
    let base = evaluate_irradiance(dir);

    // Add a specular highlight at low roughness: a bright spot in the upper
    // region This simulates a key light reflection in the environment
    let highlight_dir = [0.0f32, 0.7, -0.7]; // Approximate key light direction
    let dot = dir[0] * highlight_dir[0]
        + dir[1] * highlight_dir[1]
        + dir[2] * highlight_dir[2];
    let dot = dot.max(0.0);

    // Sharper highlight at low roughness, wider at high
    let exponent = 1.0 / (roughness * roughness + 0.01);
    let highlight = dot.powf(exponent.min(256.0));
    let highlight_strength = 0.4 * (1.0 - roughness);

    [
        base[0] + highlight * highlight_strength,
        base[1] + highlight * highlight_strength * 0.95,
        base[2] + highlight * highlight_strength * 0.85,
    ]
}

/// Generate a BRDF integration lookup table for the split-sum IBL
/// approximation.
///
/// 256x256 Rg16Float texture. X axis = NdotV, Y axis = roughness.
/// R = scale, G = bias for the Fresnel term: `specular = F0 * scale + bias`
#[allow(clippy::cast_precision_loss)] // texel indices and LUT size (256) fit in f32
fn generate_brdf_lut(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
) -> (wgpu::Texture, wgpu::TextureView) {
    let size = 256u32;

    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("BRDF LUT"),
        size: wgpu::Extent3d {
            width: size,
            height: size,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rg16Float,
        usage: wgpu::TextureUsages::TEXTURE_BINDING
            | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });

    // Generate BRDF LUT on CPU using importance sampling
    let sample_count = 256u32;
    let mut data = vec![0u8; (size * size * 4) as usize]; // 4 bytes per texel (2x f16)

    for y in 0..size {
        for x in 0..size {
            let ndot_v = (x as f32 + 0.5) / size as f32;
            let ndot_v = ndot_v.max(0.001); // Avoid division by zero
            let roughness = (y as f32 + 0.5) / size as f32;
            let roughness = roughness.max(0.01);

            let (scale, bias) = integrate_brdf(ndot_v, roughness, sample_count);

            let offset = ((y * size + x) * 4) as usize;
            let scale_f16 = half::f16::from_f32(scale);
            let bias_f16 = half::f16::from_f32(bias);
            data[offset..offset + 2].copy_from_slice(&scale_f16.to_le_bytes());
            data[offset + 2..offset + 4]
                .copy_from_slice(&bias_f16.to_le_bytes());
        }
    }

    queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture: &texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        &data,
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(size * 4),
            rows_per_image: Some(size),
        },
        wgpu::Extent3d {
            width: size,
            height: size,
            depth_or_array_layers: 1,
        },
    );

    let view = texture.create_view(&Default::default());
    (texture, view)
}

/// Integrate the BRDF for given NdotV and roughness using importance sampling.
/// Returns (scale, bias) for the split-sum approximation.
#[allow(clippy::many_single_char_names)] // standard math notation for BRDF vectors (v, h, l, a, b, g)
#[allow(clippy::cast_precision_loss)] // sample_count is small (256), fits in f32
fn integrate_brdf(
    ndot_v: f32,
    roughness: f32,
    sample_count: u32,
) -> (f32, f32) {
    use std::f32::consts::PI;

    let v = [
        (1.0 - ndot_v * ndot_v).sqrt(), // sin
        0.0,
        ndot_v, // cos
    ];

    let mut a = 0.0f32;
    let mut b = 0.0f32;

    let alpha = roughness * roughness;

    for i in 0..sample_count {
        // Hammersley sequence for quasi-random sampling
        let xi = hammersley(i, sample_count);

        // Importance sample the GGX distribution
        let phi = 2.0 * PI * xi[0];
        let cos_theta =
            ((1.0 - xi[1]) / (1.0 + (alpha * alpha - 1.0) * xi[1])).sqrt();
        let sin_theta = (1.0 - cos_theta * cos_theta).sqrt().max(0.0);

        // Halfway vector in tangent space
        let h = [sin_theta * phi.cos(), sin_theta * phi.sin(), cos_theta];

        // Light vector = reflect view around halfway
        let v_dot_h = v[0] * h[0] + v[1] * h[1] + v[2] * h[2];
        let l = [
            2.0 * v_dot_h * h[0] - v[0],
            2.0 * v_dot_h * h[1] - v[1],
            2.0 * v_dot_h * h[2] - v[2],
        ];

        let n_dot_l = l[2].max(0.0);
        let n_dot_h = cos_theta.max(0.0);
        let v_dot_h = v_dot_h.clamp(0.0, 1.0);

        if n_dot_l > 0.0 {
            let g = geometry_smith_ibl(ndot_v, n_dot_l, roughness);
            let g_vis = (g * v_dot_h) / (n_dot_h * ndot_v).max(0.0001);
            let fc = (1.0 - v_dot_h).powi(5);

            a += (1.0 - fc) * g_vis;
            b += fc * g_vis;
        }
    }

    (a / sample_count as f32, b / sample_count as f32)
}

/// Smith's geometry function for IBL (uses k = alpha^2 / 2)
fn geometry_smith_ibl(ndot_v: f32, ndot_l: f32, roughness: f32) -> f32 {
    let a = roughness * roughness;
    let k = a / 2.0;
    let ggx_v = ndot_v / (ndot_v * (1.0 - k) + k);
    let ggx_l = ndot_l / (ndot_l * (1.0 - k) + k);
    ggx_v * ggx_l
}

/// Hammersley quasi-random sequence (2D)
#[allow(clippy::cast_precision_loss)] // sample index and count are small, fit in f32
fn hammersley(i: u32, n: u32) -> [f32; 2] {
    [i as f32 / n as f32, radical_inverse_vdc(i)]
}

/// Van der Corput radical inverse (base 2)
#[allow(clippy::cast_precision_loss)] // intentional u32-to-f32 for normalized [0,1) output
fn radical_inverse_vdc(mut bits: u32) -> f32 {
    bits = bits.rotate_right(16);
    bits = ((bits & 0x5555_5555) << 1) | ((bits & 0xAAAA_AAAA) >> 1);
    bits = ((bits & 0x3333_3333) << 2) | ((bits & 0xCCCC_CCCC) >> 2);
    bits = ((bits & 0x0F0F_0F0F) << 4) | ((bits & 0xF0F0_F0F0) >> 4);
    bits = ((bits & 0x00FF_00FF) << 8) | ((bits & 0xFF00_FF00) >> 8);
    bits as f32 * 2.328_306_4e-10 // 1.0 / 0x1_0000_0000
}
