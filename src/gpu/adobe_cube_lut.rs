//! GPU [`wgpu`] 3D `Rgba16Float` texture + sampler from a CPU-decoded Adobe
//! `.cube` volume ([`crate::util::lut_adobe_cube::LutRgbCube3d`]).
//!
//! - Volume `N×N×N` texels (`N = LUT_3D_SIZE`), RGBA16F with alpha forced to 1.
//! - Linear filtering and clamp-to-edge on all axes.
//! - PR2 only uploads and holds the texture; PR3 binds it in post-process WGSL.
//!
//! ## Upload layout
//!
//! Uses [`crate::util::lut_adobe_cube::LutRgbCube3d::rgba16f_bytes_volume_order`]
//! with [`wgpu::TexelCopyBufferLayout`]:
//!
//! - `bytes_per_row = N * 8` (8 bytes per RGBA16F texel)
//! - `rows_per_image = N`
//!
//! Texel order follows `lut_adobe_cube::types` (input R varies fastest).

use crate::error::VisoError;
use crate::gpu::RenderContext;
use crate::util::lut_adobe_cube::LutRgbCube3d;

/// GPU resources for one Adobe `.cube` 3D LUT (`N×N×N` RGBA16F).
#[allow(dead_code)] // Fields used when post-process binds this LUT (PR3+).
pub(crate) struct AdobeCubeLutTexture {
    /// Keep the texture alive; the view references it.
    texture: wgpu::Texture,
    view: wgpu::TextureView,
    sampler: wgpu::Sampler,
    /// Grid edge length `N` matching [`LutRgbCube3d::size`].
    size: u32,
}

#[allow(dead_code)] // Used when post-process samples the LUT (PR3+).
impl AdobeCubeLutTexture {
    /// Create the 3D texture, upload texels, build linear/clamp sampler.
    ///
    /// # Errors
    ///
    /// [`VisoError::GpuResource`] if `lut.size` exceeds
    /// [`wgpu::Limits::max_texture_dimension_3d`].
    pub(crate) fn try_new(
        context: &RenderContext,
        lut: &LutRgbCube3d,
    ) -> Result<Self, VisoError> {
        let n = lut.size;
        let max_dim = context.device.limits().max_texture_dimension_3d;
        if n > max_dim {
            return Err(VisoError::GpuResource(format!(
                "Adobe cube LUT size {n} exceeds max_texture_dimension_3d \
                 ({max_dim})"
            )));
        }

        let extent = wgpu::Extent3d {
            width: n,
            height: n,
            depth_or_array_layers: n,
        };

        let texture = context.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Adobe cube 3D LUT"),
            size: extent,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D3,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });

        let view = texture.create_view(&wgpu::TextureViewDescriptor {
            label: Some("Adobe cube 3D LUT view"),
            dimension: Some(wgpu::TextureViewDimension::D3),
            mip_level_count: Some(1),
            ..Default::default()
        });

        let sampler = context.device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Adobe cube 3D LUT sampler"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Nearest,
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            ..Default::default()
        });

        // Packed RGBA16F bytes; same sample order as the `.cube` file.
        let bytes = lut.rgba16f_bytes_volume_order();
        debug_assert_eq!(
            bytes.len(),
            (n as usize).saturating_pow(3).saturating_mul(8)
        );

        context.queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &bytes,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(n * 8),
                rows_per_image: Some(n),
            },
            extent,
        );

        Ok(Self {
            texture,
            view,
            sampler,
            size: n,
        })
    }

    /// Bind as WGSL `texture_3d<f32>` (stores RGBA16F).
    pub(crate) fn texture_view(&self) -> &wgpu::TextureView {
        &self.view
    }

    /// Linear + clamp sampler for LUT sampling.
    pub(crate) fn sampler(&self) -> &wgpu::Sampler {
        &self.sampler
    }

    /// LUT grid size `N` (`LUT_3D_SIZE N`).
    #[must_use]
    pub(crate) fn grid_size(&self) -> u32 {
        self.size
    }
}
