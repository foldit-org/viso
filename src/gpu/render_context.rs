/// Errors that can occur during GPU context initialization.
#[derive(Debug, thiserror::Error)]
pub enum RenderContextError {
    /// Failed to create a wgpu surface from the window handle.
    #[error("surface creation failed: {0}")]
    SurfaceCreation(#[source] wgpu::CreateSurfaceError),
    /// No compatible GPU adapter found.
    #[error("no compatible GPU adapter found: {0}")]
    AdapterRequest(#[source] wgpu::RequestAdapterError),
    /// GPU device request failed (limits or features not met).
    #[error("device request failed: {0}")]
    DeviceRequest(#[source] wgpu::RequestDeviceError),
    /// Surface configuration not supported by the selected adapter.
    #[error("surface configuration not supported by adapter")]
    UnsupportedSurface,
}

/// Failure to acquire a swapchain frame for presentation.
///
/// wgpu replaced `Surface::get_current_texture`'s `Result` with a status
/// enum; this captures the acquisition-failure cases the renderer acts on.
#[derive(Debug, thiserror::Error)]
pub enum SurfaceError {
    /// Frame acquisition timed out or the window is occluded; skip the frame.
    #[error("surface frame unavailable (timeout or occluded)")]
    Timeout,
    /// Surface configuration is outdated; reconfigure and retry.
    #[error("surface outdated")]
    Outdated,
    /// Surface was lost and must be recreated.
    #[error("surface lost")]
    Lost,
    /// A validation error was raised during acquisition.
    #[error("surface validation error")]
    Validation,
}

/// Owns the core wgpu resources: device, queue, surface, and configuration.
pub struct RenderContext {
    /// The wgpu logical device.
    pub device: wgpu::Device,
    /// The wgpu command queue.
    pub queue: wgpu::Queue,
    /// The window surface for presentation (`None` in texture-only mode).
    pub surface: Option<wgpu::Surface<'static>>,
    /// Current surface configuration (format, size, present mode).
    pub config: wgpu::SurfaceConfiguration,
    /// Supersampling scale factor (1 = native, 2 = 2x SSAA).
    pub render_scale: u32,
    /// Present modes supported by the adapter+surface combination.
    supported_present_modes: Vec<wgpu::PresentMode>,
}

impl RenderContext {
    /// Create a new render context from the given window surface target and
    /// initial size.
    ///
    /// # Errors
    ///
    /// Returns `RenderContextError` if surface creation, adapter request,
    /// device request, or surface configuration fails.
    pub async fn new(
        window: impl Into<wgpu::SurfaceTarget<'static>>,
        initial_size: (u32, u32),
    ) -> Result<Self, RenderContextError> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            flags: wgpu::InstanceFlags::default().with_env(),
            ..wgpu::InstanceDescriptor::new_without_display_handle()
        });
        let surface = instance
            .create_surface(window)
            .map_err(RenderContextError::SurfaceCreation)?;

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                compatible_surface: Some(&surface),
                power_preference: wgpu::PowerPreference::HighPerformance,
                ..Default::default()
            })
            .await
            .map_err(RenderContextError::AdapterRequest)?;

        log_adapter_info(&adapter);

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("Primary Device"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
                ..Default::default()
            })
            .await
            .map_err(RenderContextError::DeviceRequest)?;

        let (config, supported_present_modes) =
            Self::configure_surface(&surface, &adapter, &device, initial_size)?;

        Ok(Self {
            device,
            queue,
            surface: Some(surface),
            config,
            render_scale: 1,
            supported_present_modes,
        })
    }

    /// Configure a created surface: default config forced to Fifo at the given
    /// size, returns the config plus the adapter's supported present modes.
    fn configure_surface(
        surface: &wgpu::Surface<'static>,
        adapter: &wgpu::Adapter,
        device: &wgpu::Device,
        size: (u32, u32),
    ) -> Result<
        (wgpu::SurfaceConfiguration, Vec<wgpu::PresentMode>),
        RenderContextError,
    > {
        let capabilities = surface.get_capabilities(adapter);

        let mut config = surface
            .get_default_config(adapter, size.0, size.1)
            .ok_or(RenderContextError::UnsupportedSurface)?;
        config.width = size.0;
        config.height = size.1;
        config.present_mode = wgpu::PresentMode::Fifo;

        surface.configure(device, &config);

        Ok((config, capabilities.present_modes))
    }

    /// Create a render context from a caller-owned device and queue while
    /// still creating the presentation surface here.
    ///
    /// The `instance` and `adapter` are borrowed only to build and configure
    /// the surface; the `device` and `queue` are taken by value and stored.
    ///
    /// # Errors
    ///
    /// Returns `RenderContextError` if surface creation or configuration
    /// fails.
    // `async` mirrors `new()` so call sites share one `block_on` convention,
    // even though this path issues no device/adapter requests to await.
    #[allow(clippy::unused_async)]
    pub async fn new_with_device(
        instance: &wgpu::Instance,
        adapter: &wgpu::Adapter,
        device: wgpu::Device,
        queue: wgpu::Queue,
        target: impl Into<wgpu::SurfaceTarget<'static>>,
        initial_size: (u32, u32),
    ) -> Result<Self, RenderContextError> {
        let surface = instance
            .create_surface(target)
            .map_err(RenderContextError::SurfaceCreation)?;

        let (config, supported_present_modes) =
            Self::configure_surface(&surface, adapter, &device, initial_size)?;

        Ok(Self {
            device,
            queue,
            surface: Some(surface),
            config,
            render_scale: 1,
            supported_present_modes,
        })
    }

    /// Create a render context from an externally-owned device and queue
    /// (no surface — for texture-only / embedded rendering).
    #[must_use]
    pub fn from_device(
        device: wgpu::Device,
        queue: wgpu::Queue,
        format: wgpu::TextureFormat,
        width: u32,
        height: u32,
    ) -> Self {
        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format,
            width,
            height,
            present_mode: wgpu::PresentMode::Fifo,
            desired_maximum_frame_latency: 2,
            alpha_mode: wgpu::CompositeAlphaMode::Auto,
            view_formats: vec![],
        };
        Self {
            device,
            queue,
            surface: None,
            config,
            render_scale: 1,
            supported_present_modes: vec![wgpu::PresentMode::Fifo],
        }
    }

    /// The surface texture format.
    pub fn format(&self) -> wgpu::TextureFormat {
        self.config.format
    }

    /// Internal render width (swapchain width * render_scale).
    pub fn render_width(&self) -> u32 {
        self.config.width * self.render_scale
    }

    /// Internal render height (swapchain height * render_scale).
    pub fn render_height(&self) -> u32 {
        self.config.height * self.render_scale
    }

    /// Reconfigure the surface for the new window size. Ignores zero-sized
    /// dimensions.
    pub fn resize(&mut self, width: u32, height: u32) {
        if width > 0 && height > 0 {
            self.config.width = width;
            self.config.height = height;
            if let Some(ref surface) = self.surface {
                surface.configure(&self.device, &self.config);
            }
        }
    }

    /// Set the surface present mode, falling back to Fifo if unsupported.
    pub fn set_present_mode(&mut self, mode: wgpu::PresentMode) {
        let effective = if self.supported_present_modes.contains(&mode) {
            mode
        } else {
            log::warn!(
                "Present mode {mode:?} not supported by adapter, falling back \
                 to Fifo"
            );
            wgpu::PresentMode::Fifo
        };
        if self.config.present_mode == effective {
            return;
        }
        self.config.present_mode = effective;
        if let Some(ref surface) = self.surface {
            surface.configure(&self.device, &self.config);
        }
    }

    /// Update the SSAA render scale from a DPI scale factor.
    ///
    /// Low-DPI displays (scale < 2.0) get 2x supersampling; HiDPI displays
    /// (scale >= 2.0) render at native resolution. Returns `true` if the
    /// render scale actually changed (caller should resize render targets).
    pub fn set_surface_scale(&mut self, scale: f64) -> bool {
        let new_scale = if scale < 2.0 { 2 } else { 1 };
        if self.render_scale == new_scale {
            return false;
        }
        self.render_scale = new_scale;
        true
    }

    /// Acquire the next swapchain texture for rendering.
    ///
    /// # Errors
    ///
    /// Returns `SurfaceError` if the surface is lost, outdated,
    /// or timed out, or if no surface is available (texture-only mode).
    pub fn get_next_frame(&self) -> Result<wgpu::SurfaceTexture, SurfaceError> {
        let Some(surface) = self.surface.as_ref() else {
            return Err(SurfaceError::Lost);
        };
        match surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(frame)
            | wgpu::CurrentSurfaceTexture::Suboptimal(frame) => Ok(frame),
            wgpu::CurrentSurfaceTexture::Timeout
            | wgpu::CurrentSurfaceTexture::Occluded => {
                Err(SurfaceError::Timeout)
            }
            wgpu::CurrentSurfaceTexture::Outdated => {
                Err(SurfaceError::Outdated)
            }
            wgpu::CurrentSurfaceTexture::Lost => Err(SurfaceError::Lost),
            wgpu::CurrentSurfaceTexture::Validation => {
                Err(SurfaceError::Validation)
            }
        }
    }

    /// Returns `true` if this context has a presentation surface.
    pub fn has_surface(&self) -> bool {
        self.surface.is_some()
    }

    /// Create a new command encoder for recording GPU commands.
    pub fn create_encoder(&self) -> wgpu::CommandEncoder {
        self.device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Render Encoder"),
            })
    }

    /// Finish the encoder and submit its command buffer to the GPU queue.
    pub fn submit(&self, encoder: wgpu::CommandEncoder) {
        let _ = self.queue.submit(std::iter::once(encoder.finish()));
    }
}

/// Log adapter name, backend, driver, and device type at startup.
///
/// Emits a warning when the selected adapter is a software (CPU) rasterizer,
/// since performance will be severely degraded.
fn log_adapter_info(adapter: &wgpu::Adapter) {
    let info = adapter.get_info();
    let driver = if info.driver.is_empty() {
        "unknown driver"
    } else {
        &info.driver
    };
    log::info!(
        "GPU adapter: {} ({:?}, {:?}, {})",
        info.name,
        info.backend,
        info.device_type,
        driver,
    );
    if info.device_type == wgpu::DeviceType::Cpu {
        log::warn!(
            "Software rasterizer selected — rendering performance will be \
             severely degraded"
        );
    }
}
