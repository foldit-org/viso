use std::borrow::Cow;

use naga_oil::compose::{
    ComposableModuleDescriptor, Composer, NagaModuleDescriptor, ShaderLanguage,
    ShaderType,
};

use crate::error::VisoError;

/// All composable (non-module) shaders in the engine.
#[derive(Debug, Clone, Copy)]
pub(crate) enum Shader {
    BloomThreshold,
    BloomBlur,
    BloomUpsample,
    Fxaa,
    Composite,
    Ssao,
    SsaoBlur,
    BackboneTube,
    Capsule,
    Sphere,
    Cone,
    Polygon,
    PickingMesh,
    PickingCapsule,
    PickingSphere,
    Isosurface,
    BackfaceDepth,
}

/// Expands each `Variant => "path"` into a match arm returning
/// `("path", include_str!("../shaders/path"))`, and collects variants
/// into a `const ALL` slice.
macro_rules! shader_registry {
    ($( $variant:ident => $path:literal ),+ $(,)?) => {
        impl Shader {
            /// Every composable shader variant, for exhaustive iteration.
            #[cfg(test)]
            const ALL: &[Self] = &[ $( Self::$variant ),+ ];

            /// (error-context path, WGSL source).
            ///
            /// The path string is NOT a filesystem path — naga-oil resolves
            /// imports via `#define_import_path` directives in WGSL source.
            /// The path is used as the wgpu debug label and in error messages.
            fn info(self) -> (&'static str, &'static str) {
                match self {
                    $( Self::$variant => ($path, include_str!(concat!("../shaders/", $path))) ),+
                }
            }
        }
    };
}

shader_registry! {
    BloomThreshold => "screen/bloom_threshold.wgsl",
    BloomBlur      => "screen/bloom_blur.wgsl",
    BloomUpsample  => "screen/bloom_upsample.wgsl",
    Fxaa           => "screen/fxaa.wgsl",
    Composite      => "screen/composite.wgsl",
    Ssao           => "screen/ssao.wgsl",
    SsaoBlur       => "screen/ssao_blur.wgsl",
    BackboneTube   => "raster/mesh/backbone_tube.wgsl",
    Capsule        => "raster/impostor/capsule.wgsl",
    Sphere         => "raster/impostor/sphere.wgsl",
    Cone           => "raster/impostor/cone.wgsl",
    Polygon        => "raster/impostor/polygon.wgsl",
    PickingMesh    => "utility/picking_mesh.wgsl",
    PickingCapsule => "utility/picking_capsule.wgsl",
    PickingSphere  => "utility/picking_sphere.wgsl",
    Isosurface     => "raster/mesh/isosurface.wgsl",
    BackfaceDepth  => "raster/mesh/backface_depth.wgsl",
}

/// Shared shader modules registered with naga-oil for `#import` support.
///
/// Order matters: modules with no dependencies first. The path strings
/// are error-context labels, not filesystem paths — naga-oil resolves
/// imports via `#define_import_path` directives in WGSL source.
const MODULE_PATHS: &[(&str, &str)] = &[
    (
        "modules/constants.wgsl",
        include_str!("../shaders/modules/constants.wgsl"),
    ),
    (
        "modules/cavity.wgsl",
        include_str!("../shaders/modules/cavity.wgsl"),
    ),
    (
        "modules/depth.wgsl",
        include_str!("../shaders/modules/depth.wgsl"),
    ),
    (
        "modules/impostor_types.wgsl",
        include_str!("../shaders/modules/impostor_types.wgsl"),
    ),
    (
        "modules/fullscreen.wgsl",
        include_str!("../shaders/modules/fullscreen.wgsl"),
    ),
    (
        "modules/camera.wgsl",
        include_str!("../shaders/modules/camera.wgsl"),
    ),
    (
        "modules/lighting.wgsl",
        include_str!("../shaders/modules/lighting.wgsl"),
    ),
    (
        "modules/ray.wgsl",
        include_str!("../shaders/modules/ray.wgsl"),
    ),
    (
        "modules/volume.wgsl",
        include_str!("../shaders/modules/volume.wgsl"),
    ),
    (
        "modules/selection.wgsl",
        include_str!("../shaders/modules/selection.wgsl"),
    ),
    (
        "modules/highlight.wgsl",
        include_str!("../shaders/modules/highlight.wgsl"),
    ),
    // pbr depends on lighting — must come after it.
    (
        "modules/pbr.wgsl",
        include_str!("../shaders/modules/pbr.wgsl"),
    ),
    // shade depends on lighting, pbr, highlight, constants.
    (
        "modules/shade.wgsl",
        include_str!("../shaders/modules/shade.wgsl"),
    ),
];

/// Wraps `naga_oil::compose::Composer` to provide shader composition with
/// `#import` support.
///
/// All WGSL shader sources are compiled in at build time via `include_str!`.
/// Consuming code references shaders by their [`Shader`] enum variant,
/// eliminating fragile string paths across the codebase.
pub(crate) struct ShaderComposer {
    composer: Composer,
}

impl ShaderComposer {
    /// Create a new composer with shared modules registered.
    pub(crate) fn new() -> Result<Self, VisoError> {
        let mut composer = Composer::default();

        for (path, source) in MODULE_PATHS {
            let _ = composer
                .add_composable_module(ComposableModuleDescriptor {
                    source,
                    file_path: path,
                    language: ShaderLanguage::Wgsl,
                    ..Default::default()
                })
                .map_err(|e| {
                    VisoError::Shader(format!(
                        "failed to register shader module '{path}': {e:?}"
                    ))
                })?;
        }

        Ok(Self { composer })
    }

    /// Compose a shader into a `wgpu::ShaderModule` ready for pipeline
    /// creation.
    pub(crate) fn compose(
        &mut self,
        device: &wgpu::Device,
        shader: Shader,
    ) -> Result<wgpu::ShaderModule, VisoError> {
        let naga_module = self.make_naga(shader)?;
        let (path, _) = shader.info();
        Ok(device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(path),
            source: wgpu::ShaderSource::Naga(Cow::Owned(naga_module)),
        }))
    }

    /// Compose a shader source into a `naga::Module` without creating a wgpu
    /// shader module. Useful for testing shader composition without a GPU
    /// device.
    #[cfg(test)]
    pub(crate) fn compose_naga(
        &mut self,
        shader: Shader,
    ) -> Result<naga::Module, VisoError> {
        self.make_naga(shader)
    }

    /// Build a `naga::Module` from a shader variant.
    fn make_naga(&mut self, shader: Shader) -> Result<naga::Module, VisoError> {
        let (path, source) = shader.info();
        self.composer
            .make_naga_module(NagaModuleDescriptor {
                source,
                file_path: path,
                shader_type: ShaderType::Wgsl,
                ..Default::default()
            })
            .map_err(|e| {
                VisoError::Shader(format!(
                    "failed to compose shader '{path}': {e}"
                ))
            })
    }
}

#[cfg(test)]
#[allow(clippy::panic, clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn test_all_shaders_compose() {
        let mut composer = ShaderComposer::new().expect("ShaderComposer::new");
        for &shader in Shader::ALL {
            let _ = composer.compose_naga(shader).unwrap_or_else(|e| {
                panic!("Shader '{shader:?}' failed to compose: {e}")
            });
        }
    }
}
