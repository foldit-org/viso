//! Ball-and-stick renderer for small molecules (ligands, ions, waters).
//!
//! Renders atoms as ray-cast sphere impostors and bonds as capsule impostors.
//! Consumes entity-level
//! [`EntityTopology`](crate::renderer::entity_topology::EntityTopology)
//! plus a positions slice; the render path never sees `&MoleculeEntity`
//! or `&Assembly`.

mod instances;

use glam::Vec3;
use molex::MoleculeType;

use crate::error::VisoError;
use crate::gpu::{RenderContext, Shader, ShaderComposer};
use crate::options::{ColorOptions, DisplayOptions, DrawingMode};
use crate::renderer::entity_topology::EntityTopology;
use crate::renderer::impostor::{
    CapsuleInstance, ImpostorPass, ShaderDef, SphereInstance,
};

/// Radius for bond capsules (thinner than protein sidechains)
pub(super) const BOND_RADIUS: f32 = 0.15;

/// Fraction of vdw_radius for ball-and-stick atom spheres
pub(super) const BALL_RADIUS_SCALE: f32 = 0.3;

/// Sphere radius for Stick mode joints (matches sidechain capsule caps).
pub(super) const STICK_SPHERE_RADIUS: f32 = 0.15;

/// Bond radius for Stick mode — matches sidechain capsule radius so
/// stick geometry has the same visual weight as sidechain bonds.
pub(super) const STICK_BOND_RADIUS: f32 = 0.3;

/// Larger spheres for ions
pub(super) const ION_RADIUS_SCALE: f32 = 0.5;

/// Small sphere for water oxygen
pub(super) const WATER_RADIUS: f32 = 0.3;

/// Perpendicular offset for double bond parallel capsules
pub(super) const DOUBLE_BOND_OFFSET: f32 = 0.2;

/// Warm beige/tan carbon tint for lipid molecules.
pub(super) const LIPID_CARBON_TINT: [f32; 3] = [0.76, 0.70, 0.50];

/// Return atom color: if a carbon tint is provided, carbon atoms use it;
/// all other elements (N, O, S, P, etc.) keep standard CPK coloring.
pub(super) fn atom_color(
    elem: molex::Element,
    carbon_tint: Option<[f32; 3]>,
) -> [f32; 3] {
    match (elem, carbon_tint) {
        (molex::Element::C, Some(tint)) => tint,
        _ => elem.cpk_color(),
    }
}

/// Pre-computed instance data for GPU upload.
pub(crate) struct PreparedBallAndStickData<'a> {
    /// Raw bytes for sphere instance data.
    pub(crate) sphere_bytes: &'a [u8],
    /// Number of sphere instances.
    pub(crate) sphere_count: u32,
    /// Raw bytes for bond capsule instance data.
    pub(crate) capsule_bytes: &'a [u8],
    /// Number of bond capsule instances.
    pub(crate) capsule_count: u32,
}

/// Output buffers for instance generation.
#[derive(Default)]
pub(super) struct InstanceCollector {
    pub(super) spheres: Vec<SphereInstance>,
    pub(super) bonds: Vec<CapsuleInstance>,
}

impl InstanceCollector {
    /// Push a visual bond capsule.
    ///
    /// `endpoints` is `[pos_a, pos_b]`; `colors` is `[color_a, color_b]`.
    pub(super) fn push_bond(
        &mut self,
        endpoints: [[f32; 3]; 2],
        radius: f32,
        colors: [[f32; 3]; 2],
        pick_id: u32,
    ) {
        let [pos_a, pos_b] = endpoints;
        self.bonds.push(CapsuleInstance {
            endpoint_a: [pos_a[0], pos_a[1], pos_a[2], radius],
            endpoint_b: [pos_b[0], pos_b[1], pos_b[2], pick_id as f32],
            color_a: [colors[0][0], colors[0][1], colors[0][2], 0.0],
            color_b: [colors[1][0], colors[1][1], colors[1][2], 0.0],
        });
    }
}

/// Find any vector perpendicular to the given vector.
pub(super) fn find_perpendicular(v: Vec3) -> Vec3 {
    if v.length_squared() < 1e-8 {
        return Vec3::X;
    }
    let candidate = if v.x.abs() < 0.9 { Vec3::X } else { Vec3::Y };
    v.cross(candidate).normalize()
}

/// Renders small molecules as ray-cast sphere + capsule impostors.
pub(crate) struct BallAndStickRenderer {
    sphere_pass: ImpostorPass<SphereInstance>,
    bond_pass: ImpostorPass<CapsuleInstance>,
}

impl BallAndStickRenderer {
    /// Create a new ball-and-stick renderer with empty buffers.
    pub(crate) fn new(
        context: &RenderContext,
        layouts: &crate::renderer::PipelineLayouts,
        shader_composer: &mut ShaderComposer,
    ) -> Result<Self, VisoError> {
        let sphere_pass = ImpostorPass::new(
            context,
            &ShaderDef {
                label: "BnS Sphere",
                shader: Shader::Sphere,
            },
            layouts,
            6,
            shader_composer,
        )?;

        let bond_pass = ImpostorPass::new(
            context,
            &ShaderDef {
                label: "BnS Bond",
                shader: Shader::Capsule,
            },
            layouts,
            6,
            shader_composer,
        )?;

        Ok(Self {
            sphere_pass,
            bond_pass,
        })
    }

    /// Generate sphere + capsule instances for a single entity using the
    /// entity's derived topology and current positions.
    ///
    /// `pick_id_offset` is the base atom pick ID for this entity so pick
    /// IDs remain globally unique across concatenated meshes.
    ///
    /// `drawing_mode` controls Stick vs BallAndStick for protein / NA
    /// entities routed through this pipeline via a per-entity drawing
    /// mode override. For non-polymer entities this is ignored.
    ///
    /// `per_residue_colors`, when provided, overrides CPK element
    /// coloring for polymer entities (Stick/BnS) — each atom is colored
    /// by its residue's backbone color (chain, SS, score, etc.).
    pub(crate) fn generate_entity_instances(
        topology: &EntityTopology,
        positions: &[Vec3],
        display: &DisplayOptions,
        colors: Option<&ColorOptions>,
        pick_id_offset: u32,
        drawing_mode: DrawingMode,
        per_residue_colors: Option<&[[f32; 3]]>,
    ) -> (Vec<SphereInstance>, Vec<CapsuleInstance>) {
        let mut out = InstanceCollector::default();
        dispatch_entity(
            topology,
            positions,
            display,
            colors,
            pick_id_offset,
            drawing_mode,
            per_residue_colors,
            &mut out,
        );
        (out.spheres, out.bonds)
    }

    /// Apply pre-computed instance data (GPU upload only, no CPU generation).
    pub(crate) fn apply_prepared(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        data: &PreparedBallAndStickData,
    ) {
        let _ = self.sphere_pass.write_bytes(
            device,
            queue,
            data.sphere_bytes,
            data.sphere_count,
        );
        let _ = self.bond_pass.write_bytes(
            device,
            queue,
            data.capsule_bytes,
            data.capsule_count,
        );
    }

    /// Draw both spheres and bonds in a single render pass.
    pub(crate) fn draw<'a>(
        &'a self,
        render_pass: &mut wgpu::RenderPass<'a>,
        bind_groups: &crate::renderer::draw_context::DrawBindGroups<'a>,
    ) {
        self.sphere_pass.draw(render_pass, bind_groups);
        self.bond_pass.draw(render_pass, bind_groups);
    }

    /// Get the sphere instance buffer (visual pass).
    pub(crate) fn sphere_buffer(&self) -> &wgpu::Buffer {
        self.sphere_pass.buffer()
    }

    /// Get the bond capsule instance buffer (visual pass).
    pub(crate) fn bond_buffer(&self) -> &wgpu::Buffer {
        self.bond_pass.buffer()
    }

    /// Get the sphere instance count.
    pub(crate) fn sphere_count(&self) -> u32 {
        self.sphere_pass.instance_count
    }

    /// Get the bond capsule instance count.
    pub(crate) fn bond_count(&self) -> u32 {
        self.bond_pass.instance_count
    }

    /// GPU buffer sizes: `(label, used_bytes, allocated_bytes)`.
    pub(crate) fn buffer_info(&self) -> Vec<(&'static str, usize, usize)> {
        vec![
            self.sphere_pass.buffer_info("BnS Spheres"),
            self.bond_pass.buffer_info("BnS Bonds"),
        ]
    }
}

/// Dispatch a single entity by molecule type to the appropriate instance
/// generator. Polymer entities produce no instances in `Cartoon` mode.
fn dispatch_entity(
    topology: &EntityTopology,
    positions: &[Vec3],
    display: &DisplayOptions,
    colors: Option<&ColorOptions>,
    atom_offset: u32,
    drawing_mode: DrawingMode,
    per_residue_colors: Option<&[[f32; 3]]>,
    out: &mut InstanceCollector,
) {
    match topology.molecule_type {
        MoleculeType::Protein | MoleculeType::DNA | MoleculeType::RNA
            if drawing_mode != DrawingMode::Cartoon =>
        {
            instances::generate_polymer_bns_instances(
                topology,
                positions,
                atom_offset,
                drawing_mode,
                per_residue_colors,
                out,
            );
        }
        MoleculeType::Ligand | MoleculeType::Cofactor => {
            let tint = (topology.molecule_type == MoleculeType::Cofactor)
                .then(|| instances::resolve_cofactor_tint(topology, colors));
            instances::generate_ligand_instances(
                topology,
                positions,
                tint,
                atom_offset,
                out,
            );
        }
        MoleculeType::Lipid => {
            let lipid_tint =
                colors.map_or(LIPID_CARBON_TINT, |c| c.lipid_carbon_tint);
            if display.lipid_ball_and_stick() {
                instances::generate_ligand_instances(
                    topology,
                    positions,
                    Some(lipid_tint),
                    atom_offset,
                    out,
                );
            } else {
                instances::generate_coarse_lipid_instances(
                    topology,
                    positions,
                    lipid_tint,
                    atom_offset,
                    out,
                );
            }
        }
        MoleculeType::Ion if display.show_ions => {
            instances::generate_ion_instances(
                topology,
                positions,
                atom_offset,
                out,
            );
        }
        MoleculeType::Water if display.show_waters => {
            instances::generate_water_instances(
                topology,
                positions,
                atom_offset,
                out,
            );
        }
        MoleculeType::Solvent if display.show_solvent => {
            let sc = colors.map_or([0.6, 0.6, 0.6], |c| c.solvent_color);
            instances::generate_solvent_instances(
                topology,
                positions,
                sc,
                atom_offset,
                out,
            );
        }
        _ => {}
    }
}
