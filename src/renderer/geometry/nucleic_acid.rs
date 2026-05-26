//! Nucleic acid ring + stem renderer.
//!
//! Renders base rings as extruded polygon impostors and stem tubes as capsule
//! impostors connecting the backbone ribbon to nucleotide ring centers.
//! Backbone ribbons are handled by `BackboneRenderer`.

use glam::Vec3;

use crate::error::VisoError;
use crate::gpu::{RenderContext, Shader, ShaderComposer};
use crate::renderer::entity_topology::ResolvedRing;
use crate::renderer::impostor::{
    CapsuleInstance, ExtrudedPolygonInstance, ImpostorPass, ShaderDef,
};

/// Default light blue-violet NA color, used as the fallback for residues
/// whose base is not NDB-recognized and wherever a per-residue color slice
/// is unavailable. The single source of truth for this constant (T4-NA-A);
/// the backbone mesh path and `entity_view` reference it here rather than
/// redefining it. Distinct from the user-configurable
/// `ColorOptions::nucleic_acid` default, which is a separate concept that
/// happens to share this value.
pub(crate) const NA_DEFAULT_COLOR: [f32; 3] = [0.45, 0.55, 0.85];

/// Stem capsule radius.
const STEM_RADIUS: f32 = 0.25;

/// Half-thickness of an extruded base-ring polygon paddle. Numerically
/// equal to [`STEM_RADIUS`] but a semantically independent quantity (a
/// paddle slab half-depth, not a tube radius) -- named separately so the
/// coincidence is explicit and either can change without the other (T5-NA-B).
const RING_HALF_THICKNESS: f32 = 0.25;

/// Renders DNA/RNA base rings and stem tubes as impostors.
pub(crate) struct NucleicAcidRenderer {
    stem_pass: ImpostorPass<CapsuleInstance>,
    ring_pass: ImpostorPass<ExtrudedPolygonInstance>,
}

impl NucleicAcidRenderer {
    /// Create a new nucleic acid renderer with empty instance buffers.
    ///
    /// Instances are uploaded later via [`Self::apply_prepared`] from the
    /// background mesh pipeline; nothing meaningful is generated here.
    pub(crate) fn new(
        context: &RenderContext,
        layouts: &crate::renderer::PipelineLayouts,
        shader_composer: &mut ShaderComposer,
    ) -> Result<Self, VisoError> {
        let stem_pass = ImpostorPass::new(
            context,
            &ShaderDef {
                label: "NA Stem",
                shader: Shader::Capsule,
            },
            layouts,
            6,
            shader_composer,
        )?;

        let ring_pass = ImpostorPass::new(
            context,
            &ShaderDef {
                label: "NA Ring",
                shader: Shader::Polygon,
            },
            layouts,
            72,
            shader_composer,
        )?;

        Ok(Self {
            stem_pass,
            ring_pass,
        })
    }

    /// Draw nucleic acid geometry into the given render pass.
    pub(crate) fn draw<'a>(
        &'a self,
        render_pass: &mut wgpu::RenderPass<'a>,
        bind_groups: &crate::renderer::draw_context::DrawBindGroups<'a>,
    ) {
        self.stem_pass.draw(render_pass, bind_groups);
        self.ring_pass.draw(render_pass, bind_groups);
    }

    /// Apply pre-computed instance data (GPU upload only, no CPU generation).
    pub(crate) fn apply_prepared(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        na: &crate::renderer::pipeline::prepared::NucleicAcidInstances,
    ) {
        // `write_bytes` returns `true` if the GPU buffer was reallocated
        // (which invalidates any externally-held bind group). Discarded
        // deliberately: `draw` rebinds from `&self` on every call, so a
        // stale external ref is never observed for the NA passes. If draw
        // is ever changed to cache a bind group, this flag must be
        // propagated like the backbone path does (T5-NA-D).
        let _ = self.stem_pass.write_bytes(
            device,
            queue,
            &na.stem_instances,
            na.stem_count,
        );
        let _ = self.ring_pass.write_bytes(
            device,
            queue,
            &na.ring_instances,
            na.ring_count,
        );
    }

    /// GPU buffer sizes: `(label, used_bytes, allocated_bytes)`.
    pub(crate) fn buffer_info(&self) -> Vec<(&'static str, usize, usize)> {
        vec![
            self.stem_pass.buffer_info("NA Stems"),
            self.ring_pass.buffer_info("NA Rings"),
        ]
    }

    // -- Instance generation --

    /// Generate capsule + polygon instances for stems and rings.
    ///
    /// Each base's stem anchors to its *own* residue's P atom (which lies
    /// on the rendered ribbon), not to the globally nearest sample of a
    /// second, undrawn spline flattened across every strand. Per-strand
    /// base-normal coherence flips each ring's Newell normal into the
    /// previous base's hemisphere along the chain, so the
    /// atom-order-dependent Newell sign can't make adjacent paddles face
    /// opposite ways.
    pub(crate) fn generate_instances(
        rings: &[ResolvedRing],
    ) -> (Vec<CapsuleInstance>, Vec<ExtrudedPolygonInstance>) {
        let mut stems = Vec::new();
        let mut ring_instances = Vec::new();

        // Per-strand normal coherence: reset at every chain boundary,
        // then flip each base's normal into the prior base's hemisphere.
        let mut prev_chain: Option<u32> = None;
        let mut prev_normal: Option<Vec3> = None;
        let mut degenerate_rings: usize = 0;
        let mut missing_p: usize = 0;

        for resolved in rings {
            let ring = &resolved.ring;

            let mut base_normal =
                crate::util::geom::newell_normal(&ring.hex_ring);
            if base_normal == Vec3::ZERO {
                degenerate_rings += 1;
                continue;
            }

            // New strand -> drop the carried hemisphere.
            if prev_chain != Some(resolved.chain_idx) {
                prev_normal = None;
                prev_chain = Some(resolved.chain_idx);
            }
            if let Some(pn) = prev_normal {
                if pn.dot(base_normal) < 0.0 {
                    base_normal = -base_normal;
                }
            }
            prev_normal = Some(base_normal);

            // Stem: owning residue's P (on the ribbon) -> ring centroid.
            if let Some(anchor) = resolved.owning_p {
                let centroid = ring.hex_ring.iter().copied().sum::<Vec3>()
                    / ring.hex_ring.len() as f32;
                stems.push(CapsuleInstance {
                    endpoint_a: [anchor.x, anchor.y, anchor.z, STEM_RADIUS],
                    endpoint_b: [centroid.x, centroid.y, centroid.z, 0.0],
                    color_a: [ring.color[0], ring.color[1], ring.color[2], 0.0],
                    color_b: [ring.color[0], ring.color[1], ring.color[2], 0.0],
                });
            } else {
                missing_p += 1;
            }

            // Hex ring (and pent ring on purines) share the coherent
            // base-plane normal so a purine's two fused rings never
            // disagree on which way the paddle faces.
            if !make_polygon_instance(
                &ring.hex_ring,
                ring.color,
                RING_HALF_THICKNESS,
                base_normal,
                &mut ring_instances,
            ) {
                degenerate_rings += 1;
            }
            if !ring.pent_ring.is_empty() {
                let _ = make_polygon_instance(
                    &ring.pent_ring,
                    ring.color,
                    RING_HALF_THICKNESS,
                    base_normal,
                    &mut ring_instances,
                );
            }
        }

        if degenerate_rings > 0 {
            log::warn!(
                "NA: {degenerate_rings} base ring(s) skipped \
                 (degenerate/out-of-range polygon) -- check for a mis-named \
                 base family"
            );
        }
        if missing_p > 0 {
            log::warn!(
                "NA: {missing_p} base(s) had no resolvable P atom; stem \
                 omitted (ring still drawn)"
            );
        }

        (stems, ring_instances)
    }
}

/// Build an `ExtrudedPolygonInstance` from a ring of 3-6 coplanar
/// positions using the caller-supplied (coherence-adjusted) plane
/// normal. Returns `false` without emitting if the vertex count is out
/// of range, so the caller can count and report the omission.
fn make_polygon_instance(
    positions: &[Vec3],
    color: [f32; 3],
    half_thickness: f32,
    normal: Vec3,
    out: &mut Vec<ExtrudedPolygonInstance>,
) -> bool {
    let n = positions.len();
    if !(3..=6).contains(&n) || normal == Vec3::ZERO {
        return false;
    }

    // Centroid: used CPU-side for degenerate vertex padding, and packed
    // into the otherwise-unused v2/v3/v4 `.w` slots so the vertex shader
    // reads it once rather than re-summing all `n` vertices for each of
    // the 72 verts per instance (T3-NA-C).
    let centroid: Vec3 = positions.iter().copied().sum::<Vec3>() / n as f32;

    // Pad to exactly 6 verts (unused slots = centroid = degenerate triangle)
    let v = |i: usize| -> Vec3 {
        if i < n {
            positions[i]
        } else {
            centroid
        }
    };

    out.push(ExtrudedPolygonInstance {
        v0: {
            let p = positions[0];
            [p.x, p.y, p.z, n as f32]
        },
        v1: {
            let p = positions[1];
            [p.x, p.y, p.z, half_thickness]
        },
        v2: {
            let p = v(2);
            [p.x, p.y, p.z, centroid.x]
        },
        v3: {
            let p = v(3);
            [p.x, p.y, p.z, centroid.y]
        },
        v4: {
            let p = v(4);
            [p.x, p.y, p.z, centroid.z]
        },
        v5: {
            let p = v(5);
            [p.x, p.y, p.z, 0.0]
        },
        normal: [normal.x, normal.y, normal.z, 0.0],
        color: [color[0], color[1], color[2], 0.0],
    });
    true
}

#[cfg(test)]
mod tests {
    use molex::NucleotideRing;

    use super::*;
    use crate::renderer::entity_topology::ResolvedRing;

    fn hexagon(z: f32) -> Vec<Vec3> {
        (0..6)
            .map(|i| {
                let a = (i as f32 / 6.0) * std::f32::consts::TAU;
                Vec3::new(a.cos(), a.sin(), z)
            })
            .collect()
    }

    fn resolved(hex: Vec<Vec3>, chain_idx: u32) -> ResolvedRing {
        ResolvedRing {
            ring: NucleotideRing {
                hex_ring: hex,
                pent_ring: Vec::new(),
                c1_prime: None,
                color: [0.5, 0.5, 0.5],
            },
            owning_p: Some(Vec3::ZERO),
            chain_idx,
        }
    }

    /// T0-NA-B: the Newell normal sign is fixed by atom traversal
    /// order, which is not chirality-stable across bases. Two co-stacked
    /// bases on one strand whose ring atoms wind in opposite directions
    /// must still emit paddle normals in the *same* hemisphere after the
    /// per-strand coherence pass -- otherwise adjacent paddles face
    /// opposite ways ("arbitrary directions").
    #[test]
    fn base_normals_coherent_across_reversed_winding() {
        let forward = hexagon(0.0);
        let mut reversed = hexagon(1.0);
        reversed.reverse(); // flips the raw Newell sign

        let rings = vec![resolved(forward, 0), resolved(reversed.clone(), 0)];
        let (_stems, polys) = NucleicAcidRenderer::generate_instances(&rings);
        assert_eq!(polys.len(), 2, "both hex rings should emit");

        let n0 = Vec3::new(
            polys[0].normal[0],
            polys[0].normal[1],
            polys[0].normal[2],
        );
        let n1 = Vec3::new(
            polys[1].normal[0],
            polys[1].normal[1],
            polys[1].normal[2],
        );
        assert!(
            n0.dot(n1) > 0.0,
            "reversed-winding base flipped the paddle normal (dot = {})",
            n0.dot(n1),
        );

        // A new strand resets coherence: the reversed base on its own
        // chain keeps its raw (opposite) sign, proving the alignment is
        // per-strand and not a global force.
        let split = vec![resolved(hexagon(0.0), 0), resolved(reversed, 1)];
        let (_s, polys2) = NucleicAcidRenderer::generate_instances(&split);
        let m0 = Vec3::new(
            polys2[0].normal[0],
            polys2[0].normal[1],
            polys2[0].normal[2],
        );
        let m1 = Vec3::new(
            polys2[1].normal[0],
            polys2[1].normal[1],
            polys2[1].normal[2],
        );
        assert!(
            m0.dot(m1) < 0.0,
            "coherence leaked across a chain boundary (dot = {})",
            m0.dot(m1),
        );
    }
}
