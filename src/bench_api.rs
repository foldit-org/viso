//! Bench-only entry points. `#[doc(hidden)]` and not part of the public
//! API: these exist solely so `benches/topology.rs` can drive the
//! `pub(crate)` per-entity topology derivation that
//! `SyncPipeline::sync_from_assembly` runs on every load and
//! topology-changing mutation. Do not call from production code.

use molex::Assembly;

use crate::engine::entity_view::derive_topology;

/// Rederive the render topology for every entity in `assembly`.
///
/// This is the per-entity work `sync_from_assembly` performs: the
/// `atom_elements` transpose, backbone/sidechain index layouts, residue
/// tables, and the bond/SS clone. Returns the total `atom_elements` count
/// so the work is not optimized away.
///
/// `ss_types` is read from the assembly as-is; populate it via
/// `Assembly::recompute_ss` before calling for a load-realistic SS clone.
#[must_use]
pub fn derive_topology_all(assembly: &Assembly) -> usize {
    let mut total = 0;
    for entity in assembly.entities() {
        let ss = assembly.ss_types(entity.id());
        let topology = std::hint::black_box(derive_topology(entity, ss));
        total += topology.atom_elements.len();
    }
    total
}
