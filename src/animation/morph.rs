//! Topology-morph sequencer: animate a residue mutation across its
//! atom-count change as three segments through a collapsed-backbone
//! waypoint, so no topology is ever resolved against a mismatched position
//! buffer.
//!
//! A mutation changes a residue's identity, so the old structure (A) and
//! new structure (B) have different atom counts and can't be interpolated
//! index-for-index. Instead the engine defers adopting B and drives, per
//! changed residue only (the rest of the structure is untouched):
//!
//!   1. COLLAPSE (on A): the changed residues' old sidechain atoms lerp into
//!      their CB stub (CA for glycine).
//!   2. EASE (on A, only when the backbone moved): the whole buffer lerps A ->
//!      B coordinates; changed residues stay collapsed at the moving stub.
//!      Unchanged atoms map B -> A by canonical per-residue offset. Skipped
//!      (`ease == 0`) for a backbone-fixed mutation.
//!   3. SWAP at the collapsed waypoint -- seamless, because there only the
//!      changed residues differ between A and B and both read as bare CB stubs
//!      -- then EXPAND the changed residues' new sidechains out of the stub via
//!      the normal queued-transition path.
//!
//! A newer snapshot arriving mid-sequence (before the swap) replaces the
//! deferred target rather than queueing; one arriving after the swap is
//! handled as a fresh morph once the expand completes.

use std::collections::HashMap;
use std::ops::Range;
use std::sync::Arc;
use std::time::Duration;

use glam::Vec3;
use molex::entity::molecule::id::EntityId;
use molex::{Assembly, MoleculeEntity};

use super::transition::Transition;
use crate::engine::VisoEngine;
use crate::util::easing::EasingFunction;

/// Which segment of the morph is currently animating.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum MorphSegment {
    /// Collapsing the changed residues' old sidechains, on the old
    /// structure (A).
    Collapse,
    /// Easing the backbone A -> B with changed residues held collapsed.
    Ease,
}

/// One entity participating in a morph, with the residue indices whose
/// identity changed between A and B.
#[derive(Clone)]
struct MorphEntity {
    id: EntityId,
    changed: Vec<u32>,
}

/// In-flight topology-morph sequence (see the module docs).
pub(crate) struct MorphSequence {
    /// Deferred target snapshot (B), adopted only at the swap. Overwritten
    /// by a newer pre-swap snapshot (coalesce-to-latest; never queued).
    target: Arc<Assembly>,
    entities: Vec<MorphEntity>,
    segment: MorphSegment,
    /// Backbone-ease segment duration; zero for a backbone-fixed mutation.
    ease: Duration,
    /// Final expand segment duration.
    expand: Duration,
}

impl VisoEngine {
    /// Intercept the snapshot-adoption path for a residue mutation. Returns
    /// `true` if a morph is in flight or was just begun (in which case it
    /// owns adoption and the caller must skip the normal path); `false` for
    /// every other update.
    pub(crate) fn advance_morph(&mut self) -> bool {
        if self.animation.morph.is_some() {
            self.drive_morph();
            true
        } else {
            self.detect_and_begin_morph()
        }
    }

    /// If the pending snapshot is a topology-morphing mutation, begin the
    /// deferred sequence (holding the target, kicking the collapse on the
    /// current structure) and return `true`. Otherwise leave the pending
    /// snapshot untouched for the normal path and return `false`.
    fn detect_and_begin_morph(&mut self) -> bool {
        let Some(pending) = self.scene.pending.clone() else {
            return false;
        };
        if pending.generation() == self.scene.last_seen_generation {
            return false;
        }

        // Read-only scan: which entities carry a morph transition and a real
        // residue-identity change with matching residue count?
        let mut entities: Vec<MorphEntity> = Vec::new();
        let (mut collapse, mut ease, mut expand) =
            (Duration::ZERO, Duration::ZERO, Duration::ZERO);
        for b in pending.entities() {
            let id = b.id();
            let Some(t) = self.animation.morph_transition(id.raw()) else {
                continue;
            };
            let Some((c, e, x)) = t.morph_durations() else {
                continue;
            };
            let Some(a) =
                self.scene.current.entities().iter().find(|e| e.id() == id)
            else {
                continue;
            };
            let Some(changed) = changed_residues(a, b) else {
                continue;
            };
            entities.push(MorphEntity { id, changed });
            // Share one timeline across entities; the longest ease wins so a
            // backbone-moving entity gets its full ease while fixed ones
            // simply don't move during it.
            collapse = collapse.max(c);
            ease = ease.max(e);
            expand = expand.max(x);
        }
        if entities.is_empty() {
            return false;
        }

        // Hold the target (deferred adoption) and kick the collapse on the
        // current structure -- no rebuild: A's topology is still cached, so
        // the per-frame animation path renders it as the sidechains fold in.
        let target = self.scene.pending.take().unwrap_or(pending);
        let collapse_t =
            Transition::eased(collapse, EasingFunction::QuadraticIn, true);
        for me in &entities {
            let Some(cur) =
                self.scene.positions.get(me.id).map(<[Vec3]>::to_vec)
            else {
                continue;
            };
            let Some(ranges) = self
                .scene
                .current
                .entities()
                .iter()
                .find(|e| e.id() == me.id)
                .and_then(|e| residue_ranges(e))
            else {
                continue;
            };
            let target_buf = build_collapse_target(&cur, &ranges, &me.changed);
            self.animation.animator.animate_entity(
                me.id,
                cur,
                target_buf,
                &collapse_t,
            );
        }
        self.animation.morph = Some(MorphSequence {
            target,
            entities,
            segment: MorphSegment::Collapse,
            ease,
            expand,
        });
        true
    }

    /// Advance an in-flight morph: coalesce a newer pending snapshot, and
    /// when the current segment's animation finishes, set up the next
    /// segment (or perform the swap and hand the expand to the normal path).
    fn drive_morph(&mut self) {
        self.coalesce_morph_target();
        let Some(morph) = self.animation.morph.as_ref() else {
            return;
        };
        let all_done = morph
            .entities
            .iter()
            .all(|me| !self.animation.animator.is_entity_animating(me.id));
        if !all_done {
            return;
        }
        let collapse_then_ease =
            morph.segment == MorphSegment::Collapse && !morph.ease.is_zero();
        if collapse_then_ease {
            self.morph_begin_ease();
        } else {
            self.morph_swap_and_expand();
        }
    }

    /// A newer snapshot arrived mid-sequence and the swap hasn't happened
    /// yet: retarget to the latest (never queue). Mid-ease, re-aim the ease
    /// at the new target ("change course").
    fn coalesce_morph_target(&mut self) {
        let newer =
            self.scene.pending.as_ref().is_some_and(|p| {
                p.generation() != self.scene.last_seen_generation
            });
        if !newer {
            return;
        }
        let Some(target) = self.scene.pending.take() else {
            return;
        };
        let in_ease = match self.animation.morph.as_mut() {
            Some(m) => {
                m.target = target;
                m.segment == MorphSegment::Ease
            }
            None => return,
        };
        if in_ease {
            self.morph_begin_ease();
        }
    }

    /// Set up segment 2: ease the buffer A -> B (changed residues held at
    /// the moving stub), re-kicking the animator per entity against the
    /// latest target.
    fn morph_begin_ease(&mut self) {
        let (target, ease, ids) = {
            let Some(m) = self.animation.morph.as_ref() else {
                return;
            };
            (
                Arc::clone(&m.target),
                m.ease,
                m.entities.iter().map(|e| e.id).collect::<Vec<_>>(),
            )
        };

        let mut kicks: Vec<(EntityId, Vec<Vec3>, Vec<Vec3>)> = Vec::new();
        let mut refreshed: Vec<(EntityId, Vec<u32>)> = Vec::new();
        for id in ids {
            let (Some(a), Some(b)) = (
                self.scene.current.entities().iter().find(|e| e.id() == id),
                target.entities().iter().find(|e| e.id() == id),
            ) else {
                continue;
            };
            let Some(changed) = changed_residues(a, b) else {
                continue;
            };
            let (Some(a_ranges), Some(b_ranges)) =
                (residue_ranges(a), residue_ranges(b))
            else {
                continue;
            };
            let b_pos = b.positions();
            let cur = self
                .scene
                .positions
                .get(id)
                .map(<[Vec3]>::to_vec)
                .unwrap_or_default();
            let ease_target =
                build_ease_target(&a_ranges, &b_ranges, &b_pos, &changed);
            kicks.push((id, cur, ease_target));
            refreshed.push((id, changed));
        }

        let ease_t = Transition::eased(ease, EasingFunction::DEFAULT, true);
        for (id, start, target_buf) in kicks {
            self.animation
                .animator
                .animate_entity(id, start, target_buf, &ease_t);
        }
        if let Some(m) = self.animation.morph.as_mut() {
            m.segment = MorphSegment::Ease;
            for (id, changed) in refreshed {
                if let Some(me) = m.entities.iter_mut().find(|e| e.id == id) {
                    me.changed = changed;
                }
            }
        }
    }

    /// Adopt the deferred target at the collapsed waypoint, seed the changed
    /// residues' new sidechains at their stub, and hand the expand off to
    /// the normal queued-transition path. Ends the sequence.
    fn morph_swap_and_expand(&mut self) {
        let Some(morph) = self.animation.morph.take() else {
            return;
        };
        let target = morph.target;

        // Recompute changed residues (pre-swap current vs latest target)
        // before adopting B, since adoption replaces the current structure.
        let mut plan: Vec<(EntityId, Vec<u32>)> = Vec::new();
        for me in &morph.entities {
            let (Some(a), Some(b)) = (
                self.scene
                    .current
                    .entities()
                    .iter()
                    .find(|e| e.id() == me.id),
                target.entities().iter().find(|e| e.id() == me.id),
            ) else {
                continue;
            };
            if let Some(changed) = changed_residues(a, b) {
                plan.push((me.id, changed));
            }
        }

        // Adopt B (this is the deferred `poll_assembly` body).
        self.sync_from_assembly(&target);
        self.scene.current = target;
        self.scene.last_seen_generation = self.scene.current.generation();

        // Seed each changed residue's new sidechain at its stub so the
        // expand lerps it outward; queue the expand transition per entity.
        let expand_t =
            Transition::eased(morph.expand, EasingFunction::QuadraticOut, true);
        let mut transitions: HashMap<u32, Transition> = HashMap::new();
        for (id, changed) in &plan {
            let ranges = self
                .scene
                .current
                .entities()
                .iter()
                .find(|e| e.id() == *id)
                .and_then(|e| residue_ranges(e));
            if let Some(ranges) = ranges {
                if let Some(pos) = self.scene.positions.get_mut(*id) {
                    seed_expand(pos, &ranges, changed);
                }
            }
            let _ = transitions.insert(id.raw(), expand_t.clone());
        }

        // Adopt any deferred non-morph transitions too; expand wins on key.
        let mut staged = self.animation.take_pending_transitions();
        staged.extend(transitions);
        self.sync_scene_to_renderers(staged);
    }
}

// ---------------------------------------------------------------------------
// Pure helpers
// ---------------------------------------------------------------------------

/// Residue indices whose identity differs between A and B, or `None` when
/// the two can't be aligned residue-for-residue (no residues, or a
/// residue-count change / indel -- the morph bails to a snap there) or when
/// nothing changed.
fn changed_residues(
    a: &MoleculeEntity,
    b: &MoleculeEntity,
) -> Option<Vec<u32>> {
    let (ar, br) = (a.residues()?, b.residues()?);
    if ar.len() != br.len() {
        return None;
    }
    let changed: Vec<u32> = ar
        .iter()
        .zip(br.iter())
        .enumerate()
        .filter(|(_, (x, y))| x.name != y.name)
        .map(|(i, _)| i as u32)
        .collect();
    (!changed.is_empty()).then_some(changed)
}

/// Per-residue atom-index ranges for a polymer entity, or `None` for a
/// non-polymer (no residues to morph).
fn residue_ranges(entity: &MoleculeEntity) -> Option<Vec<Range<usize>>> {
    Some(
        entity
            .residues()?
            .iter()
            .map(|r| r.atom_range.clone())
            .collect(),
    )
}

/// Entity-local atom index of a residue's collapse stub: CB (canonical
/// offset 4) when present, else CA (offset 1) for glycine's 4-atom residue.
fn stub_index(r: &Range<usize>) -> usize {
    if r.end - r.start >= 5 {
        r.start + 4
    } else {
        r.start + 1
    }
}

/// Segment-1 target buffer (A's layout): clone the current positions, then
/// fold each changed residue's sidechain atoms (offset 4+) into its stub.
/// Backbone and unchanged residues are left exactly where they are.
fn build_collapse_target(
    cur_pos: &[Vec3],
    a_ranges: &[Range<usize>],
    changed: &[u32],
) -> Vec<Vec3> {
    let mut target = cur_pos.to_vec();
    for &ri in changed {
        let Some(r) = a_ranges.get(ri as usize) else {
            continue;
        };
        let Some(&stub) = cur_pos.get(stub_index(r)) else {
            continue;
        };
        for atom in (r.start + 4)..r.end {
            if let Some(slot) = target.get_mut(atom) {
                *slot = stub;
            }
        }
    }
    target
}

/// Segment-2 target buffer (A's layout) mapping every atom to its B
/// position. Unchanged residues map B -> A by canonical per-residue offset
/// (atom `j` of residue `k` is the same atom on both sides). Changed
/// residues keep their backbone (offsets 0-3) but hold every sidechain atom
/// (offset 4+) at B's stub, so they stay collapsed through the ease.
fn build_ease_target(
    a_ranges: &[Range<usize>],
    b_ranges: &[Range<usize>],
    b_pos: &[Vec3],
    changed: &[u32],
) -> Vec<Vec3> {
    let a_atom_count = a_ranges.last().map_or(0, |r| r.end);
    let mut target = vec![Vec3::ZERO; a_atom_count];
    let changed: std::collections::HashSet<u32> =
        changed.iter().copied().collect();
    for (ri, (ra, rb)) in a_ranges.iter().zip(b_ranges.iter()).enumerate() {
        if changed.contains(&(ri as u32)) {
            let b_stub =
                b_pos.get(stub_index(rb)).copied().unwrap_or(Vec3::ZERO);
            for j in 0..(ra.end - ra.start) {
                let pos = if j < 4 {
                    b_pos.get(rb.start + j).copied().unwrap_or(b_stub)
                } else {
                    b_stub
                };
                if let Some(slot) = target.get_mut(ra.start + j) {
                    *slot = pos;
                }
            }
        } else {
            for j in 0..(ra.end - ra.start) {
                if let (Some(slot), Some(&p)) =
                    (target.get_mut(ra.start + j), b_pos.get(rb.start + j))
                {
                    *slot = p;
                }
            }
        }
    }
    target
}

/// Seed each changed residue's new sidechain atoms (offset 4+) at its stub,
/// in B's layout, so the expand segment can lerp them outward to their real
/// B positions.
fn seed_expand(
    positions: &mut [Vec3],
    b_ranges: &[Range<usize>],
    changed: &[u32],
) {
    for &ri in changed {
        let Some(rb) = b_ranges.get(ri as usize) else {
            continue;
        };
        let Some(stub) = positions.get(stub_index(rb)).copied() else {
            continue;
        };
        for atom in (rb.start + 4)..rb.end {
            if let Some(slot) = positions.get_mut(atom) {
                *slot = stub;
            }
        }
    }
}
