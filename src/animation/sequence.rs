//! Structural-delta animation: derive and play a per-publish `Animation`
//! that carries the scene from the current assembly (A) to a freshly
//! published one (B).
//!
//! Classification is delegated to [`molex::MoleculeEntity::diff`] (the
//! shared "delta between two states" primitive): a `MutateResidue` edit
//! marks a residue whose identity changed, a coord edit marks a plain
//! conformation move, and an `Err` (residue insert/delete) means snap.
//! Only the *geometry* of the animation — folding sidechains onto their
//! stub, holding them through a backbone ease — lives here.
//!
//! [`build_animation`] returns an ordered list of `AnimationStep`s, or
//! `None` (snap). A plain conformation change is a single
//! `AnimationStep::Lerp`. A residue mutation, whose atom set differs
//! between A and B, is sequenced through a collapsed-backbone waypoint so no
//! topology is ever resolved against a mismatched position buffer:
//!
//!   1. collapse the changed residues' old sidechains onto their CB stub,
//!   2. ease the backbone A -> B (only when it actually moved),
//!   3. `AnimationStep::AdoptTarget`: swap the structural source to B at the
//!      waypoint, where the changed residues read as bare stubs on both sides,
//!      so the atom-count change is invisible,
//!   4. expand the changed residues' new sidechains out of the stub.
//!
//! Each step owns its own target buffer; the state the expand grows out of
//! is just `collapse_sidechains` applied to B — the same fold the collapse
//! step applies to A.
//!
//! [`AnimationPlayer`] is the only in-flight state. It installs per-entity
//! runners on the [`StructureAnimator`] for `Lerp`s and signals the engine
//! to adopt B for an `AdoptTarget`. It holds no GPU handles: adoption is the
//! engine's existing rebuild path, driven off the assembly swap.

use std::ops::Range;
use std::sync::Arc;
use std::time::Duration;

use glam::Vec3;
use molex::chemistry::amino_acids::AminoAcid;
use molex::entity::molecule::id::EntityId;
use molex::ops::edit::AssemblyEdit;
use molex::{Assembly, MoleculeEntity};
use rustc_hash::FxHashMap;
use web_time::Instant;

use super::transition::Transition;
use super::StructureAnimator;
use crate::engine::scene::Scene;
use crate::util::easing::EasingFunction;

/// Largest CA displacement (Angstroms) still treated as "no backbone
/// movement". Streamed poses carry tiny float jitter even when the backbone
/// is held fixed; this keeps that jitter from adding a spurious ease step to
/// a sidechain-only edit.
const CA_EPSILON: f32 = 1e-3;

/// Plain conformation-ease duration (matches [`Transition::smooth`]).
const SMOOTH_MS: u64 = 300;
/// Collapse-segment duration for a mutation.
const COLLAPSE_MS: u64 = 150;
/// Backbone-ease-segment duration for a mutation.
const EASE_MS: u64 = 200;
/// Expand-segment duration for a mutation.
const EXPAND_MS: u64 = 150;

/// Whether a residue name is glycine, the only canonical residue with no
/// sidechain heavy atoms. Case-insensitive via molex's code parser.
fn is_glycine(name: [u8; 3]) -> bool {
    AminoAcid::from_code(name) == Some(AminoAcid::Gly)
}

/// Residue indices that changed identity between `a` and `b`, read off the
/// shared [`MoleculeEntity::diff`] primitive (the `MutateResidue` edits).
/// `Err` (a residue insert/delete, or incompatible entities) propagates as
/// `None` so the caller snaps.
fn mutated_residues(
    a: &MoleculeEntity,
    b: &MoleculeEntity,
) -> Option<(Vec<u32>, bool)> {
    let edits = a.diff(b).ok()?;
    let changed: Vec<u32> = edits
        .iter()
        .filter_map(|e| match e {
            AssemblyEdit::MutateResidue { residue_idx, .. } => {
                Some(*residue_idx as u32)
            }
            _ => None,
        })
        .collect();
    let any_coord_change = !edits.is_empty();
    Some((changed, any_coord_change))
}

/// One step of an [`Animation`].
pub(crate) enum AnimationStep {
    /// Interpolate the listed entities' positions to `targets` over
    /// `duration` with `easing`. Entities absent from the map hold still.
    Lerp {
        targets: FxHashMap<EntityId, Vec<Vec3>>,
        duration: Duration,
        easing: EasingFunction,
    },
    /// Swap the scene's structural source to this assembly (a real
    /// rebuild/adoption via the engine's normal publish path). Sits at the
    /// collapsed waypoint of a mutation, where the atom-count change is
    /// invisible.
    AdoptTarget(Arc<Assembly>),
}

/// An ordered animation produced by [`build_animation`].
pub(crate) type Animation = Vec<AnimationStep>;

// ---------------------------------------------------------------------------
// Builder
// ---------------------------------------------------------------------------

/// Pure builder: derive the animation carrying `current` (A) to `new` (B),
/// or `None` when the change should snap (a residue insert/delete, or
/// nothing actually moved).
///
/// Whole-entity gates, evaluated per changed entity then aggregated: the
/// collapse runs when any changed residue had an old (non-glycine)
/// sidechain; the ease runs when any changed entity's backbone moved past
/// [`CA_EPSILON`]; the expand runs when any changed residue gains a new
/// (non-glycine) sidechain. The `AdoptTarget` swap is present whenever any
/// residue changed identity.
pub(crate) fn build_animation(
    current: &Assembly,
    new: &Arc<Assembly>,
) -> Option<Animation> {
    struct Mutated {
        id: EntityId,
        changed: Vec<u32>,
        a_pos: Vec<Vec3>,
        a_ranges: Vec<Range<usize>>,
        b_pos: Vec<Vec3>,
        b_ranges: Vec<Range<usize>>,
        backbone_moved: bool,
        has_old_sidechain: bool,
        has_new_sidechain: bool,
    }

    let mut mutated: Vec<Mutated> = Vec::new();
    let mut moved_only: Vec<(EntityId, Vec<Vec3>)> = Vec::new();

    for b in new.entities() {
        let id = b.id();
        let Some(a) = current.entities().iter().find(|e| e.id() == id) else {
            // New entity: handled by the up-front adopt, not per-entity
            // animation.
            continue;
        };
        // `None` means the diff was non-representable (an indel): snap.
        let (changed, any_coord_change) = mutated_residues(a, b)?;
        if changed.is_empty() {
            if any_coord_change {
                moved_only.push((id, b.positions()));
            }
            continue;
        }
        let (Some(a_res), Some(b_res)) = (a.residues(), b.residues()) else {
            return None;
        };
        let has_old_sidechain = changed.iter().any(|&ri| {
            a_res.get(ri as usize).is_some_and(|r| !is_glycine(r.name))
        });
        let has_new_sidechain = changed.iter().any(|&ri| {
            b_res.get(ri as usize).is_some_and(|r| !is_glycine(r.name))
        });
        mutated.push(Mutated {
            a_ranges: a_res.iter().map(|r| r.atom_range.clone()).collect(),
            b_ranges: b_res.iter().map(|r| r.atom_range.clone()).collect(),
            a_pos: a.positions(),
            b_pos: b.positions(),
            backbone_moved: backbone_moved(a, b),
            has_old_sidechain,
            has_new_sidechain,
            id,
            changed,
        });
    }

    if mutated.is_empty() {
        if moved_only.is_empty() {
            return None;
        }
        // Plain conformation change: one ease toward B.
        let targets: FxHashMap<EntityId, Vec<Vec3>> =
            moved_only.into_iter().collect();
        return Some(vec![AnimationStep::Lerp {
            targets,
            duration: Duration::from_millis(SMOOTH_MS),
            easing: EasingFunction::DEFAULT,
        }]);
    }

    let collapse_step = mutated.iter().any(|m| m.has_old_sidechain);
    let ease_step = mutated.iter().any(|m| m.backbone_moved);
    let expand_step = mutated.iter().any(|m| m.has_new_sidechain);

    let mut steps: Animation = Vec::new();

    if collapse_step {
        let targets = mutated
            .iter()
            .map(|m| {
                (m.id, collapse_sidechains(&m.a_pos, &m.a_ranges, &m.changed))
            })
            .collect();
        steps.push(AnimationStep::Lerp {
            targets,
            duration: Duration::from_millis(COLLAPSE_MS),
            easing: EasingFunction::QuadraticIn,
        });
    }

    if ease_step {
        let targets = mutated
            .iter()
            .map(|m| {
                (
                    m.id,
                    build_ease_target(
                        &m.a_ranges,
                        &m.b_ranges,
                        &m.b_pos,
                        &m.changed,
                    ),
                )
            })
            .collect();
        steps.push(AnimationStep::Lerp {
            targets,
            duration: Duration::from_millis(EASE_MS),
            easing: EasingFunction::DEFAULT,
        });
    }

    steps.push(AnimationStep::AdoptTarget(Arc::clone(new)));

    if expand_step {
        let targets = mutated.iter().map(|m| (m.id, m.b_pos.clone())).collect();
        steps.push(AnimationStep::Lerp {
            targets,
            duration: Duration::from_millis(EXPAND_MS),
            easing: EasingFunction::QuadraticOut,
        });
    }

    Some(steps)
}

// ---------------------------------------------------------------------------
// Player
// ---------------------------------------------------------------------------

/// What the engine must do after [`AnimationPlayer::advance`].
pub(crate) enum Advance {
    /// A `Lerp` is feeding the animator (or the player idled this tick).
    /// Nothing for the engine to do beyond its usual per-frame tick.
    Running,
    /// The cursor reached an `AdoptTarget`: the engine must adopt this
    /// assembly via its normal publish/rebuild path and then call
    /// [`AnimationPlayer::seed_adopted`] before the next tick, so the
    /// upcoming expand starts from the collapsed stub.
    Adopt(Arc<Assembly>),
    /// The sequence finished; the engine drops the player.
    Done,
}

/// The only in-flight animation state: the step list plus a cursor.
pub(crate) struct AnimationPlayer {
    animation: Animation,
    cursor: usize,
    /// Whether the runners for the current `Lerp` step have been installed.
    installed: bool,
    /// Changed residues per entity captured at the `AdoptTarget` step (from
    /// A, before the swap) so the post-swap fold can re-collapse them for
    /// the expand.
    seed_plan: Vec<(EntityId, Vec<u32>)>,
}

impl AnimationPlayer {
    /// Begin playing `animation` from its first step.
    pub(crate) fn new(animation: Animation) -> Self {
        Self {
            animation,
            cursor: 0,
            installed: false,
            seed_plan: Vec::new(),
        }
    }

    /// Whether the animation contains a deferred `AdoptTarget` (a topology
    /// swap), as opposed to a plain same-topology ease.
    pub(crate) fn has_adopt(&self) -> bool {
        self.animation
            .iter()
            .any(|s| matches!(s, AnimationStep::AdoptTarget(_)))
    }

    /// Whether the cursor has passed the `AdoptTarget`, i.e. the player is
    /// in its expand tail. A new target arriving here is deferred to a fresh
    /// animation rather than re-aimed (coalesce rule).
    pub(crate) fn past_adopt(&self) -> bool {
        self.animation
            .iter()
            .position(|s| matches!(s, AnimationStep::AdoptTarget(_)))
            .is_some_and(|i| self.cursor > i)
    }

    /// Drive the active step. Called once per `update` tick from the single
    /// engine call site.
    pub(crate) fn advance(
        &mut self,
        animator: &mut StructureAnimator,
        scene: &Scene,
        _now: Instant,
    ) -> Advance {
        loop {
            let Some(step) = self.animation.get(self.cursor) else {
                return Advance::Done;
            };
            match step {
                AnimationStep::Lerp {
                    targets,
                    duration,
                    easing,
                } => {
                    if !self.installed {
                        install_lerp(
                            animator, scene, targets, *duration, *easing,
                        );
                        self.installed = true;
                    }
                    if targets
                        .keys()
                        .any(|id| animator.is_entity_animating(*id))
                    {
                        return Advance::Running;
                    }
                    // Step complete (or a no-op that installed no runners):
                    // advance and process the next step this same tick.
                    self.cursor += 1;
                    self.installed = false;
                }
                AnimationStep::AdoptTarget(b) => {
                    // Capture which residues to re-collapse for the expand,
                    // from the still-current A, before the engine swaps to B.
                    self.seed_plan = plan_seed(scene, b);
                    let adopt = Arc::clone(b);
                    self.cursor += 1;
                    self.installed = false;
                    return Advance::Adopt(adopt);
                }
            }
        }
    }

    /// After the engine has adopted B (its positions reset to full B), fold
    /// each changed residue's new sidechain back onto its stub so the next
    /// `Lerp` (the expand) starts collapsed and grows the sidechain out.
    pub(crate) fn seed_adopted(&self, scene: &mut Scene) {
        for (id, changed) in &self.seed_plan {
            let Some(ranges) = scene
                .current
                .entities()
                .iter()
                .find(|e| e.id() == *id)
                .and_then(|e| e.residues())
                .map(|res| {
                    res.iter().map(|r| r.atom_range.clone()).collect::<Vec<_>>()
                })
            else {
                continue;
            };
            let Some(pos) = scene.positions.get(*id) else {
                continue;
            };
            let collapsed = collapse_sidechains(pos, &ranges, changed);
            scene.positions.set(*id, collapsed);
        }
    }
}

/// Install a runner per target entity, interpolating its current visible
/// positions toward the step's target over `duration` with `easing`.
fn install_lerp(
    animator: &mut StructureAnimator,
    scene: &Scene,
    targets: &FxHashMap<EntityId, Vec<Vec3>>,
    duration: Duration,
    easing: EasingFunction,
) {
    let transition = Transition::eased(duration, easing, true);
    for (id, target) in targets {
        let start = scene
            .positions
            .get(*id)
            .map(<[Vec3]>::to_vec)
            .unwrap_or_default();
        animator.animate_entity(*id, start, target.clone(), &transition);
    }
}

/// The changed-residue plan for the upcoming expand: which residues to fold
/// back onto their stub, per entity, computed from the still-current A
/// against the about-to-be-adopted B.
fn plan_seed(scene: &Scene, b: &Assembly) -> Vec<(EntityId, Vec<u32>)> {
    b.entities()
        .iter()
        .filter_map(|be| {
            let id = be.id();
            let a = scene.current.entities().iter().find(|e| e.id() == id)?;
            let (changed, _) = mutated_residues(a, be)?;
            (!changed.is_empty()).then_some((id, changed))
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Geometry helpers (animation policy; the buffer math)
// ---------------------------------------------------------------------------

/// Entity-local atom index of a residue's collapse stub: CB (canonical
/// offset 4) when present, else CA (offset 1) for glycine's 4-atom residue.
fn stub_index(r: &Range<usize>) -> usize {
    if r.end - r.start >= 5 {
        r.start + 4
    } else {
        r.start + 1
    }
}

/// True when residue counts match but at least one CA moved past
/// [`CA_EPSILON`]. Per-residue CA comes from molex's canonical
/// `ProteinEntity::to_backbone`; if the two backbones can't be aligned
/// (length mismatch, non-protein) we report "not moved".
fn backbone_moved(prev: &MoleculeEntity, new: &MoleculeEntity) -> bool {
    if prev.residue_count() != new.residue_count() {
        return false;
    }
    let (Some(pp), Some(np)) = (prev.as_protein(), new.as_protein()) else {
        return false;
    };
    let (pb, nb) = (pp.to_backbone(), np.to_backbone());
    if pb.len() != nb.len() {
        return false;
    }
    pb.iter()
        .zip(nb.iter())
        .any(|(a, b)| a.ca.distance(b.ca) > CA_EPSILON)
}

/// Fold each changed residue's sidechain atoms (offset 4+) onto its stub,
/// in the given layout. Used both for the collapse step's target (on A) and
/// for the state the expand grows out of (on B) — the same fold either way.
/// Backbone and unchanged residues are left exactly where they are; glycine
/// changed residues have no sidechain atoms and contribute nothing.
fn collapse_sidechains(
    positions: &[Vec3],
    ranges: &[Range<usize>],
    changed: &[u32],
) -> Vec<Vec3> {
    let mut target = positions.to_vec();
    for &ri in changed {
        let Some(r) = ranges.get(ri as usize) else {
            continue;
        };
        let Some(&stub) = positions.get(stub_index(r)) else {
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

/// Ease target (A's layout) mapping every atom to its B position. Unchanged
/// residues map B -> A by canonical per-residue offset (atom `j` of residue
/// `k` is the same atom on both sides). Changed residues keep their backbone
/// (offsets 0-3) but hold every sidechain atom (offset 4+) at B's stub, so
/// they stay collapsed through the ease.
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
