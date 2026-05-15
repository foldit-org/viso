//! Opt-in pipeline tracing for β-sheet ribbon geometry.
//!
//! Enable with `VISO_SHEET_TRACE=1`. For each strand it prints the
//! consecutive-normal coherence after every per-residue processing
//! stage, then locates the worst final-frame discontinuity in each
//! sheet region and dumps a per-stage breakdown around it, so the
//! stage that introduces a visible artifact can be identified directly
//! instead of inferred.

use std::fmt::Write as _;

use glam::Vec3;

use super::profile::CrossSectionProfile;
use super::spline::SplinePoint;

pub(crate) fn enabled() -> bool {
    std::env::var_os("VISO_SHEET_TRACE").is_some()
}

/// Compact string of consecutive `v[i]·v[i-1]`, flagging values below
/// 0.70 (a visibly incoherent broad-face transition).
fn consecutive_dots(v: &[Vec3]) -> String {
    let mut s = String::new();
    for pair in v.windows(2) {
        let d = pair[1].dot(pair[0]);
        let mark = if d < 0.70 { '!' } else { ' ' };
        let _ = write!(s, "{mark}{d:+.2}");
    }
    s
}

/// Per-residue normal coherence for one strand at each processing stage.
pub(crate) fn trace_strand_stages(
    global_base: u32,
    start: usize,
    raw: &[Vec3],
    after_signs: &[Vec3],
    after_flatten: &[Vec3],
    after_clamp: &[Vec3],
) {
    let a = global_base as usize + start;
    let b = a + raw.len().saturating_sub(1);
    log::info!(
        "[trace] strand res {a}..={b}  (normal·prev per residue; '!' means < \
         0.70)"
    );
    log::info!("[trace]   raw     {}", consecutive_dots(raw));
    log::info!("[trace]   signs   {}", consecutive_dots(after_signs));
    log::info!("[trace]   flatten {}", consecutive_dots(after_flatten));
    log::info!("[trace]   clamp   {}", consecutive_dots(after_clamp));
}

/// Locate the worst consecutive final-normal discontinuity within sheet
/// regions of one chain and print every stage's continuity around it.
#[allow(clippy::too_many_arguments)]
pub(crate) fn trace_final_frames(
    global_base: u32,
    n_res: usize,
    tangents: &[Vec3],
    rmf: &[SplinePoint],
    spline_sheet_normals: &[Vec3],
    final_frames: &[SplinePoint],
    profiles: &[CrossSectionProfile],
) {
    let total = final_frames.len();
    if total < 2 {
        return;
    }
    let res_at = |i: usize| {
        i as f32 / (total - 1).max(1) as f32 * n_res.saturating_sub(1) as f32
    };

    let mut worst: Option<(usize, f32)> = None;
    for i in 1..total {
        let in_sheet = profiles[i].sheet_blend > 0.05
            || profiles[i - 1].sheet_blend > 0.05;
        if !in_sheet {
            continue;
        }
        let d = final_frames[i].normal.dot(final_frames[i - 1].normal);
        if worst.is_none_or(|(_, wd)| d < wd) {
            worst = Some((i, d));
        }
    }

    let Some((w, wd)) = worst else {
        log::info!("[trace] chain base={global_base}: no sheet samples");
        return;
    };
    log::info!(
        "[trace] chain base={global_base} residues={n_res} spline={total} :: \
         worst sheet finalN·prev = {wd:+.3} at spline {w} (res≈{:.1})",
        res_at(w),
    );
    log::info!(
        "[trace]    i  res |  tan·p   rmf·p  shtN·p  finN·p  finB·p | sblend  \
         width"
    );
    let lo = w.saturating_sub(3);
    let hi = (w + 4).min(total);
    for i in lo..hi {
        let (tp, rp, sp, fnp, fbp) = if i > 0 {
            (
                tangents[i].dot(tangents[i - 1]),
                rmf[i].normal.dot(rmf[i - 1].normal),
                spline_sheet_normals[i].dot(spline_sheet_normals[i - 1]),
                final_frames[i].normal.dot(final_frames[i - 1].normal),
                final_frames[i].binormal.dot(final_frames[i - 1].binormal),
            )
        } else {
            (1.0, 1.0, 1.0, 1.0, 1.0)
        };
        let mark = if i == w { '*' } else { ' ' };
        log::info!(
            "[trace] {mark}{i:>3} {:>4.0} | {tp:+.3}  {rp:+.3}  {sp:+.3}  \
             {fnp:+.3}  {fbp:+.3} | {:.2}   {:.3}",
            res_at(i),
            profiles[i].sheet_blend,
            profiles[i].width,
        );
    }
}
