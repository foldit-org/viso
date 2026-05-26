//! Beta-sheet arrowhead shaping, split out of `mesh.rs`.
//!
//! Widens/narrows the C-terminal residues of each physical strand so the
//! ribbon renders an arrowhead at the strand -> non-strand transition.

use molex::SSType;

use super::profile::CrossSectionProfile;
use crate::options::GeometryOptions;

/// Maximum number of consecutive non-sheet residues that is still
/// treated as the interior of the same physical strand. Strands are
/// frequently split into several Sheet runs by short classification
/// breaks; an arrowhead belongs only at the strand's true C-terminus,
/// not at every internal break.
const MAX_INTERIOR_GAP: usize = 2;

/// Widen and narrow the C-terminal residues of each physical beta-strand
/// to create an arrowhead at the strand -> non-strand transition.
///
/// Short interior gaps (<= [`MAX_INTERIOR_GAP`] non-sheet residues with
/// sheet resuming after) are bridged so the arrowhead is placed once, at
/// the last sheet residue of the strand:
/// - that residue (the arrow point): width -> 0.05
/// - the preceding sheet residue (the arrow shoulder): width x 1.5
pub(super) fn apply_sheet_arrows(
    ss_types: &[SSType],
    profiles: &mut [CrossSectionProfile],
    geo: &GeometryOptions,
) {
    let n = ss_types.len();
    if n == 0 {
        return;
    }

    let is_sheet = |k: usize| ss_types[k] == SSType::Sheet;

    let mut i = 0;
    while i < n {
        if !is_sheet(i) {
            i += 1;
            continue;
        }

        // Walk to the physical strand's C-terminus, stepping across
        // short interior gaps where sheet resumes within
        // MAX_INTERIOR_GAP.
        let strand_start = i;
        let mut arrow_point = i;
        i += 1;
        loop {
            while i < n && is_sheet(i) {
                arrow_point = i;
                i += 1;
            }
            let gap_end = (i + MAX_INTERIOR_GAP).min(n);
            match (i..gap_end).find(|&k| is_sheet(k)) {
                Some(k) => i = k,
                None => break,
            }
        }

        // Shoulder: the sheet residue immediately preceding the arrow
        // point (skipping any interior-gap residues between them).
        if arrow_point > strand_start {
            let mut shoulder = arrow_point - 1;
            while shoulder > strand_start && !is_sheet(shoulder) {
                shoulder -= 1;
            }
            if is_sheet(shoulder) {
                profiles[shoulder].width = geo.sheet_width * 1.5;
            }
        }
        profiles[arrow_point].width = 0.05;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sheet_profiles(
        geo: &GeometryOptions,
        n: usize,
    ) -> Vec<CrossSectionProfile> {
        (0..n)
            .map(|_| CrossSectionProfile {
                width: geo.sheet_width,
                thickness: geo.sheet_thickness,
                roundness: geo.sheet_roundness,
                radial_blend: 0.0,
                sheet_blend: 1.0,
                color: [0.5, 0.5, 0.5],
                residue_idx: 0,
            })
            .collect()
    }

    /// A short interior break that splits one physical strand into two
    /// Sheet runs must not produce a mid-strand arrowhead -- the
    /// narrowing belongs only at the strand's true C-terminus.
    #[test]
    fn sheet_arrow_bridges_interior_gap() {
        use molex::SSType::{Coil, Sheet};
        let geo = GeometryOptions::default();
        // strand spans residues 1..=7, interior gap (Coil) at residue 4.
        let ss = [
            Coil, Sheet, Sheet, Sheet, Coil, Sheet, Sheet, Sheet, Coil, Coil,
            Coil,
        ];
        let mut profiles = sheet_profiles(&geo, ss.len());
        apply_sheet_arrows(&ss, &mut profiles, &geo);

        // Narrow point at the true end, widened shoulder just before it.
        assert_eq!(profiles[7].width, 0.05, "arrow point at C-terminus");
        assert_eq!(
            profiles[6].width,
            geo.sheet_width * 1.5,
            "shoulder before the arrow point",
        );
        // The interior run-end (residue 3) must be untouched.
        assert_eq!(
            profiles[3].width, geo.sheet_width,
            "no mid-strand narrowing at the interior break",
        );
        assert_eq!(profiles[2].width, geo.sheet_width);
    }

    /// A genuine strand separation (long non-sheet stretch) still gets an
    /// arrowhead per strand.
    #[test]
    fn sheet_arrow_separates_distinct_strands() {
        use molex::SSType::{Coil, Sheet};
        let geo = GeometryOptions::default();
        let ss = [Sheet, Sheet, Sheet, Coil, Coil, Coil, Sheet, Sheet, Sheet];
        let mut profiles = sheet_profiles(&geo, ss.len());
        apply_sheet_arrows(&ss, &mut profiles, &geo);

        assert_eq!(profiles[2].width, 0.05);
        assert_eq!(profiles[1].width, geo.sheet_width * 1.5);
        assert_eq!(profiles[8].width, 0.05);
        assert_eq!(profiles[7].width, geo.sheet_width * 1.5);
    }

    /// A non-sheet gap longer than `MAX_INTERIOR_GAP` is a real strand
    /// break: each side is its own strand and gets its own arrowhead.
    #[test]
    fn sheet_arrow_splits_on_wide_gap() {
        use molex::SSType::{Coil, Sheet};
        let geo = GeometryOptions::default();
        // Gap of MAX_INTERIOR_GAP + 1 Coil residues between two strands.
        let ss = [Sheet, Sheet, Sheet, Coil, Coil, Coil, Sheet, Sheet, Sheet];
        assert!(ss[3..6].len() > MAX_INTERIOR_GAP);
        let mut profiles = sheet_profiles(&geo, ss.len());
        apply_sheet_arrows(&ss, &mut profiles, &geo);

        assert_eq!(profiles[2].width, 0.05);
        assert_eq!(profiles[8].width, 0.05);
        assert_eq!(profiles[5].width, geo.sheet_width);
    }

    /// A single-residue strand has no room for a shoulder: only the
    /// arrow point is narrowed, and indexing must not underflow.
    #[test]
    fn sheet_arrow_single_residue_strand() {
        use molex::SSType::{Coil, Sheet};
        let geo = GeometryOptions::default();
        let ss = [Coil, Sheet, Coil];
        let mut profiles = sheet_profiles(&geo, ss.len());
        apply_sheet_arrows(&ss, &mut profiles, &geo);

        assert_eq!(profiles[1].width, 0.05);
        assert_eq!(profiles[0].width, geo.sheet_width);
        assert_eq!(profiles[2].width, geo.sheet_width);
    }
}
