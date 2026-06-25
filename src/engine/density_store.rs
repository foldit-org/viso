//! Storage for electron density maps loaded into the engine.
//!
//! Density maps are not entities (no atoms, residues, chains). They get
//! their own store, analogous to `ConstraintSpecs` for bands/pulls.

use molex::entity::surface::Density;

/// Default sigma for computing the initial threshold at load time.
const DEFAULT_SIGMA: f32 = 3.0;

/// Default density mesh color (blue).
const DEFAULT_COLOR: [f32; 3] = [0.3, 0.5, 0.8];

/// Default density opacity.
const DEFAULT_OPACITY: f32 = 0.35;

/// A single density map entry with display parameters.
pub(crate) struct DensityEntry {
    /// The parsed density map (owns the 3D grid).
    pub(crate) map: Density,
    /// Raw density threshold for isosurface extraction.
    pub(crate) threshold: f32,
    /// Whether this map is visible.
    pub(crate) visible: bool,
    /// Mesh color RGB.
    pub(crate) color: [f32; 3],
    /// Mesh opacity (0.0 = fully transparent, 1.0 = opaque).
    pub(crate) opacity: f32,
}

/// Manages all loaded density maps with dirty tracking.
pub(crate) struct DensityStore {
    entries: Vec<(u32, DensityEntry)>,
    next_id: u32,
}

impl DensityStore {
    /// Create an empty density store.
    pub(crate) fn new() -> Self {
        Self {
            entries: Vec::new(),
            next_id: 0,
        }
    }

    /// Add a density map. Returns the assigned ID.
    pub(crate) fn add(&mut self, map: Density) -> u32 {
        let id = self.next_id;
        self.next_id += 1;
        let threshold = map.sigma_level(DEFAULT_SIGMA);
        self.entries.push((
            id,
            DensityEntry {
                map,
                threshold,
                visible: true,
                color: DEFAULT_COLOR,
                opacity: DEFAULT_OPACITY,
            },
        ));
        id
    }

    /// Remove a density map by ID. Returns `true` if found.
    pub(crate) fn remove(&mut self, id: u32) -> bool {
        let len = self.entries.len();
        self.entries.retain(|(eid, _)| *eid != id);
        self.entries.len() != len
    }

    /// Get a density entry by ID.
    pub(crate) fn get(&self, id: u32) -> Option<&DensityEntry> {
        self.entries
            .iter()
            .find(|(eid, _)| *eid == id)
            .map(|(_, e)| e)
    }

    /// Get a mutable density entry by ID.
    fn get_mut(&mut self, id: u32) -> Option<&mut DensityEntry> {
        self.entries
            .iter_mut()
            .find(|(eid, _)| *eid == id)
            .map(|(_, e)| e)
    }

    /// Set the raw density threshold for a density map.
    pub(crate) fn set_threshold(&mut self, id: u32, threshold: f32) {
        if let Some(entry) = self.get_mut(id) {
            entry.threshold = threshold;
        }
    }

    /// Set visibility for a density map.
    pub(crate) fn set_visible(&mut self, id: u32, visible: bool) {
        if let Some(entry) = self.get_mut(id) {
            entry.visible = visible;
        }
    }

    /// Set color for a density map.
    pub(crate) fn set_color(&mut self, id: u32, color: [f32; 3]) {
        if let Some(entry) = self.get_mut(id) {
            entry.color = color;
        }
    }

    /// Set opacity for a density map (0.0–1.0).
    pub(crate) fn set_opacity(&mut self, id: u32, opacity: f32) {
        if let Some(entry) = self.get_mut(id) {
            entry.opacity = opacity.clamp(0.0, 1.0);
        }
    }

    /// All density entries (visible and hidden).
    pub(crate) fn all_entries(
        &self,
    ) -> impl Iterator<Item = (u32, &DensityEntry)> {
        self.entries.iter().map(|(id, e)| (*id, e))
    }

    /// All visible density entries.
    pub(crate) fn visible_entries(
        &self,
    ) -> impl Iterator<Item = (u32, &DensityEntry)> {
        self.entries
            .iter()
            .filter(|(_, e)| e.visible)
            .map(|(id, e)| (*id, e))
    }
}
