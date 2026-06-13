use std::collections::HashMap;

use serde::{Deserialize, Serialize};

/// Color palette options for molecular visualization.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct ColorOptions {
    /// RGB tint applied to lipid carbon atoms.
    pub lipid_carbon_tint: [f32; 3],
    /// RGB color for hydrophobic sidechain atoms.
    pub hydrophobic_sidechain: [f32; 3],
    /// RGB color for hydrophilic sidechain atoms.
    pub hydrophilic_sidechain: [f32; 3],
    /// RGB color for nucleic acid backbone.
    pub nucleic_acid: [f32; 3],
    /// Default RGB color for constraint bands.
    pub band_default: [f32; 3],
    /// RGB color for backbone constraint bands.
    pub band_backbone: [f32; 3],
    /// RGB color for disulfide constraint bands.
    pub band_disulfide: [f32; 3],
    /// RGB color for hydrogen-bond constraint bands.
    pub band_hbond: [f32; 3],
    /// RGB color for solvent molecules.
    pub solvent_color: [f32; 3],
    /// Per-cofactor carbon tint keyed by 3-letter residue name.
    pub cofactor_tints: HashMap<String, [f32; 3]>,
}

impl Default for ColorOptions {
    fn default() -> Self {
        let mut cofactor_tints = HashMap::new();
        let _ = cofactor_tints.insert("CLA".to_owned(), [0.2, 0.7, 0.3]);
        let _ = cofactor_tints.insert("CHL".to_owned(), [0.2, 0.6, 0.35]);
        let _ = cofactor_tints.insert("BCR".to_owned(), [0.9, 0.5, 0.1]);
        let _ = cofactor_tints.insert("BCB".to_owned(), [0.9, 0.5, 0.1]);
        let _ = cofactor_tints.insert("HEM".to_owned(), [0.7, 0.15, 0.15]);
        let _ = cofactor_tints.insert("HEC".to_owned(), [0.7, 0.15, 0.15]);
        let _ = cofactor_tints.insert("HEA".to_owned(), [0.7, 0.15, 0.15]);
        let _ = cofactor_tints.insert("HEB".to_owned(), [0.7, 0.15, 0.15]);
        let _ = cofactor_tints.insert("PHO".to_owned(), [0.5, 0.7, 0.3]);
        let _ = cofactor_tints.insert("PL9".to_owned(), [0.6, 0.5, 0.2]);
        let _ = cofactor_tints.insert("PLQ".to_owned(), [0.6, 0.5, 0.2]);

        Self {
            lipid_carbon_tint: [0.76, 0.70, 0.50],
            hydrophobic_sidechain: [0.95, 0.6, 0.2],
            hydrophilic_sidechain: [0.3, 0.5, 0.9],
            nucleic_acid: [0.45, 0.55, 0.85],
            band_default: [0.5, 0.0, 0.5],
            band_backbone: [1.0, 0.75, 0.0],
            band_disulfide: [0.5, 1.0, 0.0],
            band_hbond: [0.0, 0.75, 1.0],
            solvent_color: [0.6, 0.6, 0.6],
            cofactor_tints,
        }
    }
}

impl ColorOptions {
    /// Look up cofactor carbon tint by 3-letter residue name. Falls back to
    /// neutral gray.
    #[must_use]
    pub fn cofactor_tint(&self, res_name: &str) -> [f32; 3] {
        self.cofactor_tints
            .get(res_name.trim())
            .copied()
            .unwrap_or([0.5, 0.5, 0.5])
    }
}
