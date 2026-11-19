//! Owned, flattened description of a Doppler map and dataset, suitable for
//! being held across repeated `opus`/`tropus` calls during MEM iteration
//! (`doppler_core::op`/`tr` take borrowed slices for one-shot calls; these
//! structs own their data so a single instance can be reused across many
//! MEM iterations).

use doppler_core::{DefaultParams, Itype, Nxyz};

/// Everything about the Doppler map side needed to drive repeated `op`/`tr`
/// calls and default-image recomputation during MEM iteration.
pub struct DopplerImages {
    pub nxyz: Vec<Nxyz>,
    pub vxy: Vec<f64>,
    pub vz: Vec<f64>,
    pub wavel: Vec<Vec<f64>>,
    pub gamma: Vec<Vec<f32>>,
    pub scale: Vec<Vec<f32>>,
    pub itype: Vec<Itype>,
    pub tzero: f64,
    pub period: f64,
    pub quad: f64,
    pub vfine: f64,
    pub sfac: f64,
    /// Per-image default-computation parameters (one per image, matching
    /// `nxyz`), used for both the initial default and (when needed) the
    /// per-iteration "mode 30" recompute.
    pub defaults: Vec<DefaultParams>,
}

/// Everything about the dataset side needed to drive repeated `op`/`tr`
/// calls during MEM iteration.
pub struct DopplerData {
    pub wave: Vec<f64>,
    pub nwave: Vec<usize>,
    pub nspec: Vec<usize>,
    pub time: Vec<Vec<f64>>,
    pub expose: Vec<Vec<f32>>,
    pub nsub: Vec<Vec<i32>>,
    pub fwhm: Vec<f64>,
}
