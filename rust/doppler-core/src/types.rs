//! Data structures mirroring the vectors/vector-of-vectors used by the
//! original C++ `op`/`tr` routines.

use crate::error::DopplerError;

/// Dimensions of a single image (`Nxyz` in the original C++).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Nxyz {
    pub nx: usize,
    pub ny: usize,
    pub nz: usize,
}

impl Nxyz {
    pub fn new(nx: usize, ny: usize, nz: usize) -> Self {
        Self { nx, ny, nz }
    }

    /// Total number of pixels in the image.
    pub fn ntot(&self) -> usize {
        self.nx * self.ny * self.nz
    }
}

/// Image "type", determining how it is combined with orbital phase, as per
/// the `Itype` enum in the original C++ code.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Itype {
    PUnit,
    NUnit,
    PSine,
    NSine,
    PCosine,
    NCosine,
    PSine2,
    NSine2,
    PCosine2,
    NCosine2,
}

impl Itype {
    /// Multiplicative factor applied to the projected image given the
    /// orbital `phase` (in cycles), matching the `switch` statement in the
    /// original `op`/`tr` functions.
    #[inline]
    pub fn factor(&self, phase: f64) -> f64 {
        use std::f64::consts::PI;
        let cosp = (2. * PI * phase).cos();
        let sinp = (2. * PI * phase).sin();
        match self {
            Itype::PUnit => 1.,
            Itype::NUnit => -1.,
            Itype::PSine => sinp,
            Itype::NSine => -sinp,
            Itype::PCosine => cosp,
            Itype::NCosine => -cosp,
            Itype::PSine2 => (4. * PI * phase).sin(),
            Itype::NSine2 => -(4. * PI * phase).sin(),
            Itype::PCosine2 => (4. * PI * phase).cos(),
            Itype::NCosine2 => -(4. * PI * phase).cos(),
        }
    }
}

/// Determines the per-dataset array lengths (`nwave`, `nspec`) line up
/// with the per-dataset vectors (`time`/`expose`/`nsub`/`fwhm`) and that
/// `wave` has the expected total length, matching the dimension checks
/// implicit in the original C++ `read_data`/`op`/`tr`.
pub(crate) fn validate_data(
    wave: &[f64],
    nwave: &[usize],
    nspec: &[usize],
    time: &[Vec<f64>],
    expose: &[Vec<f32>],
    nsub: &[Vec<i32>],
    fwhm: &[f64],
) -> Result<(), DopplerError> {
    let nd = nspec.len();
    if nwave.len() != nd {
        return Err(DopplerError::LengthMismatch(
            "nwave and nspec must have the same length".into(),
        ));
    }
    for (name, v) in [
        ("time", time.len()),
        ("expose", expose.len()),
        ("nsub", nsub.len()),
        ("fwhm", fwhm.len()),
    ] {
        if v != nd {
            return Err(DopplerError::LengthMismatch(format!(
                "{name} must have length {nd}, got {v}"
            )));
        }
    }
    for nd_idx in 0..nd {
        let ns = nspec[nd_idx];
        if time[nd_idx].len() != ns || expose[nd_idx].len() != ns || nsub[nd_idx].len() != ns {
            return Err(DopplerError::LengthMismatch(format!(
                "dataset {nd_idx}: time/expose/nsub must have length {ns}"
            )));
        }
    }
    let expect: usize = nwave.iter().zip(nspec).map(|(&w, &s)| w * s).sum();
    if wave.len() != expect {
        return Err(DopplerError::LengthMismatch(format!(
            "wave has length {}, expected {}",
            wave.len(),
            expect
        )));
    }
    Ok(())
}

pub(crate) fn validate_images(
    nxyz: &[Nxyz],
    vxy: &[f64],
    vz: &[f64],
    wavel: &[Vec<f64>],
    gamma: &[Vec<f32>],
    scale: &[Vec<f32>],
    itype: &[Itype],
) -> Result<(), DopplerError> {
    let ni = nxyz.len();
    for (name, len) in [
        ("vxy", vxy.len()),
        ("vz", vz.len()),
        ("wavel", wavel.len()),
        ("gamma", gamma.len()),
        ("scale", scale.len()),
        ("itype", itype.len()),
    ] {
        if len != ni {
            return Err(DopplerError::LengthMismatch(format!(
                "{name} must have length {ni} (one per image), got {len}"
            )));
        }
    }
    for i in 0..ni {
        if wavel[i].len() != gamma[i].len() || wavel[i].len() != scale[i].len() {
            return Err(DopplerError::LengthMismatch(format!(
                "image {i}: wavel/gamma/scale must have matching lengths"
            )));
        }
    }
    Ok(())
}
