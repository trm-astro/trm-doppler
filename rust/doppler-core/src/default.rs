//! Port of the per-image "default image" computation shared by the C++
//! `doppler_comdef()` Python entry point and the `mode == 30` branch of
//! `doppler_memit()`'s iteration loop (the step that recomputes MEM's
//! default/prior image from the current reconstruction before each
//! iteration).

use crate::gaussdef::gaussdef;
use crate::types::Nxyz;
use crate::EFAC;

/// Default-image option for one image, matching `trm.doppler.map.Default`'s
/// `option` values (`DefOpt::Uniform` = 1, etc. in the original C++ `enum
/// DefOpt`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DefOpt {
    /// Uniform default: the mean pixel value (times `bias`) everywhere.
    Uniform,
    /// Gaussian-blurred version of the current image (2D).
    Gauss2D,
    /// Gaussian-blurred version of the current image (3D), optionally
    /// "squeezed" towards a Gaussian in Vz.
    Gauss3D,
}

/// Per-image parameters controlling how its default is computed, matching
/// the fields of a `trm.doppler.map.Default` object.
#[derive(Debug, Clone, Copy)]
pub struct DefaultParams {
    pub option: DefOpt,
    /// Factor to pull the default level up/down; must be > 0.
    pub bias: f64,
    /// FWHM blur in the Vx-Vy plane, km/s (Gauss2D/Gauss3D only).
    pub fwhmxy: f64,
    /// FWHM blur in Vz, km/s (Gauss3D only).
    pub fwhmz: f64,
    /// 3D squeeze-towards-Vz-Gaussian factor (Gauss3D only); <= 0 disables.
    pub squeeze: f64,
    /// FWHM of the squeezed component, km/s (Gauss3D only).
    pub sqfwhm: f64,
}

/// Computes the default image for one image, matching the per-image body
/// of the C++ `for(size_t nim=0; nim<nxyz.size(); nim++)` loop in
/// `doppler_comdef`/`doppler_memit`'s mode-30 branch.
pub fn compute_default(image: &[f32], nxyz: Nxyz, vxy: f64, vz: f64, params: &DefaultParams) -> Vec<f32> {
    match params.option {
        DefOpt::Uniform => {
            let npix = nxyz.ntot();
            let ave =
                image.iter().map(|&v| v as f64).sum::<f64>() / npix as f64 * params.bias;
            vec![ave as f32; npix]
        }

        DefOpt::Gauss2D | DefOpt::Gauss3D => {
            let fx = params.fwhmxy / vxy;
            let fy = params.fwhmxy / vxy;
            let fz = if nxyz.nz > 1 { params.fwhmz / vz } else { 0. };

            let mut output = if params.option == DefOpt::Gauss3D && params.squeeze > 0. {
                squeeze_towards_vz_gaussian(image, nxyz, vz, params.squeeze, params.sqfwhm, fx, fy, fz)
            } else {
                gaussdef(image, nxyz, fx, fy, fz)
            };

            if params.bias != 1.0 {
                for v in output.iter_mut() {
                    *v = (*v as f64 * params.bias) as f32;
                }
            }
            output
        }
    }
}

/// Implements the `def[nim] == GAUSS3D && squeeze[nim] > 0.` branch: blends
/// a Vx-Vy-only blur of a per-(x,y)-column Gaussian-in-Vz profile with a
/// full 3D blur of the original image, matching the original C++ exactly.
///
/// Note: the original C++ also computes a per-column weighted-mean Vz
/// (`vzmean`) here, apparently intending to centre the Vz Gaussian on it,
/// but then never uses that value (the Gaussian is always centred on
/// `vz == 0`); that computation is a no-op in the original and is omitted
/// here since it does not affect the result.
#[allow(clippy::too_many_arguments)]
fn squeeze_towards_vz_gaussian(
    image: &[f32],
    nxyz: Nxyz,
    vz: f64,
    squeeze: f64,
    sqfwhm: f64,
    fx: f64,
    fy: f64,
    fz: f64,
) -> Vec<f32> {
    let (nx, ny, nz) = (nxyz.nx, nxyz.ny, nxyz.nz);
    let nxy = nx * ny;
    let sigma = sqfwhm / EFAC;

    let mut tptr = vec![0f64; nxyz.ntot()];
    for iy in 0..ny {
        for ix in 0..nx {
            let ixyp = iy * nx + ix;

            let mut znorm = 0.0;
            for iz in 0..nz {
                znorm += image[ixyp + iz * nxy] as f64;
            }

            let mut znorm2 = 0.0;
            let mut vslice = -vz * (nz as f64 - 1.) / 2.;
            for iz in 0..nz {
                let g = (-((vslice / sigma).powi(2)) / 2.).exp();
                tptr[ixyp + iz * nxy] = g;
                znorm2 += g;
                vslice += vz;
            }

            let sfac = znorm / znorm2 * squeeze;
            for iz in 0..nz {
                tptr[ixyp + iz * nxy] *= sfac;
            }
        }
    }

    let tptr_f32: Vec<f32> = tptr.iter().map(|&v| v as f32).collect();
    // blur the squeezed Vz-Gaussian component in Vx-Vy only (not Vz)
    let ttptr = gaussdef(&tptr_f32, nxyz, fx, fy, 0.0);
    // blur the original image over all three dimensions
    let mut optr = gaussdef(image, nxyz, fx, fy, fz);

    for (o, t) in optr.iter_mut().zip(ttptr.iter()) {
        *o = ((1. - squeeze) * *o as f64 + *t as f64) as f32;
    }
    optr
}
