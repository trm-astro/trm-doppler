//! MEM iteration driver: bridges `doppler-core`'s `op`/`tr` projection
//! routines to the external `memsys` crate's `MemTransform`-based MEM
//! (Skilling & Bryan maximum entropy) implementation, mirroring
//! `doppler_memit()` from the original C++ `doppler.cc`.
//!
//! See [`memsys`] (currently a local placeholder, `../memsys-stub`; see
//! its doc comment) for the MEM algorithm itself -- this crate only owns
//! the Doppler-specific projection (`doppler_core::op`/`tr`) and the
//! default-image recomputation (`doppler_core::compute_default`), wiring
//! them into `memsys::memprm`'s iteration loop.

mod error;
mod problem;
mod transform;

pub use error::DopplerMemError;
pub use problem::{DopplerData, DopplerImages};
pub use transform::DopplerTransform;

use doppler_core::{compute_default, DefaultParams, Nxyz};
use ndarray::Array1;
use memsys::MemGblst;

/// Diagnostics and final image returned by [`run_memit`], mirroring the
/// output parameters of the original C++ `Mem::memprm`.
#[derive(Debug, Clone)]
pub struct MemitResult {
    /// Final reconstructed image (concatenated across all images, same
    /// layout as the `image` input).
    pub image: Vec<f32>,
    /// Constraint (~reduced chi-squared) value on the final iteration.
    pub constraint: f64,
    /// Gradient-direction misfit ("test") on the final iteration; MEM is
    /// considered converged once this drops below `tlim`.
    pub test: f64,
    /// Entropy value on the final iteration.
    pub entropy: f64,
    /// Number of iterations actually performed (<= `niter`, if convergence
    /// was reached early).
    pub iterations: i32,
}

/// Runs up to `niter` MEM iterations on `image`, given `images`/`data`
/// describing the Doppler map/dataset, with observed fluxes `flux` and
/// uncertainties `ferr`. Mirrors `doppler_memit()`, including its mode
/// selection (falling back to recomputing the default image each
/// iteration -- "mode 30" in the original code -- whenever any image uses
/// a non-uniform default or a bias != 1) and its convergence test (`test <
/// tlim && c <= caim`).
#[allow(clippy::too_many_arguments)]
pub fn run_memit(
    images: &DopplerImages,
    data: &DopplerData,
    image: &[f32],
    flux: &[f32],
    ferr: &[f32],
    niter: i32,
    caim: f64,
    tlim: f64,
    rmax: f64,
) -> Result<MemitResult, DopplerMemError> {
    let nmod: usize = images.nxyz.iter().map(|n| n.ntot()).sum();
    let ndata: usize = data
        .nwave
        .iter()
        .zip(&data.nspec)
        .map(|(&w, &s)| w * s)
        .sum();

    if image.len() != nmod {
        return Err(DopplerMemError::LengthMismatch(format!(
            "image has length {}, expected {nmod} (sum of image pixel counts)",
            image.len()
        )));
    }
    if flux.len() != ndata || ferr.len() != ndata {
        return Err(DopplerMemError::LengthMismatch(format!(
            "flux/ferr must both have length {ndata} (sum of nwave*nspec), got {}/{}",
            flux.len(),
            ferr.len()
        )));
    }
    if niter < 1 {
        return Err(DopplerMemError::InvalidParameter("niter must be >= 1".into()));
    }
    if caim <= 0. {
        return Err(DopplerMemError::InvalidParameter("caim must be > 0".into()));
    }
    if tlim >= 0.5 {
        return Err(DopplerMemError::InvalidParameter("tlim must be < 0.5".into()));
    }
    if rmax >= 0.5 {
        return Err(DopplerMemError::InvalidParameter("rmax must be < 0.5".into()));
    }

    // mode = 30 whenever any image needs its default recomputed each
    // iteration (non-uniform default, or a bias pulling it away from 1),
    // else 10 -- matching `doppler_memit`'s mode selection exactly.
    let mode = if images
        .defaults
        .iter()
        .any(|d| d.option != doppler_core::DefOpt::Uniform || d.bias != 1.0)
    {
        30
    } else {
        10
    };

    // accuracies: 2/sigma^2/ndata for unmasked points (ferr > 0), passed
    // through unchanged otherwise, matching doppler_memit's treatment of
    // Mem::Gbl::kb[21].
    let accuracies: Vec<f64> = ferr
        .iter()
        .map(|&e| {
            if e > 0. {
                2. / (e as f64).powi(2) / ndata.max(1) as f64
            } else {
                e as f64
            }
        })
        .collect();

    let default_model = recompute_defaults(images, image)?;

    let mut gbl = MemGblst::new(
        to_f64(image),
        to_f64(flux),
        Array1::from(accuracies),
        to_f64(&default_model),
        Array1::zeros(nmod),
    );

    let mut transform = DopplerTransform::new(images, data);

    let mut acc = 1.0;
    let (mut c, mut test, mut s) = (f64::INFINITY, f64::INFINITY, 0.0);
    let mut iterations = 0;

    for it in 0..niter {
        if mode == 30 {
            let current: Vec<f32> = gbl.map.iter().map(|&v| v as f32).collect();
            let new_default = recompute_defaults(images, &current)?;
            gbl.default_model = to_f64(&new_default);
        }

        let (c_out, test_out, _cnew, s_out, _rnew, _snew, _sumf) =
            memsys::memprm(&mut gbl, &mut transform, mode, 20, caim, rmax, 1.0, &mut acc);
        c = c_out;
        test = test_out;
        s = s_out;
        iterations = it + 1;

        if test < tlim && c <= caim {
            break;
        }
    }

    let final_image: Vec<f32> = gbl.map.iter().map(|&v| v as f32).collect();

    Ok(MemitResult {
        image: final_image,
        constraint: c,
        test,
        entropy: s,
        iterations,
    })
}

fn to_f64(v: &[f32]) -> Array1<f64> {
    Array1::from(v.iter().map(|&x| x as f64).collect::<Vec<_>>())
}

/// Recomputes the default image for every image in `images`, given the
/// current concatenated image buffer `image`, matching the per-image loop
/// body shared by `doppler_comdef()` and `doppler_memit()`'s mode-30 step.
fn recompute_defaults(images: &DopplerImages, image: &[f32]) -> Result<Vec<f32>, DopplerMemError> {
    let nmod: usize = images.nxyz.iter().map(|n: &Nxyz| n.ntot()).sum();
    if image.len() != nmod {
        return Err(DopplerMemError::LengthMismatch(format!(
            "image has length {}, expected {nmod}",
            image.len()
        )));
    }

    let mut out = vec![0f32; nmod];
    let mut offset = 0;
    for (i, nxyz) in images.nxyz.iter().enumerate() {
        let npix = nxyz.ntot();
        let slice = &image[offset..offset + npix];
        let params: &DefaultParams = &images.defaults[i];
        let def = compute_default(slice, *nxyz, images.vxy[i], images.vz[i], params);
        out[offset..offset + npix].copy_from_slice(&def);
        offset += npix;
    }
    Ok(out)
}
