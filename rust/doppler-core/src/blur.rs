//! FFT-based instrumental blurring, using `realfft`/`rustfft` in place of
//! the original code's `fftw3` calls.

use realfft::{ComplexToReal, RealFftPlanner, RealToComplex};
use rustfft::num_complex::Complex64;
use std::sync::Arc;

use crate::EFAC;

/// Reusable FFT plan pair for real-valued signals of length `nfine`,
/// equivalent to the `pforw`/`pback` plans created once in the original C++
/// `op`/`tr` (outside the parallel region, since `fftw_plan` creation is not
/// thread-safe). `realfft` plans are `Send + Sync` and, like `fftw_execute`
/// on a shared plan with distinct buffers, can be driven concurrently from
/// multiple threads provided each thread uses its own scratch/output
/// buffers (see [`BlurPlan::scratch`]).
pub struct BlurPlan {
    pub nfine: usize,
    pub nfft: usize,
    forward: Arc<dyn RealToComplex<f64>>,
    inverse: Arc<dyn ComplexToReal<f64>>,
}

impl BlurPlan {
    pub fn new(nfine: usize) -> Self {
        let mut planner = RealFftPlanner::<f64>::new();
        let forward = planner.plan_fft_forward(nfine);
        let inverse = planner.plan_fft_inverse(nfine);
        Self {
            nfine,
            nfft: nfine / 2 + 1,
            forward,
            inverse,
        }
    }

    /// Allocate a per-thread scratch space for use with this plan.
    pub fn scratch(&self) -> BlurScratch {
        BlurScratch {
            fine: self.forward.make_input_vec(),
            spec: self.forward.make_output_vec(),
            fwd_scratch: self.forward.make_scratch_vec(),
            bwd_scratch: self.inverse.make_scratch_vec(),
        }
    }

    /// Forward real->complex transform of `scratch.fine`, result left in
    /// `scratch.spec`. Note: `realfft`'s r2c transform overwrites its input
    /// buffer as scratch space, matching `fftw`'s behaviour of `fine` being
    /// usable afterwards only via the c2r transform below.
    pub fn forward(&self, scratch: &mut BlurScratch) {
        self.forward
            .process_with_scratch(&mut scratch.fine, &mut scratch.spec, &mut scratch.fwd_scratch)
            .expect("realfft forward transform failed");
    }

    /// Inverse complex->real transform of `scratch.spec` into
    /// `scratch.fine`. `realfft`'s c2r does *not* apply the `1/n`
    /// normalisation (matching raw `fftw_execute`), so the blurring kernel
    /// normalisation (done once, up front, exactly as in the C++ code) is
    /// what makes the overall round trip come out normalised.
    pub fn inverse(&self, scratch: &mut BlurScratch) {
        self.inverse
            .process_with_scratch(&mut scratch.spec, &mut scratch.fine, &mut scratch.bwd_scratch)
            .expect("realfft inverse transform failed");
    }
}

/// Per-thread/per-call scratch buffers for use with a [`BlurPlan`].
pub struct BlurScratch {
    pub fine: Vec<f64>,
    pub spec: Vec<Complex64>,
    fwd_scratch: Vec<Complex64>,
    bwd_scratch: Vec<Complex64>,
}

/// Computes the number of pixels needed either side of centre for the
/// Gaussian blurring kernel given `fwhm` (km/s), `vfine` (km/s/pixel), i.e.
/// `nblurr` from the original code (total blur width is `2*nblurr-1`).
pub fn nblurr_for(fwhm: f64, vfine: f64) -> usize {
    let psigma = fwhm / vfine / EFAC;
    (6. * psigma).round() as usize + 1
}

/// Builds and forward-transforms a normalised Gaussian blurring kernel of
/// half-width `half_width` pixels (total kernel width `2*half_width-1`) and
/// standard deviation `sigma` pixels, using `plan`'s FFT size as the
/// zero-padded transform length. This is the core of the `blurr[]`
/// construction shared by `op`/`tr`'s instrumental-resolution blurring and
/// `gaussdef`'s positional blurring in the original C++ code; both compute
/// an identical normalised, symmetric, zero-centred Gaussian kernel and FFT
/// it once for repeated use as a convolution multiplier.
pub fn gaussian_kernel_fft(
    plan: &BlurPlan,
    scratch: &mut BlurScratch,
    half_width: usize,
    sigma: f64,
) -> Vec<Complex64> {
    let nfine = plan.nfine;

    let blurr = &mut scratch.fine;
    blurr.iter_mut().for_each(|v| *v = 0.);
    let mut norm = 1.0;
    blurr[0] = 1.0;
    for m in 1..half_width {
        let prf = (-((m as f64 / sigma).powi(2)) / 2.).exp();
        let n = nfine - m;
        blurr[m] = prf;
        blurr[n] = prf;
        norm += 2. * prf;
    }
    norm *= nfine as f64;
    blurr[0] /= norm;
    for m in 1..half_width {
        let n = nfine - m;
        blurr[m] /= norm;
        blurr[n] /= norm;
    }

    plan.forward(scratch);
    scratch.spec.clone()
}

/// Builds and forward-transforms the normalised Gaussian blurring kernel
/// for one dataset, equivalent to computing `blurr[]` and taking `pblurr`
/// in the original C++ code. Returns the kernel's FFT (length `nfft`).
pub fn blurring_fft(plan: &BlurPlan, scratch: &mut BlurScratch, fwhm: f64, vfine: f64) -> Vec<Complex64> {
    let psigma = fwhm / vfine / EFAC;
    let nblurr = nblurr_for(fwhm, vfine);
    gaussian_kernel_fft(plan, scratch, nblurr, psigma)
}
