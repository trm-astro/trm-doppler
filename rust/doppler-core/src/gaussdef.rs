//! Port of the C++ `gaussdef()` function: separable 3-axis Gaussian
//! blurring of a (2D or 3D) image, used to construct "default" images for
//! MEM (the prior towards which the entropy term pulls the reconstruction).
//!
//! Unlike `op`/`tr`'s blurring (one 1-D convolution of a velocity-space
//! projection), `gaussdef` blurs the image itself, independently along each
//! of its axes that has more than one pixel, each as a 1-D FFT convolution
//! with its own kernel width/plan (since the three axes can have different
//! lengths and different requested FWHMs).

use crate::blur::{gaussian_kernel_fft, BlurPlan};
use crate::types::Nxyz;
use crate::EFAC;

/// Applies a separable Gaussian blur to `input` (dimensions `nxyz`),
/// blurring along X (if `nxyz.nx > 1 && fwhmx >= 0`), then Y, then Z, in
/// that order, exactly as the original C++ `gaussdef()` did. `fwhmx`,
/// `fwhmy`, `fwhmz` are in pixel units (i.e. already divided by the
/// relevant `vxy`/`vz` pixel scale by the caller). Returns a newly
/// allocated output buffer; a negative fwhm for a given axis skips
/// blurring along that axis.
pub fn gaussdef(input: &[f32], nxyz: Nxyz, fwhmx: f64, fwhmy: f64, fwhmz: f64) -> Vec<f32> {
    let mut output = input.to_vec();

    if nxyz.nx > 1 && fwhmx >= 0. {
        blur_axis(&mut output, nxyz, Axis::X, fwhmx);
    }
    if nxyz.ny > 1 && fwhmy >= 0. {
        blur_axis(&mut output, nxyz, Axis::Y, fwhmy);
    }
    if nxyz.nz > 1 && fwhmz >= 0. {
        blur_axis(&mut output, nxyz, Axis::Z, fwhmz);
    }

    output
}

#[derive(Clone, Copy)]
enum Axis {
    X,
    Y,
    Z,
}

/// Blurs `data` (dimensions `nxyz`, flattened row-major x-fastest as in the
/// rest of this crate) along one `axis` with the given FWHM (pixels),
/// in-place. One FFT plan/kernel is built for the whole axis and reused
/// across every 1-D line along it, matching the original C++ code's
/// `pblurr`/`pforw`/`pback` plans being created once per axis, outside the
/// loop over lines.
fn blur_axis(data: &mut [f32], nxyz: Nxyz, axis: Axis, fwhm: f64) {
    let (n, nlines, line_stride) = match axis {
        Axis::X => (nxyz.nx, nxyz.ny * nxyz.nz, 1usize),
        Axis::Y => (nxyz.ny, nxyz.nx * nxyz.nz, nxyz.nx),
        Axis::Z => (nxyz.nz, nxyz.nx * nxyz.ny, nxyz.nx * nxyz.ny),
    };

    // Work out buffer size needed for the FFTs: axis length plus margin for
    // the blurring kernel's half-width, rounded up to a power of two.
    let nadd = 2 * (3. * fwhm + 1.) as usize;
    let ntot = n + nadd;
    let ntot_pow2 = 1usize << (ntot as f64).log2().ceil() as u32;

    let plan = BlurPlan::new(ntot_pow2);
    let sigma = fwhm / EFAC;
    let half_width = nadd / 2;

    let mut kernel_scratch = plan.scratch();
    let bfft = gaussian_kernel_fft(&plan, &mut kernel_scratch, half_width.max(1), sigma);

    // Base offset of each 1-D line along `axis`: for X, lines are indexed by
    // (iy, iz); for Y, by (ix, iz); for Z, by (ix, iy). `line_base` maps a
    // line index 0..nlines to the flat offset of that line's first pixel.
    let (nx, ny) = (nxyz.nx, nxyz.ny);
    let line_base = |line: usize| -> usize {
        match axis {
            Axis::X => {
                let iy = line % ny;
                let iz = line / ny;
                (iz * ny + iy) * nx
            }
            Axis::Y => {
                let ix = line % nx;
                let iz = line / nx;
                iz * nx * ny + ix
            }
            Axis::Z => {
                let ix = line % nx;
                let iy = line / nx;
                iy * nx + ix
            }
        }
    };

    for line in 0..nlines {
        let base = line_base(line);
        let mut scratch = plan.scratch();
        scratch.fine.iter_mut().for_each(|v| *v = 0.);
        for i in 0..n {
            scratch.fine[i] = data[base + i * line_stride] as f64;
        }

        plan.forward(&mut scratch);
        for k in 0..plan.nfft {
            scratch.spec[k] *= bfft[k];
        }
        plan.inverse(&mut scratch);

        for i in 0..n {
            data[base + i * line_stride] = scratch.fine[i] as f32;
        }
    }
}
