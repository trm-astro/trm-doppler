//! Port of the `op` (image -> data) and `tr` (data -> image, transpose of
//! `op`) projection routines from the original C++ `doppler.cc`.
//!
//! The algorithm, comments and variable names below closely follow the
//! original implementation so that the two can be compared line-by-line;
//! only the FFT backend (rustfft/realfft in place of fftw3) and threading
//! model (rayon in place of OpenMP) have changed.

use rayon::prelude::*;

use crate::blur::{blurring_fft, nblurr_for, BlurPlan};
use crate::error::DopplerError;
use crate::types::{validate_data, validate_images, Itype, Nxyz};
use crate::CKMS;

/// Image-to-data projection: the heart of Doppler tomography. Projects one
/// or more images to compute data corresponding to each spectrum.
///
/// Arguments::
///
///  image  : multiple images, stored in a 1D array for compatibility with
///           mem routines (input)
///  nxyz   : dimensions of each image, one/image, (input)
///  vxy    : Vx-Vy pixel size(s), one/image, (input)
///  vz     : Vz pixel size(s), one/image, (input)
///  wavel  : central wavelengths for each image, vector/image, (input)
///  gamma  : systemic velocities for each wavelength, vector/image, (input)
///  scale  : scaling factors for each wavelength, vector/image, (input)
///  itype  : types for each image, (input)
///  tzero  : ephemeris zero point, same units as the times (input)
///  period : ephemeris period, same units as the times (input)
///  quad   : quadratic term of ephemeris, same units as the times (input)
///  vfine  : km/s to use for internal fine array (input)
///  sfac   : global scaling factor
///
///  data   : computed data, again as a 1D array for mem routines (output)
///  wave   : wavelength array, matching the data array, (input)
///  nwave  : number of wavelengths, one/dataset [effectively an X dimension],
///           (input)
///  nspec  : number of spectra, one/dataset [effectively a Y dimension], (input)
///  time   : central times for each spectrum (input)
///  expose : exposure lengths for each spectrum (input)
///  nsub   : sub-division factors for each spectrum (input)
///  fwhm   : FWHM resolutions, kms/s, one/dataset, (input)
///
/// Note the arguments 'image' through to 'sfac' are associated with the
/// Doppler map file, while the rest are associated with the data. 'data' is
/// the only output argument.
///
#[allow(clippy::too_many_arguments)]
pub fn op(
    image: &[f32],
    nxyz: &[Nxyz],
    vxy: &[f64],
    vz: &[f64],
    wavel: &[Vec<f64>],
    gamma: &[Vec<f32>],
    scale: &[Vec<f32>],
    itype: &[Itype],
    tzero: f64,
    period: f64,
    quad: f64,
    vfine: f64,
    sfac: f64,
    data: &mut [f32],
    wave: &[f64],
    nwave: &[usize],
    nspec: &[usize],
    time: &[Vec<f64>],
    expose: &[Vec<f32>],
    nsub: &[Vec<i32>],
    fwhm: &[f64],
) -> Result<(), DopplerError> {
    // check that the inputs for map (nx, velocity scales etc) have consistent shape.
    validate_images(nxyz, vxy, vz, wavel, gamma, scale, itype)?;
    // check that the inputs for data (nwave, nspec, time etc) have consistent shape.
    validate_data(wave, nwave, nspec, time, expose, nsub, fwhm)?;
    let (nfine, nfine_pow2, nblurr) = fine_array_size(nxyz, vxy, vz, nspec, fwhm, vfine);
    // adjust vmax to be the maximum velocity of the fine array measured from zero.
    let vmax = nfine as f64 * vfine / 2.;

    let ndpix: usize = nwave.iter().zip(nspec).map(|(&w, &s)| w * s).sum();
    if data.len() != ndpix {
        return Err(DopplerError::LengthMismatch(format!(
            "data has length {}, expected {ndpix}",
            data.len()
        )));
    }

    let plan = BlurPlan::new(nfine_pow2);
    // set output data to zero everywhere.
    data.iter_mut().for_each(|v| *v = 0.);

    let mut data_off = 0usize;
    let mut wave_off = 0usize;

    for nd in 0..nspec.len() {
        let mut kernel_scratch = plan.scratch();
        let bfft = blurring_fft(&plan, &mut kernel_scratch, fwhm[nd], vfine);

        let mut iptr_off = 0usize;
        for ni in 0..nxyz.len() {
            let nx = nxyz[ni].nx;
            let ny = nxyz[ni].ny;
            let nz = nxyz[ni].nz;
            let this_image = &image[iptr_off..iptr_off + nx * ny * nz];
            iptr_off += nx * ny * nz;

            if skip_dataset_image(&wavel[ni], &gamma[ni], wave, wave_off, nwave[nd], nspec[nd], vmax) {
                continue;
            }

            let vxyi = vxy[ni];
            let vzi = if nz > 1 { vz[ni] } else { 0. };

            let dslice = &mut data[data_off..data_off + nwave[nd] * nspec[nd]];
            let wslice = &wave[wave_off..wave_off + nwave[nd] * nspec[nd]];

            dslice
                .par_chunks_mut(nwave[nd])
                .zip(wslice.par_chunks(nwave[nd]))
                .enumerate()
                .for_each(|(ns, (dptr, wptr))| {
                    let mut scratch = plan.scratch();
                    let mut tfine = vec![0f64; nfine];
                    scratch.fine.iter_mut().for_each(|v| *v = 0.);

                    let ntdiv = nsub[nd][ns].max(1);
                    for nt in 0..ntdiv {
                        tfine.iter_mut().for_each(|v| *v = 0.);

                        let phase = orbital_phase(
                            time[nd][ns],
                            expose[nd][ns] as f64,
                            nt,
                            ntdiv,
                            tzero,
                            period,
                            quad,
                        );
                        let itfac = itype[ni].factor(phase);
                        use std::f64::consts::PI;
                        let cosp = (2. * PI * phase).cos();
                        let sinp = (2. * PI * phase).sin();

                        let pxstep = -vxyi / vfine * cosp;
                        let pystep = vxyi / vfine * sinp;
                        let (mut pzoff, pzstep) = if nz > 1 {
                            (
                                (nfine as f64 - 1.) / 2. + vzi * (nz as f64 + 1.) / 2. / vfine,
                                -vzi / vfine,
                            )
                        } else {
                            ((nfine as f64 - 1.) / 2., 0.)
                        };

                        let mut iiptr = 0usize;
                        for _iz in 0..nz {
                            pzoff += pzstep;
                            let mut pyoff = pzoff - pystep * (ny as f64 + 1.) / 2.;
                            for _iy in 0..ny {
                                pyoff += pystep;
                                let mut pxoff = pyoff - pxstep * (nx as f64 + 1.) / 2.;
                                for _ix in 0..nx {
                                    pxoff += pxstep;
                                    let nf = pxoff.round() as isize;
                                    if nf >= 0 && (nf as usize) < nfine {
                                        tfine[nf as usize] += this_image[iiptr] as f64;
                                    }
                                    iiptr += 1;
                                }
                            }
                        }

                        let weight = if ntdiv > 1 && (nt == 0 || nt == ntdiv - 1) {
                            itfac * vxyi.powi(2) / (2 * (ntdiv - 1)) as f64
                        } else {
                            itfac * vxyi.powi(2) / (ntdiv - 1).max(1) as f64
                        };
                        for nf in 0..nfine {
                            scratch.fine[nf] += weight * tfine[nf];
                        }
                    }

                    // Apply blurring via the FFT convolution theorem.
                    plan.forward(&mut scratch);
                    for k in 0..plan.nfft {
                        scratch.spec[k] *= bfft[k];
                    }
                    plan.inverse(&mut scratch);
                    let fine = &scratch.fine;

                    // Add the blurred, projected spectrum into the output
                    // data for every wavelength/line associated with this
                    // image.
                    for k in 0..wavel[ni].len() {
                        let wv = wavel[ni][k];
                        let gm = gamma[ni][k] as f64;
                        let sc = sfac * scale[ni][k] as f64;

                        let mut v1 = CKMS * ((wptr[0] + wptr[1]) / 2. / wv) - gm;
                        for m in 1..nwave[nd] - 1 {
                            let v2 = CKMS * ((wptr[m] + wptr[m + 1]) / 2. / wv - 1.) - gm;

                            if v1 < vmax && v2 > -vmax {
                                let fp1 = v1 / vfine + (nfine as f64 - 1.) / 2.;
                                let fp2 = v2 / vfine + (nfine as f64 - 1.) / 2.;

                                let ifp1 = ((fp1 + 0.5).ceil() as isize).clamp(0, nfine as isize) as usize;
                                let ifp2 = ((fp2 + 0.5).floor() as isize).clamp(0, nfine as isize) as usize;

                                let mut sum: f64 = fine[ifp1..ifp2].iter().sum();
                                if ifp1 > 0 {
                                    sum += (ifp1 as f64 - 0.5 - fp1) * fine[ifp1 - 1];
                                }
                                if ifp2 < nfine {
                                    sum += (fp2 - ifp2 as f64 + 0.5) * fine[ifp2];
                                }

                                dptr[m] += (sc * sum / ((v2 - v1).abs() / wptr[m])) as f32;
                            }
                            v1 = v2;
                        }
                    }
                });
        }

        let _ = nblurr; // kept for documentation/debug parity with C++
        data_off += nspec[nd] * nwave[nd];
        wave_off += nspec[nd] * nwave[nd];
    }

    Ok(())
}

/// Data-to-image projection: the transpose of [`op`].
///
/// This is the Rust equivalent of the C++ `tr()` function.
#[allow(clippy::too_many_arguments)]
pub fn tr(
    image: &mut [f32],
    nxyz: &[Nxyz],
    vxy: &[f64],
    vz: &[f64],
    wavel: &[Vec<f64>],
    gamma: &[Vec<f32>],
    scale: &[Vec<f32>],
    itype: &[Itype],
    tzero: f64,
    period: f64,
    quad: f64,
    vfine: f64,
    sfac: f64,
    data: &[f32],
    wave: &[f64],
    nwave: &[usize],
    nspec: &[usize],
    time: &[Vec<f64>],
    expose: &[Vec<f32>],
    nsub: &[Vec<i32>],
    fwhm: &[f64],
) -> Result<(), DopplerError> {
    validate_images(nxyz, vxy, vz, wavel, gamma, scale, itype)?;
    validate_data(wave, nwave, nspec, time, expose, nsub, fwhm)?;
    let (nfine, nfine_pow2, _nblurr) = fine_array_size(nxyz, vxy, vz, nspec, fwhm, vfine);
    // adjust vmax to be the maximum velocity of the fine array measured from zero.
    let vmax = nfine as f64 * vfine / 2.;

    let nipix: usize = nxyz.iter().map(|n| n.ntot()).sum();
    if image.len() != nipix {
        return Err(DopplerError::LengthMismatch(format!(
            "image has length {}, expected {nipix}",
            image.len()
        )));
    }

    let plan = BlurPlan::new(nfine_pow2);
    // set output image to zero everywhere. 
    image.iter_mut().for_each(|v| *v = 0.);

    let mut data_off = 0usize;
    let mut wave_off = 0usize;
    // Per-image byte offsets into `image`, recomputed per dataset exactly
    // as the C++ iptr walk does (images always visited in the same order).
    for nd in 0..nspec.len() {
        let mut kernel_scratch = plan.scratch();
        let bfft = blurring_fft(&plan, &mut kernel_scratch, fwhm[nd], vfine);

        let mut iptr_off = 0usize;
        for ni in 0..nxyz.len() {
            let nx = nxyz[ni].nx;
            let ny = nxyz[ni].ny;
            let nz = nxyz[ni].nz;
            let npix_i = nx * ny * nz;
            let image_start = iptr_off;
            iptr_off += npix_i;

            if skip_dataset_image(&wavel[ni], &gamma[ni], wave, wave_off, nwave[nd], nspec[nd], vmax) {
                continue;
            }

            let vxyi = vxy[ni];
            let vzi = if nz > 1 { vz[ni] } else { 0. };

            let dslice = &data[data_off..data_off + nwave[nd] * nspec[nd]];
            let wslice = &wave[wave_off..wave_off + nwave[nd] * nspec[nd]];

            // Accumulate per-spectrum image contributions into thread-local
            // buffers, then sum them into the shared image at the end, to
            // keep the inner loop race-free without locking every pixel.
            let ns_count = nspec[nd];
            let partials: Vec<Vec<f32>> = (0..ns_count)
                .into_par_iter()
                .map(|ns| {
                    let dptr = &dslice[ns * nwave[nd]..(ns + 1) * nwave[nd]];
                    let wptr = &wslice[ns * nwave[nd]..(ns + 1) * nwave[nd]];

                    let mut scratch = plan.scratch();
                    scratch.fine.iter_mut().for_each(|v| *v = 0.);

                    for k in 0..wavel[ni].len() {
                        let wv = wavel[ni][k];
                        let gm = gamma[ni][k] as f64;
                        let sc = sfac * scale[ni][k] as f64;

                        let mut v1 = CKMS * ((wptr[0] + wptr[1]) / 2. / wv) - gm;
                        for m in 1..nwave[nd] - 1 {
                            let v2 = CKMS * ((wptr[m] + wptr[m + 1]) / 2. / wv - 1.) - gm;

                            if v1 < vmax && v2 > -vmax {
                                let fp1 = v1 / vfine + (nfine as f64 - 1.) / 2.;
                                let fp2 = v2 / vfine + (nfine as f64 - 1.) / 2.;

                                let ifp1 = ((fp1 + 0.5).ceil() as isize).clamp(0, nfine as isize) as usize;
                                let ifp2 = ((fp2 + 0.5).floor() as isize).clamp(0, nfine as isize) as usize;

                                let add = sc * dptr[m] as f64 / ((v2 - v1).abs() / wptr[m]);
                                for nf in ifp1..ifp2 {
                                    scratch.fine[nf] += add;
                                }
                                if ifp1 > 0 {
                                    scratch.fine[ifp1 - 1] += (ifp1 as f64 - 0.5 - fp1) * add;
                                }
                                if ifp2 < nfine {
                                    scratch.fine[ifp2] += (fp2 - ifp2 as f64 + 0.5) * add;
                                }
                            }
                            v1 = v2;
                        }
                    }

                    // Blur (same convolution as op).
                    plan.forward(&mut scratch);
                    for k in 0..plan.nfft {
                        scratch.spec[k] *= bfft[k];
                    }
                    plan.inverse(&mut scratch);

                    let mut image_part = vec![0f32; npix_i];

                    let ntdiv = nsub[nd][ns].max(1);
                    for nt in 0..ntdiv {
                        let phase = orbital_phase(
                            time[nd][ns],
                            expose[nd][ns] as f64,
                            nt,
                            ntdiv,
                            tzero,
                            period,
                            quad,
                        );
                        let itfac = itype[ni].factor(phase);
                        use std::f64::consts::PI;
                        let cosp = (2. * PI * phase).cos();
                        let sinp = (2. * PI * phase).sin();

                        let pxstep = -vxyi / vfine * cosp;
                        let pystep = vxyi / vfine * sinp;
                        let pzstep = if nz > 1 { -vzi / vfine } else { 0. };

                        let weight = if ntdiv > 1 && (nt == 0 || nt == ntdiv - 1) {
                            itfac * vxyi.powi(2) / (2 * (ntdiv - 1)) as f64
                        } else {
                            itfac * vxyi.powi(2) / (ntdiv - 1).max(1) as f64
                        };
                        let tfine: Vec<f64> = scratch.fine[..nfine].iter().map(|v| weight * v).collect();

                        let mut pzoff = (nfine as f64 - 1.) / 2. - pzstep * (nz as f64 + 1.) / 2.;
                        let mut iiptr = 0usize;
                        for _iz in 0..nz {
                            pzoff += pzstep;
                            let mut pyoff = pzoff - pystep * (ny as f64 + 1.) / 2.;
                            for _iy in 0..ny {
                                pyoff += pystep;
                                let mut pxoff = pyoff - pxstep * (nx as f64 + 1.) / 2.;
                                for _ix in 0..nx {
                                    pxoff += pxstep;
                                    let nf = pxoff.round() as isize;
                                    if nf >= 0 && (nf as usize) < nfine {
                                        image_part[iiptr] += tfine[nf as usize] as f32;
                                    }
                                    iiptr += 1;
                                }
                            }
                        }
                    }

                    image_part
                })
                .collect();

            let target = &mut image[image_start..image_start + npix_i];
            for part in partials {
                for (t, p) in target.iter_mut().zip(part.iter()) {
                    *t += *p;
                }
            }
        }

        data_off += nspec[nd] * nwave[nd];
        wave_off += nspec[nd] * nwave[nd];
    }

    Ok(())
}

/// Computes the orbital phase for a sub-exposure sample, with two
/// Newton-Raphson corrections for the quadratic ephemeris term, matching
/// the original C++ code exactly.
#[inline]
fn orbital_phase(
    time_mid: f64,
    expose: f64,
    nt: i32,
    ntdiv: i32,
    tzero: f64,
    period: f64,
    quad: f64,
) -> f64 {
    let tsub = time_mid
        + expose * (nt as f64 - (ntdiv as f64 - 1.) / 2.) / (ntdiv - 1).max(1) as f64;
    let mut phase = (tsub - tzero) / period;
    for _ in 0..2 {
        let corr = tzero + period * phase + quad * phase.powi(2) - tsub;
        let deriv = period + 2. * quad * phase;
        phase -= corr / deriv;
    }
    phase
}

/// Determines the fine-array parameters shared by `op`/`tr`: `nfine` (the
/// minimum size needed to cover the maximum projected velocity plus
/// blurring margin), `NFINE` (`nfine` rounded up to a power of two, for
/// efficient FFTs) and `nblurr` (blurring kernel half-width in fine
/// pixels).
fn fine_array_size(
    nxyz: &[Nxyz],
    vxy: &[f64],
    vz: &[f64],
    nspec: &[usize],
    fwhm: &[f64],
    vfine: f64,
) -> (usize, usize, usize) {
    let nblurr = fwhm.iter().map(|&f| nblurr_for(f, vfine)).max().unwrap_or(0);
    debug_assert_eq!(nspec.len(), fwhm.len());

    let mut vmax = 0f64;
    for (i, n) in nxyz.iter().enumerate() {
        let (nx, ny, nz) = (n.nx as f64, n.ny as f64, n.nz as f64);
        let v = if n.nz > 1 {
            (vxy[i].powi(2) * (nx.powi(2) + ny.powi(2)) + (vz[i] * nz).powi(2)).sqrt()
        } else {
            (vxy[i].powi(2) * (nx.powi(2) + ny.powi(2))).sqrt()
        };
        vmax = vmax.max(v);
    }

    // nfine is the number of pixels needed for the fine array at its maximum
    // which should be enough in all cases even if its too many on occasion.
    // We add in extra pixels beyond the maximum velocity in order to allow
    // for blurring.
    let nfine = 5usize.max((vmax / vfine).ceil() as usize + 2 * nblurr);


    // The blurring is carried out with FFTs requiring zero padding.  Thus the
    // actual number of pixels to grab is the nearest power of 2 larger than
    // nfine+nblurr.
    let nfine_pow2 = 1usize << ((nfine + nblurr) as f64).log2().ceil() as u32;
    (nfine, nfine_pow2, nblurr)
}

/// Returns true if this dataset/image pair can be skipped entirely because
/// none of the image's spectral lines fall within the data's wavelength
/// range for any spectrum -- mirrors the `skip` logic in the original code.
fn skip_dataset_image(
    wavel: &[f64],
    gamma: &[f32],
    wave: &[f64],
    wave_off: usize,
    nwave: usize,
    nspec: usize,
    vmax: f64,
) -> bool {
    for (wv, &gm) in wavel.iter().zip(gamma.iter()) {
        let gm = gm as f64;
        for ns in 0..nspec {
            let wptr = &wave[wave_off + ns * nwave..wave_off + (ns + 1) * nwave];
            let w1 = wptr[0];
            let w2 = wptr[nwave - 1];
            let v1 = CKMS * (w1 / wv - 1.) - gm;
            let v2 = CKMS * (w2 / wv - 1.) - gm;
            if v1 < vmax && v2 > -vmax {
                return false;
            }
        }
    }
    true
}
