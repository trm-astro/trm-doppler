//! PyO3 binding for `doppler_mem::run_memit`, matching the calling
//! convention of the original C++ `doppler.memit()`.

use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;

use doppler_core::DefaultParams;
use doppler_mem::{run_memit, DopplerData, DopplerImages};

use crate::params::{build_image_params, defopt_from_i32};

/// `memit(image, nxyz, vxy, vz, wavel, gamma, scale, itype, tzero, period,
///     quad, vfine, sfac, def_option, def_bias, def_fwhmxy, def_fwhmz,
///     def_squeeze, def_sqfwhm, flux, ferr, wave, nwave, nspec, time,
///     expose, nsub, fwhm, niter, caim, tlim, rmax)`
///
/// Runs up to `niter` MEM iterations starting from `image`, given `flux`/
/// `ferr` as the target data and its uncertainties. `def_*` are one value
/// per image (matching `nxyz`), giving each image's
/// `trm.doppler.map.Default` parameters. Returns `(image, constraint,
/// test, entropy, iterations)`, mirroring `doppler_memit()`'s behaviour
/// (including its "mode 30" default-recompute step, triggered whenever any
/// image has a non-uniform default or `bias != 1`).
///
/// Note: the current MEM iteration itself is provided by a temporary local
/// placeholder for the `memsys` crate (see `doppler-mem`/`memsys-stub`)
/// pending `trm-astro/rust-mem` publishing its agreed library API; the
/// returned diagnostics should not yet be treated as scientifically
/// authoritative.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn memit<'py>(
    py: Python<'py>,
    image: PyReadonlyArray1<f32>,
    nxyz: Vec<(usize, usize, usize)>,
    vxy: Vec<f64>,
    vz: Vec<f64>,
    wavel: Vec<Vec<f64>>,
    gamma: Vec<Vec<f32>>,
    scale: Vec<Vec<f32>>,
    itype: Vec<i32>,
    tzero: f64,
    period: f64,
    quad: f64,
    vfine: f64,
    sfac: f64,
    def_option: Vec<i32>,
    def_bias: Vec<f64>,
    def_fwhmxy: Vec<f64>,
    def_fwhmz: Vec<f64>,
    def_squeeze: Vec<f64>,
    def_sqfwhm: Vec<f64>,
    flux: PyReadonlyArray1<f32>,
    ferr: PyReadonlyArray1<f32>,
    wave: PyReadonlyArray1<f64>,
    nwave: Vec<usize>,
    nspec: Vec<usize>,
    time: Vec<Vec<f64>>,
    expose: Vec<Vec<f32>>,
    nsub: Vec<Vec<i32>>,
    fwhm: Vec<f64>,
    niter: i32,
    caim: f64,
    tlim: f64,
    rmax: f64,
) -> PyResult<(Bound<'py, PyArray1<f32>>, f64, f64, f64, i32)> {
    let (nxyz, vxy, vz, wavel, gamma, scale, itype) =
        build_image_params(nxyz, vxy, vz, wavel, gamma, scale, itype)?;

    let ni = nxyz.len();
    for (name, len) in [
        ("def_option", def_option.len()),
        ("def_bias", def_bias.len()),
        ("def_fwhmxy", def_fwhmxy.len()),
        ("def_fwhmz", def_fwhmz.len()),
        ("def_squeeze", def_squeeze.len()),
        ("def_sqfwhm", def_sqfwhm.len()),
    ] {
        if len != ni {
            return Err(PyRuntimeError::new_err(format!(
                "{name} must have length {ni} (one per image), got {len}"
            )));
        }
    }
    let defaults = (0..ni)
        .map(|i| {
            Ok(DefaultParams {
                option: defopt_from_i32(def_option[i])?,
                bias: def_bias[i],
                fwhmxy: def_fwhmxy[i],
                fwhmz: def_fwhmz[i],
                squeeze: def_squeeze[i],
                sqfwhm: def_sqfwhm[i],
            })
        })
        .collect::<PyResult<Vec<_>>>()?;

    let images = DopplerImages {
        nxyz,
        vxy,
        vz,
        wavel,
        gamma,
        scale,
        itype,
        tzero,
        period,
        quad,
        vfine,
        sfac,
        defaults,
    };
    let data = DopplerData {
        wave: wave.as_slice()?.to_vec(),
        nwave,
        nspec,
        time,
        expose,
        nsub,
        fwhm,
    };

    let image = image.as_slice()?;
    let flux = flux.as_slice()?;
    let ferr = ferr.as_slice()?;

    let result = py
        .allow_threads(|| run_memit(&images, &data, image, flux, ferr, niter, caim, tlim, rmax))
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

    Ok((
        PyArray1::from_vec_bound(py, result.image),
        result.constraint,
        result.test,
        result.entropy,
        result.iterations,
    ))
}
