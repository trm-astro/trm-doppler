//! PyO3 bindings exposing the `doppler-core`/`doppler-mem` numerical
//! routines to Python, mirroring the calling convention of the retired
//! `_doppler` C extension's `op`/`tr`/`comdef`/`memit`-based routines
//! (formerly `trm/doppler/doppler.cc`, now removed in favour of this
//! Rust implementation).
//!
//! Rather than reaching into arbitrary Python `Map`/`Data` objects via the
//! Python C API (as the original code did with `read_map`/`read_data`), the
//! functions here take plain, explicit arguments built from numpy arrays and
//! Python lists. The existing pure-Python `trm.doppler.map`/`trm.doppler.data`
//! classes are expected to marshal their state into these calls; this keeps
//! the Rust/Python boundary simple and avoids duplicating attribute-walking
//! logic that is already better expressed in Python.

mod default_image;
mod memit;
mod params;

use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;

use doppler_core::{op as core_op, tr as core_tr};
use params::build_image_params;

/// `op(image, nxyz, vxy, vz, wavel, gamma, scale, itype, tzero, period,
///     quad, vfine, sfac, wave, nwave, nspec, time, expose, nsub, fwhm)`
///
/// Computes the data equivalent to the given image(s), returning a new
/// numpy array of the projected flux. See `doppler_core::op` for details.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn op<'py>(
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
    wave: PyReadonlyArray1<f64>,
    nwave: Vec<usize>,
    nspec: Vec<usize>,
    time: Vec<Vec<f64>>,
    expose: Vec<Vec<f32>>,
    nsub: Vec<Vec<i32>>,
    fwhm: Vec<f64>,
) -> PyResult<Bound<'py, PyArray1<f32>>> {
    let (nxyz, vxy, vz, wavel, gamma, scale, itype) =
        build_image_params(nxyz, vxy, vz, wavel, gamma, scale, itype)?;

    let image = image.as_slice()?;
    let wave = wave.as_slice()?;
    let ndpix: usize = nwave.iter().zip(&nspec).map(|(&w, &s)| w * s).sum();
    let mut data = vec![0f32; ndpix];

    py.allow_threads(|| {
        core_op(
            image, &nxyz, &vxy, &vz, &wavel, &gamma, &scale, &itype, tzero, period, quad, vfine,
            sfac, &mut data, wave, &nwave, &nspec, &time, &expose, &nsub, &fwhm,
        )
    })
    .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

    Ok(PyArray1::from_vec_bound(py, data))
}

/// `tr(image_npix, nxyz, vxy, vz, wavel, gamma, scale, itype, tzero, period,
///     quad, vfine, sfac, data, wave, nwave, nspec, time, expose, nsub,
///     fwhm)`
///
/// Computes the transpose of [`op`]: projects data back onto image space,
/// returning a new numpy array containing the (additive) image update. See
/// `doppler_core::tr` for details.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn tr<'py>(
    py: Python<'py>,
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
    data: PyReadonlyArray1<f32>,
    wave: PyReadonlyArray1<f64>,
    nwave: Vec<usize>,
    nspec: Vec<usize>,
    time: Vec<Vec<f64>>,
    expose: Vec<Vec<f32>>,
    nsub: Vec<Vec<i32>>,
    fwhm: Vec<f64>,
) -> PyResult<Bound<'py, PyArray1<f32>>> {
    let (nxyz, vxy, vz, wavel, gamma, scale, itype) =
        build_image_params(nxyz, vxy, vz, wavel, gamma, scale, itype)?;

    let data = data.as_slice()?;
    let wave = wave.as_slice()?;
    let nipix: usize = nxyz.iter().map(|n| n.ntot()).sum();
    let mut image = vec![0f32; nipix];

    py.allow_threads(|| {
        core_tr(
            &mut image, &nxyz, &vxy, &vz, &wavel, &gamma, &scale, &itype, tzero, period, quad,
            vfine, sfac, data, wave, &nwave, &nspec, &time, &expose, &nsub, &fwhm,
        )
    })
    .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

    Ok(PyArray1::from_vec_bound(py, image))
}

/// PyO3 module: `trm.doppler._doppler_rs`.
#[pymodule]
fn _doppler_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(op, m)?)?;
    m.add_function(wrap_pyfunction!(tr, m)?)?;
    m.add_function(wrap_pyfunction!(default_image::default_image, m)?)?;
    m.add_function(wrap_pyfunction!(memit::memit, m)?)?;
    Ok(())
}
