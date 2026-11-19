//! PyO3 binding for `doppler_core::compute_default`, the building block
//! behind the original C++ `doppler.comdef()`.

use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;

use doppler_core::{compute_default, DefaultParams, Nxyz};

use crate::params::defopt_from_i32;

/// `default_image(image, nx, ny, nz, vxy, vz, option, bias, fwhmxy, fwhmz,
///     squeeze, sqfwhm)`
///
/// Computes the MEM default/prior image for a single image, matching the
/// per-image body of the original C++ `doppler_comdef()`. `option` is 1
/// (Uniform), 2 (Gauss2D) or 3 (Gauss3D), matching
/// `trm.doppler.map.Default.option`.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn default_image<'py>(
    py: Python<'py>,
    image: PyReadonlyArray1<f32>,
    nx: usize,
    ny: usize,
    nz: usize,
    vxy: f64,
    vz: f64,
    option: i32,
    bias: f64,
    fwhmxy: f64,
    fwhmz: f64,
    squeeze: f64,
    sqfwhm: f64,
) -> PyResult<Bound<'py, PyArray1<f32>>> {
    let nxyz = Nxyz::new(nx, ny, nz);
    let params = DefaultParams {
        option: defopt_from_i32(option)?,
        bias,
        fwhmxy,
        fwhmz,
        squeeze,
        sqfwhm,
    };

    let image = image.as_slice()?;
    if image.len() != nxyz.ntot() {
        return Err(PyRuntimeError::new_err(format!(
            "image has length {}, expected {} (nx*ny*nz)",
            image.len(),
            nxyz.ntot()
        )));
    }

    let out = py.allow_threads(|| compute_default(image, nxyz, vxy, vz, &params));
    Ok(PyArray1::from_vec_bound(py, out))
}
