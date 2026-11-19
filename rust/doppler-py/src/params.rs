//! Shared argument-marshalling helpers used by the PyO3 entry points.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use doppler_core::{DefOpt, Itype, Nxyz};

pub fn itype_from_i32(v: i32) -> PyResult<Itype> {
    Ok(match v {
        1 => Itype::PUnit,
        2 => Itype::NUnit,
        3 => Itype::PSine,
        4 => Itype::NSine,
        5 => Itype::PCosine,
        6 => Itype::NCosine,
        7 => Itype::PSine2,
        8 => Itype::NSine2,
        9 => Itype::PCosine2,
        10 => Itype::NCosine2,
        other => {
            return Err(PyValueError::new_err(format!(
                "unrecognised image type code {other}"
            )))
        }
    })
}

pub fn defopt_from_i32(v: i32) -> PyResult<DefOpt> {
    Ok(match v {
        1 => DefOpt::Uniform,
        2 => DefOpt::Gauss2D,
        3 => DefOpt::Gauss3D,
        other => {
            return Err(PyValueError::new_err(format!(
                "unrecognised default-image option code {other} (expected 1=Uniform, 2=Gauss2D, 3=Gauss3D)"
            )))
        }
    })
}

/// Bundles the per-image "Map" arguments shared by `op`/`tr`, taking
/// simple Python lists/arrays instead of a `Map` object directly.
#[allow(clippy::too_many_arguments)]
pub fn build_image_params(
    nxyz: Vec<(usize, usize, usize)>,
    vxy: Vec<f64>,
    vz: Vec<f64>,
    wavel: Vec<Vec<f64>>,
    gamma: Vec<Vec<f32>>,
    scale: Vec<Vec<f32>>,
    itype: Vec<i32>,
) -> PyResult<(
    Vec<Nxyz>,
    Vec<f64>,
    Vec<f64>,
    Vec<Vec<f64>>,
    Vec<Vec<f32>>,
    Vec<Vec<f32>>,
    Vec<Itype>,
)> {
    let nxyz = nxyz
        .into_iter()
        .map(|(x, y, z)| Nxyz::new(x, y, z))
        .collect();
    let itype = itype
        .into_iter()
        .map(itype_from_i32)
        .collect::<PyResult<Vec<_>>>()?;
    Ok((nxyz, vxy, vz, wavel, gamma, scale, itype))
}
