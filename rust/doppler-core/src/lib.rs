//! Numerical core of `trm.doppler`.
//!
//! This crate is a Rust port of the `op`/`tr` image<->data projection
//! routines and the `gaussdef`-based default-image computation that formed
//! the computational heart of the original C++ `doppler.cc` Python
//! extension. `op`/`tr` implement the forward ("opus") and transpose
//! ("tropus") projections used by Doppler tomography's maximum-entropy
//! image reconstruction.
//!
//! FFT-based blurring, previously implemented with `fftw3`, is implemented
//! here with [`rustfft`]/[`realfft`]. MEM iteration itself (the Skilling &
//! Bryan algorithm, `Mem::memprm` in the original C++) is out of scope for
//! this crate; it is provided by an external `memsys` crate, which this
//! crate's `op`/`tr` are designed to be driven by via that crate's
//! `MemTransform` trait (see the `doppler-mem` crate for the glue).

mod blur;
mod default;
mod error;
mod gaussdef;
mod project;
mod types;

pub use default::{compute_default, DefOpt, DefaultParams};
pub use error::DopplerError;
pub use gaussdef::gaussdef;
pub use types::{Itype, Nxyz};

/// Speed of light in km/s.
pub const CKMS: f64 = 299_792.458;

/// Ratio of FWHM to sigma for a Gaussian.
pub const EFAC: f64 = 2.354_820_045;

pub use project::{op, tr};
