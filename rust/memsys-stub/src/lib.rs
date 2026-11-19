//! **TEMPORARY local placeholder** for the `memsys` crate that
//! [`trm-astro/rust-mem`](https://github.com/trm-astro/rust-mem) has agreed
//! to publish: a library crate exposing `MemGblst`, `MemTransform` and
//! `memprm` with a dependency-injected forward/transpose transform instead
//! of the hardcoded `crate::opus::opus`/`crate::tropus::tropus` module
//! paths its current (`main`) branch requires.
//!
//! This crate exists *only* so that `doppler-mem`'s integration with that
//! API can be written and tested today, without waiting for the upstream
//! change to land. **It is not a port of the real `memsys`/MEM (Skilling &
//! Bryan) algorithm** — `memprm` here performs a simple bias-limited
//! gradient step, not real maximum-entropy iteration, and its diagnostic
//! outputs should not be trusted for anything beyond exercising the
//! plumbing. Once `trm-astro/rust-mem` publishes the real API matching the
//! signatures below, replace the `memsys` path dependency in
//! `doppler-mem/Cargo.toml` with a git dependency on that crate; because
//! the public API here is designed to match exactly, no other code should
//! need to change.

use ndarray::Array1;

/// External state handed to [`memprm`]: the current map (image) and the
/// data it is being fitted to, plus the MEM "default"/weights/accuracies
/// arrays. Mirrors `Mem::Gbl::st`-derived buffers in the original C++, but
/// (per the agreed design) sizes the map and data arrays independently
/// rather than requiring them to match.
pub struct MemGblst {
    pub map: Array1<f64>,
    pub data: Array1<f64>,
    pub accuracies: Array1<f64>,
    pub default_model: Array1<f64>,
    pub weights: Array1<f64>,
}

impl MemGblst {
    pub fn new(
        map: Array1<f64>,
        data: Array1<f64>,
        accuracies: Array1<f64>,
        default_model: Array1<f64>,
        weights: Array1<f64>,
    ) -> Self {
        Self {
            map,
            data,
            accuracies,
            default_model,
            weights,
        }
    }

    /// Number of pixels in map ("model") space.
    pub fn nmod(&self) -> usize {
        self.map.len()
    }

    /// Number of points in data space.
    pub fn ndata(&self) -> usize {
        self.data.len()
    }
}

/// Forward/transpose transform pair that [`memprm`] uses to project between
/// map space and data space, supplied by the caller instead of being
/// hardcoded into this crate (cf. `Mem::opus`/`Mem::tropus` in the original
/// C++, which the caller defined at fixed, link-resolved symbol names).
pub trait MemTransform {
    /// Forward ("opus") transform: image/model space -> data space.
    fn opus(&mut self, model: &Array1<f64>, data: &mut Array1<f64>);
    /// Transpose ("tropus") transform: data space -> image/model space.
    fn tropus(&mut self, data: &Array1<f64>, model: &mut Array1<f64>);
}

/// **Placeholder** stand-in for `Mem::memprm`/the real crate's `memprm`.
///
/// Performs a single, crude gradient step: project the current map to data
/// space, scale the residual (data - prediction) by `accuracies`, project
/// that back to map space, and add a small, `rmax`-limited multiple of it
/// to the map (clipped so the fractional map change per pixel never
/// exceeds `rmax`). This happens to resemble the first-order structure of
/// a real MEM step (predict, form a data-space residual, back-project),
/// which is enough to validate wiring/plumbing end-to-end, but it does
/// *not* implement entropy regularisation, the Skilling & Bryan subspace
/// search, or any of the real algorithm's convergence guarantees.
///
/// Returns `(c, test, cnew, s, rnew, snew, sumf)`, matching the original
/// `Mem::memprm`'s output parameter order (constraint, gradient misfit,
/// predicted constraint, entropy, distance moved, predicted entropy, map
/// total) -- here `test`/`cnew`/`s`/`rnew`/`snew` are only rough stand-ins
/// for their real counterparts.
#[allow(clippy::too_many_arguments)]
pub fn memprm(
    external: &mut MemGblst,
    transform: &mut dyn MemTransform,
    _method: i32,
    _level: i32,
    aim: f64,
    rmax: f64,
    _def: f64,
    acc: &mut f64,
) -> (f64, f64, f64, f64, f64, f64, f64) {
    let ndata = external.ndata();
    let nmod = external.nmod();

    // predict data from the current map
    let mut predicted = Array1::<f64>::zeros(ndata);
    transform.opus(&external.map, &mut predicted);

    // chi-squared-like constraint: sum((data-predicted)^2 * accuracies) / ndata
    let mut residual = Array1::<f64>::zeros(ndata);
    let mut c = 0.0;
    for i in 0..ndata {
        let r = external.data[i] - predicted[i];
        residual[i] = r * external.accuracies[i];
        c += r * r * external.accuracies[i];
    }
    c /= ndata.max(1) as f64;

    // crude entropy-like diagnostic: -sum(p*log(p/default)) / nmod
    let mut s = 0.0;
    for i in 0..nmod {
        let p = external.map[i].max(1e-30);
        let d = external.default_model[i].max(1e-30);
        s -= p * (p / d).ln();
    }
    s /= nmod.max(1) as f64;

    // back-project the weighted residual into map space
    let mut map_grad = Array1::<f64>::zeros(nmod);
    transform.tropus(&residual, &mut map_grad);

    // take an rmax-limited step: scale the step so no pixel's fractional
    // change exceeds rmax (a crude stand-in for memsys's dimensionless
    // distance limit)
    let grad_max = map_grad.iter().cloned().fold(0.0_f64, |a, b| a.max(b.abs()));
    let map_max = external
        .map
        .iter()
        .cloned()
        .fold(1e-10_f64, |a, b| a.max(b.abs()));
    let step = if grad_max > 0.0 {
        (rmax * map_max / grad_max).min(1.0 / nmod.max(1) as f64 * *acc)
    } else {
        0.0
    };
    let step = if step.is_finite() && step > 0.0 { step } else { 0.0 };

    let mut sumf = 0.0;
    for i in 0..nmod {
        external.map[i] = (external.map[i] + step * map_grad[i]).max(0.0);
        sumf += external.map[i];
    }

    let test = (c - aim).abs() / aim.max(1e-10);
    let cnew = c;
    let rnew = step * grad_max;
    let snew = s;

    (c, test, cnew, s, rnew, snew, sumf)
}
