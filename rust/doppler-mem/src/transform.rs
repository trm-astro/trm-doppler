//! Adapter implementing `memsys::MemTransform` in terms of
//! `doppler_core::op`/`tr`.

use ndarray::Array1;

use crate::problem::{DopplerData, DopplerImages};

/// Bridges [`memsys::MemTransform`]'s `f64` `ndarray` buffers to
/// `doppler_core::op`/`tr`'s flat `f32` slices, borrowing the (owned,
/// repeatedly-reused) Doppler map/dataset description for the lifetime of
/// a MEM run.
pub struct DopplerTransform<'a> {
    images: &'a DopplerImages,
    data: &'a DopplerData,
}

impl<'a> DopplerTransform<'a> {
    pub fn new(images: &'a DopplerImages, data: &'a DopplerData) -> Self {
        Self { images, data }
    }
}

impl<'a> memsys::MemTransform for DopplerTransform<'a> {
    fn opus(&mut self, model: &Array1<f64>, data: &mut Array1<f64>) {
        let image: Vec<f32> = model.iter().map(|&v| v as f32).collect();
        let mut out = vec![0f32; data.len()];

        doppler_core::op(
            &image,
            &self.images.nxyz,
            &self.images.vxy,
            &self.images.vz,
            &self.images.wavel,
            &self.images.gamma,
            &self.images.scale,
            &self.images.itype,
            self.images.tzero,
            self.images.period,
            self.images.quad,
            self.images.vfine,
            self.images.sfac,
            &mut out,
            &self.data.wave,
            &self.data.nwave,
            &self.data.nspec,
            &self.data.time,
            &self.data.expose,
            &self.data.nsub,
            &self.data.fwhm,
        )
        .expect("doppler_core::op failed: invalid problem dimensions");

        for (d, v) in data.iter_mut().zip(out) {
            *d = v as f64;
        }
    }

    fn tropus(&mut self, data: &Array1<f64>, model: &mut Array1<f64>) {
        let flux: Vec<f32> = data.iter().map(|&v| v as f32).collect();
        let mut out = vec![0f32; model.len()];

        doppler_core::tr(
            &mut out,
            &self.images.nxyz,
            &self.images.vxy,
            &self.images.vz,
            &self.images.wavel,
            &self.images.gamma,
            &self.images.scale,
            &self.images.itype,
            self.images.tzero,
            self.images.period,
            self.images.quad,
            self.images.vfine,
            self.images.sfac,
            &flux,
            &self.data.wave,
            &self.data.nwave,
            &self.data.nspec,
            &self.data.time,
            &self.data.expose,
            &self.data.nsub,
            &self.data.fwhm,
        )
        .expect("doppler_core::tr failed: invalid problem dimensions");

        for (m, v) in model.iter_mut().zip(out) {
            *m = v as f64;
        }
    }
}
