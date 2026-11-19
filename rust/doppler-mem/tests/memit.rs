//! Integration test for the `doppler-mem` plumbing: verifies `run_memit`
//! wires `doppler_core::op`/`tr` through to `memsys::memprm` correctly end
//! to end (constructing `MemGblst`, driving iterations, and returning a
//! finite, correctly-shaped image). This does **not** assert anything
//! about MEM/entropy correctness, since `memsys` is currently a
//! placeholder (see `../../memsys-stub/src/lib.rs`) -- only that the
//! integration compiles, runs, and produces sane output.

use doppler_core::{DefOpt, DefaultParams, Itype, Nxyz};
use doppler_mem::{run_memit, DopplerData, DopplerImages};

fn small_problem() -> (DopplerImages, DopplerData) {
    let nx = 6;
    let ny = 6;
    let nwave = 12;
    let nspec = 6;

    let wv0 = 6560.0;
    let dwv = 0.6;
    let mut wave = Vec::with_capacity(nwave * nspec);
    for _s in 0..nspec {
        for i in 0..nwave {
            wave.push(wv0 + dwv * i as f64);
        }
    }

    let images = DopplerImages {
        nxyz: vec![Nxyz::new(nx, ny, 1)],
        vxy: vec![120.0],
        vz: vec![0.0],
        wavel: vec![vec![6563.0]],
        gamma: vec![vec![0.0]],
        scale: vec![vec![1.0]],
        itype: vec![Itype::PUnit],
        tzero: 0.0,
        period: 1.0,
        quad: 0.0,
        vfine: 25.0,
        sfac: 1.0,
        defaults: vec![DefaultParams {
            option: DefOpt::Uniform,
            bias: 1.0,
            fwhmxy: 0.0,
            fwhmz: 0.0,
            squeeze: 0.0,
            sqfwhm: 0.0,
        }],
    };

    let data = DopplerData {
        wave,
        nwave: vec![nwave],
        nspec: vec![nspec],
        time: vec![(0..nspec).map(|i| i as f64 * 0.15).collect()],
        expose: vec![vec![0.05f32; nspec]],
        nsub: vec![vec![1i32; nspec]],
        fwhm: vec![40.0],
    };

    (images, data)
}

#[test]
fn run_memit_produces_finite_image() {
    let (images, data) = small_problem();
    let nmod: usize = images.nxyz.iter().map(|n| n.ntot()).sum();
    let ndata: usize = data.nwave.iter().zip(&data.nspec).map(|(&w, &s)| w * s).sum();

    // start from a flat image and some synthetic "observed" flux
    let image = vec![1.0f32; nmod];
    let flux = vec![0.5f32; ndata];
    let ferr = vec![0.1f32; ndata];

    let result = run_memit(&images, &data, &image, &flux, &ferr, 5, 1.0, 1e-4, 0.2)
        .expect("run_memit should succeed");

    assert_eq!(result.image.len(), nmod);
    assert!(result.image.iter().all(|v| v.is_finite()));
    assert!(result.constraint.is_finite());
    assert!(result.iterations >= 1 && result.iterations <= 5);
}

#[test]
fn run_memit_mode_30_runs_with_gaussian_default() {
    let (mut images, data) = small_problem();
    // force mode=30 (non-uniform default)
    images.defaults[0] = DefaultParams {
        option: DefOpt::Gauss2D,
        bias: 1.0,
        fwhmxy: 2.0,
        fwhmz: 0.0,
        squeeze: 0.0,
        sqfwhm: 0.0,
    };

    let nmod: usize = images.nxyz.iter().map(|n| n.ntot()).sum();
    let ndata: usize = data.nwave.iter().zip(&data.nspec).map(|(&w, &s)| w * s).sum();
    let image = vec![1.0f32; nmod];
    let flux = vec![0.5f32; ndata];
    let ferr = vec![0.1f32; ndata];

    let result = run_memit(&images, &data, &image, &flux, &ferr, 3, 1.0, 1e-4, 0.2)
        .expect("run_memit should succeed with mode 30");
    assert!(result.image.iter().all(|v| v.is_finite()));
}

#[test]
fn run_memit_rejects_mismatched_image_length() {
    let (images, data) = small_problem();
    let ndata: usize = data.nwave.iter().zip(&data.nspec).map(|(&w, &s)| w * s).sum();
    let bad_image = vec![1.0f32; 3]; // wrong length
    let flux = vec![0.5f32; ndata];
    let ferr = vec![0.1f32; ndata];

    let err = run_memit(&images, &data, &bad_image, &flux, &ferr, 1, 1.0, 1e-4, 0.2)
        .expect_err("should reject mismatched image length");
    assert!(err.to_string().contains("length"));
}
