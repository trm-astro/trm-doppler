//! Sanity tests for the `op`/`tr` projection routines, notably checking
//! that `tr` is the adjoint (transpose) of `op`, i.e. for random image `x`
//! and data `y`:  <op(x), y> == <x, tr(y)>  (dot-product/adjoint test),
//! which is the standard correctness check for tomographic projection
//! operators used in MEM reconstruction.

use doppler_core::{op, tr, Itype, Nxyz};

struct Scenario {
    nxyz: Vec<Nxyz>,
    vxy: Vec<f64>,
    vz: Vec<f64>,
    wavel: Vec<Vec<f64>>,
    gamma: Vec<Vec<f32>>,
    scale: Vec<Vec<f32>>,
    itype: Vec<Itype>,
    tzero: f64,
    period: f64,
    quad: f64,
    vfine: f64,
    sfac: f64,
    wave: Vec<f64>,
    nwave: Vec<usize>,
    nspec: Vec<usize>,
    time: Vec<Vec<f64>>,
    expose: Vec<Vec<f32>>,
    nsub: Vec<Vec<i32>>,
    fwhm: Vec<f64>,
}

fn small_scenario() -> Scenario {
    let nx = 8;
    let ny = 8;
    let nwave = 16;
    let nspec = 10;

    // simple linear wavelength grid around 6563 A (H-alpha) with enough
    // width to see Doppler shifts at typical velocities
    let wv0 = 6560.0;
    let dwv = 0.5;
    let mut wave = Vec::with_capacity(nwave * nspec);
    for _s in 0..nspec {
        for i in 0..nwave {
            wave.push(wv0 + dwv * i as f64);
        }
    }

    let time: Vec<f64> = (0..nspec).map(|i| i as f64 * 0.1).collect();
    let expose = vec![0.05f32; nspec];
    let nsub = vec![1i32; nspec];

    Scenario {
        nxyz: vec![Nxyz::new(nx, ny, 1)],
        vxy: vec![100.0],
        vz: vec![0.0],
        wavel: vec![vec![6563.0]],
        gamma: vec![vec![0.0]],
        scale: vec![vec![1.0]],
        itype: vec![Itype::PUnit],
        tzero: 0.0,
        period: 1.0,
        quad: 0.0,
        vfine: 20.0,
        sfac: 1.0,
        wave,
        nwave: vec![nwave],
        nspec: vec![nspec],
        time: vec![time],
        expose: vec![expose],
        nsub: vec![nsub],
        fwhm: vec![50.0],
    }
}

fn run_op(s: &Scenario, image: &[f32]) -> Vec<f32> {
    let ndpix: usize = s.nwave.iter().zip(&s.nspec).map(|(&w, &n)| w * n).sum();
    let mut data = vec![0f32; ndpix];
    op(
        image, &s.nxyz, &s.vxy, &s.vz, &s.wavel, &s.gamma, &s.scale, &s.itype, s.tzero, s.period,
        s.quad, s.vfine, s.sfac, &mut data, &s.wave, &s.nwave, &s.nspec, &s.time, &s.expose,
        &s.nsub, &s.fwhm,
    )
    .expect("op should succeed");
    data
}

fn run_tr(s: &Scenario, data: &[f32]) -> Vec<f32> {
    let nipix: usize = s.nxyz.iter().map(|n| n.ntot()).sum();
    let mut image = vec![0f32; nipix];
    tr(
        &mut image, &s.nxyz, &s.vxy, &s.vz, &s.wavel, &s.gamma, &s.scale, &s.itype, s.tzero,
        s.period, s.quad, s.vfine, s.sfac, data, &s.wave, &s.nwave, &s.nspec, &s.time, &s.expose,
        &s.nsub, &s.fwhm,
    )
    .expect("tr should succeed");
    image
}

#[test]
fn op_runs_and_is_finite() {
    let s = small_scenario();
    let nipix: usize = s.nxyz.iter().map(|n| n.ntot()).sum();
    // simple centred blob
    let mut image = vec![0f32; nipix];
    image[nipix / 2] = 1.0;

    let data = run_op(&s, &image);
    assert!(data.iter().all(|v| v.is_finite()));
    assert!(data.iter().any(|&v| v != 0.0), "op output should be non-trivial");
}

#[test]
fn tr_runs_and_is_finite() {
    let s = small_scenario();
    let ndpix: usize = s.nwave.iter().zip(&s.nspec).map(|(&w, &n)| w * n).sum();
    let mut data = vec![0f32; ndpix];
    // pick an interior wavelength pixel (not the first/last of a spectrum,
    // which are deliberately excluded from the projection, matching the
    // original C++ behaviour)
    let nwave = s.nwave[0];
    data[nwave + nwave / 2] = 1.0;

    let image = run_tr(&s, &data);
    assert!(image.iter().all(|v| v.is_finite()));
    assert!(image.iter().any(|&v| v != 0.0), "tr output should be non-trivial");
}

#[test]
fn tr_is_adjoint_of_op() {
    // Dot-product test: <op(x), y> should equal <x, tr(y)> for the
    // projection operator and its transpose, up to floating point error.
    let s = small_scenario();
    let nipix: usize = s.nxyz.iter().map(|n| n.ntot()).sum();
    let ndpix: usize = s.nwave.iter().zip(&s.nspec).map(|(&w, &n)| w * n).sum();

    // deterministic pseudo-random fill so the test is reproducible
    let mut state: u64 = 0x2545F4914F6CDD1D;
    let mut rnd = move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        (state >> 11) as f64 / (1u64 << 53) as f64
    };

    let x: Vec<f32> = (0..nipix).map(|_| rnd() as f32).collect();
    let y: Vec<f32> = (0..ndpix).map(|_| rnd() as f32).collect();

    let op_x = run_op(&s, &x);
    let tr_y = run_tr(&s, &y);

    let lhs: f64 = op_x.iter().zip(&y).map(|(&a, &b)| a as f64 * b as f64).sum();
    let rhs: f64 = x.iter().zip(&tr_y).map(|(&a, &b)| a as f64 * b as f64).sum();

    let scale = lhs.abs().max(rhs.abs()).max(1.0);
    assert!(
        (lhs - rhs).abs() / scale < 1e-3,
        "adjoint mismatch: <op(x),y>={lhs}, <x,tr(y)>={rhs}"
    );
}

#[test]
fn op_rejects_mismatched_data_vectors() {
    // time/expose/nsub inconsistent with nspec should be a clear error,
    // not a panic/out-of-bounds index.
    let mut s = small_scenario();
    s.time[0].pop();

    let nipix: usize = s.nxyz.iter().map(|n| n.ntot()).sum();
    let ndpix: usize = s.nwave.iter().zip(&s.nspec).map(|(&w, &n)| w * n).sum();
    let image = vec![0f32; nipix];
    let mut data = vec![0f32; ndpix];

    let err = op(
        &image, &s.nxyz, &s.vxy, &s.vz, &s.wavel, &s.gamma, &s.scale, &s.itype, s.tzero,
        s.period, s.quad, s.vfine, s.sfac, &mut data, &s.wave, &s.nwave, &s.nspec, &s.time,
        &s.expose, &s.nsub, &s.fwhm,
    )
    .expect_err("op should reject a mismatched time vector");
    assert!(err.to_string().contains("time/expose/nsub"));
}

#[test]
fn tr_rejects_mismatched_wave_length() {
    let mut s = small_scenario();
    s.wave.pop();

    let nipix: usize = s.nxyz.iter().map(|n| n.ntot()).sum();
    let ndpix: usize = s.nwave.iter().zip(&s.nspec).map(|(&w, &n)| w * n).sum();
    let mut image = vec![0f32; nipix];
    let data = vec![0f32; ndpix];

    let err = tr(
        &mut image, &s.nxyz, &s.vxy, &s.vz, &s.wavel, &s.gamma, &s.scale, &s.itype, s.tzero,
        s.period, s.quad, s.vfine, s.sfac, &data, &s.wave, &s.nwave, &s.nspec, &s.time,
        &s.expose, &s.nsub, &s.fwhm,
    )
    .expect_err("tr should reject a mismatched wave length");
    assert!(err.to_string().contains("wave has length"));
}
