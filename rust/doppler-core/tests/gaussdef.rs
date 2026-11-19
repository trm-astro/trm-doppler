//! Sanity tests for `gaussdef`/`compute_default`.

use doppler_core::{compute_default, gaussdef, DefOpt, DefaultParams, Nxyz};

#[test]
fn gaussdef_preserves_total_flux_2d() {
    // A Gaussian blur (FFT convolution with a normalised kernel) should
    // conserve total flux away from the edges; for a single spike well
    // clear of the boundary this should hold to good precision.
    let nxyz = Nxyz::new(32, 32, 1);
    let mut image = vec![0f32; nxyz.ntot()];
    image[16 * 32 + 16] = 100.0;

    let blurred = gaussdef(&image, nxyz, 3.0, 3.0, 0.0);
    assert!(blurred.iter().all(|v| v.is_finite()));

    let total_before: f64 = image.iter().map(|&v| v as f64).sum();
    let total_after: f64 = blurred.iter().map(|&v| v as f64).sum();
    assert!(
        (total_before - total_after).abs() / total_before < 1e-3,
        "flux not conserved: before={total_before}, after={total_after}"
    );

    // the blur should have spread the spike out, so the peak drops and
    // neighbouring pixels become non-zero
    assert!(blurred[16 * 32 + 16] < image[16 * 32 + 16]);
    assert!(blurred[16 * 32 + 17] > 0.0);
}

#[test]
fn gaussdef_skips_negative_fwhm_axis() {
    let nxyz = Nxyz::new(16, 16, 1);
    let mut image = vec![0f32; nxyz.ntot()];
    image[8 * 16 + 8] = 5.0;

    // negative fwhmy should leave the y blur untouched (only x blurred)
    let blurred = gaussdef(&image, nxyz, 2.0, -1.0, 0.0);
    assert!(blurred.iter().all(|v| v.is_finite()));
    // the exact centre row should be unaffected in y, i.e. row 7 and 9
    // (one above/below the spike) must remain exactly zero since no
    // blurring happened along y
    assert_eq!(blurred[7 * 16 + 8], 0.0);
    assert_eq!(blurred[9 * 16 + 8], 0.0);
}

#[test]
fn uniform_default_is_mean_times_bias() {
    let nxyz = Nxyz::new(4, 4, 1);
    let image: Vec<f32> = (0..16).map(|i| i as f32).collect();
    let params = DefaultParams {
        option: DefOpt::Uniform,
        bias: 0.5,
        fwhmxy: 0.0,
        fwhmz: 0.0,
        squeeze: 0.0,
        sqfwhm: 0.0,
    };
    let default = compute_default(&image, nxyz, 1.0, 0.0, &params);
    let expected = (0..16).sum::<i32>() as f64 / 16.0 * 0.5;
    assert!(default.iter().all(|&v| (v as f64 - expected).abs() < 1e-5));
}

#[test]
fn gauss3d_default_runs_and_is_finite() {
    let nxyz = Nxyz::new(8, 8, 5);
    let mut image = vec![1f32; nxyz.ntot()];
    image[2 * 64 + 3 * 8 + 3] = 50.0;

    let params = DefaultParams {
        option: DefOpt::Gauss3D,
        bias: 1.0,
        fwhmxy: 2.0,
        fwhmz: 1.5,
        squeeze: 0.3,
        sqfwhm: 2.0,
    };
    let default = compute_default(&image, nxyz, 1.0, 1.0, &params);
    assert_eq!(default.len(), nxyz.ntot());
    assert!(default.iter().all(|v| v.is_finite()));
}
