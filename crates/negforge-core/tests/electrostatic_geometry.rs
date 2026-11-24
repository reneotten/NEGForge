//! Manufactured electrostatic invariants and explicit-area SCF regressions.
use negforge_core::constants::{E, EPS_0};
use negforge_core::device::{Device, DeviceParams};
use negforge_core::selfconsistent::{solve_self_consistent, SelfConsistentOptions};

fn small_device() -> Device {
    Device::new(DeviceParams {
        a: 0.5,
        l_ch: 8.0,
        l_ds: 8.0,
        auto_size_contacts: false,
        v_g: 0.5,
        d_e: 0.002,
        ..Default::default()
    })
}

#[test]
fn explicit_area_converts_per_spin_density_to_spin_summed_volume_charge() {
    let mut device = small_device();
    let n = 0.25;
    assert_eq!(device.effective_area_m2, None);
    assert_eq!(device.mobile_charge_density(n), -E * n * 1e9);
    device.set_effective_area_m2(Some(20e-18)).unwrap();
    let expected = -2.0 * E * n * 1e9 / 20e-18;
    assert!((device.mobile_charge_density(n) / expected - 1.0).abs() < 1e-14);
    device.set_effective_area_m2(Some(40e-18)).unwrap();
    assert!((device.mobile_charge_density(n) / expected - 0.5).abs() < 1e-14);
    for invalid in [0.0, -1.0, f64::NAN, f64::INFINITY] {
        assert!(device.set_effective_area_m2(Some(invalid)).is_err());
        device.effective_area_m2 = Some(invalid);
        assert!(device.validate().is_err());
    }
    device.set_effective_area_m2(None).unwrap();
    assert_eq!(device.mobile_charge_density(n), -E * n * 1e9);
}

#[test]
fn uniform_charge_has_exact_screened_poisson_shift_and_electron_sign() {
    let mut device = small_device();
    device.psi_g.fill(0.0);
    device.psi_bi.fill(0.0);
    device.set_effective_area_m2(Some(20e-18)).unwrap();
    let rho = device.mobile_charge_density(0.1);
    device.rho.fill(rho);
    device.calc_potential();
    // Constant solution: -U/lambda² = rho * 1e-18 / epsilon.
    let expected = -rho * 1e-18 * device.lambda.powi(2) / (EPS_0 * device.params.k_si);
    assert!(expected > 0.0);
    for &u in &device.psi_f {
        assert!((u / expected - 1.0).abs() < 1e-11);
    }
    device.rho.fill(0.0);
    device.params.n_dot = -rho;
    device.calc_potential();
    for &u in &device.psi_f {
        assert!((u / expected + 1.0).abs() < 1e-11);
    }
}

#[test]
fn finite_area_scf_really_iterates_and_suppresses_injected_charge() {
    let options = SelfConsistentOptions {
        max_iterations: 300,
        tolerance: 1e-7,
        mixing: 0.15,
        green_energy_fraction: 1.0,
        eta: 0.002,
    };
    let mut legacy = small_device();
    let legacy_result = solve_self_consistent(&mut legacy, &options).unwrap();
    let mut finite = small_device();
    finite.set_effective_area_m2(Some(20e-18)).unwrap();
    let result = solve_self_consistent(&mut finite, &options).unwrap();
    let channel = finite.n_left..finite.n_right;
    let original: f64 = legacy_result.electron_density[channel.clone()].iter().sum();
    let screened: f64 = result.electron_density[channel.clone()].iter().sum();
    let shift = channel
        .map(|i| finite.psi_f[i] - legacy.psi_f[i])
        .fold(0.0, f64::max);
    eprintln!("finite-area SCF: legacy_iterations={}, finite_iterations={}, residual={:.3e} eV, max_channel_shift={:.6} eV, channel_density_ratio={:.6}", legacy_result.iterations, result.iterations, result.residual, shift, screened / original);
    assert!(
        result.iterations > 1,
        "physical feedback must not converge on its first iteration"
    );
    assert!(
        shift > 1e-3,
        "finite-area electrons must raise the barrier appreciably"
    );
    assert!(
        screened < 0.95 * original,
        "repulsive feedback must suppress occupation"
    );
    assert!(result.residual < options.tolerance);
    // Verify the actual undamped target, not just a tiny mixed step.
    let psi = finite.psi_f.clone();
    finite.rho = result
        .electron_density
        .iter()
        .map(|&n| finite.mobile_charge_density(n))
        .collect();
    finite.calc_potential();
    let residual = psi
        .iter()
        .zip(&finite.psi_f)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f64::max);
    assert!((residual - result.residual).abs() < 1e-12);
}
