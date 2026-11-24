use negforge_core::constants::{E, EPS_0, H};
use negforge_core::device::{Device, DeviceParams};
use negforge_core::selfconsistent::{solve_self_consistent, SelfConsistentOptions};
use negforge_core::sweep::{subthreshold_swing, sweep_v_ds, sweep_v_g, IvPoint};

#[test]
fn undefined_subthreshold_swing_is_not_reported_as_infinity() {
    let points = [
        IvPoint {
            voltage: 0.0,
            current: 1.0,
        },
        IvPoint {
            voltage: 1.0,
            current: 1.0,
        },
    ];
    assert!(subthreshold_swing(&points, 0.0, 1.0).is_nan());
}

#[test]
fn reverse_bias_energy_bound_covers_both_reservoirs() {
    let mut d = small_device();
    d.set_v_ds(-2.0);
    assert!(d.e_max > d.e_fd);
}

fn small_device() -> Device {
    Device::new(DeviceParams {
        a: 1.0,
        l_ch: 4.0,
        l_ds: 4.0,
        auto_size_contacts: false,
        d_e: 0.02,
        ..Default::default()
    })
}

#[test]
fn charge_source_uses_si_to_nm_conversion_and_electron_energy_sign() {
    let mut d = small_device();
    d.calc_potential();
    let baseline = d.psi_f.clone();
    let rho = -1e5;
    d.rho.fill(rho);
    d.calc_potential();
    let shift = -d.lambda.powi(2) * rho * 1e-18 / (EPS_0 * d.params.k_si);
    for (new, old) in d.psi_f.iter().zip(baseline) {
        assert!(
            (new - old - shift).abs() < 1e-10,
            "{} != {}",
            new - old,
            shift
        );
    }
}

#[test]
fn reservoir_reference_and_landauer_current_are_in_amperes() {
    let mut d = small_device();
    d.params.e_fs = 0.2;
    d.set_v_ds(0.0);
    d.psi_0 = -10.0;
    assert_eq!(
        d.calc_thermionic_current(),
        0.0,
        "equilibrium must have zero current"
    );
    for bias in [-0.1, 0.1] {
        d.set_v_ds(bias);
        d.psi_0 = -10.0;
        let expected = 2.0 * E * E / H * bias;
        assert!((d.calc_thermionic_current() / expected - 1.0).abs() < 1e-10);
    }
}

#[test]
fn tiny_mixing_cannot_claim_a_non_fixed_point_converged() {
    let mut d = small_device();
    d.rho.fill(-1e5);
    let opts = SelfConsistentOptions {
        max_iterations: 1,
        mixing: 1e-12,
        tolerance: 1e-8,
        ..Default::default()
    };
    assert!(solve_self_consistent(&mut d, &opts).is_err());
}

#[test]
fn invalid_sweep_ranges_error_without_mutating_device() {
    for (lo, hi, step) in [
        (0.0, 1.0, -0.1),
        (1.0, 0.0, 0.1),
        (f64::NAN, 1.0, 0.1),
        (0.0, 1.0, f64::INFINITY),
    ] {
        let mut d = small_device();
        assert!(sweep_v_ds(&mut d, lo, hi, step, None).is_err());
        assert_eq!(d.params.v_ds, 0.0);
    }
}

#[test]
fn sweep_never_overshoots_requested_endpoint() {
    let mut d = small_device();
    let points = sweep_v_g(&mut d, 0.0, 0.26, 0.1, None).unwrap();
    assert_eq!(points.len(), 3);
    assert!(points
        .iter()
        .all(|p| p.voltage <= 0.26 && p.current.is_finite()));
}

#[test]
fn scf_rejects_nonfinite_inputs_instead_of_converging() {
    for tolerance in [f64::INFINITY, f64::NAN, 0.0, -1.0] {
        let mut d = small_device();
        let opts = SelfConsistentOptions {
            tolerance,
            ..Default::default()
        };
        assert!(solve_self_consistent(&mut d, &opts).is_err());
    }
    let mut d = small_device();
    d.rho[0] = f64::NAN;
    assert!(solve_self_consistent(&mut d, &SelfConsistentOptions::default()).is_err());
}

#[test]
fn zero_temperature_barrier_edge_is_finite() {
    let mut d = small_device();
    d.params.t = 0.0;
    d.set_v_ds(0.1);
    d.psi_0 = d.params.e_fs;
    assert_eq!(d.calc_thermionic_current(), 0.0);
}

#[test]
fn changing_length_resizes_all_grid_dependent_state() {
    let mut d = small_device();
    d.calc_potential();
    d.set_l_ch(8.0);
    assert_eq!(d.psi_f.len(), d.n);
    assert!(d.validate().is_ok());
}

#[test]
fn sweep_rejects_invalid_device_before_resetting_charge() {
    let mut d = small_device();
    d.params.d_e = f64::NAN;
    assert!(sweep_v_g(&mut d, 0.0, 0.1, 0.1, None).is_err());
}

#[test]
fn scf_returned_green_matches_returned_device_iterate() {
    let mut d = small_device();
    d.rho.fill(-1.0);
    let opts = SelfConsistentOptions {
        tolerance: 1.0,
        ..Default::default()
    };
    let result = solve_self_consistent(&mut d, &opts).unwrap();
    let direct = negforge_core::negf::green_function_sweep(&d, &result.green.energies, opts.eta);
    assert_eq!(result.green.g_col_source, direct.g_col_source);
}

#[test]
fn unbiased_geometry_and_potential_are_mirror_symmetric() {
    let mut d = small_device();
    d.calc_potential();
    for i in 0..d.n {
        assert_eq!(d.psi_bi[i], d.psi_bi[d.n - 1 - i]);
        assert!((d.psi_f[i] - d.psi_f[d.n - 1 - i]).abs() < 1e-12);
    }
}

#[test]
fn scf_rejects_empty_or_unbounded_energy_windows() {
    let mut d = small_device();
    d.rho.fill(-1e8);
    assert!(matches!(
        solve_self_consistent(&mut d, &SelfConsistentOptions::default()),
        Err(negforge_core::error::NegForgeError::InvalidParameter(_))
    ));
}
