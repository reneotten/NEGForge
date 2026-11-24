use negforge_core::{resources, selfconsistent::solve_self_consistent, sweep};
use negforge_core::{Device, DeviceParams, SelfConsistentOptions};

fn device(v_g: f64, t: f64) -> Device {
    Device::new(DeviceParams {
        a: 0.5,
        l_ch: 40.0,
        l_ds: 4.0,
        auto_size_contacts: false,
        d_e: 0.001,
        v_g,
        t,
        ..Default::default()
    })
}

#[test]
fn depleted_scf_gate_and_drain_sweeps() {
    for (v_g, t) in [(0.0, 0.0), (0.0, 10.0), (-2.0, 300.0)] {
        let mut dev = device(v_g, t);
        let opts = SelfConsistentOptions::default();
        let gate = sweep::sweep_v_g(&mut dev, v_g - 0.1, v_g, 0.05, Some(&opts)).unwrap();
        assert_eq!(gate.len(), 3);
        assert!(gate.iter().all(|p| p.current == 0.0));
        let drain = sweep::sweep_v_ds(&mut dev, 0.0, 0.1, 0.05, Some(&opts)).unwrap();
        assert_eq!(drain.len(), 3);
        assert!(drain
            .iter()
            .all(|p| p.current.is_finite() && p.current >= 0.0));
    }
}

#[test]
fn positive_window_smaller_than_step_solves_and_uses_cell_width() {
    let mut dev = device(1.0, 0.0);
    dev.params.d_e = 10.0;
    let result = solve_self_consistent(&mut dev, &SelfConsistentOptions::default()).unwrap();
    assert_eq!(result.green.energies.len(), 1);
    let min = dev.psi_f.iter().copied().fold(f64::INFINITY, f64::min);
    // One midpoint in [min, max] gives width = 2*(mid-min), not d_e.
    let width = 2.0 * (result.green.energies[0] - min);
    assert!(width > 0.0 && width < dev.params.d_e);
    let expected = negforge_core::charge::electron_density(
        &result.green,
        dev.t_hop,
        dev.params.a,
        dev.psi_f[0],
        dev.psi_f[dev.n - 1],
        dev.params.e_fs,
        dev.e_fd,
        0.0,
        width,
    );
    assert_eq!(expected, result.electron_density);
    assert!(expected.iter().any(|n| *n > 0.0));
}

#[test]
fn midpoint_inputs_and_resource_limits_remain_distinct() {
    assert_eq!(resources::midpoint_count(1.0, 1.0, 0.001).unwrap(), 0);
    assert_eq!(resources::midpoint_count(0.0, 0.0002, 0.001).unwrap(), 1);
    for (min, max, step) in [
        (2.0, 1.0, 0.001),
        (0.0, 1.0, 0.0),
        (f64::NAN, 1.0, 0.001),
        (0.0, f64::INFINITY, 0.001),
    ] {
        let msg = resources::midpoint_count(min, max, step)
            .unwrap_err()
            .to_string();
        assert!(msg.contains("finite"));
        assert!(!msg.contains("resource budget"));
    }
    assert!(resources::midpoint_count(0.0, 1.0, 1e-12)
        .unwrap_err()
        .to_string()
        .contains("resource budget"));
    assert!(resources::check_working_set(97, 100_000).is_err());
    assert!(resources::check_working_set(resources::MAX_SITES + 1, 1).is_err());
    let mut dev = device(1.0, 300.0);
    dev.params.d_e = 1e-12;
    assert!(
        solve_self_consistent(&mut dev, &SelfConsistentOptions::default())
            .err()
            .unwrap()
            .to_string()
            .contains("resource budget")
    );
}
