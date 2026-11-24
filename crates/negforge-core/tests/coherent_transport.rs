use negforge_core::constants::{E, H, K_B};
use negforge_core::device::{Device, DeviceParams};
use num_complex::Complex64 as C;

fn wire(length: f64) -> Device {
    Device::new(DeviceParams {
        a: 0.5,
        l_ch: length,
        l_ds: 1.0,
        auto_size_contacts: false,
        ..Default::default()
    })
}

fn barrier(sites: usize) -> Device {
    let mut d = wire(sites as f64 * 0.5);
    d.psi_f[2..2 + sites].fill(0.3);
    d.psi_0 = 0.3;
    d.params.e_fs = 0.12;
    d.set_v_ds(0.06);
    d
}

// Exact discrete rectangular-barrier scattering formula, independent of G.
fn rectangular_transmission(energy: f64, height: f64, sites: usize, t: f64) -> f64 {
    let cos_q = 1.0 - (energy - height) / (2.0 * t);
    let ratio = if cos_q > 1.0 {
        (sites as f64 * cos_q.acosh()).sinh() / cos_q.acosh().sinh()
    } else if cos_q < -1.0 {
        (sites as f64 * (-cos_q).acosh()).sinh() / (-cos_q).acosh().sinh()
    } else if cos_q.abs() == 1.0 {
        sites as f64
    } else {
        (sites as f64 * cos_q.acos()).sin() / cos_q.acos().sin()
    };
    let sin_k_sq = 1.0 - (1.0 - energy / (2.0 * t)).powi(2);
    1.0 / (1.0 + height.powi(2) * ratio.powi(2) / (4.0 * t * t * sin_k_sq))
}

// Pivoted dense Gaussian solve, with independently constructed retarded leads.
#[allow(clippy::needless_range_loop)] // Keep the dense reference's matrix algebra explicit.
fn dense_transmission(d: &Device, energy: f64) -> f64 {
    let n = d.n;
    let t = d.t_hop;
    let sigma = |u: f64| {
        let x = energy - 2.0 * t - u;
        C::new(x / 2.0, -(4.0 * t * t - x * x).sqrt() / 2.0)
    };
    let s = sigma(d.psi_f[0]);
    let r = sigma(d.psi_f[n - 1]);
    let mut a = vec![vec![C::new(0.0, 0.0); n + 1]; n];
    for i in 0..n {
        a[i][i] = C::new(energy - 2.0 * t - d.psi_f[i], 0.0);
        if i > 0 {
            a[i][i - 1] = C::new(t, 0.0);
        }
        if i + 1 < n {
            a[i][i + 1] = C::new(t, 0.0);
        }
    }
    a[0][0] -= s;
    a[n - 1][n - 1] -= r;
    a[n - 1][n] = C::new(1.0, 0.0);
    for col in 0..n {
        let pivot = (col..n)
            .max_by(|&i, &j| a[i][col].norm().total_cmp(&a[j][col].norm()))
            .unwrap();
        a.swap(col, pivot);
        let p = a[col][col];
        for k in col..=n {
            a[col][k] /= p;
        }
        let row = a[col].clone();
        for i in 0..n {
            if i == col {
                continue;
            }
            let factor = a[i][col];
            for k in col..=n {
                a[i][k] -= factor * row[k];
            }
        }
    }
    4.0 * s.im * r.im * a[0][n].norm_sqr()
}

#[test]
fn rectangular_barrier_matches_scattering_formula_and_dense_inverse() {
    for sites in [1, 3, 8, 25] {
        let d = barrier(sites);
        for energy in [0.01, 0.1, 0.299, 0.3, 0.31, 0.5] {
            let actual = d.transmission_spectrum(&[energy]).unwrap()[0];
            let analytic = rectangular_transmission(energy, 0.3, sites, d.t_hop);
            let dense = dense_transmission(&d, energy);
            assert!(
                (actual / analytic - 1.0).abs() < 2e-11,
                "sites={sites} E={energy}: {actual} != {analytic}"
            );
            assert!((actual / dense - 1.0).abs() < 2e-11);
        }
    }
}

#[test]
fn tunneling_depends_on_width_and_survives_zero_temperature() {
    let mut thin = barrier(3);
    let mut thick = barrier(12);
    thin.params.t = 0.0;
    thick.params.t = 0.0;
    let a = thin.calc_negf_current().unwrap();
    let b = thick.calc_negf_current().unwrap();
    println!("zero-T barrier: width 1.5nm I={a:.12e} A; width 6nm I={b:.12e} A");
    assert!(a > b * 1000.0 && b > 0.0);
    assert_eq!(thin.calc_thermionic_current(), 0.0);
    assert_eq!(thin.calc_current(), a);
}

#[test]
fn current_energy_step_converges_to_independent_barrier_integral() {
    let mut d = barrier(8);
    let kt = K_B * d.params.t / E;
    let lo = 0.0;
    let hi = 4.0 * d.t_hop;
    let panels = 200_000;
    let width = (hi - lo) / panels as f64;
    let reference: f64 = (0..panels)
        .map(|i| {
            let energy = lo + (i as f64 + 0.5) * width;
            let fs = 1.0 / (1.0 + ((energy - d.params.e_fs) / kt).exp());
            let fd = 1.0 / (1.0 + ((energy - d.e_fd) / kt).exp());
            rectangular_transmission(energy, 0.3, 8, d.t_hop) * (fs - fd) * width
        })
        .sum::<f64>()
        * 2.0
        * E
        * E
        / H;
    let mut values = Vec::new();
    for step in [0.02, 0.01, 0.005] {
        d.params.d_e = step;
        values.push(d.calc_negf_current().unwrap());
    }
    println!("8-site barrier I(dE=.02,.01,.005)={values:?}; independent={reference:.12e} A");
    assert!((values[2] / reference - 1.0).abs() < 1e-5);
    assert!((values[2] - reference).abs() < (values[0] - reference).abs());
}

#[test]
fn current_integrates_finite_band_and_both_bias_signs() {
    let mut d = wire(5.0);
    d.params.t = 0.0;
    d.params.e_fs = 0.2;
    for bias in [0.1, -0.1, 0.0] {
        d.set_v_ds(bias);
        let expected = 2.0 * E * E / H * bias;
        let current = d.calc_negf_current().unwrap();
        println!("wire bias={bias}: I={current:.12e} A");
        assert!(
            (current - expected).abs() < 1e-15,
            "{current} != {expected}"
        );
    }
    d.params.e_fs = 10.0;
    d.set_v_ds(0.1);
    assert_eq!(
        d.calc_negf_current().unwrap(),
        0.0,
        "no occupied difference inside finite band"
    );
}

#[test]
fn finite_temperature_wire_matches_exact_finite_band_integral() {
    let mut d = wire(20.0);
    for bias in [0.1, -0.1, 1e-10] {
        d.set_v_ds(bias);
        let kt = K_B * d.params.t / E;
        let primitive =
            |mu: f64| kt * ((mu / kt).exp().ln_1p() - ((mu - 4.0 * d.t_hop) / kt).exp().ln_1p());
        let expected = 2.0 * E * E / H * (primitive(d.params.e_fs) - primitive(d.e_fd));
        assert!((d.calc_negf_current().unwrap() / expected - 1.0).abs() < 2e-6);
    }
    d.set_v_ds(0.0);
    d.psi_f[4] = 0.5;
    assert_eq!(d.calc_negf_current().unwrap(), 0.0);
}

#[test]
fn perfect_wire_has_unit_transmission_without_absorption() {
    for length in [2.0, 1000.0] {
        let d = wire(length);
        let t = d.t_hop;
        let es = [-t, 0.0, 0.01 * t, t, 2.0 * t, 3.99 * t, 4.0 * t, 5.0 * t];
        let ts = d.transmission_spectrum(&es).unwrap();
        for (i, value) in ts.iter().enumerate() {
            let expected = if (2..6).contains(&i) { 1.0 } else { 0.0 };
            assert!(
                (value - expected).abs() < 1e-10,
                "length={length} E={} T={value}",
                es[i]
            );
        }
    }
}

#[test]
fn evanescent_lead_and_disjoint_bands_have_no_transport() {
    let mut d = wire(3.0);
    d.psi_f[d.n - 1] = 0.2;
    assert_eq!(d.transmission_spectrum(&[0.1]).unwrap(), [0.0]);
    d.set_v_ds(0.1);
    d.psi_f[d.n - 1] = 5.0 * d.t_hop;
    assert_eq!(d.calc_negf_current().unwrap(), 0.0);
}

#[test]
fn transmission_and_current_reject_invalid_state() {
    let mut d = wire(3.0);
    assert!(d.transmission_spectrum(&[f64::NAN]).is_err());
    assert_eq!(d.transmission_spectrum(&[]).unwrap(), Vec::<f64>::new());
    d.psi_f[1] = f64::NAN;
    assert!(d.calc_negf_current().is_err());
    assert!(d.transmission_spectrum(&[0.1]).is_err());
    d.psi_f[1] = 0.0;
    d.set_v_ds(0.1);
    d.params.d_e = 1e-12;
    assert!(d.calc_negf_current().is_err());
}

#[test]
fn unequal_leads_match_dense_and_are_reciprocal_and_gauge_covariant() {
    let mut d = wire(3.0);
    for i in 0..d.n {
        d.psi_f[i] = 0.08 * i as f64 / (d.n - 1) as f64;
    }
    d.psi_f[3] += 0.17;
    let es = [0.09, 0.15, 0.3, 0.6];
    let ts = d.transmission_spectrum(&es).unwrap();
    for (&energy, &actual) in es.iter().zip(&ts) {
        assert!((actual / dense_transmission(&d, energy) - 1.0).abs() < 1e-12);
    }
    d.psi_f.reverse();
    for (&a, b) in ts.iter().zip(d.transmission_spectrum(&es).unwrap()) {
        assert!((a - b).abs() < 1e-12);
    }
    d.set_v_ds(-0.05);
    let current = d.calc_negf_current().unwrap();
    for u in &mut d.psi_f {
        *u += 0.7;
    }
    d.params.e_fs += 0.7;
    d.e_fd += 0.7;
    assert!((d.calc_negf_current().unwrap() / current - 1.0).abs() < 1e-11);
}

#[test]
fn sweeps_propagate_coherent_integration_errors() {
    let mut d = wire(3.0);
    d.params.d_e = 1e-12;
    assert!(negforge_core::sweep::sweep_v_ds(&mut d, 0.1, 0.1, 0.1, None).is_err());
}

#[test]
fn opaque_barrier_does_not_overflow() {
    let mut d = wire(1000.0);
    d.psi_f[1..d.n - 1].fill(1.0);
    assert_eq!(d.transmission_spectrum(&[0.1]).unwrap(), [0.0]);
}
