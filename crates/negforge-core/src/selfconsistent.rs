//! Self-consistent Poisson <-> NEGF loop.
//!
//! Solve electrostatics, compute the NEGF electron density, convert it to
//! charge, mix it into `rho`, and repeat. Convergence uses the undamped
//! fixed-point potential residual; exhausting the iteration budget returns
//! an error rather than an unconverged result.
//!
//! ## Charge-density units
//!
//! `calc_potential`'s RHS divides `rho + N_dot` by `EPS_0 * k_si`, with
//! `EPS_0` in SI units (F/m). For that division to land back in the same
//! eV/nm^2 ballpark as the other RHS term (`(Psi_g + Psi_bi) / lambda^2`),
//! `rho` is a charge density in C/m^3; the SI source must also be scaled
//! by `1e-18` to match the nm^-2 Laplacian.
//! This loop converts the NEGF electron density to a
//! volume charge density before assigning it to `rho`:
//!
//! 1. [`charge::electron_density`] returns a linear density in nm^-1
//!    (natural units for this model's nm-scaled grid).
//! 2. Convert to m^-1 (`* 1e9`). With an explicit `effective_area_m2`,
//!    multiply by two for spin and divide by that area to obtain m^-3.
//!    With no area, use the per-spin, implicit 1 m^2 normalization; this
//!    default is not a physical nanoscale cross-section.
//! 3. Multiply by `-e` to obtain negative electronic charge in C/m^3.
//!    [`Device::mobile_charge_density`] performs this conversion.
//!
//! `N_dot` (fixed dopant charge) is left as-is (default zero); if it is
//! used with a nonzero value it should be supplied in the same C/m^3 units
//! for consistency.
//!
//! ## Validity: mixing, tolerance, and convergence are numerical, not physical
//!
//! `mixing` (linear/Picard mixing of the charge-density update) and `eta`
//! are numerical stabilization knobs, not physical parameters. Mixing
//! affects the convergence rate, not the fixed point. Finite `eta` changes
//! the evaluated density, so broadening and grid refinement must be tested
//! separately from fixed-point convergence. Resolve sharp resonances by
//! refining `d_e` relative to their widths (including `eta`); no universal
//! minimum eta is a physical convergence criterion. A run that fails to
//! converge (`NotConverged`) is not
//! evidence the device physics is invalid — it may just need more
//! `max_iterations`, a smaller `mixing`, or a larger `eta`. Conversely,
//! convergence is a necessary but not sufficient condition for physical
//! correctness: a converged fixed point still inherits every approximation
//! described in `device.rs`, `negf.rs` and `charge.rs` (ballistic
//! transport, natural-length electrostatics, no explicit spin-degeneracy
//! factor in the charge density, etc.).

use crate::charge;
#[cfg(test)]
use crate::constants::E as ELEMENTARY_CHARGE;
use crate::device::Device;
use crate::error::{NegForgeError, Result};
use crate::negf::{self, GreenFunctionResult};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SelfConsistentOptions {
    /// Maximum number of Poisson<->NEGF iterations before giving up.
    pub max_iterations: usize,
    /// Converged when the undamped Poisson/NEGF fixed-point potential
    /// residual (eV) drops below this, independent of the mixing factor.
    pub tolerance: f64,
    /// Linear mixing factor in `(0, 1]` applied to the charge-density
    /// update (simple damping to stabilize the fixed-point iteration).
    pub mixing: f64,
    /// Fraction of the span from min(psi_f) to device.e_max (not a fraction
    /// of an absolute energy). The upper limit is raised to cover both Fermi
    /// levels, then capped at the highest lead band top. Values below one may
    /// truncate thermal tails; use one for the epsilon-based cutoff and test
    /// convergence. The default is 0.7.
    pub green_energy_fraction: f64,
    /// Imaginary broadening used for the NEGF evaluations inside the loop.
    /// Should be several times the energy-grid step (`device.params.d_e`)
    /// so that resonances are resolved smoothly across iterations instead
    /// of spiking whenever a grid point happens to land near a pole — see
    /// the module docs and [`crate::negf::DEFAULT_ETA`]. Increase this if
    /// the loop fails to converge; decrease for sharper resonance
    /// resolution (at the cost of a noisier, potentially non-converging
    /// iteration).
    pub eta: f64,
}

impl Default for SelfConsistentOptions {
    fn default() -> Self {
        Self {
            max_iterations: 50,
            tolerance: 1e-6,
            mixing: 0.3,
            green_energy_fraction: 0.7,
            eta: 0.08,
        }
    }
}

pub struct SelfConsistentResult {
    pub iterations: usize,
    pub residual: f64,
    pub green: GreenFunctionResult,
    pub electron_density: Vec<f64>,
}

fn negf_energy_grid(device: &Device, fraction: f64) -> Result<Vec<f64>> {
    let e_min = device.psi_f.iter().cloned().fold(f64::INFINITY, f64::min);
    // Fractional truncation acts on a span, never on an absolute energy:
    // shifting all potentials and reservoir energies must shift this grid.
    // Do not let that truncation remove the occupied interval below either mu.
    let mu_high = device.params.e_fs.max(device.e_fd);
    let band_top = device.psi_f[0].max(device.psi_f[device.n - 1]) + 4.0 * device.t_hop;
    let e_max = (e_min + fraction * (device.e_max - e_min))
        .max(mu_high)
        .min(band_top);
    let d_e = device.params.d_e;
    let count = crate::resources::energy_count(e_min, e_max, d_e)?;
    crate::resources::check_working_set(device.n, count)?;
    // Reuse the conservatively checked count as the number of midpoint cells.
    // Their width is <= d_e and covers the complete window without sampling
    // its singular band-edge endpoint or rounding away the final partial cell.
    let width = (e_max - e_min) / count as f64;
    Ok((0..count)
        .map(|k| e_min + (k as f64 + 0.5) * width)
        .collect())
}

/// Run the self-consistent Poisson<->NEGF loop on `device`, mutating its
/// `rho` and `psi_f` in place. Returns diagnostics plus the final NEGF
/// sweep and electron density on success.
pub fn solve_self_consistent(
    device: &mut Device,
    opts: &SelfConsistentOptions,
) -> Result<SelfConsistentResult> {
    if !(0.0 < opts.mixing && opts.mixing <= 1.0) {
        return Err(NegForgeError::InvalidParameter(format!(
            "mixing must be in (0, 1], got {}",
            opts.mixing
        )));
    }

    if opts.max_iterations == 0
        || !opts.tolerance.is_finite()
        || opts.tolerance <= 0.0
        || !opts.eta.is_finite()
        || opts.eta <= 0.0
        || !opts.green_energy_fraction.is_finite()
        || opts.green_energy_fraction <= 0.0
        || opts.green_energy_fraction > 1.0
    {
        return Err(NegForgeError::InvalidParameter(
            "SCF requires positive finite tolerance, eta, iterations and energy fraction in (0, 1]"
                .into(),
        ));
    }
    device.validate()?;

    device.calc_potential();

    let mut last_residual = f64::INFINITY;
    for iteration in 1..=opts.max_iterations {
        if !device.psi_f.iter().all(|v| v.is_finite()) {
            return Err(NegForgeError::InvalidParameter(
                "non-finite SCF potential".into(),
            ));
        }
        let energies = negf_energy_grid(device, opts.green_energy_fraction)?;
        // The SCF grid contains equal-width midpoint cells, not endpoint
        // endpoint rectangles with weight params.d_e.
        let quadrature_width = energies[1] - energies[0];
        let green = negf::green_function_sweep(device, &energies, opts.eta);
        let n_electron = charge::electron_density(
            &green,
            device.t_hop,
            device.params.a,
            device.psi_f[0],
            device.psi_f[device.n - 1],
            device.params.e_fs,
            device.e_fd,
            device.params.t,
            quadrature_width,
        );

        if n_electron.len() != device.n || !n_electron.iter().all(|v| v.is_finite() && *v >= 0.0) {
            return Err(NegForgeError::InvalidParameter(
                "invalid NEGF electron density".into(),
            ));
        }
        let rho_before = device.rho.clone();
        let psi_before = device.psi_f.clone();
        device.rho = n_electron
            .iter()
            .map(|&n| device.mobile_charge_density(n))
            .collect();

        device.calc_potential();
        if !device.psi_f.iter().all(|v| v.is_finite()) {
            return Err(NegForgeError::InvalidParameter(
                "non-finite SCF target potential".into(),
            ));
        }

        let residual = psi_before
            .iter()
            .zip(device.psi_f.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        last_residual = residual;

        if residual < opts.tolerance {
            // These fields and the returned NEGF data all describe the
            // same iterate, whose undamped fixed-point residual was tested.
            device.rho = rho_before;
            device.psi_f = psi_before;
            device.psi_0 = device
                .psi_f
                .iter()
                .copied()
                .fold(f64::NEG_INFINITY, f64::max);
            return Ok(SelfConsistentResult {
                iterations: iteration,
                residual,
                green,
                electron_density: n_electron,
            });
        }
        for (rho_i, old) in device.rho.iter_mut().zip(rho_before) {
            *rho_i = old + opts.mixing * (*rho_i - old);
        }
        device.calc_potential();
    }

    Err(NegForgeError::NotConverged {
        iterations: opts.max_iterations,
        residual: last_residual,
        tolerance: opts.tolerance,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::DeviceParams;

    fn window_device() -> Device {
        let mut device = Device::new(DeviceParams {
            a: 0.5,
            l_ch: 3.0,
            l_ds: 2.0,
            auto_size_contacts: false,
            d_e: 0.001,
            ..Default::default()
        });
        device.psi_f.fill(-0.2);
        device
    }

    #[test]
    fn scf_energy_window_is_gauge_covariant() {
        let device = window_device();
        let original = negf_energy_grid(&device, 0.7).unwrap();
        for shift in [-3.0, 2.0] {
            let mut shifted = device.clone();
            shifted.psi_f.iter_mut().for_each(|p| *p += shift);
            shifted.params.e_fs += shift;
            shifted.e_fd += shift;
            shifted.e_max += shift;
            let translated = negf_energy_grid(&shifted, 0.7).unwrap();
            assert_eq!(original.len(), translated.len());
            for (a, b) in original.iter().zip(translated) {
                assert!((a + shift - b).abs() < 1e-12);
            }
        }
    }

    #[test]
    fn scf_fraction_does_not_cut_occupied_reverse_bias_states() {
        let mut device = window_device();
        device.set_v_ds(-0.4);
        device.psi_f.fill(0.0);
        device.params.t = 0.0;
        device.e_max = device.params.e_fs.max(device.e_fd);
        let energies = negf_energy_grid(&device, 0.7).unwrap();
        // At zero T all energies up to the higher reservoir mu are occupied.
        // Fractional truncation must not cut into that interval.
        assert!(energies.last().unwrap() >= &(device.e_fd - device.params.d_e));
    }

    #[test]
    fn scf_grid_samples_cell_interiors_not_band_edges() {
        let mut device = window_device();
        device.psi_f.fill(0.0);
        device.e_max = 0.1;
        device.params.e_fs = 0.05;
        device.e_fd = 0.05;
        device.params.d_e = 0.03;
        let energies = negf_energy_grid(&device, 1.0).unwrap();
        let width = energies[1] - energies[0];
        assert!((energies[0] - width / 2.0).abs() < 1e-14);
        assert!((energies.last().unwrap() + width / 2.0 - 0.1).abs() < 1e-14);
        assert!(width <= device.params.d_e);
    }

    /// Independent infinite-chain state counting: each reservoir fills one
    /// direction, n=(k_Fs+k_Fd)/(2*pi*a). This checks the SCF quadrature and
    /// its weight, rather than just comparing two implementations of G.
    #[test]
    fn scf_quadrature_matches_nonequilibrium_chain_state_counting() {
        let mut device = window_device();
        device.psi_f.fill(0.0);
        device.params.t = 0.0;
        device.params.e_fs = 0.15;
        device.e_fd = 0.45;
        device.e_max = 0.45;
        // Avoid an integer cell-count boundary: roundoff under a gauge shift
        // may legitimately add/remove one cell there (a quadrature error).
        device.params.d_e = 0.00021;
        let k = |mu: f64| (1.0 - mu / (2.0 * device.t_hop)).acos();
        let expected = (k(device.params.e_fs) + k(device.e_fd))
            / (2.0 * std::f64::consts::PI * device.params.a);
        let evaluate = |d: &Device| {
            let energies = negf_energy_grid(d, 0.7).unwrap();
            let width = energies[1] - energies[0];
            let green = negf::green_function_sweep(d, &energies, 1e-10 * d.t_hop);
            charge::electron_density(
                &green,
                d.t_hop,
                d.params.a,
                d.psi_f[0],
                d.psi_f[d.n - 1],
                d.params.e_fs,
                d.e_fd,
                0.0,
                width,
            )
        };
        let original = evaluate(&device);
        for &n in &original {
            assert!((n - expected).abs() < 0.007, "{n} vs analytic {expected}");
        }
        for shift in [-1.25, 0.75] {
            let mut shifted = device.clone();
            shifted.psi_f.fill(shift);
            shifted.params.e_fs += shift;
            shifted.e_fd += shift;
            shifted.e_max += shift;
            for (a, b) in original.iter().zip(evaluate(&shifted)) {
                assert!(
                    (a - b).abs() < 1e-9,
                    "gauge-dependent occupation: {a} vs {b}"
                );
            }
        }
        // Raising both chemical potentials past the finite lattice band must
        // fill exactly one per-spin orbital/site, not truncate at 0.7*mu.
        device.params.e_fs = 2.0;
        device.e_fd = 2.0;
        device.e_max = 2.0;
        for n in evaluate(&device) {
            assert!((n * device.params.a - 1.0).abs() < 0.008);
        }
    }

    #[test]
    fn scf_finite_temperature_matches_independent_momentum_integral() {
        let mut device = window_device();
        device.psi_f.fill(0.0);
        device.params.e_fs = 0.04;
        device.e_fd = 0.3;
        device.params.t = 300.0;
        device.params.d_e = 0.00021;
        let theta = crate::constants::K_B * device.params.t / ELEMENTARY_CHARGE;
        device.e_max = device.e_fd - theta * device.params.epsilon.ln();
        // Momentum-space quadrature has no band-edge DOS singularities and
        // does not use the NEGF solver or its energy quadrature.
        let samples = 50_000;
        let expected: f64 = (0..samples)
            .map(|i| {
                let k = std::f64::consts::PI * (i as f64 + 0.5) / samples as f64;
                let energy = 2.0 * device.t_hop * (1.0 - k.cos());
                let f = |mu: f64| 1.0 / (1.0 + ((energy - mu) / theta).exp());
                (f(device.params.e_fs) + f(device.e_fd)) / 2.0
            })
            .sum::<f64>()
            / (samples as f64 * device.params.a);
        let energies = negf_energy_grid(&device, 1.0).unwrap();
        let width = energies[1] - energies[0];
        let green = negf::green_function_sweep(&device, &energies, 1e-10 * device.t_hop);
        let density = charge::electron_density(
            &green,
            device.t_hop,
            device.params.a,
            0.0,
            0.0,
            device.params.e_fs,
            device.e_fd,
            device.params.t,
            width,
        );
        for n in density {
            assert!((n - expected).abs() < 0.007, "{n} vs k-space {expected}");
        }
    }

    #[test]
    fn converges_for_a_small_device() {
        let mut device = Device::new(DeviceParams {
            a: 1.0,
            l_ch: 10.0,
            l_ds: 10.0,
            auto_size_contacts: false,
            d_e: 0.01,
            ..Default::default()
        });

        let opts = SelfConsistentOptions {
            max_iterations: 100,
            tolerance: 1e-5,
            mixing: 0.3,
            green_energy_fraction: 0.7,
            eta: 0.04,
        };
        let result = solve_self_consistent(&mut device, &opts).expect("should converge");
        assert!(result.residual < opts.tolerance);
        assert_eq!(result.electron_density.len(), device.n);
        // Structurally guaranteed (electron_density is a sum of |G|^2-weighted
        // terms, see charge.rs) regardless of the contact self-energy sign
        // convention in negf.rs — worth locking in as a regression check
        // since it's the one physical positivity requirement the loop
        // actually depends on.
        assert!(result
            .electron_density
            .iter()
            .all(|v| v.is_finite() && *v >= 0.0));
        assert!(device.psi_f.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn rejects_invalid_mixing() {
        let mut device = Device::new(DeviceParams {
            a: 1.0,
            l_ch: 10.0,
            l_ds: 10.0,
            auto_size_contacts: false,
            ..Default::default()
        });
        let opts = SelfConsistentOptions {
            mixing: 0.0,
            ..Default::default()
        };
        assert!(solve_self_consistent(&mut device, &opts).is_err());
    }
}
