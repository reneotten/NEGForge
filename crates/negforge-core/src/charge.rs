//! NEGF-derived charge density, feeding the self-consistent electrostatic
//! loop.
//!
//! Each contact contributes its energy-dependent Fermi occupation weighted
//! by its own propagating-band injection. Evanescent leads shift the
//! retarded Hamiltonian but inject no particles.
//!
//! The return value is a linear number density in nm^-1. Multiplication by
//! `1e9` converts to m^-1; conversion to volume charge additionally requires
//! a specified transverse area and multiplication by `-e`. A choice of units
//! cannot repair a missing cross-section or SI/nm conversion in Poisson.
//!
//! ## Validity: normalization and what's still a simplification
//!
//! Every term summed into `n` is `(non-negative broadening) * (Fermi
//! occupation in (0,1)) * |G|^2`, so `n(x) >= 0` always holds by
//! construction, independent of any Green's-function sign convention.
//!
//! `gamma_s`/`gamma_d` here are `t_hop * sin(k)`, i.e. *half* of the
//! standard contact broadening `Gamma = -2*Im(Sigma) = 2*t_hop*sin(k)`
//! (see `negf.rs` for the self-energy `Sigma` this comes from); combined
//! with this module's `1/pi` prefactor (rather than the more commonly
//! quoted `Gamma * .../(2*pi)`), the two halvings cancel and the result
//! matches the textbook formula
//! `n(x) = (1/2*pi) * integral[Gamma_S(E) f_S(E) |G_{x,1}|^2 +
//! Gamma_D(E) f_D(E) |G_{x,N}|^2] dE` exactly. One thing that is *not* a
//! mere normalization choice: there is no explicit spin-degeneracy factor
//! of two here (unlike `calc_current`'s explicit `2*e/h`). Thus this API
//! returns a **per-spin**, single-orbital linear density in nm^-1. A Fermi
//! probability cannot encode spin degeneracy: multiply by two downstream
//! when a spin-summed density is required.
//!
//! The rectangular sum assumes a uniform grid with the caller's `d_e`;
//! grid endpoints, finite window truncation, and narrow resonances require
//! explicit quadrature convergence checks. Only contact-injected charge is
//! included: finite device eta has no matching injection, and isolated bound
//! states have no occupation prescription. See `crate::negf`.

use crate::constants::{E, K_B};
use crate::negf::GreenFunctionResult;

/// Compute the electron number density `n(x)` from a Green's-function
/// sweep. The result is per spin in nm^-1; see the module docs for units
/// and quadrature assumptions.
#[allow(clippy::too_many_arguments)]
pub fn electron_density(
    green: &GreenFunctionResult,
    t_hop: f64,
    a: f64,
    psi_f_source: f64,
    psi_f_drain: f64,
    e_fs: f64,
    e_fd: f64,
    temperature: f64,
    d_e: f64,
) -> Vec<f64> {
    let n_sites = green.g_col_source[0].len();
    let mut n = vec![0.0; n_sites];

    assert!(temperature.is_finite() && temperature >= 0.0);
    let fermi = |energy: f64, mu: f64| {
        if temperature == 0.0 {
            if energy < mu {
                1.0
            } else if energy > mu {
                0.0
            } else {
                0.5
            }
        } else {
            1.0 / (((energy - mu) * E / (K_B * temperature)).exp() + 1.0)
        }
    };
    let f_s = |energy: f64| fermi(energy, e_fs);
    let f_d = |energy: f64| fermi(energy, e_fd);

    for (k, &energy) in green.energies.iter().enumerate() {
        let gamma_s = if energy > psi_f_source {
            t_hop * green.k_sa[k].re.sin()
        } else {
            0.0
        };
        let gamma_d = if energy > psi_f_drain {
            t_hop * green.k_da[k].re.sin()
        } else {
            0.0
        };

        let weight_s = gamma_s * f_s(energy);
        let weight_d = gamma_d * f_d(energy);

        for (i, n_i) in n.iter_mut().enumerate().take(n_sites) {
            *n_i += weight_s * green.g_col_source[k][i] + weight_d * green.g_col_drain[k][i];
        }
    }

    for v in n.iter_mut() {
        *v = *v / std::f64::consts::PI * d_e / a;
    }
    n
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use num_complex::Complex64;

    #[test]
    fn equilibrium_density_matches_analytic_infinite_chain_per_spin() {
        use crate::device::{Device, DeviceParams};
        use crate::negf::green_function_sweep;
        let mut device = Device::new(DeviceParams {
            a: 1.0,
            l_ch: 3.0,
            l_ds: 2.0,
            auto_size_contacts: false,
            ..Default::default()
        });
        device.psi_f.fill(0.0);
        let t = device.t_hop;
        // Midpoint quadrature avoids the integrable band-edge singularities.
        let steps = 2000;
        let d_e = 2.0 * t / steps as f64;
        let energies: Vec<_> = (0..steps).map(|i| (i as f64 + 0.5) * d_e).collect();
        let green = green_function_sweep(&device, &energies, 1e-12 * t);
        let n = electron_density(&green, t, 1.0, 0.0, 0.0, 2.0 * t, 2.0 * t, 0.0, d_e);
        // Half a filled single-spin band contains 1/2 electron per site.
        for value in n {
            assert_relative_eq!(value, 0.5, epsilon = 0.004);
        }
    }

    #[test]
    fn zero_temperature_fermi_edge_has_half_occupation() {
        let green = GreenFunctionResult {
            energies: vec![1.0, 2.0, 3.0],
            g_diag: vec![vec![0.0]; 3],
            g_col_source: vec![vec![1.0]; 3],
            g_col_drain: vec![vec![0.0]; 3],
            k_sa: vec![Complex64::new(std::f64::consts::FRAC_PI_2, 0.0); 3],
            k_da: vec![Complex64::new(0.0, 0.0); 3],
        };
        let n = electron_density(&green, 1.0, 2.0, 0.0, 0.0, 2.0, 2.0, 0.0, 1.0);
        assert_relative_eq!(n[0], 1.5 / (2.0 * std::f64::consts::PI), epsilon = 1e-14);
    }
}
