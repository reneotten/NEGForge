//! Non-Equilibrium Green's Function (NEGF) transport calculation.
//!
//! Builds the tight-binding Hamiltonian for the potential profile `psi_f`, attaches
//! open-boundary self-energies at the source/drain contacts, and computes
//! the retarded Green's function `G^r(E) = [(E + i*eta) I - H]^-1` for a
//! grid of energies.
//!
//! Energy-resolved coherent transmission uses a separate scaled-continuant
//! recurrence, O(N) work and O(1) scratch per energy. It evaluates the exact
//! real-energy lead broadenings and introduces **no** device absorbing eta;
//! density/LDOS broadening below does not enter the Landauer current.
//!
//! Only the diagonal (local density of states) and the first/last columns
//! (needed for the injected charge density) are needed. The O(N) methods are:
//!
//! - left- and right-connected recursive Green's function sweeps for the full
//!   diagonal, and
//! - a direct tridiagonal solve (O(N), via `tridiag::solve_complex`) for
//!   each of the two boundary columns, sidestepping the (easy to get wrong)
//!   off-diagonal recursive Green's function formulas entirely.
//!
//! Both pieces are cross-checked against a dense reference solver in the
//! test module below (that check validates the O(N) algorithms reproduce
//! whatever matrix is actually constructed; it does not by itself validate
//! that the matrix's physics — the self-energy formula, sign convention,
//! etc. — is right, which is discussed below).
//!
//! ## Validity of the tight-binding discretization
//!
//! The on-site/hopping parameters (`2*t_hop + Psi_f`, `-t_hop`, with
//! `t_hop = hbar^2 / (2 m_eff a^2)`) are the standard finite-difference
//! discretization of the effective-mass Schrodinger equation, and reproduce
//! the continuum parabolic dispersion `E = hbar^2 k^2 / (2 m_eff)` only for
//! small `k*a` (long wavelength compared to the grid spacing). A
//! tight-binding chain's true dispersion is `E = 2*t_hop*(1 - cos(k*a))`,
//! bounded above by a bandwidth of `4*t_hop`; states populated near or above
//! that bandwidth are lattice artifacts with no continuum-model
//! counterpart, not real physics. In practice this means `a` must be chosen
//! fine enough that the energy window actually swept (`psi_0` up to
//! `e_max`) stays well below `4*t_hop` — worth checking (e.g. by halving
//! `a` and confirming the results are unchanged) before trusting results at
//! a new device scale.
//!
//! ## Contact self-energies and physical limits
//!
//! Semi-infinite leads have on-site energy `epsilon = 2*t + psi_contact`.
//! Their retarded self-energy solves `Sigma^2 - (E-epsilon)*Sigma + t^2=0`.
//! Inside the band the root has negative imaginary part; outside either
//! band edge the real root must decay (`|Sigma| <= t`). Evanescent leads
//! therefore contribute a real shift even though their injection is zero.
//! The implementation selects these roots explicitly, avoiding complex-acos
//! branch ambiguities and cancellation far outside the band.
//!
//! Returned `k_sa`/`k_da` use the reflected-acos convention; they
//! are not the canonical physical wavevectors outside the propagating band.
//! Self-energies use real E with the retarded 0+ prescription, while eta is
//! applied only to the finite device. This is an absorbing regularization,
//! not a conserving dephasing model: charge includes contact injection but
//! no matching lesser self-energy for eta. Finite eta suppresses occupation
//! and changes the fixed point; test convergence as eta and dE decrease.
//! Contact-only injection also cannot assign an occupation to true bound
//! states disconnected from both lead continua.

use num_complex::Complex64;

use crate::device::Device;
use crate::tridiag;

/// Default imaginary broadening added to the energy, `E + i*eta`, standing
/// in for an infinitesimal escape rate. Suitable for a local-density-of-states plot at a
/// fixed potential.
///
/// The self-consistent loop may choose a larger eta to smooth resonances.
/// That changes the density (absorption without compensating injection),
/// rather than merely accelerating an otherwise identical calculation.
/// Energy resolution and eta convergence must be checked independently.
pub const DEFAULT_ETA: f64 = 1e-8;

/// Result of a [`green_function_sweep`] call: retarded-Green's-function
/// quantities on an energy grid, one entry per energy point.
pub struct GreenFunctionResult {
    /// Energy grid, eV.
    pub energies: Vec<f64>,
    /// `-Im(G_ii(E)) / a` for every site `i`, indexed `[energy][site]`.
    /// Divide by pi for the per-spin LDOS per unit length, in (eV nm)^-1.
    pub g_diag: Vec<Vec<f64>>,
    /// `|G_{i,0}(E)|^2` for every site `i`, indexed `[energy][site]`
    /// (source-injected column).
    pub g_col_source: Vec<Vec<f64>>,
    /// `|G_{i,N-1}(E)|^2` for every site `i`, indexed `[energy][site]`
    /// (drain-injected column).
    pub g_col_drain: Vec<Vec<f64>>,
    /// Source contact wavevector at each energy (complex: evanescent below
    /// the contact band edge).
    pub k_sa: Vec<Complex64>,
    /// Drain contact wavevector at each energy.
    pub k_da: Vec<Complex64>,
}

/// Run the NEGF sweep over `energies` for the device's current `psi_f`,
/// with imaginary broadening `eta` (see [`DEFAULT_ETA`] for guidance on
/// choosing it). The caller chooses the energy grid.
pub fn green_function_sweep(device: &Device, energies: &[f64], eta: f64) -> GreenFunctionResult {
    assert!(
        eta.is_finite() && eta > 0.0,
        "eta must be finite and positive"
    );
    assert!(
        energies.iter().all(|e| e.is_finite()),
        "energies must be finite"
    );
    assert!(device.n > 0 && device.psi_f.len() == device.n);
    assert!(device.psi_f.iter().all(|p| p.is_finite()));
    assert!(device.t_hop.is_finite() && device.t_hop > 0.0);
    assert!(device.params.a.is_finite() && device.params.a > 0.0);
    // Empty sweeps have no rows but still require valid sites and budgeted
    // scratch. Conservatively check one row; do not relax global limits.
    crate::resources::check_working_set(device.n, energies.len().max(1))
        .expect("NEGF grid exceeds practical resource budget");
    let n = device.n;
    let t = device.t_hop;
    let a = device.params.a;
    let psi_0s = device.psi_f[0];
    let psi_0d = device.psi_f[n - 1];

    // Bulk on-site energies (before contact self-energy is applied).
    let bulk_diag: Vec<f64> = device.psi_f.iter().map(|p| 2.0 * t + p).collect();
    let c1 = bulk_diag[0];
    let c_n = bulk_diag[n - 1];
    let off_diag = vec![t; n - 1]; // constant real off-diagonal of (E+i eta)I - H

    let mut g_diag = Vec::with_capacity(energies.len());
    let mut g_col_source = Vec::with_capacity(energies.len());
    let mut g_col_drain = Vec::with_capacity(energies.len());
    let mut k_sa = Vec::with_capacity(energies.len());
    let mut k_da = Vec::with_capacity(energies.len());

    for &e in energies {
        let ka_s = Complex64::new(-(2.0 * t - e + psi_0s) / (2.0 * t), 0.0).acos();
        let ka_d = Complex64::new(-(2.0 * t - e + psi_0d) / (2.0 * t), 0.0).acos();
        k_sa.push(ka_s);
        k_da.push(ka_d);

        let mut diag: Vec<Complex64> = bulk_diag
            .iter()
            .map(|&d| Complex64::new(e, eta) - Complex64::new(d, 0.0))
            .collect();

        // Evanescent leads have zero broadening but a nonzero real shift.
        // Subtraction (not assignment) also handles both leads on one site.
        diag[0] -= contact_self_energy(e, c1, t);
        diag[n - 1] -= contact_self_energy(e, c_n, t);

        let full_diag = full_diagonal(&off_diag, &diag);

        let mut e_source = vec![Complex64::new(0.0, 0.0); n];
        e_source[0] = Complex64::new(1.0, 0.0);
        let col_source = tridiag::solve_complex(&off_diag, &diag, &off_diag, &e_source);

        let mut e_drain = vec![Complex64::new(0.0, 0.0); n];
        e_drain[n - 1] = Complex64::new(1.0, 0.0);
        let col_drain = tridiag::solve_complex(&off_diag, &diag, &off_diag, &e_drain);

        g_diag.push(full_diag.iter().map(|g| -g.im / a).collect());
        g_col_source.push(col_source.iter().map(|g| g.norm_sqr()).collect());
        g_col_drain.push(col_drain.iter().map(|g| g.norm_sqr()).collect());
    }

    GreenFunctionResult {
        energies: energies.to_vec(),
        g_diag,
        g_col_source,
        g_col_drain,
        k_sa,
        k_da,
    }
}

/// Coherent single-mode transmission at real energy, without absorbing eta.
/// The scaled continuant is the determinant of (E-H-Sigma)/t; its inverse
/// gives |t G_1N|. Renormalizing BOTH adjacent determinants on every site
/// avoids overflowing an opaque barrier or dividing by a connected-chain pole.
/// O(N) time, O(1) scratch per energy (no energy-by-site storage).
pub(crate) fn coherent_transmission(device: &Device, energy: f64) -> crate::error::Result<f64> {
    use crate::error::NegForgeError;
    let fail = || {
        NegForgeError::InvalidParameter(format!(
            "non-finite or non-unitary coherent transmission at E={energy}"
        ))
    };
    let t = device.t_hop;
    let source = energy - device.psi_f[0];
    let drain = energy - device.psi_f[device.n - 1];
    // Exact real-energy broadenings, including the finite lattice upper edge.
    // At a band edge the open-channel transmission is defined as zero.
    if source <= 0.0 || drain <= 0.0 || source >= 4.0 * t || drain >= 4.0 * t {
        return Ok(0.0);
    }
    let gs = (source / t * (4.0 - source / t)).sqrt();
    let gd = (drain / t * (4.0 - drain / t)).sqrt();
    let ss = Complex64::new(source / (2.0 * t) - 1.0, -gs / 2.0);
    let sd = Complex64::new(drain / (2.0 * t) - 1.0, -gd / 2.0);
    let mut previous = Complex64::new(1.0, 0.0);
    let mut older = Complex64::new(0.0, 0.0);
    let mut log_scale = 0.0;
    for (i, &potential) in device.psi_f.iter().enumerate() {
        let mut diagonal = Complex64::new((energy - potential) / t - 2.0, 0.0);
        if i == 0 {
            diagonal -= ss;
        }
        if i + 1 == device.n {
            diagonal -= sd;
        }
        let next = diagonal * previous - older;
        let scale = next.norm().max(previous.norm());
        if !scale.is_finite() || scale == 0.0 {
            return Err(fail());
        }
        older = previous / scale;
        previous = next / scale;
        log_scale += scale.ln();
    }
    let log_t = gs.ln() + gd.ln() - 2.0 * (log_scale + previous.norm().ln());
    let value = log_t.exp();
    if !value.is_finite() || value > 1.0 + 1e-8 {
        return Err(fail());
    }
    // Only roundoff-sized overshoots of the single-channel unitarity bound.
    Ok(value.min(1.0))
}

/// Retarded semi-infinite-chain self-energy at real E (lead eta = 0+).
/// Choose the decaying quadratic root outside BOTH band edges; rationalize
/// that root to avoid subtracting nearly equal numbers far from the band.
fn contact_self_energy(energy: f64, onsite: f64, t: f64) -> Complex64 {
    let x = (energy - onsite) / (2.0 * t);
    if x.abs() <= 1.0 {
        Complex64::new(t * x, -t * ((1.0 - x) * (1.0 + x)).sqrt())
    } else {
        let root = (x.abs() - 1.0).sqrt() * (x.abs() + 1.0).sqrt();
        Complex64::new(t / (x + x.signum() * root), 0.0)
    }
}

/// Full diagonal of `A^-1` for a tridiagonal `A` with constant real
/// off-diagonal `t` (i.e. `A[i][i+1] = A[i+1][i] = t` for all `i`) and
/// complex diagonal `diag`, via the standard left-connected recursive
/// Green's function sweep.
fn full_diagonal(off_diag: &[f64], diag: &[Complex64]) -> Vec<Complex64> {
    let n = diag.len();
    let mut g_left = vec![Complex64::new(0.0, 0.0); n];
    g_left[0] = Complex64::new(1.0, 0.0) / diag[0];
    for i in 1..n {
        let t = off_diag[i - 1];
        g_left[i] = Complex64::new(1.0, 0.0) / (diag[i] - g_left[i - 1] * (t * t));
    }

    // A second connected sweep avoids catastrophic cancellation in
    // g_left + g_left^2*t^2*full at poles of a truncated chain.
    let mut g_right = vec![Complex64::new(0.0, 0.0); n];
    g_right[n - 1] = Complex64::new(1.0, 0.0) / diag[n - 1];
    for i in (0..n - 1).rev() {
        let t = off_diag[i];
        g_right[i] = Complex64::new(1.0, 0.0) / (diag[i] - t * t * g_right[i + 1]);
    }
    (0..n)
        .map(|i| {
            let mut denominator = diag[i];
            if i > 0 {
                denominator -= off_diag[i - 1].powi(2) * g_left[i - 1];
            }
            if i + 1 < n {
                denominator -= off_diag[i].powi(2) * g_right[i + 1];
            }
            Complex64::new(1.0, 0.0) / denominator
        })
        .collect()
}

#[cfg(test)]
#[allow(clippy::needless_range_loop)]
mod tests {
    use super::*;
    use crate::device::{Device, DeviceParams};
    use approx::assert_relative_eq;

    /// Dense reference: build the full complex tridiagonal matrix and
    /// invert it via Gauss-Jordan elimination, for cross-checking the O(N)
    /// recursive/solve-based implementation above on small systems.
    fn dense_inverse(off_diag: &[f64], diag: &[Complex64]) -> Vec<Vec<Complex64>> {
        let n = diag.len();
        let mut a = vec![vec![Complex64::new(0.0, 0.0); n]; n];
        for i in 0..n {
            a[i][i] = diag[i];
            if i > 0 {
                a[i][i - 1] = Complex64::new(off_diag[i - 1], 0.0);
            }
            if i < n - 1 {
                a[i][i + 1] = Complex64::new(off_diag[i], 0.0);
            }
        }
        let mut inv = vec![vec![Complex64::new(0.0, 0.0); n]; n];
        for i in 0..n {
            inv[i][i] = Complex64::new(1.0, 0.0);
        }
        for col in 0..n {
            let mut pivot = col;
            for row in (col + 1)..n {
                if a[row][col].norm() > a[pivot][col].norm() {
                    pivot = row;
                }
            }
            a.swap(col, pivot);
            inv.swap(col, pivot);
            let pivot_val = a[col][col];
            for k in 0..n {
                a[col][k] /= pivot_val;
                inv[col][k] /= pivot_val;
            }
            let pivot_row_a = a[col].clone();
            let pivot_row_inv = inv[col].clone();
            for row in 0..n {
                if row == col {
                    continue;
                }
                let factor = a[row][col];
                for k in 0..n {
                    a[row][k] -= factor * pivot_row_a[k];
                    inv[row][k] -= factor * pivot_row_inv[k];
                }
            }
        }
        inv
    }

    #[test]
    #[should_panic(expected = "eta must be finite and positive")]
    fn negative_eta_is_rejected() {
        let device = Device::new(DeviceParams::default());
        green_function_sweep(&device, &[0.1], -0.01);
    }

    #[test]
    fn one_site_accumulates_both_contacts() {
        let mut device = Device::new(DeviceParams::default());
        device.n = 1;
        device.psi_f = vec![0.0];
        let t = device.t_hop;
        let eta = 0.2 * t;
        let result = green_function_sweep(&device, &[2.0 * t], eta);
        // At midband each lead has Sigma=-it, so G=1/[i(eta+2t)].
        assert_relative_eq!(
            result.g_diag[0][0],
            1.0 / (device.params.a * (eta + 2.0 * t)),
            epsilon = 1e-12
        );
        assert_relative_eq!(
            result.g_col_source[0][0],
            1.0 / (eta + 2.0 * t).powi(2),
            epsilon = 1e-12
        );
    }

    #[test]
    fn uniform_chain_matches_infinite_chain_resolvent() {
        let mut device = Device::new(DeviceParams {
            a: 1.0,
            l_ch: 3.0,
            l_ds: 2.0,
            auto_size_contacts: false,
            ..Default::default()
        });
        device.psi_f.fill(0.0);
        let t = device.t_hop;
        // The diagonal of the infinite chain is 1/sqrt((E-2t)^2-4t^2).
        // Include both evanescent regimes, which still have a real self-energy.
        for e in [-t, 0.3 * t, 2.0 * t, 3.7 * t, 5.0 * t] {
            let eta = 1e-7 * t;
            let result = green_function_sweep(&device, &[e], eta);
            let x = e - 2.0 * t;
            let sigma = if x.abs() < 2.0 * t {
                Complex64::new(x / 2.0, -(4.0 * t * t - x * x).sqrt() / 2.0)
            } else {
                Complex64::new((x - x.signum() * (x * x - 4.0 * t * t).sqrt()) / 2.0, 0.0)
            };
            let mut diag = vec![Complex64::new(x, eta); device.n];
            diag[0] -= sigma;
            diag[device.n - 1] -= sigma;
            let dense = dense_inverse(&vec![t; device.n - 1], &diag);
            for i in 0..device.n {
                assert_relative_eq!(result.g_diag[0][i], -dense[i][i].im, epsilon = 1e-8);
                assert_relative_eq!(
                    result.g_col_source[0][i],
                    dense[i][0].norm_sqr(),
                    max_relative = 1e-9
                );
                assert_relative_eq!(
                    result.g_col_drain[0][i],
                    dense[i][device.n - 1].norm_sqr(),
                    max_relative = 1e-9
                );
            }
            if x.abs() < 2.0 * t {
                let expected = 1.0 / (4.0 * t * t - x * x).sqrt();
                assert_relative_eq!(
                    result.g_diag[0][device.n / 2],
                    expected,
                    max_relative = 1e-5
                );
            }
        }
    }

    #[test]
    fn diagonal_near_a_connected_pole_matches_dense() {
        let diag = [Complex64::new(0.0, 1e-10); 2];
        let actual = full_diagonal(&[1.0], &diag);
        let expected = dense_inverse(&[1.0], &diag);
        for i in 0..2 {
            assert_relative_eq!(actual[i].im, expected[i][i].im, max_relative = 1e-12);
        }
    }

    #[test]
    fn recursive_diagonal_matches_dense_inverse() {
        let n = 9;
        let off_diag: Vec<f64> = vec![0.37; n - 1];
        let diag: Vec<Complex64> = (0..n)
            .map(|i| Complex64::new(-1.5 - 0.3 * i as f64, 1e-6))
            .collect();

        let recursive = full_diagonal(&off_diag, &diag);
        let dense = dense_inverse(&off_diag, &diag);

        for i in 0..n {
            assert_relative_eq!(recursive[i].re, dense[i][i].re, epsilon = 1e-8);
            assert_relative_eq!(recursive[i].im, dense[i][i].im, epsilon = 1e-8);
        }
    }

    #[test]
    fn boundary_columns_match_dense_inverse() {
        let n = 9;
        let off_diag: Vec<f64> = vec![0.37; n - 1];
        let diag: Vec<Complex64> = (0..n)
            .map(|i| Complex64::new(-1.5 - 0.3 * i as f64, 1e-6))
            .collect();

        let mut e0 = vec![Complex64::new(0.0, 0.0); n];
        e0[0] = Complex64::new(1.0, 0.0);
        let col0 = tridiag::solve_complex(&off_diag, &diag, &off_diag, &e0);

        let mut e_last = vec![Complex64::new(0.0, 0.0); n];
        e_last[n - 1] = Complex64::new(1.0, 0.0);
        let col_last = tridiag::solve_complex(&off_diag, &diag, &off_diag, &e_last);

        let dense = dense_inverse(&off_diag, &diag);

        for i in 0..n {
            assert_relative_eq!(col0[i].re, dense[i][0].re, epsilon = 1e-8);
            assert_relative_eq!(col0[i].im, dense[i][0].im, epsilon = 1e-8);
            assert_relative_eq!(col_last[i].re, dense[i][n - 1].re, epsilon = 1e-8);
            assert_relative_eq!(col_last[i].im, dense[i][n - 1].im, epsilon = 1e-8);
        }
    }

    #[test]
    fn green_function_sweep_runs_and_is_finite() {
        let mut device = Device::new(DeviceParams {
            a: 1.0,
            l_ch: 10.0,
            l_ds: 10.0,
            auto_size_contacts: false,
            ..Default::default()
        });
        device.calc_potential();

        let e_min = device.psi_f.iter().cloned().fold(f64::INFINITY, f64::min);
        let energies: Vec<f64> = (0..20).map(|i| e_min + i as f64 * 0.01).collect();
        let result = green_function_sweep(&device, &energies, DEFAULT_ETA);

        assert_eq!(result.g_diag.len(), energies.len());
        for row in &result.g_diag {
            assert_eq!(row.len(), device.n);
            assert!(row.iter().all(|v| v.is_finite()));
        }
        for row in &result.g_col_source {
            assert!(row.iter().all(|v| v.is_finite() && *v >= 0.0));
        }
    }

    /// Documents the corrected sign claim from the module docs' "Contact
    /// self-energies" section: for the branch of `ka_s`/`ka_d` this module
    /// actually computes (`ka in (0, pi)`, the reflected branch), the
    /// self-energy `sigma = t_hop * exp(-i*ka)` — as constructed in
    /// [`green_function_sweep`] — has `Im(sigma) <= 0`, as required for a
    /// causal, absorbing (not amplifying) contact.
    #[test]
    fn contact_self_energy_has_non_positive_imaginary_part_for_propagating_modes() {
        let t = 1.69_f64;
        for ka in [0.2, 0.7_f64, std::f64::consts::FRAC_PI_2, 2.5, 3.0] {
            let sigma = t * (-(Complex64::i() * Complex64::new(ka, 0.0))).exp();
            assert!(
                sigma.im <= 0.0,
                "expected Im(sigma) <= 0 at ka={ka}, got {}",
                sigma.im
            );
            assert!((sigma.im + t * ka.sin()).abs() < 1e-12);
        }
    }

    /// With the corrected self-energy sign, `g_diag` (the `-Im(G_ii)/a`
    /// LDOS proxy) should be non-negative — the defining property of a
    /// proper (causal) local density of states, and the concrete symptom
    /// checked at the self-consistent-loop broadening.
    #[test]
    fn g_diag_is_non_negative_at_self_consistent_loop_broadening() {
        let mut device = Device::new(DeviceParams {
            a: 0.5,
            l_ch: 20.0,
            l_ds: 20.0,
            auto_size_contacts: false,
            ..Default::default()
        });
        device.calc_potential();

        let energies: Vec<f64> = (0..200).map(|i| device.psi_0 + i as f64 * 0.005).collect();
        let result = green_function_sweep(&device, &energies, 0.08);

        let worst = result
            .g_diag
            .iter()
            .flat_map(|row| row.iter().cloned())
            .fold(f64::INFINITY, f64::min);
        assert!(
            worst >= -1e-9,
            "expected non-negative g_diag, got worst={worst}"
        );
    }
}
