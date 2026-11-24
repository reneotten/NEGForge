//! Device geometry, electrostatics and ballistic-current model.
//!
//! The model is a 1D
//! source-channel-drain chain (source and drain treated as long, heavily
//! screened contact regions) with:
//!
//! - an electrostatic potential solved from a 1D "natural length" (`lambda`)
//!   approximation rather than a full 2D/3D Poisson equation, common in compact
//!   models for thin-body/double-gate MOSFETs;
//! - a ballistic (Landauer) drain current from Fermi-Dirac occupation of the
//!   source and drain reservoirs.
//!
//! Units: lengths in nm, energies/potentials in eV,
//! temperature in Kelvin. `Psi_g`/`Psi_bi`/`Psi_f` store *potential energy*
//! (already in eV), not electrostatic potential in volts — consistent with
//! `E_f`/`E_g` also being in eV.
//!
//! ## Validity of the approximations
//!
//! - **Natural-length electrostatics** (`Device::compute_lambda`) is a
//!   thin-body/double-gate compact-model reduction of the full 2D/3D Poisson
//!   equation. It assumes the transverse potential profile is parabolic
//!   (valid when the body is thin compared to the channel length) and breaks
//!   down for bulk (thick-body) MOSFETs, where the transverse charge
//!   distribution is not well approximated by a single decay length. It is
//!   also a *classical* charge model in the transverse direction — it does
//!   not resolve transverse-mode (subband) quantization from confinement in
//!   `d_ch`, unlike the longitudinal NEGF treatment in `negf.rs`.
//! - **Coherent single-mode Landauer current** ([`Device::calc_negf_current`])
//!   includes elastic reflection and sub-barrier tunneling through `psi_f`,
//!   with conserving real-energy semi-infinite leads and no absorbing eta.
//!   It omits inelastic scattering, transverse subbands and valley degeneracy.
//!   [`Device::calc_thermionic_current`] retains the analytic top-of-the-barrier
//!   comparator, `T(E) = theta(E - psi_0)`, which omits quantum reflection,
//!   tunneling and the discretized leads' finite upper band edges.
//! - **Effective-mass, single-valley, parabolic band** (`m_eff`, used in
//!   `t_hop` here and in the NEGF hopping parameter). Real silicon has
//!   multiple equivalent conduction-band valleys (a valley-degeneracy factor
//!   this model does not include) and a non-parabolic dispersion away from
//!   the band edge; `m_eff` is a single fitting parameter standing in for
//!   both effects, so results should be read as illustrating trends, not as
//!   quantitatively predicting absolute currents for a specific real device.
//!   Spin degeneracy *is* included, via the explicit factor of 2 in
//!   `calc_current`'s `2*e/h` prefactor (the standard single-mode Landauer
//!   conductance quantum).

use crate::constants::{E, EPS_0, H_BAR, K_B, M_E};
use crate::error::{NegForgeError, Result};
use crate::tridiag;

/// Configuration for a [`Device`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DeviceParams {
    /// Grid spacing, nm.
    pub a: f64,
    /// Fermi energy, eV.
    pub e_f: f64,
    /// Band gap, eV.
    pub e_g: f64,
    /// Drain-source voltage, V.
    pub v_ds: f64,
    /// Gate voltage (Psi_g = -e*V_g), V.
    pub v_g: f64,
    /// Oxide thickness, nm.
    pub d_ox: f64,
    /// Channel (body) thickness, nm.
    pub d_ch: f64,
    /// Relative permittivity of the channel (silicon by default).
    pub k_si: f64,
    /// Relative permittivity of the oxide.
    pub k_ox: f64,
    /// Gate geometry factor (number of gates).
    pub geo: f64,
    /// Channel length, nm.
    pub l_ch: f64,
    /// If `true` (default), the
    /// source/drain contact length is auto-sized to `floor(lambda) * 15`
    /// instead of using `l_ds` directly.
    pub auto_size_contacts: bool,
    /// Source/drain contact region length, nm. Only used verbatim when
    /// `auto_size_contacts` is `false`.
    pub l_ds: f64,
    /// Fixed dopant charge density in C/m^3, added to the mobile charge
    /// density in the electrostatic solve.
    pub n_dot: f64,
    /// Fermi-function tolerance used to bound the ballistic energy window.
    pub epsilon: f64,
    /// Source Fermi level, eV.
    pub e_fs: f64,
    /// Temperature, K.
    pub t: f64,
    /// Energy integration step, eV.
    pub d_e: f64,
    /// Effective mass, kg.
    pub m_eff: f64,
}

impl Default for DeviceParams {
    fn default() -> Self {
        Self {
            a: 0.5,
            e_f: 0.15,
            e_g: 1.0,
            v_ds: 0.0,
            v_g: 0.0,
            d_ox: 5.0,
            d_ch: 5.0,
            k_si: 11.2,
            k_ox: 3.9,
            geo: 1.0,
            l_ch: 40.0,
            auto_size_contacts: true,
            l_ds: 40.0,
            n_dot: 0.0,
            epsilon: 10e-15,
            e_fs: 0.05,
            t: 300.0,
            d_e: 0.001,
            m_eff: 0.9 * M_E,
        }
    }
}

/// A 1D ballistic-MOSFET device: geometry, electrostatics and derived
/// transport quantities.
#[derive(Debug, Clone)]
pub struct Device {
    pub params: DeviceParams,

    /// Explicit transverse active area in m² for spin-summed SCF charge.
    /// `None` selects the per-spin, implicit 1 m² normalization;
    /// it is not a physical device cross-section. No width is inferred
    /// from `d_ch`, `geo`, or the longitudinal grid.
    pub effective_area_m2: Option<f64>,

    /// Natural (screening) length, nm.
    pub lambda: f64,
    /// Number of grid points along the channel.
    pub n: usize,
    /// First grid index (0-based) belonging to the channel region.
    pub n_left: usize,
    /// First grid index (0-based) belonging to the drain region.
    pub n_right: usize,

    /// Gate potential energy profile, eV.
    pub psi_g: Vec<f64>,
    /// Built-in potential energy profile, eV.
    pub psi_bi: Vec<f64>,
    /// Mobile charge density term entering the electrostatic solve. Zero
    /// until a self-consistent (NEGF-fed) solve updates it.
    pub rho: Vec<f64>,
    /// Solved electrostatic potential energy profile, eV.
    pub psi_f: Vec<f64>,
    /// `max(psi_f)`, used as the lower bound of the ballistic energy window.
    pub psi_0: f64,

    /// Drain Fermi level, eV (`e_fs - V_ds`).
    pub e_fd: f64,
    /// Upper bound of the ballistic energy integration window, eV.
    pub e_max: f64,
    /// Nearest-neighbor hopping parameter, eV.
    pub t_hop: f64,
}

impl Device {
    /// Select a physical transverse area, or select implicit unit-area normalization.
    /// This does not infer a device width or change the per-spin NEGF API.
    pub fn set_effective_area_m2(&mut self, area: Option<f64>) -> Result<()> {
        Self::validate_effective_area(area)?;
        self.effective_area_m2 = area;
        Ok(())
    }

    fn validate_effective_area(area: Option<f64>) -> Result<()> {
        if area.is_some_and(|a| !a.is_finite() || a <= 0.0) {
            return Err(NegForgeError::InvalidParameter(
                "effective_area_m2 must be positive and finite when supplied".into(),
            ));
        }
        Ok(())
    }

    /// Convert per-spin linear density (nm^-1) to signed volume charge (C/m³).
    /// With explicit area, both spins contribute: `-2 e n 1e9 / area`.
    /// `None` retains the old `-e n 1e9` numerical normalization exactly;
    /// it is not a physically specified cross-section. Validate the device
    /// and the finite nonnegative input before numerical work.
    pub fn mobile_charge_density(&self, per_spin_nm_inv: f64) -> f64 {
        match self.effective_area_m2 {
            Some(area) => -2.0 * E * per_spin_nm_inv * 1e9 / area,
            None => -E * per_spin_nm_inv * 1e9,
        }
    }

    /// Validate parameters and public vector state before numerical work.
    pub fn validate(&self) -> Result<()> {
        Self::validate_effective_area(self.effective_area_m2)?;
        crate::resources::check_sites(self.n)?;
        for (name, value) in [
            ("a", self.params.a),
            ("d_ox", self.params.d_ox),
            ("d_ch", self.params.d_ch),
            ("k_si", self.params.k_si),
            ("k_ox", self.params.k_ox),
            ("geo", self.params.geo),
            ("l_ch", self.params.l_ch),
            ("d_e", self.params.d_e),
            ("m_eff", self.params.m_eff),
            ("lambda", self.lambda),
            ("t_hop", self.t_hop),
        ] {
            if !value.is_finite() || value <= 0.0 {
                return Err(NegForgeError::InvalidParameter(format!(
                    "{name} must be positive and finite"
                )));
            }
        }
        if !self.params.t.is_finite()
            || self.params.t < 0.0
            || !self.params.l_ds.is_finite()
            || self.params.l_ds < 0.0
            || !self.params.epsilon.is_finite()
            || self.params.epsilon <= 0.0
            || self.params.epsilon >= 1.0
            || ![
                self.params.e_f,
                self.params.e_g,
                self.params.e_fs,
                self.params.v_ds,
                self.params.v_g,
                self.params.n_dot,
                self.e_fd,
                self.e_max,
            ]
            .iter()
            .all(|v| v.is_finite())
            || self.n < 2
            || self.n_left >= self.n_right
            || self.n_right > self.n
            || [&self.psi_g, &self.psi_bi, &self.rho, &self.psi_f]
                .iter()
                .any(|v| v.len() != self.n || !v.iter().all(|x| x.is_finite()))
        {
            return Err(NegForgeError::InvalidParameter(
                "invalid device geometry, energies or vector state".into(),
            ));
        }
        Ok(())
    }

    /// Build a device from geometry, material, and reservoir parameters.
    pub fn new(mut params: DeviceParams) -> Self {
        let lambda = Self::compute_lambda(&params);
        if params.auto_size_contacts {
            params.l_ds = (lambda.floor()) * 15.0;
        }

        let (n, n_left, n_right) = Self::compute_grid(&params);
        let e_fd = params.e_fs - params.v_ds;
        let e_max = params.e_fs.max(e_fd) - K_B * params.t * params.epsilon.ln() / E;
        let t_hop = H_BAR.powi(2) / (2.0 * params.m_eff * (params.a * 1e-9).powi(2) * E);

        let mut device = Self {
            params,
            effective_area_m2: None,
            lambda,
            n,
            n_left,
            n_right,
            psi_g: vec![0.0; n],
            psi_bi: vec![0.0; n],
            rho: vec![0.0; n],
            psi_f: vec![0.0; n],
            psi_0: 0.0,
            e_fd,
            e_max,
            t_hop,
        };
        device.init_vectors();
        device
    }

    /// `lambda = sqrt(k_si/k_ox * d_ch * d_ox / geo)`, the scale length over
    /// which source/drain potentials leak into the channel (see module
    /// docs). Only meaningful in the thin-body regime this formula was
    /// derived for; not a substitute for a full Poisson solve when `d_ch` is
    /// not small compared to `l_ch`.
    fn compute_lambda(params: &DeviceParams) -> f64 {
        (params.k_si / params.k_ox * params.d_ch * params.d_ox / params.geo).sqrt()
    }

    /// Check the effective (including auto-sized contacts) site count without
    /// allocating. Shared by construction and Python parameter validation.
    pub fn checked_site_count(params: &DeviceParams) -> Result<usize> {
        let mut effective = *params;
        let lambda = Self::compute_lambda(params);
        if effective.auto_size_contacts {
            effective.l_ds = lambda.floor() * 15.0;
        }
        if !lambda.is_finite() || lambda <= 0.0 {
            return Err(NegForgeError::InvalidParameter(
                "invalid screening length".into(),
            ));
        }
        Self::checked_grid(&effective).map(|(n, _, _)| n)
    }

    fn checked_grid(params: &DeviceParams) -> Result<(usize, usize, usize)> {
        let intervals = ((2.0 * params.l_ds + params.l_ch) / params.a).floor();
        if !params.a.is_finite()
            || params.a <= 0.0
            || !params.l_ds.is_finite()
            || params.l_ds < 0.0
            || !params.l_ch.is_finite()
            || params.l_ch <= 0.0
            || !intervals.is_finite()
            || intervals < 1.0
            || intervals >= crate::resources::MAX_SITES as f64
        {
            return Err(NegForgeError::InvalidParameter(
                "geometry exceeds practical site budget".into(),
            ));
        }
        let n = (intervals as usize)
            .checked_add(1)
            .ok_or_else(|| NegForgeError::InvalidParameter("site count overflow".into()))?;
        crate::resources::check_sites(n)?;
        let n_left = (params.l_ds / params.a).floor() as usize;
        let n_right = n
            .checked_sub(n_left)
            .ok_or_else(|| NegForgeError::InvalidParameter("invalid contact count".into()))?;
        if n_left >= n_right {
            return Err(NegForgeError::InvalidParameter("empty channel".into()));
        }
        Ok((n, n_left, n_right))
    }

    fn compute_grid(params: &DeviceParams) -> (usize, usize, usize) {
        Self::checked_grid(params).expect("invalid device geometry or site budget")
    }

    /// Rebuild `psi_g`, `psi_bi` and reset `rho` from the current
    /// parameters and region boundaries.
    pub fn init_vectors(&mut self) {
        self.psi_g = vec![0.0; self.n];
        self.psi_bi = vec![0.0; self.n];
        self.rho = vec![0.0; self.n];

        // Keep the source and drain contact site counts identical. The
        for i in self.n_left..self.n_right {
            self.psi_g[i] = -self.params.v_g;
            self.psi_bi[i] = self.params.e_f + self.params.e_g / 2.0;
        }
        // Drain: n_right..n.
        for i in self.n_right..self.n {
            self.psi_g[i] = 0.0;
            self.psi_bi[i] = -self.params.v_ds;
        }
        // Source region (0..n_left) stays at zero.
    }

    /// Build the tridiagonal second-difference operator with the
    /// reflective (Neumann-like) boundary doubling used for the
    /// electrostatic solve, scaled by `1/a^2`. Returns `(sub, diag, sup)`.
    fn laplacian(&self) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let n = self.n;
        let inv_a2 = 1.0 / self.params.a.powi(2);
        let diag = vec![-2.0 * inv_a2; n];
        let mut sup = vec![inv_a2; n - 1];
        let mut sub = vec![inv_a2; n - 1];
        sup[0] = 2.0 * inv_a2;
        sub[n - 2] = 2.0 * inv_a2;
        (sub, diag, sup)
    }

    /// Solve for the self-consistent-length electrostatic potential energy
    /// profile given the current `rho`, storing the result in `psi_f` and
    /// updating `psi_0 = max(psi_f)`.
    pub fn calc_potential(&mut self) {
        let (sub, mut diag, sup) = self.laplacian();
        let inv_lambda2 = 1.0 / self.lambda.powi(2);
        for d in diag.iter_mut() {
            *d -= inv_lambda2;
        }

        let rhs: Vec<f64> = (0..self.n)
            .map(|i| {
                // rho is C/m^3; the Laplacian is nm^-2. Electron potential
                // energy U/e = -phi has source +rho/epsilon.
                (self.rho[i] + self.params.n_dot) * 1e-18 / (EPS_0 * self.params.k_si)
                    - inv_lambda2 * (self.psi_g[i] + self.psi_bi[i])
            })
            .collect();

        self.psi_f = tridiag::solve_real(&sub, &diag, &sup, &rhs);
        self.psi_0 = self.psi_f.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    }

    /// Coherent, conserving single-mode transmission, in input energy order.
    /// Uses the endpoint potentials as the two semi-infinite lead band bottoms.
    /// No artificial eta is applied. Empty input returns an empty spectrum.
    pub fn transmission_spectrum(&self, energies: &[f64]) -> Result<Vec<f64>> {
        self.validate()?;
        if energies.len() > crate::resources::MAX_ENERGIES
            || !energies.iter().all(|e| e.is_finite())
        {
            return Err(NegForgeError::InvalidParameter(
                "invalid transmission energy grid".into(),
            ));
        }
        energies
            .iter()
            .map(|&e| crate::negf::coherent_transmission(self, e))
            .collect()
    }

    /// Coherent spin-degenerate single-mode Landauer current, amperes.
    /// Integrates only the overlapping finite lead bands and the occupied
    /// Fermi-difference window (tail cutoff `epsilon`). Four-point Gaussian
    /// quadrature uses panels no wider than `d_e`; refine d_e for resonances.
    /// Transmission is streamed, with no absorbing eta or density sweep.
    pub fn calc_negf_current(&self) -> Result<f64> {
        self.validate()?;
        let mu_s = self.params.e_fs;
        let mu_d = self.e_fd;
        if mu_s == mu_d {
            return Ok(0.0);
        }
        let kt = K_B * self.params.t / E;
        let tail = -kt * self.params.epsilon.ln();
        let lo = self.psi_f[0]
            .max(self.psi_f[self.n - 1])
            .max(mu_s.min(mu_d) - tail);
        let hi = (self.psi_f[0] + 4.0 * self.t_hop)
            .min(self.psi_f[self.n - 1] + 4.0 * self.t_hop)
            .min(mu_s.max(mu_d) + tail);
        if !lo.is_finite() || !hi.is_finite() {
            return Err(NegForgeError::InvalidParameter(
                "non-finite current integration bounds".into(),
            ));
        }
        if hi <= lo {
            return Ok(0.0);
        }
        let count = ((hi - lo) / self.params.d_e).ceil().max(1.0);
        if !count.is_finite() || count > crate::resources::MAX_ENERGIES as f64 / 4.0 {
            return Err(NegForgeError::InvalidParameter(
                "current integration exceeds energy budget".into(),
            ));
        }
        let panels = count as usize;
        let width = (hi - lo) / panels as f64;
        // Stable positive f(high mu)-f(low mu), avoiding cancellation in
        // filled states and preserving tiny-bias linear response.
        let delta = (mu_s - mu_d).abs();
        let difference = |energy: f64| {
            if kt == 0.0 {
                return 1.0;
            }
            let x = (energy - mu_s.max(mu_d)) / kt;
            let y = (energy - mu_s.min(mu_d)) / kt;
            let numerator = -(-delta / kt).exp_m1();
            if y <= 0.0 {
                numerator * y.exp() / ((1.0 + x.exp()) * (1.0 + y.exp()))
            } else if x >= 0.0 {
                numerator * (-x).exp() / ((1.0 + (-x).exp()) * (1.0 + (-y).exp()))
            } else {
                numerator / ((1.0 + x.exp()) * (1.0 + (-y).exp()))
            }
        };
        let mut sum = 0.0;
        for i in 0..panels {
            let center = lo + (i as f64 + 0.5) * width;
            for (node, weight) in [
                (-0.8611363115940526, 0.3478548451374538),
                (-0.3399810435848563, 0.6521451548625461),
                (0.3399810435848563, 0.6521451548625461),
                (0.8611363115940526, 0.3478548451374538),
            ] {
                let energy = center + 0.5 * width * node;
                sum +=
                    weight * crate::negf::coherent_transmission(self, energy)? * difference(energy);
            }
        }
        let current =
            2.0 * E * E / crate::constants::H * (mu_s - mu_d).signum() * sum * width / 2.0;
        if !current.is_finite() {
            return Err(NegForgeError::InvalidParameter(
                "non-finite coherent current".into(),
            ));
        }
        Ok(current)
    }

    /// Default coherent current, amperes. Legacy infallible interface:
    /// numerical/validation errors are explicit panics, never thermionic fallbacks.
    /// Use `calc_negf_current` to propagate errors instead.
    pub fn calc_current(&self) -> f64 {
        self.calc_negf_current()
            .expect("coherent current calculation failed")
    }

    /// Analytic top-of-the-barrier thermionic comparator (not coherent NEGF).
    /// Spin-degenerate current in amperes; step T(E)=1 above psi_0.
    pub fn calc_thermionic_current(&self) -> f64 {
        let kt = K_B * self.params.t / E;
        let integral = |mu: f64| {
            if kt == 0.0 {
                return (mu - self.psi_0).max(0.0);
            }
            let x = (mu - self.psi_0) / kt;
            (mu - self.psi_0).max(0.0) + kt * (-x.abs()).exp().ln_1p()
        };
        2.0 * E * E / crate::constants::H * (integral(self.params.e_fs) - integral(self.e_fd))
    }

    pub fn set_v_ds(&mut self, v: f64) {
        self.params.v_ds = v;
        self.e_fd = self.params.e_fs - v;
        self.e_max =
            self.params.e_fs.max(self.e_fd) - K_B * self.params.t * self.params.epsilon.ln() / E;
        self.init_vectors();
    }

    pub fn set_v_g(&mut self, v: f64) {
        self.params.v_g = v;
        self.init_vectors();
    }

    pub fn set_l_ch(&mut self, l: f64) {
        let mut params = self.params;
        params.l_ch = l;
        let (n, n_left, n_right) = Self::compute_grid(&params);
        self.params = params;
        self.n = n;
        self.n_left = n_left;
        self.n_right = n_right;
        self.init_vectors();
        self.psi_f = vec![0.0; self.n];
        self.psi_0 = 0.0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn lambda_matches_closed_form() {
        let params = DeviceParams::default();
        let dev = Device::new(params);
        let expected = (params.k_si / params.k_ox * params.d_ch * params.d_ox / params.geo).sqrt();
        assert_relative_eq!(dev.lambda, expected, epsilon = 1e-12);
    }

    #[test]
    fn zero_bias_zero_gate_gives_symmetric_flat_contacts() {
        // With V_g = V_ds = 0 and no charge, psi_bi is zero everywhere
        // outside the channel and the channel barrier is E_f + E_g/2 — the
        // solved potential should be flat (zero) far into the contacts,
        // since there is no driving term there.
        let dev = Device::new(DeviceParams::default());
        assert_relative_eq!(dev.psi_g[0], 0.0);
        assert_relative_eq!(dev.psi_bi[0], 0.0);
        assert_relative_eq!(dev.psi_bi[dev.n - 1], 0.0);
    }

    #[test]
    fn calc_potential_is_finite_and_bounded_by_source_terms() {
        let mut dev = Device::new(DeviceParams::default());
        dev.calc_potential();
        assert_eq!(dev.psi_f.len(), dev.n);
        assert!(dev.psi_f.iter().all(|v| v.is_finite()));
        // The channel barrier term is E_f + E_g/2; the solved potential
        // (a screened, smoothed version of the driving terms) should not
        // wildly overshoot it.
        let max_drive = dev.params.e_f + dev.params.e_g / 2.0;
        assert!(dev.psi_0 <= max_drive * 1.01);
    }

    #[test]
    fn increasing_v_ds_increases_current() {
        let mut dev = Device::new(DeviceParams::default());
        dev.calc_potential();
        let i0 = dev.calc_current();

        dev.set_v_ds(0.3);
        dev.calc_potential();
        let i1 = dev.calc_current();

        assert!(
            i1 > i0,
            "current should increase with drain bias: {i0} -> {i1}"
        );
    }

    #[test]
    fn increasing_v_g_increases_current_ballistic_mosfet() {
        let mut dev = Device::new(DeviceParams::default());
        dev.set_v_ds(0.3);
        dev.calc_potential();
        let i0 = dev.calc_current();

        dev.set_v_g(0.3);
        dev.calc_potential();
        let i1 = dev.calc_current();

        assert!(
            i1 > i0,
            "current should increase with gate bias: {i0} -> {i1}"
        );
    }
}
