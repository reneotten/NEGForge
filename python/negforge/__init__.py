"""NEGForge: 1D ballistic-MOSFET electrostatics + NEGF transport simulator.

The numerical engine (self-consistent Poisson/NEGF solve, tridiagonal linear
algebra, recursive Green's-function evaluation) lives in Rust
(``negforge-core``, exposed here via PyO3 as ``negforge._negforge``).
This package is a thin, notebook-friendly Python wrapper around it: numpy
arrays instead of raw tuples, keyword-argument device construction, and a
small `IVCurve` helper for bias sweeps. See the top-level repository README
for the physics background, unit conventions, and known limitations.
"""

from __future__ import annotations

import numpy as np

from ._negforge import Device as _RustDevice

__all__ = ["Device", "IVCurve"]


class IVCurve:
    """Voltage/current arrays from a bias sweep ([`Device.sweep_v_g`] /
    [`Device.sweep_v_ds`])."""

    def __init__(self, voltage, current):
        self.voltage = np.asarray(voltage, dtype=float)
        self.current = np.asarray(current, dtype=float)
        if self.voltage.ndim != 1 or self.current.ndim != 1 or self.voltage.shape != self.current.shape:
            raise ValueError("voltage and current must be matching one-dimensional arrays")
        if not np.isfinite(self.voltage).all():
            raise ValueError("voltages must be finite")

    def __iter__(self):
        return iter((self.voltage, self.current))

    def __repr__(self):
        return f"IVCurve({len(self.voltage)} points)"

    def subthreshold_swing(self, v_min: float = 0.0, v_max: float = 0.4) -> float:
        """Subthreshold swing from a `log10(I)` vs voltage linear fit over
        `[v_min, v_max]`."""
        mask = (self.voltage >= v_min) & (self.voltage <= v_max)
        if mask.sum() < 2:
            raise ValueError("not enough points in [v_min, v_max] to fit a swing")
        voltage, current = self.voltage[mask], self.current[mask]
        if not np.isfinite(voltage).all() or not np.isfinite(current).all() or (current <= 0).any():
            raise ValueError("swing requires finite voltages and strictly positive finite currents")
        if np.ptp(voltage) == 0 or np.ptp(current) == 0:
            raise ValueError("swing requires varying voltage and current")
        log_i = np.log10(current)
        slope, _ = np.polyfit(voltage, log_i, 1)
        if not np.isfinite(slope) or slope <= np.finfo(float).eps:
            raise ValueError("swing requires a positive, resolvable slope")
        return float(1.0 / slope)


class Device:
    """A 1D ballistic MOSFET device.

    Thin, Pythonic wrapper around the Rust engine: numpy arrays for
    profiles, and dataclass-like results instead of raw tuples. Keyword
    arguments match the Rust `DeviceParams` field names (see the top-level
    README for units and defaults), e.g.::

        dev = negforge.Device(v_ds=0.3, v_g=0.2, l_ch=40.0)
        dev.solve_self_consistent()
        current = dev.calc_current()
    """

    def __init__(self, **kwargs):
        self._inner = _RustDevice(**kwargs)
        self._inner.calc_potential()

    def __repr__(self):
        return f"Device(n={self.n}, a={self.a} nm, screening_length={self.screening_length:.3f} nm)"

    # -- electrostatics ----------------------------------------------------

    def calc_potential(self) -> "Device":
        """Decoupled electrostatic solve (`rho` unchanged)."""
        self._inner.calc_potential()
        return self

    def solve_self_consistent(
        self,
        max_iterations: int = 50,
        tolerance: float = 1e-6,
        mixing: float = 0.3,
        eta: float = 0.08,
        green_energy_fraction: float = 0.7,
    ) -> tuple[int, float]:
        """Run the self-consistent Poisson<->NEGF loop. Returns
        `(iterations, residual)` on success; raises `RuntimeError` if it
        does not converge within `max_iterations`.
        """
        return self._inner.solve_self_consistent(
            max_iterations, tolerance, mixing, eta, green_energy_fraction
        )

    # -- bias control --------------------------------------------------------

    def set_v_ds(self, v: float) -> "Device":
        """Set drain bias, clear charge feedback and recompute decoupled potential."""
        self._inner.set_v_ds(v)
        return self

    def set_v_g(self, v: float) -> "Device":
        """Set gate bias, clear charge feedback and recompute decoupled potential."""
        self._inner.set_v_g(v)
        return self

    def set_l_ch(self, l: float) -> "Device":
        """Resize the grid, clear charge feedback and recompute decoupled potential."""
        self._inner.set_l_ch(l)
        return self

    # -- observables -----------------------------------------------------------

    def calc_current(self) -> float:
        """Coherent NEGF/Landauer current through the solved potential, A."""
        return self._inner.calc_current()

    def calc_negf_current(self) -> float:
        """Explicit name for the coherent current returned by `calc_current`."""
        return self._inner.calc_negf_current()

    def calc_thermionic_current(self) -> float:
        """Ideal step-transmission current, A (no tunnelling or reflection).

        This analytic comparator is not the default transport model.
        """
        return self._inner.calc_thermionic_current()

    def transmission_spectrum(self, energies) -> np.ndarray:
        """Dimensionless coherent transmission at 1D input energies in eV.

        Returns a float array with the same shape and order as `energies`.
        Energies need not be sorted; empty inputs return an empty array.
        """
        energies = np.asarray(energies, dtype=float)
        if energies.ndim != 1 or not np.isfinite(energies).all():
            raise ValueError("energies must be a finite one-dimensional array")
        return np.asarray(self._inner.transmission_spectrum(energies.tolist()), dtype=float)

    @property
    def psi_f(self) -> np.ndarray:
        """Solved electrostatic potential energy profile, eV."""
        return np.asarray(self._inner.psi_f)

    @property
    def rho(self) -> np.ndarray:
        """Charge density term entering the electrostatic solve."""
        return np.asarray(self._inner.rho)

    @property
    def positions_nm(self) -> np.ndarray:
        return np.asarray(self._inner.positions_nm())

    @property
    def n(self) -> int:
        return self._inner.n

    @property
    def a(self) -> float:
        return self._inner.a

    @property
    def screening_length(self) -> float:
        """Natural (screening) length lambda, nm."""
        return self._inner.screening_length

    def local_density_of_states(self):
        """NEGF sweep at the device's current potential.

        Returns energies of shape `(M,)` and spectral values of shape `(M, n)`
        over the tight-binding band envelope from min(U) to max(U) + 4t_hop
        in eV. This range is independent of reservoir occupation. Divide
        the returned `-Im(G_ii)/a` by pi for the per-spin LDOS per unit length.
        """
        energies, ldos = self._inner.local_density_of_states()
        return np.asarray(energies), np.asarray(ldos)

    def sweep_v_g(self, v_min: float, v_max: float, step: float, self_consistent: bool = False) -> IVCurve:
        """Gate-voltage sweep at the device's current drain bias."""
        voltage, current = self._inner.sweep_v_g(v_min, v_max, step, self_consistent)
        return IVCurve(voltage, current)

    def sweep_v_ds(self, v_min: float, v_max: float, step: float, self_consistent: bool = False) -> IVCurve:
        """Drain-voltage sweep at the device's current gate bias."""
        voltage, current = self._inner.sweep_v_ds(v_min, v_max, step, self_consistent)
        return IVCurve(voltage, current)
