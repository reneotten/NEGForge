"""Integration tests: run against an installed maturin wheel, not source mocks."""

import json
from importlib.metadata import metadata
from pathlib import Path

import numpy as np
import pytest

from negforge import Device, IVCurve
from negforge._negforge import Device as RustDevice


SMALL = dict(a=1.0, l_ch=10.0, l_ds=10.0, auto_size_contacts=False, d_e=0.01)


def test_python_version_metadata_matches_pyo3_support():
    assert metadata("negforge")["Requires-Python"].replace(" ", "") == ">=3.9,<3.15"


def test_profiles_and_ldos_are_finite_numpy_arrays():
    dev = Device(**SMALL, v_ds=0.3, v_g=0.2)
    assert dev.n == 31
    np.testing.assert_allclose(dev.positions_nm, np.arange(dev.n) * dev.a)
    for profile in (dev.psi_f, dev.rho):
        assert profile.shape == (dev.n,)
        assert np.isfinite(profile).all()
    saved = dev.psi_f.copy()
    dev.psi_f[:] = 999
    np.testing.assert_array_equal(dev.psi_f, saved)
    energies, ldos = dev.local_density_of_states()
    assert ldos.shape == (energies.size, dev.n)
    assert np.isfinite(ldos).all()
    assert (ldos >= 0).all()


def test_setters_recompute_profiles_and_current():
    dev = Device(**SMALL)
    assert dev.calc_current() == 0
    assert dev.set_v_ds(0.3).set_v_g(0.2) is dev
    expected = Device(**SMALL, v_ds=0.3, v_g=0.2)
    np.testing.assert_allclose(dev.psi_f, expected.psi_f)
    assert dev.calc_current() == pytest.approx(expected.calc_current())
    assert dev.set_l_ch(15.0) is dev
    assert dev.psi_f.shape == (dev.n,)
    assert dev.rho.shape == (dev.n,)


@pytest.mark.parametrize("name", ["sweep_v_g", "sweep_v_ds"])
def test_sweep_matches_individual_solves(name):
    dev = Device(**SMALL, v_ds=0.3)
    curve = getattr(dev, name)(0.0, 0.2, 0.1)
    voltage, current = curve
    np.testing.assert_allclose(voltage, [0, 0.1, 0.2])
    expected = []
    for v in voltage:
        single = Device(**SMALL, v_ds=0.3)
        getattr(single, "set_" + name.removeprefix("sweep_"))(v)
        single.calc_potential()
        expected.append(single.calc_current())
    np.testing.assert_allclose(current, expected)


@pytest.mark.parametrize("voltage,current", [([[0, 1]], [1, 2]), ([0, 1], [1]), ([0, np.nan], [1, 2])])
def test_ivcurve_rejects_invalid_arrays(voltage, current):
    with pytest.raises(ValueError):
        IVCurve(voltage, current)


def test_subthreshold_swing_known_exponential():
    voltage = np.linspace(0, 0.4, 9)
    assert IVCurve(voltage, 10 ** (voltage / 0.06)).subthreshold_swing() == pytest.approx(0.06)


@pytest.mark.parametrize("current", [[0, 1, 2], [-1, 1, 2], [1, np.nan, 2], [1, np.inf, 2], [1, 1, 1]])
def test_subthreshold_swing_rejects_undefined_fit(current):
    with pytest.raises(ValueError):
        IVCurve([0, 0.1, 0.2], current).subthreshold_swing()


def test_self_consistent_convergence_and_nonconvergence():
    dev = Device(**SMALL)
    iterations, residual = dev.solve_self_consistent(max_iterations=100, tolerance=1e-5, eta=0.04)
    assert 0 < iterations <= 100
    assert np.isfinite(residual) and residual < 1e-5
    assert np.isfinite(dev.rho).all()
    with pytest.raises(RuntimeError, match="did not converge"):
        # Zero built-in barrier keeps the tiny SI-consistent charge response
        # above roundoff relative to the initial zero potential.
        Device(**SMALL, e_f=0.0, e_g=0.0).solve_self_consistent(max_iterations=1, tolerance=1e-30)


def test_invalid_solver_options_are_value_errors():
    with pytest.raises(ValueError, match="mixing"):
        Device(**SMALL).solve_self_consistent(mixing=0)


def test_native_sweeps_have_same_optional_argument_as_wrapper():
    dev = RustDevice(**SMALL)
    voltage, current = dev.sweep_v_g(0, 0.1, 0.1)
    assert len(voltage) == len(current) == 2


def test_notebook_code_is_valid_python():
    notebook = json.loads((Path(__file__).parents[1] / "notebooks/negforge_demo.ipynb").read_text())
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] == "code":
            compile("".join(cell["source"]), f"notebook-cell-{index}", "exec")


@pytest.mark.parametrize("kwargs", [{"d_e": -0.01}, {"t": 0}, {"v_g": float("nan")}, {"epsilon": 1.0}])
def test_constructor_rejects_invalid_parameters(kwargs):
    with pytest.raises(ValueError):
        Device(**kwargs)


@pytest.mark.parametrize("device_type", [Device, RustDevice])
@pytest.mark.parametrize("area", [0.0, -1.0, float("nan"), float("inf"), -float("inf")])
def test_constructor_rejects_invalid_effective_area(device_type, area):
    with pytest.raises(ValueError, match="effective_area_m2"):
        device_type(**SMALL, effective_area_m2=area)


def test_finite_area_scf_raises_channel_barrier():
    geometry = dict(a=0.5, l_ch=8.0, l_ds=8.0, auto_size_contacts=False,
                    v_g=0.5, d_e=0.002)
    options = dict(max_iterations=300, tolerance=1e-7, mixing=0.15,
                   eta=0.002, green_energy_fraction=1.0)
    legacy = Device(**geometry)
    legacy.solve_self_consistent(**options)
    finite = Device(**geometry, effective_area_m2=20e-18)
    iterations, residual = finite.solve_self_consistent(**options)
    channel = (finite.positions_nm >= 8) & (finite.positions_nm <= 16)
    assert iterations > 1
    assert residual < options["tolerance"]
    assert np.max(finite.psi_f[channel] - legacy.psi_f[channel]) > 1e-3
    assert np.isfinite(finite.rho).all()


@pytest.mark.parametrize("device_type", [Device, RustDevice])
@pytest.mark.parametrize("fraction", [0.0, -0.1, 1.1, float("nan"), float("inf")])
def test_invalid_green_energy_fraction_is_value_error(device_type, fraction):
    with pytest.raises(ValueError):
        device_type(**SMALL).solve_self_consistent(green_energy_fraction=fraction)


@pytest.mark.parametrize("name", ["sweep_v_g", "sweep_v_ds"])
def test_self_consistent_sweep_preserves_finite_area_geometry(name):
    geometry = dict(SMALL, v_ds=0.1, v_g=0.5, effective_area_m2=100e-18)
    dev = Device(**geometry)
    curve = getattr(dev, name)(0.1, 0.2, 0.1, self_consistent=True)
    expected = []
    for voltage in curve.voltage:
        single = Device(**geometry)
        getattr(single, "set_" + name.removeprefix("sweep_"))(voltage)
        single.solve_self_consistent()
        expected.append(single.calc_current())
    np.testing.assert_allclose(curve.current, expected, rtol=1e-10, atol=1e-20)
    np.testing.assert_allclose(dev.psi_f, single.psi_f, rtol=1e-10, atol=1e-12)
    legacy = Device(**{k: v for k, v in geometry.items() if k != "effective_area_m2"})
    getattr(legacy, "set_" + name.removeprefix("sweep_"))(curve.voltage[-1])
    legacy.solve_self_consistent()
    assert np.max(dev.psi_f - legacy.psi_f) > 1e-5


def test_invalid_setter_preserves_device():
    dev = Device(**SMALL)
    before = dev.psi_f.copy()
    with pytest.raises(ValueError):
        dev.set_v_g(float("nan"))
    np.testing.assert_array_equal(dev.psi_f, before)
