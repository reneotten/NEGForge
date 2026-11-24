"""Depleted charge is not an absence of available spectral states."""
import numpy as np
import pytest

from negforge import Device


def device(**kwargs):
    params = dict(a=0.5, l_ch=40.0, l_ds=4.0, auto_size_contacts=False, d_e=0.001)
    params.update(kwargs)
    return Device(**params)


@pytest.mark.parametrize("v_g,t", [(0.0, 0.0), (0.0, 10.0), (-2.0, 300.0)])
def test_depleted_scf_and_unoccupied_ldos(v_g, t):
    dev = device(v_g=v_g, t=t)
    assert dev.n == 97
    iterations, residual = dev.solve_self_consistent()
    assert 1 <= iterations <= 50
    assert residual < 1e-6
    np.testing.assert_array_equal(dev.rho, np.zeros(97))
    assert dev.calc_current() == 0.0
    energies, ldos = dev.local_density_of_states()
    # Spectral window covers the full local tight-binding band envelope,
    # independently of reservoir occupation (not the empty SCF charge grid).
    assert energies.ndim == 1 and energies.size > 1
    assert ldos.shape == (energies.size, 97)
    assert np.isfinite(ldos).all() and (ldos >= 0).all()
    assert (ldos > 0).any()
    assert energies[0] > dev.psi_f.min()
    assert energies[-1] > dev.psi_f.max()
    assert np.diff(energies).max() <= 0.001 * (1 + 1e-10)
    assert dev.transmission_spectrum([]).shape == (0,)


@pytest.mark.parametrize("t,v_g", [(0.0, 0.0), (10.0, 0.0), (300.0, -2.0)])
def test_depleted_self_consistent_bias_sweeps(t, v_g):
    dev = device(t=t, v_g=v_g)
    gate = dev.sweep_v_g(v_g - 0.1, v_g, 0.05, self_consistent=True)
    assert gate.voltage.shape == (3,)
    np.testing.assert_array_equal(gate.current, np.zeros(3))
    drain = dev.sweep_v_ds(0.0, 0.1, 0.05, self_consistent=True)
    assert drain.voltage.shape == (3,)
    assert np.isfinite(drain.current).all()
    assert (drain.current >= 0).all()


@pytest.mark.parametrize("t", [-1.0, float("nan"), float("inf")])
def test_invalid_temperature_remains_an_input_error(t):
    with pytest.raises(ValueError, match="t.*finite|temperature"):
        device(t=t)


def test_single_cell_windows_preserve_python_shapes():
    dev = device(v_g=1.0, t=0.0, d_e=10.0)
    _, residual = dev.solve_self_consistent()
    assert residual < 1e-6
    energies, ldos = dev.local_density_of_states()
    assert energies.shape == (1,)
    assert ldos.shape == (1, 97)
    assert np.isfinite(ldos).all()


def test_ldos_large_grid_is_still_budget_limited():
    with pytest.raises(ValueError, match="resource budget"):
        device(d_e=1e-12).local_density_of_states()
