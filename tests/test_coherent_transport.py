"""Installed-wheel coverage of coherent transport and the thermionic comparator."""

import numpy as np
import pytest

from negforge import Device
from negforge._negforge import Device as RustDevice


SMALL = dict(a=1.0, l_ch=10.0, l_ds=10.0, auto_size_contacts=False, d_e=0.002)


@pytest.mark.parametrize("device_type", [Device, RustDevice])
def test_default_current_is_explicit_coherent_current(device_type):
    dev = device_type(**SMALL, v_ds=0.2, v_g=0.2)
    dev.calc_potential()
    coherent = dev.calc_negf_current()
    assert np.isfinite(coherent) and coherent > 0
    assert dev.calc_current() == coherent
    comparator = dev.calc_thermionic_current()
    assert np.isfinite(comparator) and comparator > 0
    assert not np.isclose(coherent, comparator, rtol=1e-3, atol=1e-20)


@pytest.mark.parametrize("energies", [[], [0.8, -0.1, 0.2, 0.8], np.linspace(-0.2, 1.0, 17)[::2]])
def test_transmission_is_matching_one_dimensional_array(energies):
    dev = Device(**SMALL, v_ds=0.2, v_g=0.2)
    spectrum = dev.transmission_spectrum(energies)
    assert isinstance(spectrum, np.ndarray)
    assert spectrum.shape == np.asarray(energies).shape
    assert spectrum.dtype == np.dtype(float)
    assert np.isfinite(spectrum).all()
    assert ((spectrum >= 0) & (spectrum <= 1 + 1e-10)).all()
    native = RustDevice(**SMALL, v_ds=0.2, v_g=0.2)
    native.calc_potential()
    np.testing.assert_allclose(spectrum, native.transmission_spectrum(list(energies)))


@pytest.mark.parametrize("energies", [0.2, [[0.1, 0.2]], [np.nan], [np.inf], [-np.inf]])
def test_transmission_rejects_invalid_energies(energies):
    with pytest.raises(ValueError):
        Device(**SMALL).transmission_spectrum(energies)


@pytest.mark.parametrize("energies", [[np.nan], [np.inf], [-np.inf]])
def test_native_transmission_rejects_nonfinite_energies(energies):
    with pytest.raises(ValueError):
        RustDevice(**SMALL).transmission_spectrum(energies)


@pytest.mark.parametrize("name", ["sweep_v_g", "sweep_v_ds"])
@pytest.mark.parametrize("self_consistent", [False, True])
def test_sweeps_use_coherent_core(name, self_consistent):
    geometry = dict(SMALL, v_ds=0.2, v_g=0.2)
    curve = getattr(Device(**geometry), name)(0.1, 0.2, 0.1, self_consistent=self_consistent)
    expected, thermionic = [], []
    for voltage in curve.voltage:
        dev = Device(**geometry)
        getattr(dev, "set_" + name.removeprefix("sweep_"))(voltage)
        if self_consistent:
            dev.solve_self_consistent()
        expected.append(dev.calc_negf_current())
        thermionic.append(dev.calc_thermionic_current())
    np.testing.assert_allclose(curve.current, expected, rtol=1e-10, atol=1e-20)
    assert not np.allclose(curve.current, thermionic, rtol=1e-3, atol=1e-20)


@pytest.mark.parametrize("device_type", [Device, RustDevice])
def test_coherent_current_equilibrium_and_bias_sign(device_type):
    dev = device_type(**SMALL)
    dev.calc_potential()
    assert dev.calc_current() == 0
    for bias in (-0.2, 0.2):
        dev.set_v_ds(bias)
        assert np.sign(dev.calc_current()) == np.sign(bias)


@pytest.mark.parametrize("device_type", [Device, RustDevice])
@pytest.mark.parametrize("method", ["calc_current", "calc_negf_current"])
def test_coherent_current_propagates_energy_budget_errors(device_type, method):
    dev = device_type(**dict(SMALL, d_e=1e-20), v_ds=0.2)
    dev.calc_potential()
    with pytest.raises(ValueError, match="energy budget"):
        getattr(dev, method)()


@pytest.mark.parametrize("name", ["sweep_v_g", "sweep_v_ds"])
def test_sweeps_propagate_coherent_energy_budget_errors(name):
    dev = Device(**dict(SMALL, d_e=1e-20), v_ds=0.2)
    with pytest.raises(ValueError, match="energy budget"):
        getattr(dev, name)(0.1, 0.2, 0.1)
