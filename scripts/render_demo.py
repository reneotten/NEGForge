#!/usr/bin/env python3
"""Computed notebook/gallery figures. Run with the repo's .venv314 Python.

No transport surrogate is used: full rendering requires the coherent Python API.
Use --electrostatics-only to render the independently available first three figures.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import matplotlib
# Backend selection belongs to the CLI, never to an imported figure library.
if __name__ == "__main__":
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.patches import Rectangle
import numpy as np
import negforge

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "figures"
Q, H, KB = 1.602176634e-19, 6.62607015e-34, 1.380649e-23
HBAR, MASS = 1.054571817e-34, 0.9 * 9.1093837015e-31
G0 = 2 * Q**2 / H  # A/eV when integrating energies numerically in eV
PARAMS = dict(a=0.25, l_ch=8.0, l_ds=8.0, auto_size_contacts=False,
              v_g=0.5, v_ds=0.1, d_e=0.002, t=300.0, e_fs=0.05,
              d_ch=5.0, d_ox=5.0, geo=1.0)
AREA = 20e-18  # Explicit caller assumption, not inferred from body thickness.
SCF = dict(max_iterations=400, tolerance=1e-7, mixing=0.1,
           eta=0.002, green_energy_fraction=1.0)
plt.rcParams.update({"font.size": 10, "figure.dpi": 130,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.2})


def require_transport():
    missing = [name for name in ("transmission_spectrum", "calc_thermionic_current")
               if not hasattr(negforge.Device, name)]
    if missing:
        raise RuntimeError("Installed negforge lacks " + ", ".join(missing)
                           + ". Rebuild the updated extension in .venv314, then run:\n"
                           + ".venv314/bin/python scripts/render_demo.py")


def device(*, scf=False, **changes):
    p = PARAMS | changes
    d = negforge.Device(**p, effective_area_m2=AREA)
    status = d.solve_self_consistent(**SCF) if scf else None
    return d, status


def electrostatic_examples():
    dec, _ = device()
    sc, status = device(scf=True)
    # Separates numerical quadrature change from eta-dependent fixed point.
    fine, fine_status = device(scf=True, d_e=0.001)
    eta_device = negforge.Device(**(PARAMS | {"d_e": 0.001}), effective_area_m2=AREA)
    eta_status = eta_device.solve_self_consistent(**(SCF | {"eta": 0.001}))
    unit = negforge.Device(**PARAMS)
    unit_status = unit.solve_self_consistent(**SCF)
    stats = {
        "parameters": PARAMS, "effective_area_m2": AREA, "scf_options": SCF,
        "sites": dec.n, "screening_length_nm": dec.screening_length,
        "decoupled_barrier_eV": float(dec.psi_f.max()),
        "scf_barrier_eV": float(sc.psi_f.max()),
        "max_feedback_eV": float(np.max(sc.psi_f - dec.psi_f)),
        "min_feedback_eV": float(np.min(sc.psi_f - dec.psi_f)),
        "scf_iterations": int(status[0]), "scf_residual_eV": float(status[1]),
        "peak_abs_charge_C_m3": float(np.max(np.abs(sc.rho))),
        "implicit_unit_area_iterations": int(unit_status[0]),
        "implicit_unit_area_max_feedback_eV": float(np.max(np.abs(unit.psi_f-dec.psi_f))),
        "fixed_eta_half_dE_barrier_eV": float(fine.psi_f.max()),
        "fixed_eta_half_dE_residual_eV": float(fine_status[1]),
        "half_eta_half_dE_barrier_eV": float(eta_device.psi_f.max()),
        "half_eta_half_dE_residual_eV": float(eta_status[1]),
    }
    assert status[1] < SCF["tolerance"]
    assert np.isfinite(sc.psi_f).all() and np.isfinite(sc.rho).all()
    assert np.min(sc.rho) < 0 and np.max(sc.rho) <= 0
    assert np.min(sc.psi_f-dec.psi_f) > 0, "Electron feedback must raise energy"
    return dec, sc, fine, eta_device, stats


def geometry_figure(dec):
    fig, (ax, pot) = plt.subplots(2, 1, figsize=(9, 6), layout="constrained")
    for left, width, color, label in [(0, 8, "#b7d9ef", "Source"),
                                      (8, 8, "#bce2c4", "Channel"),
                                      (16, 8, "#f0c9b2", "Drain")]:
        ax.add_patch(Rectangle((left, 0), width, 5, color=color, ec="white"))
        ax.text(left + width/2, 2.5, label, ha="center", va="center")
    ax.add_patch(Rectangle((8, 5), 8, 5, color="#e5e5e5"))
    ax.add_patch(Rectangle((8, 10), 8, 1, color="#758195"))
    ax.text(12, 7.5, "oxide: 5 nm", ha="center", va="center")
    ax.text(12, 11.5, "gate: 0.5 V; geo = 1", ha="center")
    ax.set(xlim=(0, 24), ylim=(-1, 13), xlabel="longitudinal position x (nm)",
           ylabel="transverse coordinate (nm)", title="1D transport grid / assumed transverse geometry")
    ax.grid(False)
    pot.plot(dec.positions_nm, dec.psi_f, color="#1765a0", label="decoupled electron energy U(x)")
    for edge in [8, 16]:
        pot.axvline(edge, color="gray", ls=":")
    pot.set(xlabel="x (nm)", ylabel="U (eV)", title=f"{dec.n} sites; a = 0.25 nm; natural length = {dec.screening_length:.2f} nm")
    pot.legend(loc="lower left")
    fig.suptitle("Caller-supplied SCF area = 20 nm²; no transverse modes are resolved", fontsize=12)
    return fig


def feedback_figure(dec, sc, fine, eta_device):
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.1), layout="constrained")
    x = dec.positions_nm
    axes[0].plot(x, dec.psi_f, label="electrostatic (rho = 0)")
    axes[0].plot(x, sc.psi_f, label="finite-area SCF")
    axes[0].set(ylabel="electron energy U (eV)", title="Charge raises the energy")
    axes[0].legend(fontsize=8)
    axes[1].plot(x, sc.rho / 1e5)
    axes[1].set(ylabel="mobile charge (10⁵ C/m³)", title="Spin-summed charge / 20 nm²")
    for d, label, style in [(sc, "dE = eta = 2 meV", "-"), (fine, "dE = 1; eta = 2 meV", "--"),
                            (eta_device, "dE = eta = 1 meV", "-")]:
        axes[2].plot(x, d.psi_f, style, label=label)
    axes[2].set(ylabel="U (eV)", title="Quadrature / eta sensitivity")
    axes[2].legend(fontsize=8, loc="lower center")
    for ax in axes:
        ax.set_xlabel("x (nm)")
        for edge in [8, 16]:
            ax.axvline(edge, color="gray", ls=":", lw=0.8)
    fig.suptitle("SCF convergence is not eta → 0 convergence; eta absorbs without reinjection", fontsize=12)
    return fig


def ldos_figure(dec, sc, stats):
    pairs = [d.local_density_of_states() for d in (dec, sc)]
    arrays = [np.asarray(z) / np.pi for _, z in pairs]
    assert all(np.isfinite(z).all() and z.max() > 0 and z.min() >= 0 for z in arrays)
    vmax = max(float(np.quantile(z, 0.995)) for z in arrays)
    norm = LogNorm(vmin=vmax*1e-5, vmax=vmax)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), layout="constrained")
    stats["ldos"] = []
    for ax, d, (energy, _), z, title in zip(axes, (dec, sc), pairs, arrays,
                                         ("Decoupled electrostatics", "Finite-area SCF potential")):
        mesh = ax.pcolormesh(d.positions_nm, energy, np.maximum(z, np.finfo(float).tiny),
                             norm=norm, shading="auto", cmap="inferno", rasterized=True)
        ax.plot(d.positions_nm, d.psi_f, color="cyan", lw=1.3, label="U(x)")
        ax.axhline(PARAMS["e_fs"], color="white", ls="--", lw=0.8, label="source chemical potential")
        ax.set(xlabel="x (nm)", ylabel="energy E (eV)", title=title,
               ylim=(min(float(pair[0][0]) for pair in pairs), 0.25))
        ax.legend(fontsize=7, loc="upper right", facecolor="#303030", labelcolor="white", framealpha=0.85)
        stats["ldos"].append({"shape": list(z.shape), "maximum_per_eV_nm": float(z.max()),
                               "fraction_above_1e_minus_6_peak": float(np.mean(z > z.max()*1e-6))})
    fig.colorbar(mesh, ax=axes, label="per-spin LDOS (eV nm)⁻¹; shared log scale", extend="both")
    fig.suptitle("LDOS = −Im Gᵣᵢᵢ / (πa): available states, not flux or transmission", fontsize=12)
    return fig


def spectrum(d, step=0.00025):
    # Common propagating lead band and thermally relevant upper window.
    hop = HBAR**2 / (2*MASS*(d.a*1e-9)**2*Q)
    low = max(float(d.psi_f[0]), float(d.psi_f[-1]))
    high = min(float(d.psi_f[0])+4*hop, float(d.psi_f[-1])+4*hop,
               PARAMS["e_fs"] + 35*KB*PARAMS["t"]/Q)
    count = int(np.ceil((high-low)/step))
    # Midpoints avoid singular band edges and permit rectangle integration.
    de = (high-low)/count
    energy = low + (np.arange(count)+0.5)*de
    trans = np.asarray(d.transmission_spectrum(energy), dtype=float)
    assert trans.shape == energy.shape and np.isfinite(trans).all()
    assert trans.min() >= -1e-12 and trans.max() <= 1+1e-8
    return energy, trans, de


def occupations(energy, v_ds=0.1):
    kt = KB*PARAMS["t"]/Q
    fs = 1/(1+np.exp(np.clip((energy-PARAMS["e_fs"])/kt, -700, 700)))
    fd = 1/(1+np.exp(np.clip((energy-(PARAMS["e_fs"]-v_ds))/kt, -700, 700)))
    return fs, fd


def transport_figure(dec, sc, stats):
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.3), layout="constrained")
    spectral = {}
    for d, label, color in [(dec, "electrostatic", "#2467a4"), (sc, "finite-area SCF", "#bd512a")]:
        e, t, de = spectrum(d)
        fs, fd = occupations(e)
        j = G0*t*(fs-fd)
        barrier = float(d.psi_f.max())
        axes[0].semilogy(e, np.maximum(t, 1e-18), color=color, label=label)
        axes[0].axvline(barrier, color=color, ls=":", lw=1, label=f"{label} barrier")
        axes[2].plot(e, j*1e6, color=color, label=label)
        sub = e < barrier
        axes[2].fill_between(e, 0, j*1e6, where=sub, color=color, alpha=.18)
        total = float(np.sum(j)*de)
        coarse_e, coarse_t, coarse_de = spectrum(d, step=0.0005)
        cs, cd = occupations(coarse_e)
        coarse = float(G0*np.sum(coarse_t*(cs-cd))*coarse_de)
        api = float(d.calc_current())
        assert np.isfinite(api) and api > 0
        # Independent integration check, not an equality between two API calls.
        mismatch = abs(total-api)/api
        assert mismatch < .03, f"Spectral integral/API disagree by {mismatch:.2%}"
        spectral[label] = dict(coherent_api_A=api, integrated_spectrum_A=total,
                              thermionic_A=float(d.calc_thermionic_current()),
                              integration_relative_difference=mismatch,
                              half_step_relative_change=abs(total-coarse)/total,
                              sub_barrier_current_A=float(np.sum(j[sub])*de),
                              sub_barrier_fraction=float(np.sum(j[sub])*de/total),
                              max_T=float(t.max()), min_T=float(t.min()),
                              common_lead_band_min_eV=float(max(d.psi_f[0], d.psi_f[-1])))
    window = np.linspace(-.15, .3, 1000)
    fs, fd = occupations(window)
    axes[1].plot(window, fs, label="f source")
    axes[1].plot(window, fd, label="f drain")
    axes[1].plot(window, fs-fd, "k--", label="Fermi window: f source − f drain")
    axes[1].set(ylabel="occupation / window", ylim=(-.03, 1.03), title="Reservoir occupation at 300 K")
    axes[0].set(ylabel="coherent transmission T(E)", ylim=(1e-10, 1.15), title="Finite T below the barrier")
    axes[2].set(ylabel="spectral current (µA/eV)", title="Shading: E below each barrier")
    for ax in axes:
        ax.set_xlabel("energy E (eV)")
        ax.set_xlim(-.02, .3)
        ax.legend(fontsize=7)
    axes[1].set_xlim(-.15, .3)  # Include both chemical potentials, especially mu_d = -0.05 eV.
    stats["spectral_current"] = spectral
    fig.suptitle("I = (2e²/h) ∫ T(E) [f source − f drain] dE (E in eV); Vds = 0.1 V", fontsize=12)
    return fig


def iv_figure(stats):
    fig, grid = plt.subplots(2, 2, figsize=(11, 6.5), layout="constrained",
                             sharex="col", gridspec_kw={"height_ratios": [3, 1]})
    axes, ratio_axes = grid[0], grid[1]
    rows = []
    for ax, name, values in [(axes[0], "v_g", np.linspace(.3, .6, 7)),
                             (axes[1], "v_ds", np.array([0., .05, .1, .15]))]:
        curves = {(scf, model): [] for scf in (False, True) for model in ("coherent", "thermionic")}
        for value in values:
            for scf in (False, True):
                d, status = device(scf=scf, **{name: float(value)})
                currents = {"coherent": float(d.calc_current()), "thermionic": float(d.calc_thermionic_current())}
                assert all(np.isfinite(i) and i >= -1e-18 for i in currents.values())
                if name == "v_ds" and value == 0:
                    assert max(abs(i) for i in currents.values()) < 1e-18
                rows.append(dict(sweep=name, voltage_V=float(value), scf=scf,
                                 barrier_eV=float(d.psi_f.max()), **{k+"_A": v for k,v in currents.items()},
                                 iterations=int(status[0]) if status else 0,
                                 residual_eV=float(status[1]) if status else None))
                for model, current in currents.items():
                    curves[scf, model].append(current*1e6)
        for (scf, model), current in curves.items():
            label = ("finite-area SCF" if scf else "electrostatic") + ": " + model
            ax.plot(values, current, marker="o", ms=3, color="#bd512a" if scf else "#2467a4",
                    ls="-" if model == "coherent" else "--", label=label)
        ratio_ax = ratio_axes[0 if name == "v_g" else 1]
        for scf in (False, True):
            thermal = np.asarray(curves[scf, "thermionic"])
            coherent = np.asarray(curves[scf, "coherent"])
            ratio = np.divide(coherent, thermal, out=np.full_like(thermal, np.nan), where=thermal != 0)
            ratio_ax.plot(values, ratio, "o-", ms=3, color="#bd512a" if scf else "#2467a4")
        ratio_ax.axhline(1, color="gray", ls=":", lw=1)
        ratio_ax.set(ylabel="coherent /\nthermionic", xlabel="Vg (V)" if name == "v_g" else "Vds (V)")
        ax.set(ylabel="spin-summed current (µA)",
               title="Transfer: Vds = 0.1 V" if name == "v_g" else "Output: Vg = 0.5 V")
        if name == "v_g":
            ax.set_yscale("log")
        ax.legend(fontsize=7)
    stats["iv_points"] = rows
    fig.suptitle("Coherent transport vs T = Θ(E − max U) thermionic comparator", fontsize=12)
    return fig


def write_gallery(stats):
    text = """# Coherent-transport calculation gallery

Regenerate from the repository root using the installed updated Python extension:

```bash
.venv314/bin/python scripts/render_demo.py
```

The output-free [notebook](../../notebooks/negforge_demo.ipynb) executes the same
functions. `sanity_checks.json` contains inputs, solver residuals, LDOS statistics,
spectral integration checks, and every I–V point. Failed assertions or SCF solves
stop rendering; missing transport APIs never trigger surrogate curves.
`--electrostatics-only` is an explicit partial-render option.

Use the [setup instructions](../../README.md#python-package--notebook) to create
and activate `.venv314`. The CLI selects the headless Agg backend; importing its
figure helpers leaves the caller's backend unchanged. The notebook explicitly
selects inline PNG display. Verify actual notebook images in a fresh kernel with:

```bash
.venv314/bin/python -m pytest -q -s tests/test_notebook.py
```

The regression requires six executed code cells and five decoded, nonblank
`image/png` display outputs; existing gallery files cannot satisfy it.

## 1. Device geometry and electrostatics

![Geometry and computed potential](01_geometry.png)

The schematic depicts the declared input, not a transverse simulation. The 1D
chain has 8 nm source, channel, and drain regions; spacing 0.25 nm; 97 sites.
Body and oxide thicknesses are 5 nm, `geo=1`, temperature 300 K, Vg=0.5 V,
Vds=0.1 V. The caller supplies 20 nm² for SCF volume charge, independently
of body thickness; current remains a spin-summed single-mode current.
Endpoint band energies are calculated, not pinned to reservoir chemical
potentials (mu source=0.05 eV, mu drain=-0.05 eV).

## 2. Explicit-area feedback and numerical sensitivity

![Electrostatic and finite-area SCF comparison](02_feedback.png)

Electrons raise their potential energy through repulsive feedback. Negative
rho is spin-summed volume charge, in C/m³. Implicit unit-area charge is not a
specified physical cross-section. The iteration residual, quadrature error,
and absorbing eta dependence are separate numerical questions.

| Computed quantity | Value |
|---|---:|
"""
    for label, key in [("Natural length (nm)", "screening_length_nm"),
                       ("Electrostatic barrier (eV)", "decoupled_barrier_eV"),
                       ("Finite-area SCF barrier (eV)", "scf_barrier_eV"),
                       ("SCF iterations", "scf_iterations"),
                       ("SCF residual (eV)", "scf_residual_eV"),
                       ("Maximum feedback (eV)", "max_feedback_eV"),
                       ("Peak charge magnitude (C/m³)", "peak_abs_charge_C_m3"),
                       ("Implicit unit-area max feedback (eV)", "implicit_unit_area_max_feedback_eV"),
                       ("Barrier: dE=1, eta=2 meV (eV)", "fixed_eta_half_dE_barrier_eV"),
                       ("Barrier: dE=eta=1 meV (eV)", "half_eta_half_dE_barrier_eV")]:
        text += f"| {label} | {stats[key]:.9g} |\n"
    text += """
The baseline uses dE=eta=2 meV, mixing=0.1, tolerance=1e-7 eV, up to
400 iterations, and the full stored SCF energy bound. Halving dE at fixed eta
barely shifts the barrier; reducing eta shifts it visibly. This is **not** a
claim of eta→0 convergence. Eta absorbs without compensating injection and
is not tunneling or a conserving scattering/dephasing model.

## 3. Available states and potential overlay

![Per-spin LDOS with potential overlay](03_ldos.png)

LDOS is `-Im G_ii/(pi*a)`, in (eV nm)⁻¹ per spin; it is neither particle
flux nor transmission. Cyan is U(x), white is the source chemical potential.
Both panels use one log normalization and show the same low-energy window
through 0.25 eV. The cap is the larger 99.5th percentile; full sampled arrays
extend beyond this display window. White below the SCF energy grid is
**unsampled**, not zero LDOS. The LDOS helper's fixed retarded broadening is
separate from the declared SCF eta; narrow features require grid checks.
"""
    if stats["transport_rendered"]:
        text += """
## 4. Coherent transmission, Fermi window, and spectral current

![Transmission, reservoir occupation and spectral current](04_transport.png)

T(E) is computed by `transmission_spectrum`, restricted to the common
propagating lead band. Dotted lines mark each barrier maximum. Finite
transmission below that maximum is sub-barrier transport, not a conclusion
inferred from LDOS. The small sub-barrier interval in the SCF device reflects
its raised source endpoint band edge, not a missing occupied reservoir window.
The Fermi window weights available transmission; shaded spectral-current
areas lie below the corresponding barrier.

For numerical energies in eV, `j(E)=(2e²/h)*T(E)*(f_source-f_drain)` in A/eV.
The extra e converts the integration measure, and spin is counted exactly once.

| Potential | Coherent API (µA) | Thermionic (µA) | Below-barrier fraction | Spectral/API relative difference | Half-step relative change |
|---|---:|---:|---:|---:|---:|
"""
        for label, result in stats["spectral_current"].items():
            text += (f"| {label} | {result['coherent_api_A']*1e6:.8g} | "
                     f"{result['thermionic_A']*1e6:.8g} | {result['sub_barrier_fraction']:.6g} | "
                     f"{result['integration_relative_difference']:.4g} | "
                     f"{result['half_step_relative_change']:.4g} |\n")
        text += """
The independent midpoint spectrum uses an approximately 0.25 meV energy step,
compares with 0.5 meV, and integrates within the common lead band through
35 kBT above the highest chemical potential or the band ceiling. Assertions
check finite 0≤T≤1, positive biased current, and spectral/API agreement within
3%; the actual errors are tabulated above, not merely the pass threshold.

## 5. Transfer and output results

![Computed I–V comparisons and model ratios](05_iv.png)

Every point starts with a fresh device. Solid: coherent `calc_current()`;
dashed: `calc_thermionic_current()` with T=Theta(E-max U). Color selects
potential model, and bottom panels resolve their small current differences.
The 0/0 ratio at equilibrium is deliberately omitted; both zero-bias currents
are asserted zero. All potential/bias calculations and their residuals are
recorded in the JSON. No broad-range subthreshold-swing fit is assumed.

Coherent minus thermionic current is **not the sub-barrier current**:
reflection and finite lead bandwidth also change above-barrier transport.
"""
    else:
        text += "\n**Partial render:** coherent spectra and I–V results have not been computed.\n"
    text += """
## Model limits

These are computed examples, not calibrated transistor predictions.
Natural-length electrostatics, finite contacts, effective-mass grid dispersion,
eta-dependent charge, absent transverse/valley structure, and isolated
bound-state occupations limit interpretation. A small iterative residual does
not remove those limitations. See [physics](../PHYSICS.md) and
[explicit-area normalization](../ELECTROSTATIC_GEOMETRY.md).
"""
    (OUT / "README.md").write_text(text)


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / (name+".png"), dpi=170)
    plt.close(fig)
    print(OUT / (name+".png"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--electrostatics-only", action="store_true")
    args = parser.parse_args()
    if not args.electrostatics_only:
        require_transport()
    dec, sc, fine, eta_device, stats = electrostatic_examples()
    save(geometry_figure(dec), "01_geometry")
    save(feedback_figure(dec, sc, fine, eta_device), "02_feedback")
    save(ldos_figure(dec, sc, stats), "03_ldos")
    stats["transport_rendered"] = not args.electrostatics_only
    if not args.electrostatics_only:
        save(transport_figure(dec, sc, stats), "04_transport")
        save(iv_figure(stats), "05_iv")
    (OUT / "sanity_checks.json").write_text(json.dumps(stats, indent=2)+"\n")
    write_gallery(stats)
    print(json.dumps(stats, indent=2))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(2)
