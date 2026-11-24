# NEGForge

A 1D ballistic-MOSFET electrostatics + NEGF (Non-Equilibrium Green's
Function) transport simulator. The numerical engine is Rust; the frontend
is a Jupyter notebook, talking to the engine through a native PyO3
extension module.

The engine couples electrostatics to NEGF-derived electron density through
a self-consistent fixed-point solve and evaluates ballistic Landauer current.

## Architecture

```
crates/negforge-core/   Pure-Rust physics engine (no Python dependency)
crates/negforge-py/     PyO3 bindings, compiled to the native module
                         negforge._negforge
python/negforge/        Pythonic wrapper package (numpy arrays, keyword
                         device construction, IVCurve helper)
notebooks/               Jupyter notebook frontend
legacy_matlab/           Original MATLAB code, kept for reference
docs/PHYSICS.md          Physics tutorial and current implementation guide
```

`negforge-core` has no PyO3/Python dependency at all — it's a normal Rust
library with its own unit and integration tests, usable from any Rust
program. `negforge-py` is a thin binding layer on top of it.

### `negforge-core` modules

- `constants` — physical constants (elementary charge, k_B, h, hbar,
  electron mass, and vacuum permittivity).
- `tridiag` — real and complex adjacent-partial-pivoting tridiagonal solvers, used
  by both the electrostatics and NEGF modules.
- `device` — device geometry/parameters and the electrostatic
  (`calc_potential`), coherent-current (`calc_current`), and analytic
  thermionic-comparator (`calc_thermionic_current`) calculations.
- `negf` — the NEGF retarded Green's-function calculation, using
  left- and right-connected recursive Green's-function
  sweeps (O(N)) for the diagonal (local density of states) and a direct
  tridiagonal solve (O(N)) for each of the two contact-column quantities
  needed for the charge density — both cross-checked against a dense
  reference solver in the test suite.
- `charge` — NEGF-derived per-spin electron density with energy-dependent
  reservoir occupations and separate source/drain injection bands.
- `selfconsistent` — the Poisson&harr;NEGF self-consistency loop.
- `sweep` — bias sweeps (`sweep_v_g`, `sweep_v_ds`) and subthreshold-swing
  extraction.

## Building and running

### Rust engine only

```bash
cargo build --locked -p negforge-core
cargo test --locked -p negforge-core
```

### Python package + notebook

Requires Rust/Cargo **1.85 or newer** and **CPython 3.14** (the primary build and validation
target). CPython 3.9–3.13 remain supported; substitute the interpreter below
when using an older version. PyO3 0.30.1 supports 3.14 natively: do not set
`PYO3_USE_ABI3_FORWARD_COMPATIBILITY` or use an older interpreter as a bypass.

```bash
python3.14 -m venv .venv314 && source .venv314/bin/activate
python -m pip install --upgrade pip
python -m pip install "maturin>=1.9.4,<2.0"
maturin build --release --locked --interpreter "$(command -v python)" --out target/wheels/cp314
python -m pip install target/wheels/cp314/negforge-*.whl
python -m pip install "negforge[notebook]==0.1.0" pytest
jupyter notebook notebooks/negforge_demo.ipynb
```

The wheel contains `crates/negforge-py` as the native module
`negforge._negforge`; `python/negforge/__init__.py` wraps it in a
friendlier, numpy-returning API (`negforge.Device`, `negforge.IVCurve`).
For iterative development, `maturin develop --release --locked` installs an
editable extension instead. Rebuild after changing Rust sources. Always run
Jupyter from the activated environment so its kernel uses the same interpreter.

## Physics model

Source-channel-drain chain, 1D, with source/drain treated as long, heavily
screened contact regions:

- **Electrostatics**: not a full 2D/3D Poisson solve, but a 1D "natural
  length" (`lambda`) approximation common in compact models for
  thin-body/double-gate MOSFETs. `lambda = sqrt(k_si/k_ox * d_ch * d_ox /
  geo)`. The electrostatic operator is tridiagonal with a reflective
  (Neumann-like) boundary condition at both ends, solved via O(N) elimination
  with adjacent partial pivoting. Charge inputs `rho` and `n_dot`
  are C/m³; their SI Poisson source is multiplied by `1e-18` for the nm grid.
  By default SCF uses per-spin density over an implicit **1 m² area**, making
  feedback extremely weak. Rust callers can set `effective_area_m2` explicitly
  for spin-summed volume charge; no width is inferred from `d_ch`. See
  [`docs/ELECTROSTATIC_GEOMETRY.md`](docs/ELECTROSTATIC_GEOMETRY.md).
- **Coherent current (default)**: `calc_current` and the gate/drain bias
  sweeps use the spin-degenerate, single-mode Landauer integral with Caroli
  transmission, `T(E) = Gamma_S(E) Gamma_D(E) |G^r_1N(E)|²`. With energies
  integrated in **eV**, `I = (2e²/h) integral T(E) [f_S(E)-f_D(E)] dE`
  returns **A**, not current per width. The reservoirs use
  `e_fd = e_fs - v_ds`. The potential profile determines tunneling,
  reflection, and coherent resonances; it is not reduced to its maximum.
- **Thermionic comparator**: `calc_thermionic_current` instead assumes
  `T(E) = step(E-max(psi_f))`. Its analytic integral to infinity is
  `I = (2e²/h) [F(e_fs)-F(e_fd)]`, with
  `F(mu) = kT_eV * log(1+exp((mu-max(psi_f))/kT_eV))` evaluated stably.
  At zero temperature, `F(mu) = max(mu-max(psi_f), 0)`. Only this comparator
  is independent of energy-grid truncation at a fixed potential.
- **NEGF**: a tight-binding discretization (on-site energy `2t + Psi_f(x)`,
  hopping `-t`) with open-boundary self-energies at the two contacts,
  giving the retarded Green's function, local density of states, and
  the contact-injected electron density.

Units throughout: lengths in nm, energies and
potentials in eV, temperature in Kelvin.

### Validity of the model — read before trusting a number

Each approximation below is documented in more detail (with the reasoning
and, where relevant, a test demonstrating it) in the corresponding source
module's doc comments.

- **Natural-length electrostatics** (`device.rs`) assumes a thin body with a
  parabolic transverse potential profile. Reasonable for thin-body/
  double-gate/gate-all-around MOSFETs; not valid for bulk (thick-body)
  devices. It is also a classical (non-quantized) transverse charge model,
  in tension with the fully-quantized longitudinal NEGF treatment — this is
  a first-order, not a fully self-consistent 2D, model.
- **Coherent Landauer current** (`calc_current`) includes elastic quantum
  reflection, sub-barrier tunneling, and interference, but no phonon,
  impurity, or other dissipative scattering and no dephasing. It assumes
  ideal equilibrium reservoirs and a static, single-particle potential.
  The separate `calc_thermionic_current` comparator discards tunneling and
  reflection above the barrier; it is not the default sweep current.
- **LDOS is not transmitted flux**: standing waves and resonances in a
  scattering-state LDOS plot show locally available states, not a current
  density or a transmission probability. Caroli transmission also requires
  contact broadenings and the end-to-end Green's function. A large LDOS
  resonance can be weakly coupled to a contact and carry little current.
- **Current convergence**: refine energy resolution and integration coverage
  to resolve lead band edges, Fermi edges, and narrow resonances; check the
  spatial grid and the small-broadening limit separately. SCF density
  broadening is not a physical scattering model or evidence of converged
  two-terminal current. Fixed-point convergence alone does not establish
  current convergence.
- **Effective-mass, single-valley, parabolic band** (`m_eff`, used for both
  the ballistic hopping parameter and the NEGF tight-binding parameter):
  real silicon has multiple equivalent valleys and a non-parabolic band
  away from the edge; `m_eff` is a single fitting parameter standing in for
  both, so absolute currents should be read as illustrative trends rather
  than device-accurate predictions.
- **Tight-binding NEGF discretization** (`negf.rs`) reproduces the continuum
  parabolic dispersion only for grid spacings fine enough that the swept
  energy range stays well below the chain's `4*t_hop` bandwidth; worth
  checking via a grid-refinement test at new device scales.
- **Retarded leads** use the causal quadratic self-energy root inside the
  band and the decaying real root outside **both** band edges. Evanescent
  leads shift the Hamiltonian but inject no charge. Returned wavevectors
  use a reflected-acos convention; they do not select the root.
- **Per-spin density:** `electron_density` returns single-orbital, per-spin
  linear density in nm⁻¹. Default unit-area SCF does not double it; explicit-area
  SCF does. The returned density itself is never spin-summed.
- **Finite eta is absorptive:** it is added only to the finite device,
  without matching injection. The SCF default `eta=0.08` eV smooths
  resonances but suppresses occupation and changes the fixed point.
  Convergence in eta, energy step, and energy window must be tested
  separately. Contact-only injection does not populate isolated bound states.
- **SCF convergence** tests the undamped fixed-point potential residual,
  before charge mixing. Reducing `mixing` cannot alone satisfy tolerance.
  With the 1 m² area, fast convergence is not evidence of substantial screening.

The test suite checks selected qualitative trends and numerical invariants,
not quantitative transistor accuracy. In particular, weak-feedback SCF
convergence and non-negative density do not validate the charge normalization
or the finite-eta limit. Treat results as educational, not device-accurate
predictions.


## Testing

- `crates/negforge-core/src/*.rs` — unit tests per module, including
  cross-checks of the O(N) tridiagonal/recursive-Green's-function solvers
  against dense (Gaussian-elimination / full-matrix-inversion) reference
  implementations on small systems.
- `crates/negforge-core/tests/self_consistent_realistic_device.rs` — an
  integration test at the model's default (non-toy) device scale,
  confirming the self-consistent loop converges and reproduces the
  expected ballistic-MOSFET trend (current increasing with gate bias).
- `crates/negforge-core/tests/physical_regressions.rs` — SI/nm source and
  electron-energy sign, absolute Landauer units and reservoir reference,
  undamped residual, returned-iterate consistency, and input/range validation.
- Unit regressions cover analytic per-spin chain occupation, evanescent
  lead shifts, near-pole diagonal stability, and zero/tiny-pivot solvers.
- `tests/test_python.py` — installed Python API and validation regressions.
- `tests/test_notebook.py` — fresh-kernel frontend regression using the active
  Python interpreter. Executes all six cells and requires exactly five decoded,
  nonblank `image/png` display outputs (not merely saved gallery artifacts).
  Also checks that importing the gallery preserves the caller's backend.
  Requires the optional notebook dependencies; Rust tests alone do not exercise
  the extension or notebook.

```bash
cargo test --locked -p negforge-core
cargo test --locked --workspace
# In the activated environment, after building/installing the extension:
python -m pytest -q tests
# Focused frontend check; prints the five-image count:
python -m pytest -q -s tests/test_notebook.py
# Optional executed copy (nbconvert alone does not assert rich image outputs):
jupyter nbconvert --to notebook --execute notebooks/negforge_demo.ipynb \
  --output negforge_demo.executed.ipynb --output-dir "$TMPDIR"
```

Notebook execution needs the optional notebook dependencies and its Python
kernel must use the environment containing the compiled extension. Keep
execution outputs out of the source notebook.
