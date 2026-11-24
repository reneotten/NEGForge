# NEGForge — Physics and Numerical Methods Tutorial

This document is a tutorial walkthrough of both the *physics* and the
*numerical methods* behind NEGForge: a 1D ballistic-MOSFET electrostatics +
NEGF (Non-Equilibrium Green's Function) transport simulator. It is written
to stand on its own — you should not need the reference textbooks open to
follow it — but it points to where each idea is developed in full so you
can go deeper, and it points at the exact source file and function that
implements each piece, so you can move between "why" and "how" freely.

**How to read this document.** Parts I–V follow the model's own data flow,
each one bottom-to-top: the physics problem being solved, the equations
that solve it, the discretization that turns those equations into linear
algebra, and the algorithm that solves that linear algebra efficiently.
Part VI collects validity/limitations in one place. Part VII explains the
testing strategy. If you only want the destination, jump to Part VIII's
glossary and the code-to-section index below; if you want the journey,
read start to finish.

| Code | What it is | Covered in |
|---|---|---|
| `crates/negforge-core/src/tridiag.rs` | Adjacent-partial-pivoting tridiagonal solvers | Part I.4 |
| `crates/negforge-core/src/device.rs` | Electrostatics (`calc_potential`), coherent current (`calc_current`), thermionic comparator (`calc_thermionic_current`) | Parts I, II |
| `crates/negforge-core/src/negf.rs` | Tight-binding Hamiltonian, contact self-energies, retarded Green's function | Part III |
| `crates/negforge-core/src/charge.rs` | NEGF-derived electron density | Part IV |
| `crates/negforge-core/src/selfconsistent.rs` | Poisson↔NEGF fixed-point loop | Part V |

Primary references used throughout:

- S. Datta, *Quantum Transport: Atom to Transistor*, Cambridge University
  Press, 2005 — Chapter 2 ("A Simple Solver") for the electrostatic model;
  Chapters 8–10 for the NEGF transport formalism.
- S. Datta, *Electronic Transport in Mesoscopic Systems*, Cambridge
  University Press, 1995 — background on the Landauer/NEGF picture of
  transport.
- M. Lundstrom & J. Guo, *Nanoscale Transistors: Device Physics, Modeling
  and Simulation*, Springer, 2006 — short-channel electrostatics, the
  "natural length" theory, and the top-of-the-barrier ballistic MOSFET
  model.
- Y. Taur & T. H. Ning, *Fundamentals of Modern VLSI Devices*, Cambridge
  University Press — the scale-length / natural-length model for
  double-gate and thin-body MOSFETs.
- A. Svizhenko, M. P. Anantram, T. R. Govindan, B. Biegel, R. Venugopal,
  "Two-dimensional quantum mechanical modeling of nanotransistors,"
  *J. Appl. Phys.* **91**, 2343 (2002) — the recursive Green's function
  (RGF) algorithm used for the O(N) NEGF sweep.
- R. Lake, G. Klimeck, R. C. Bowen, D. Jovanovic, "Single and multiband
  modeling of quantum electron transport through layered semiconductor
  devices," *J. Appl. Phys.* **81**, 7845 (1997) — another standard RGF
  reference, and the contact self-energy / broadening-function formalism.

---

# Part I — Electrostatics: from 3-D Poisson's equation to a 1-D solve

## I.1 What problem is being solved?

The device is a gate-controlled transistor (thin-body / double-gate /
gate-all-around MOSFET) split into three regions along the transport
direction `x`:

```
 |<------ source ------>|<------ channel ------>|<------ drain ------>|
 x = 0                 x = d_S            x = d_S + d_ch             x = N·a
        (grounded, V=0)      (under the gate)          (biased at V_ds)
```

The gate sits over the channel only (self-aligned gate assumption). We want
the electrostatic potential energy `φ(x)` along the device, because it is
what opens or closes the channel to current, and — the point of this whole
codebase — it becomes the on-site potential energy in the NEGF Hamiltonian
(Part III), and is in turn corrected by the charge density NEGF computes
(Part V). This chapter covers the linear reduced Poisson solve at fixed
charge density, the electrostatic ingredient of the coupled model.

## I.2 From 3-D electrostatics to a 1-D ODE: the natural length λ

**Continuum derivation (SI voltage convention):** in this subsection only,
`phi` denotes electrostatic voltage in V and lengths are in m. At §I.3 we
switch to the code's electron potential energy in eV and nm grid, explicitly
changing the source sign and scale. The exact problem is 2-D (or 3-D)
Poisson's equation inside the semiconductor body, with boundary conditions
set by capacitive coupling to the gate through the oxide:

$$\nabla^2 \phi(x,y) = -\frac{\rho(x,y)}{\varepsilon_0 \varepsilon_{si}}$$

with a Robin-type boundary condition at the gate/oxide/silicon interface
(continuity of displacement field across the oxide, no free charge in the
oxide):

$$\varepsilon_{ox}\frac{\phi_g - \phi(x, y_{surface})}{t_{ox}} = \varepsilon_{si}\left.\frac{\partial \phi}{\partial y}\right|_{surface}$$

Solving the full 2-D problem numerically at every bias point is expensive.
Instead, following Taur & Ning and Datta's *Quantum Transport* §2.4, we
assume the potential is *parabolic in the transverse direction* `y`
(reasonable for a thin body between one or two gates) and satisfies the
interface boundary condition above. Substituting that ansatz into Poisson's
equation and integrating out `y` collapses the 2-D problem into a single
1-D ODE for the **centerline potential** `φ(x)`:

$$\frac{d^2\phi}{dx^2} - \frac{\phi(x)}{\lambda^2} = -\frac{\rho(x)}{\varepsilon_0\varepsilon_{si}} - \frac{\phi_g(x) + \phi_{bi}(x)}{\lambda^2}$$

where

$$\lambda = \sqrt{\frac{\varepsilon_{si}}{\varepsilon_{ox}}\cdot\frac{t_{ch}\, t_{ox}}{g}}$$

is the **natural length** (a.k.a. scale length), `t_ch` the body/channel
thickness, `t_ox` the oxide thickness, and `g` a geometry factor (1 for a
single gate, 2 for a symmetric double gate, etc. — the code exposes this as
`geo`).

**Physical reading of the ODE.** Far from the source/drain junctions (many
`λ` away), the `d²φ/dx²` term is negligible and `φ → φ_g + φ_bi`: the gate
fully sets the potential, exactly as in a long-channel MOSFET. Near a
junction, the fixed potential imposed by the heavily doped source/drain
"leaks" into the channel over a decay length `λ`. If the channel length `L`
is only a few `λ`, the source and drain potentials overlap in the middle of
the channel and the gate partially loses control — this *is* the
microscopic origin of short-channel effects: threshold voltage roll-off and
drain-induced barrier lowering (DIBL). `λ` is therefore the single number
that tells you, for a given `t_ch`/`t_ox`/gate geometry, how short a
channel can be made before electrostatic control degrades — which is why
the source/drain extension regions are sized as multiples of `λ`
(`l_ds = floor(lambda)*15` for automatically sized contacts in
`device.rs`), so boundaries lie far from the channel.

**Validity.** This is a thin-body approximation: it assumes the transverse
potential profile is well-described by a single parabola, which holds when
the body is thin compared to the channel length, and degrades for bulk
(thick-body) MOSFETs where the transverse charge distribution isn't
captured by one decay length. It is also a *classical*, non-quantized
charge model in the transverse direction, in tension with the fully
quantum-mechanical longitudinal treatment NEGF gives the same device in
Part III — the overall model is a first-order approximation, not a fully
self-consistent multi-dimensional quantum simulation.

## I.3 Discretization

Discretize `x` on a uniform grid with spacing `a` and replace the second
derivative with the standard 3-point finite-difference stencil:

$$\left.\frac{d^2\phi}{dx^2}\right|_i \approx \frac{\phi_{i+1} - 2\phi_i + \phi_{i-1}}{a^2}$$

which turns the ODE into a linear system `A φ = b` for the vector of
grid-point potentials, where `A` is tridiagonal (`-2/a²` on the diagonal,
`1/a²` on the off-diagonals, from the Laplacian, with `-1/λ²` added to
every diagonal entry — `device::Device::laplacian` builds exactly this),
and

$$b_i = \frac{10^{-18}(\rho_i+N_{dot})}{\varepsilon_0\varepsilon_{si}} - \frac{\phi_{g,i} + \phi_{bi,i}}{\lambda^2}$$

Here `phi` denotes electron potential energy in eV, not voltage. The
continuum voltage Poisson equation above is a derivation in SI units;
for electron energy `U/e=-V` its charge-source sign reverses to positive.
The factor `1e-18` converts the SI source (V/m²) to the nm⁻² operator's
units. A negative electronic charge therefore raises the electron-energy
barrier. `rho` and `N_dot` are both C/m³.

(`device::Device::calc_potential` assembles this right-hand side and calls
the tridiagonal solver).

**Boundary conditions.** The far ends of the source and drain extensions
get **Neumann (zero-field) boundaries**, `dφ/dx = 0` — physically, "the
potential has flattened out by the time you're this far from the channel,
so there's no field left." This is implemented with the classic
ghost-point trick: reflecting the mirror point `φ_{-1} = φ_1` across the
boundary turns the stencil at the edge from `(φ_2 - 2φ_1 + φ_0)` into
`(2φ_2 - 2φ_1)`, i.e. doubling the corner off-diagonal entry of `A`
(`sup[0] = sub[n-2] = 2/a²` in `Device::laplacian`).

## I.4 Tutorial: Thomas elimination and the implemented pivoted solver

**Educational derivation:** the formulas below describe *unpivoted*
Thomas elimination, which requires nonzero, adequately sized reduced
pivots. The production `solve_real` and `solve_complex` instead use
**adjacent partial pivoting**: compare the diagonal pivot with the
subdiagonal and interchange adjacent rows when needed. This introduces
at most a second superdiagonal, retaining O(N) storage and work while
handling nonsingular systems with zero or tiny leading pivots. Empty
and scalar systems are supported; dimension mismatches and exactly
singular systems panic. These APIs do not estimate conditioning.

`A φ = b` is an `N×N` system, but `A` is tridiagonal (only 3 nonzero
diagonals). Solving it with general-purpose Gaussian elimination costs
`O(N^3)`; because a tridiagonal matrix only has `O(N)` nonzero entries to
begin with, a specialized elimination — the **Thomas algorithm** — solves
it in `O(N)`. This motivates `tridiag::solve_real` and its complex
counterpart `solve_complex`, reused in Part III; their implemented
pivoting extension is described above.

Write the system as, for each row `i`:

$$\text{sub}_{i-1}\, x_{i-1} + \text{diag}_i\, x_i + \text{sup}_i\, x_{i+1} = \text{rhs}_i$$

**Forward elimination.** Starting from row 0, eliminate `x_{i-1}` from row
`i` using row `i-1`'s (already-reduced) equation. Tracking only the
modified coefficients (since the matrix stays tridiagonal, elimination
never creates new nonzero entries — the key property that makes this
`O(N)` instead of `O(N^3)`):

$$c'_0 = \frac{\text{sup}_0}{\text{diag}_0}, \qquad d'_0 = \frac{\text{rhs}_0}{\text{diag}_0}$$

$$c'_i = \frac{\text{sup}_i}{\text{diag}_i - \text{sub}_{i-1} c'_{i-1}}, \qquad d'_i = \frac{\text{rhs}_i - \text{sub}_{i-1} d'_{i-1}}{\text{diag}_i - \text{sub}_{i-1} c'_{i-1}}$$

After this forward pass, row `i` reads `x_i + c'_i x_{i+1} = d'_i` — each
row now involves only `x_i` and `x_{i+1}`.

**Back substitution.** Starting from the last row (which has no `x_{i+1}`
term, so `x_{N-1} = d'_{N-1}` directly), walk backward:

$$x_{N-1} = d'_{N-1}, \qquad x_i = d'_i - c'_i\, x_{i+1}$$

Both passes are a single `O(N)` loop, so the whole solve is `O(N)` — this
is what makes it practical to call `calc_potential` once per
self-consistent iteration (Part V) without the electrostatics solve itself
becoming the bottleneck. The production complex solver applies the
pivoted variant with complex arithmetic to the NEGF boundary columns.

**Validated by:** `tridiag::tests::thomas_matches_dense_gaussian_elimination`
cross-checks this against a dense (`O(N^3)`) Gaussian-elimination reference
on a 12-site system. Separate pivoting regressions exercise zero/tiny
leading pivots, complex residuals, and empty/scalar cases.

## I.5 Worked example

`device::DeviceParams::default()` uses `d_ch=5 nm`, `d_ox=5 nm`,
`k_si=11.2`, `k_ox=3.9`, and `geo=1`:

$$\lambda=\sqrt{(11.2/3.9)\,5\,5}\ \mathrm{nm}. $$

Reducing body or oxide thickness, or increasing the geometry factor,
shortens the natural length and improves gate control at a fixed channel
length. The automatically sized contacts extend well beyond this scale.

**What the potential profile means.** With `V_g = 0` and `V_ds > 0` you
should see: a flat, low region in the source, a barrier peak under the
gate (height ≈ mid-gap `E_f + E_g/2`, attenuated somewhat near the
junctions by the `1/λ²` coupling to the drain), and a step down to
`-V_ds` in the drain.

- **Increase `V_g`** (more negative `phi_g`/`Psi_g` in the channel, the
  n-type convention used here) → the channel barrier drops → the
  transistor turns "on."
- **Increase `V_ds`** with the channel short relative to `λ` → the barrier
  peak itself drops (not just the drain-side potential) → this is DIBL,
  visible directly as a change in `max(phi_f)` while sweeping `V_ds` at
  fixed `V_g`.
- **Shrink `d_ch`/`l_ch`** relative to `λ` → the flat "gate is in control"
  plateau at the top of the barrier shrinks or disappears, replaced by a
  rounded peak — the discretized signature of losing electrostatic
  control.

---

# Part II — Coherent current: the Landauer–Caroli formula

## II.1 Physics and default current

At a fixed potential profile `psi_f(x)`, `Device::calc_current` evaluates
coherent two-terminal transport through the tight-binding Hamiltonian of
Part III. The default is **not** a step at the maximum barrier. Elastic
reflection, sub-barrier tunneling, and quantum interference are included
through the energy-dependent **Caroli transmission**:

$$\mathcal T(E)=\operatorname{Tr}[\Gamma_S G^r\Gamma_D G^a]
=\Gamma_S(E)\Gamma_D(E)\,|G^r_{1N}(E)|^2.$$

The scalar expression applies to this single-orbital chain with contacts
attached at its endpoints. `G^a=(G^r)^\dagger` and
`Gamma_alpha=-2 Im(Sigma_alpha)`; the contact broadenings vanish outside
the respective lead's propagating band. Thus end-to-end transmission
requires propagating states in **both** leads. Evanescent states within
the device still mediate tunneling through a barrier.

The spin-degenerate Landauer current is

$$I=\frac{2e^2}{h}\int_{\text{energy in eV}}
\mathcal T(E)\,[f_S(E)-f_D(E)]\,dE,$$

where `f_alpha(E)=1/[exp((E-mu_alpha)/theta)+1]`,
`theta=k_B T/e` in eV, `mu_S=e_fs`, and `mu_D=e_fs-v_ds`.
At zero temperature these are Fermi steps, with half occupation at the
edge. Positive current corresponds to net source-to-drain electron
transport in this convention; reversing the reservoir imbalance reverses
the current. Equal reservoir occupations give zero current.

The gate and drain sweeps (`sweep_v_g`, `sweep_v_ds`) use `calc_current` at
each bias point. Their self-consistency option selects whether the
potential comes from an electrostatics-only solve or the coupled
Poisson–NEGF iteration; it does not select a thermionic current model.

**Assumptions.** This is a static, single-particle, coherent elastic model
with ideal equilibrium reservoirs. There is no phonon or impurity
scattering, dissipative energy relaxation, or conserving dephasing model.
Elastic scattering from the potential is included, so “ballistic” must
not be read as unit transmission. The current is for one orbital mode
with two spins, not a sum over transverse subbands or valleys.

## II.2 Analytic thermionic comparator

`Device::calc_thermionic_current` is a separate top-of-the-barrier
comparator. It replaces the actual transmission by
`T(E)=step(E-psi_0)`, where `psi_0=max(psi_f)`: no tunneling below the
barrier and perfect transmission above it. Let `theta=k_B T/e` in eV.
Then

$$F(\mu)=\theta\ln\!\left(1+e^{(\mu-\psi_0)/\theta}\right),
\qquad I_{\mathrm{thermionic}}=\frac{2e^2}{h}
[F(\mu_S)-F(\mu_D)].$$

The softplus is evaluated stably; at `T=0` its exact limit is
`F(mu)=max(mu-psi_0,0)`. This comparator integrates analytically to infinity
and is independent of `d_e` and the numerical energy window **at a fixed
potential**. Those controls can still affect an SCF-derived potential.
The comparator is not used as the default current in bias sweeps.

Differences from coherent current can reflect tunneling, above-barrier
reflection, resonances, and the finite lead bands of the discretized
Hamiltonian. The comparator is not a rigorous upper or lower bound on
coherent current.

## II.3 Current units and numerical convergence

The Landauer prefactor `2e/h` assumes the energy integral is in Joules.
With the numerical energy variable in eV, `dE_J=e*dE_eV` supplies another
factor of `e`: use **`2e²/h`**, giving current in **amperes**. Transmission
and Fermi occupations are dimensionless; the factor two counts spin.
There is no extra `1e-3`, grid-spacing, or device-area factor. This is a
single-mode current, not A/m or a current density.

The coherent integral must resolve contact band edges, the reservoir
occupation window (including thermal tails), and narrow transmission
resonances. Check energy resolution and window coverage independently,
then refine the spatial grid to check the effective-mass limit. Lead
band overlap limits the support of the transmission, but does not ensure
that a chosen quadrature resolves its structure. At low temperature,
resolve Fermi edges rather than relying on thermal smoothing.

Check sensitivity to any imaginary regularization and the small-broadening
limit. The finite-device eta used to smooth an SCF density is absorbing
without compensating injection (§III.4); it must not be interpreted as a
physical scattering rate or silently substituted for a converged coherent
flux calculation. An SCF residual below tolerance says nothing by itself
about current quadrature accuracy. Convergence should be assessed for
the observables of interest, including subthreshold current, not just
for a visually smooth LDOS or I–V curve.

## II.4 The SCF density energy window

For the NEGF grid, the stored cutoff in eV is
`e_max = max(e_fs,e_fd) - theta*ln(epsilon)`, covering the thermal tails
of both reservoirs also under reverse bias. The SCF integration interval is

```text
L = min(psi_f)
mu_high = max(e_fs, e_fd)
B = max(psi_f[0], psi_f[N-1]) + 4*t_hop
U = min(B, max(mu_high, L + green_energy_fraction*(e_max-L)))
```

All terms are energies in eV. A common shift of the potential, both
chemical potentials and `e_max` translates `[L,U]` without changing the
occupations or Green's functions. Applying `green_energy_fraction` to an
absolute energy instead of a span would violate this energy-reference
invariance. The lower endpoint includes the lead band bottoms; the cap
`B` removes energies where neither lead injects particles. It does not
assign occupation to bound states outside the lead continua.

The default fraction `0.7` cannot cut below the higher Fermi level while
that level lies within the injecting bands, but may truncate thermally
occupied states above it. Only fraction `1.0` uses the full epsilon-based
thermal cutoff, and finite `epsilon` controls a Fermi-probability tail,
not a rigorous density-error bound near resonances. Compare fractions,
refine `d_e`, and reduce eta separately before trusting the density.
The finite lattice bandwidth is a property of the discretized model;
covering it does not establish the continuum effective-mass limit.

---

# Part III — NEGF: quantum transport via the retarded Green's function

The default coherent current in Part II and the contact-injected charge
in Part IV use the same tight-binding, open-contact NEGF framework.
The retarded Green's function describes propagation and spectral
structure; reservoir occupations then determine net current and charge.
The analytic thermionic comparator alone discards interference and
resonance structure.

## III.1 From the Schrödinger equation to a tight-binding chain

Start from the effective-mass, 1-D time-independent Schrödinger equation
with electron potential energy `U(x)` as the potential term. For an
electrostatic voltage `V(x)`, `U_J=-e V`; the code already stores
`psi_f=U_J/e` in eV, so it must not be multiplied by `-e` again:

$$-\frac{\hbar^2}{2m^*}\frac{d^2\psi}{dx^2} + U(x)\,\psi = E\,\psi$$

Discretize on the same grid as the electrostatics (spacing `a`), replacing
the second derivative with the same 3-point stencil as §I.3. Collecting
terms gives a matrix eigenvalue-like equation `H\psi = E\psi` with a
**tight-binding chain** Hamiltonian:

$$H_{ii} = 2t + U(x_i), \qquad H_{i,i\pm1} = -t, \qquad t = \frac{\hbar^2}{2 m^* a^2}$$

(`Device::new` computes `t` into the `t_hop` field;
`negf::green_function_sweep`'s `bulk_diag = 2*t + psi_f` and
`off_diag = t` build exactly this — note the off-diagonal of the *matrix
being inverted*, `(E+iη)I - H`, is `-H_{i,i±1} = +t`, positive, which is
what the code stores).

**Validity: how fine does `a` need to be?** An infinite tight-binding chain
with this `H` has dispersion `E(k) = 2t(1-\cos(ka))` — bounded above by a
bandwidth of `4t`, unlike the true continuum's unbounded parabola
`E = ħ²k²/2m*`. The two agree only for small `k·a` (Taylor-expand
`\cos(ka) ≈ 1 - (ka)^2/2` to see `E(k) ≈ t(ka)^2 = ħ²k²/2m*`, recovering
the continuum dispersion). So `a` needs to be fine enough that the energy
range actually swept (`psi_0` to `e_max`, roughly) stays well below `4t` —
worth checking (e.g. halve `a` and confirm results are unchanged) before
trusting NEGF results at a new device scale.

## III.2 Open boundaries: contact self-energies

A real device isn't an isolated chain — the source and drain are
semi-infinite reservoirs. Modeling that directly (an infinite matrix) isn't
possible, but its *effect* on the finite simulated region can be captured
exactly by an energy-dependent correction to the boundary sites' on-site
energy, called a **self-energy** `Σ(E)`. This is the standard NEGF
open-boundary technique (Datta, *Quantum Transport*, §8.2–8.3).

For a semi-infinite 1-D tight-binding chain (same `t`, on-site energy
`2t + U_{contact}`) attached at a boundary site, the self-energy has a
closed form in terms of the chain's own dispersion. Solving
`E = 2t(1-\cos(ka)) + U_{contact}` for `k` gives

$$\cos(k a) = \frac{2t + U_{contact} - E}{2t}$$

taking the branch `k·a \in [0,\pi]` for energies inside the contact's
propagating band, `U_{contact} \le E \le U_{contact} + 4t`. The self-energy
of the semi-infinite lead, evaluated at the boundary site, is then

$$\Sigma(E) = -t\,e^{i k(E) a}$$

**Causality requirement.** A contact that only lets particles escape into
the lead (absorbing, not amplifying) must have `Im(Σ) ≤ 0` — equivalently,
the **broadening function**

$$\Gamma(E) = -2\,\mathrm{Im}\big[\Sigma(E)\big] \ge 0$$

must be non-negative. This is not a sign convention you get to pick freely:
it is required for the resulting Green's function to actually be the
*retarded* one (poles in the lower half of the complex-`E` plane), and for
physical quantities built from it (density of states, injected charge) to
come out non-negative. Concretely: for a propagating mode
(`0 < k(E)a < \pi`), `\mathrm{Im}[-t e^{ika}] = -t\sin(ka) \le 0` — good,
this checks out for the formula above.

**Implemented root selection.** Write `epsilon_c=2t+U_contact` and
`x=(E-epsilon_c)/(2t)`. The surface self-energy obeys
`Sigma²-(E-epsilon_c)Sigma+t²=0`. The code uses

$$\Sigma(E)=\begin{cases}
tx-it\sqrt{(1-x)(1+x)}, & |x|\le1,\\
\displaystyle\frac{t}{x+\operatorname{sgn}(x)\sqrt{|x|-1}\sqrt{|x|+1}}, & |x|>1.
\end{cases}$$

Inside the band this is the causal root; outside **both** band edges it
is the decaying real root (`|Sigma|<=t`). The rationalized expression
avoids cancellation far outside the band. An evanescent lead contributes
a real Hamiltonian shift even though its injection `Gamma` is zero.
Both self-energies are subtracted from the boundary diagonal, including
both contributions if a low-level one-site calculation is requested.

**Educational branch identity, not the production root algorithm:**
returned `ka_s = acos(-(2*t-E+psi_0s)/(2*t))` uses a reflected
wavevector. Inside the band `ka_s=pi-k(E)a`, hence
`t exp(-i ka_s)=-t exp(i k(E)a)`. This identity is not used to select
the production self-energy.
Outside the band the returned complex wavevector is not a canonical
physical wavevector. Uniform-chain and one-site regressions test the
actual assembled operator, in addition to causality and LDOS positivity.

## III.3 The retarded Green's function, LDOS, and broadening

With the Hamiltonian and self-energies assembled, the central object is the
**retarded Green's function**

$$G^r(E) = \big[(E+i\eta)I - H - \Sigma_S(E) - \Sigma_D(E)\big]^{-1}$$

where `η` is a small positive broadening (§III.4) and `Σ_S`, `Σ_D` are
nonzero only at the source/drain boundary sites. Two derived quantities
describe spectral structure and contact coupling:

- **Local density of states**, `D(x,E) = -\mathrm{Im}[G^r_{xx}(E)]/(\pi)`
  (per unit length, hence the extra `/a` the code applies —
  `GreenFunctionResult::g_diag` stores `-\mathrm{Im}(G_{ii})/a`, i.e. the
  LDOS without the `1/\pi` normalization).
- **Broadening / injection**, `\Gamma_{S,D}(E) = -2\,\mathrm{Im}[\Sigma_{S,D}(E)]`,
  which for our contact self-energy (§III.2) works out to
  `\Gamma_{S,D}(E) = 2t\sin(k_{S,D}(E)\,a)` for propagating contact modes —
  this is what drives the charge-density calculation in Part IV.

### Scattering-state LDOS is not transmitted flux

The open-contact Green's function contains scattering states: an incident
wave can interfere with its reflected wave, producing standing-wave LDOS
patterns and resonant spectral peaks. Here “scattering” means elastic
scattering by the device potential, not added phonon or impurity
self-energies. The physical per-spin LDOS is `g_diag/pi`, in
`1/(eV nm)`; raw `g_diag` plots omit this factor unless explicitly normalized.

LDOS measures locally available states, not their occupation or direction
of travel. Even at equilibrium there can be a large LDOS and zero net
current. Contact-resolved spectral weights `Gamma_alpha*|G_i,alpha|²`
enter the density with occupations (Part IV), whereas transmitted flux
uses the end-to-end Caroli expression and the occupation **difference**
(Part II). A high LDOS resonance need not transmit efficiently if its
coupling to one contact is weak. Do not interpret LDOS color intensity
as transmission probability or current density.

## III.4 The artificial broadening `η`

`η` in `(E+iη)I - H - \Sigma_S - \Sigma_D` is a mathematical device, not a
physical scattering rate: it nudges the poles of `G^r` off the real axis so
the matrix stays invertible, in the same spirit as the `+i0^+`
prescription in the Feynman propagator. `negf::DEFAULT_ETA = 1e-8` is
appropriate for a
one-off local-density-of-states plot at a fixed potential, where you *want*
to see sharp, well-resolved resonances.

A narrow resonance can be missed or spuriously overweighted by a coarse
energy grid. `SelfConsistentOptions::eta` defaults to `0.08` eV to smooth
such resonances. This is **absorbing regularization, not conserving
dephasing**: eta is applied only to the finite device, while lead
self-energies use real energy with the retarded `0+` prescription. There
is no matching lesser/in-scattering self-energy for eta. Consequently
finite eta removes occupation, changes the evaluated density, and can
change the SCF fixed point; it is not merely a convergence accelerator.

A successful iteration at default eta does not establish the small-eta
physical limit. Refine eta, `d_e`, and the energy window separately,
resolving resonances before drawing quantitative conclusions. Contact-only
injection also supplies no occupation prescription for true bound states
disconnected from both lead continua. No universal minimum eta establishes
physical convergence.

## III.5 Tutorial: the recursive Green's function (RGF) algorithm

Directly forming and inverting the `N×N` matrix `(E+iη)I - H - \Sigma_S -
\Sigma_D` with dense Gaussian elimination costs `O(N^3)` *per energy point*, which is the dominant cost of a
bias sweep or a self-consistent loop that calls this once per iteration.
But only two things are actually needed from the full inverse: its
diagonal (`g_diag`, the LDOS) and its first/last columns (the source/drain
-injected quantities `g_col_source`/`g_col_drain` that Part IV needs). Both
can be obtained in `O(N)` by exploiting the matrix's tridiagonal structure
— the standard **recursive Green's function** (RGF) technique used
throughout the mesoscopic-transport literature (Svizhenko et al. 2002;
Lake et al. 1997).

**The boundary columns** (`negf::green_function_sweep`'s `col_source`,
`col_drain`) are the easy half: the `i`-th column of `M^{-1}` is, by
definition, the solution `x` of `M x = e_i` for the `i`-th standard basis
vector. Since `M` is tridiagonal, this is exactly `tridiag::solve_complex`
from §I.4 — one `O(N)` adjacent-partial-pivoting solve per column, no
separate inverse-column derivation needed.

**The full diagonal** (`negf::full_diagonal`) needs a genuinely different
technique, since there's no single right-hand side that extracts "the
diagonal" from a linear solve. The implementation is two sweeps:

*Forward sweep* — build a **left-connected** Green's function `g_left[i]`,
defined as the (single-site) Green's function of the truncated chain
consisting of sites `{0,...,i}` only (i.e. as if site `i+1` didn't exist).
This is exactly the same recursion as the Thomas algorithm's forward
elimination (§I.4) applied to this matrix, viewed as a continued fraction —
each new site's effective on-site energy is renormalized by its coupling
to the (already-resolved) chain to its left:

$$g_{\text{left}}[0] = \frac{1}{\text{diag}_0}, \qquad
g_{\text{left}}[i] = \frac{1}{\text{diag}_i - t_{i-1}^2\, g_{\text{left}}[i-1]}$$

*Right-connected sweep* — independently resolve the chain to the right:

$$g_R[N-1]=1/d_{N-1},\qquad g_R[i]=\frac{1}{d_i-t_i^2g_R[i+1]}.$$

The implemented **full diagonal** combines both embeddings:

$$G_{ii}=\frac{1}{d_i-t_{i-1}^2g_L[i-1]-t_i^2g_R[i+1]},$$

omitting the nonexistent left/right term at the endpoints. Here `d_i`
is the diagonal of the matrix being inverted. Both sweeps and the final
combination are O(N).

**Educational identity (not used in production):** the equivalent
backward reconstruction `full[i]=g_L[i]+g_L[i]^2*t_i^2*full[i+1]`, seeded
by `full[N-1]=g_L[N-1]`, is exact algebraically but can catastrophically
cancel near a pole of a truncated chain. The two-connected-sweep
implementation avoids that reconstruction; the near-connected-pole
regression checks this stability improvement against a dense inverse.

**Validated by:** `negf::tests::recursive_diagonal_matches_dense_inverse`
and `negf::tests::boundary_columns_match_dense_inverse` cross-check both
pieces above against a dense (`O(N^3)`) Gauss-Jordan matrix inversion on a
9-site deterministic complex tridiagonal system, to within `1e-8` absolute
tolerance — the appropriate way to validate a recursive numerical identity
like this one, and the reason this document states the recursion rather
than re-deriving the (slightly intricate) exact equivalence between the
above and a direct Cramer's-rule expansion from scratch.

## III.6 Code walkthrough: `negf::green_function_sweep`

For each energy `E` in the caller-supplied grid:

1. Compute the contact wavevectors `ka_s`, `ka_d` (§III.2) — complex in
   general, real only inside each contact's propagating band.
2. Build the bulk diagonal `(E+iη) - (2t + \psi_f(x_i))` for every site.
3. Subtract both contact self-energies (§III.2) at all energies, retaining
   the real evanescent shifts below and above the propagating bands.
4. Get the full diagonal via the RGF sweep (§III.5) → `g_diag`.
5. Solve for the source/drain-injected columns via two tridiagonal solves
   (§III.5) → `g_col_source`, `g_col_drain` (stored as `|G|^2`, the squared
   magnitudes Part IV's charge-injection formula needs).

The energy grid is caller-selected. The SCF loop uses the interval in
§II.4 and equal-width midpoint cells with widths no larger than `d_e`.
Each sample lies at a cell center; the density sum weights it by the
actual cell width. This covers the complete selected interval without
sampling its lower endpoint or discarding a final fractional cell.
Band edges inside the interval, Fermi steps at zero temperature, and
narrow resonances still require grid-refinement checks. An LDOS plot can
use a different caller-selected grid, but is not automatically a density
quadrature.

---

# Part IV — From Green's functions to charge density

## IV.1 Physics

The **electron density** at position `x` is built from the NEGF
positive correlation function `G^n=-iG^<`, related to the retarded
Green's function and the contacts' occupation via the standard NEGF result

$$G^n(E) = G^r(E)\,\big[\Sigma_S^{in}(E) + \Sigma_D^{in}(E)\big]\,G^a(E), \qquad \Sigma_{S,D}^{in}(E) = \Gamma_{S,D}(E)\, f_{S,D}(E)$$

("in-scattering" from each contact, weighted by that contact's occupation
`f_{S,D}`, §II.1). For our boundary-only self-energies this reduces to
`G^n_{xx}(E) = \Gamma_S(E) f_S(E) |G^r_{x,1}(E)|^2 + \Gamma_D(E) f_D(E)
|G^r_{x,N}(E)|^2` — exactly the two `|G|^2` boundary columns Part III
computed — and the electron density is

$$n_i = \frac{1}{a}\int \frac{dE}{2\pi}\Big[\Gamma_S(E) f_S(E)\, |G^r_{i,1}(E)|^2 + \Gamma_D(E) f_D(E)\, |G^r_{i,N}(E)|^2\Big].$$

The `1/a` converts site occupation to per-spin linear density in nm⁻¹.
The standard lesser Green's function is `G^<=iG^n`; using the positive
correlation avoids confusing an imaginary-valued lesser function with
occupation.

**Why this is non-negative by construction.** Every factor —
`Γ_{S,D} ≥ 0` (§III.2's causality requirement), `f_{S,D} \in (0,1)`
(Fermi-Dirac), `|G|^2 \ge 0` — is non-negative, so `n(x) \ge 0` always
holds, *regardless* of any Green's-function sign convention question. This
is the one positivity guarantee the self-consistent loop (Part V) actually
leans on.

## IV.2 Code walkthrough: `charge::electron_density`

`charge::electron_density` implements the integral above as a uniform
weighted sum over the energy grid Part III swept. Its `d_e` argument is
the **quadrature weight in eV**, not necessarily the device's requested
maximum grid step. For the SCF midpoint grid it is the actual cell width.
A general nonuniform energy grid needs a different quadrature rather than
passing a single width to this API. Two normalization details matter:

- `gamma_s = t_hop * sin(k)` is *half* of `Γ_S = 2t\sin(k)` from §III.3;
  combined with the function's `1/pi` prefactor (rather than the `1/2π`
  the formula above has), the two halvings cancel exactly:
  `t\sin(k)/\pi = (2t\sin(k))/(2\pi)`, reproducing the textbook formula's
  magnitude exactly, just factored differently.
- The result is explicitly **per spin, single orbital**, in nm⁻¹; a Fermi
  probability cannot encode spin degeneracy. Multiply by two downstream
  for spin-summed density. The current includes spin degeneracy but the
  default unit-area SCF charge conversion does not; explicit-area SCF does.
  The analytic half-filled uniform
  chain regression fixes this normalization at `n=1/(2a)` per spin.

Each energy sample uses `f_S(E)` and `f_D(E)` at that energy. Source and
drain injection are separately restricted to their own propagating bands;
an evanescent contact injects no particles.

---

# Part V — Closing the loop: self-consistent Poisson↔NEGF iteration

## V.1 Physics: why self-consistency matters

A single electrostatic solve at fixed charge and a subsequent NEGF density
calculation do not establish self-consistency. Mobile charge screens the
electrostatic potential, and the potential determines where charge
accumulates. `selfconsistent::solve_self_consistent` evaluates this feedback:

1. Solve electrostatics (`calc_potential`) for the current `\rho`.
2. Run the NEGF sweep (Part III) at the resulting potential.
3. Compute the electron density (Part IV) from that sweep.
4. Convert it to a charge density and mix it into `\rho` (§V.2).
5. Repeat from step 1 until the potential profile stops changing (or a
   caller-supplied iteration budget runs out).

## V.2 Numerical method: Picard iteration with linear mixing

This is a **fixed-point iteration** (also called Picard iteration): define
a map `F: \rho \mapsto \rho'` (steps 1–4 above), and look for `\rho^*` with
`F(\rho^*) = \rho^*`. Applying `F` repeatedly (`\rho_{n+1} = F(\rho_n)`)
converges to a fixed point only if `F` is a contraction near it; blindly
substituting the full NEGF-computed density at every step
(`\rho_{n+1} = F(\rho_n)` exactly) tends to overshoot and oscillate or
diverge for this kind of strongly-coupled electrostatic/charge problem — a
well-known issue with naive Poisson–Schrödinger/NEGF self-consistency, not
specific to this codebase.

The standard fix, used here, is **linear (Picard) mixing**: instead of
jumping all the way to the newly computed target, take a damped step
toward it,

$$\rho_{n+1} = \rho_n + \alpha\,(\rho_{n+1}^{\text{target}} - \rho_n), \qquad \alpha = \texttt{mixing} \in (0, 1]$$

(`SelfConsistentOptions::mixing`, default `0.3`). Smaller `α` is more
stable but converges more slowly (more iterations to reach a given
tolerance); `α = 1` recovers plain, undamped fixed-point iteration.
Convergence is declared when the **undamped fixed-point potential
residual** `max(abs(Poisson(rho_target)-psi_current))` drops below
`tolerance` (default `1e-6` eV), before applying mixing. Thus arbitrarily
small mixing cannot fake convergence by making a tiny step. If
`max_iterations` (default 50) is exhausted, `NotConverged` reports the
last undamped residual.

**Mixing changes convergence rate, not the fixed point. Finite eta
changes the density and potentially the fixed point** (§III.4). A
`NotConverged` result may require more iterations, a smaller mixing factor,
or better energy resolution; increasing eta can stabilize a different,
more absorptive problem, not validate the original one. Conversely,
fixed-point convergence does not validate the physical approximations.

## V.3 Charge-density units

Two separate conversions are essential; neither substitutes for the other:

1. `charge::electron_density` returns **per-spin linear density** in nm⁻¹.
2. Convert to m⁻¹ (`*1e9`) and divide by a transverse area to obtain m⁻³.
   Rust callers can supply `Device::effective_area_m2`; no width or area is
   inferred from `d_ch` or gate geometry. With `None` (default), SCF uses an
   implicit **1 m²** area and per-spin charge.
3. Multiply by `-e`. With explicit area `A`, the spin-summed target is
   `rho_target=-2*e*n_electron*1e9/A` in C/m³. The default without an area
   uses `rho_target=-e*n_electron*1e9`. `Device::mobile_charge_density`
   performs this conversion; the NEGF density remains per spin in both modes.
4. In `calc_potential`, the source is
   `(rho+n_dot)*1e-18/(EPS_0*k_si)`, not simply `rho/epsilon`. The factor
   `1e-18` converts the SI m⁻² source to the nm⁻² operator (§I.3).

The default unit-area convention dilutes a 1-D mode over an enormous area relative
to a nanoscale channel, making mobile-charge feedback extremely weak.
The loop is closed algorithmically, but rapid convergence under this
normalization does not demonstrate appreciable physical screening. A
quantitative transistor model requires an explicit physical area and
convergence studies. The uniform transverse-density approximation and
finite-eta limitations remain even with explicit area. See
[`ELECTROSTATIC_GEOMETRY.md`](ELECTROSTATIC_GEOMETRY.md) for the Rust API and
manufactured-source and finite-area regression checks.

`N_dot` (fixed dopant charge) is left as-is (default zero, undoped body);
if used with a nonzero value it should be supplied in the same C/m³ units.

## V.4 Code walkthrough: `selfconsistent::solve_self_consistent`

```text
validate_device_and_options()
calc_potential()                          // uses existing rho (initially zero)
loop (up to max_iterations):
    energies   = negf_energy_grid(...)    // midpoint cells in [L,U], see II.4
    width      = energies[1] - energies[0] // actual quadrature weight in eV
    green      = green_function_sweep(energies, eta)
    n_electron = electron_density(green, ..., width)  // per spin, nm^-1
    rho_before, psi_before = rho.clone(), psi_f.clone()
    rho        = map(device.mobile_charge_density, n_electron) // C/m^3
    calc_potential()                     // undamped target potential
    residual   = max(abs(psi_f - psi_before))
    if residual < tolerance:
        restore rho_before, psi_before, and barrier
        return Ok(residual, green, n_electron)
    rho = rho_before + mixing * (rho - rho_before)
    calc_potential()                     // next mixed iterate
return Err(NotConverged)
```

On success, device potential, returned Green's functions, and returned
density describe the **same tested iterate**. Stored `rho` is the iterate's
mixed input; its target inferred from the returned density agrees only to
the fixed-point potential tolerance. The SCF grid rejects empty,
under-resolved, nonfinite, and over-budget windows instead of silently
returning a misleading density.

---

# Part VI — Validity and known limitations, all in one place

Each item below is documented in more detail (with the reasoning, and
where relevant, a regression test) in the corresponding source module's
doc comments; this is a consolidated summary.

- **Natural-length electrostatics** (Part I.2): thin-body approximation,
  not valid for bulk/thick-body devices; a classical (non-quantized)
  transverse charge model in tension with the quantum-mechanical
  longitudinal NEGF treatment.
- **Coherent Landauer–Caroli current** (Part II.1): includes elastic
  reflection, tunneling, and interference, but no dissipative scattering,
  energy relaxation, or dephasing. Assumes static single-particle
  electrostatics and equilibrium reservoirs. The separate thermionic
  comparator (Part II.2), not the default, assumes a step transmission.
- **Spectral weight versus flux** (Part III.3): scattering-state LDOS is
  not transmission or current density. Large spectral peaks alone do
  not establish appreciable current.
- **Independent convergence checks** (Part II.3): SCF fixed-point
  convergence is distinct from current energy integration, density
  integration, small-eta, and spatial-grid convergence.
- **Effective-mass, single-valley, parabolic band** (`m_eff`, Parts II–III):
  stands in for silicon's multiple equivalent valleys and non-parabolic
  dispersion away from the band edge. Treat absolute currents as
  illustrative trends, not device-accurate predictions. Spin degeneracy
  *is* included in both current models via `2e²/h` for eV integration.
- **Tight-binding discretization range of validity** (Part III.1): valid
  only while the swept energy range stays well below the chain's `4t`
  bandwidth; check via grid refinement at new device scales.
- **Retarded leads** (Part III.2): causal propagating root and decaying
  real roots outside both band edges; evanescent real shifts are retained.
- **Absorbing eta** (Part III.4): no compensating injection, so eta changes
  occupation and potentially the fixed point. Contact-only density does
  not prescribe occupations for isolated bound states.
- **Per-spin charge** (Part IV.2): normalization is verified against an
  analytic uniform chain; it is not spin-summed despite spin-degenerate
  current. Fermi probabilities cannot hide a degeneracy factor.
- **SCF area normalization** (Part V.3): the default per-spin, implicit 1 m²
  area yields very weak feedback. Explicit area selects spin-summed volume
  charge but still assumes uniform transverse density. Fixed-point convergence
  alone does not validate screening, eta, or energy quadrature.

Passing qualitative trends and positivity tests does not validate the
charge normalization, physical screening, or the finite-eta limit. These
limitations can affect quantitative results even at a converged fixed
point. Read results as educational illustrations, not validated
predictions for a real device.

---

# Part VII — Testing strategy: numerics vs. physics

The test suite (`cargo test --workspace`) validates two genuinely different
things, and it's worth knowing which is which when reading or extending it:

- **Numerical correctness** — "does the fast algorithm compute the same
  thing as a slow, obviously-correct reference?" `tridiag`'s pivoted
  solver is checked against dense Gaussian elimination
  (§I.4); `negf`'s recursive Green's function sweep and boundary-column
  solves are checked against a dense Gauss-Jordan matrix inversion
  (§III.5). These tests would catch an algebra mistake in the `O(N)`
  algorithms even if the underlying physics model were entirely different.
- **Physical correctness** — "does the model behave the way the physics
  says it should?" Current increasing with gate bias and with drain bias
  (§II, `device::tests`), the self-consistent loop converging to a stable,
  non-negative charge density (§V, `selfconsistent::tests`), and the local density of states staying
  non-negative (`negf::tests::g_diag_is_non_negative_at_self_consistent_loop_broadening`)
  and the contact self-energy carrying the causally-required sign
  (`negf::tests::contact_self_energy_has_non_positive_imaginary_part_for_propagating_modes`).
  These tests wouldn't catch a subtle O(N) algorithm bug (a wrong-but
  -internally-consistent fast algorithm could still pass them), which is
  why both kinds of test exist side by side.

`crates/negforge-core/tests/self_consistent_realistic_device.rs` runs the
full pipeline (Parts I–V) at the model's actual default device scale rather
than a toy size, specifically to catch issues that only appear once the
NEGF energy grid and tight-binding bandwidth are at realistic proportions.
`physical_regressions.rs` separately checks SI/nm charge-source scaling
and electron-energy sign, absolute current in A and shared reservoir
reference, undamped residuals under tiny mixing, matching returned NEGF
and device iterates, symmetric geometry, and invalid inputs/ranges.
Unit tests cover analytic per-spin occupation, uniform-chain lead shifts,
near-pole diagonal stability, and pivoted real/complex systems.
These are targeted invariants, not quantitative validation of a real MOSFET.

See the README setup instructions: use Rust/Cargo plus CPython 3.14
(the primary validation target; CPython 3.9–3.13 is also supported)
and build/install the PyO3 extension before running the Python suite.

```bash
cargo test --locked -p negforge-core
cargo test --locked --workspace
python -m pytest -q tests
jupyter nbconvert --to notebook --execute notebooks/negforge_demo.ipynb \
  --output negforge_demo.executed.ipynb --output-dir "$TMPDIR"
```

The Python tests (`tests/test_python.py`) and notebook exercise paths not
covered by Rust tests. Notebook execution requires the optional notebook
dependencies and a kernel using the environment containing the extension.
Write executed output outside the source notebook; execution is a separate
verification step, not implied by a passing Rust suite.

---

# Part VIII — Glossary

| Symbol / term | Meaning | Units in this codebase |
|---|---|---|
| `φ`, `Psi_f` | Self-consistent electrostatic potential *energy* (not volts) | eV |
| `λ`, `lambda` | Natural (screening) length | nm |
| `a` | Grid spacing | nm |
| `t`, `t_hop` | Tight-binding hopping parameter, `ħ²/(2m*a²)` | eV |
| `η`, `eta` | Artificial imaginary broadening, `(E+iη)I - H` | eV |
| `Σ_S`, `Σ_D` | Contact retarded self-energies | eV |
| `Γ_S`, `Γ_D` | Contact broadening functions, `-2 Im(Σ)` | eV |
| `G^r` | Retarded Green's function | eV⁻¹ (energy-domain) |
| `f_S`, `f_D` | Source/drain Fermi-Dirac occupation | dimensionless |
| `T(E)`, `mathcal T(E)` | Coherent Caroli transmission probability | dimensionless |
| `I` | Spin-degenerate single-mode current | A (not current per width) |
| `k_B`, `h`, `ħ`, `e`, `m_e` | Physical constants (`constants.rs`) | SI |
| `ρ`, `rho` | Charge density fed into the Poisson solve | C/m³ |
| `n(x)` | Per-spin, single-orbital linear electron density from NEGF | nm⁻¹ (`electron_density`'s return value) |
| `N_dot` | Fixed (dopant) charge term | same units as `rho` |
| `mixing`, `α` | Linear/Picard mixing factor for the self-consistent loop | dimensionless, `(0,1]` |

Lengths throughout are in nm, energies and potentials in eV, temperature in
Kelvin.
