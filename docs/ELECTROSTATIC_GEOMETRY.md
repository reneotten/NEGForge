# Explicit-area electrostatic feedback

## Units and sign

`psi_f` is electron potential energy in eV, numerically `-V` for an
electrostatic voltage `V` in volts. It is not voltage and must not acquire
another factor of `-e` when put into the Hamiltonian. The natural-length
Poisson operator uses nm:

```
(d²/dx_nm² - 1/lambda_nm²) psi_f
    = (rho + n_dot) * 1e-18 / (EPS_0 * k_si)
      - (psi_g + psi_bi) / lambda_nm²
```

Both `rho` and signed fixed dopant charge `n_dot` are C/m³. The `1e-18`
converts the SI m⁻² source to nm⁻². For a uniform source and zero gate/built-in
profile the exact discrete Neumann solution is
`psi_f = -rho * 1e-18 * lambda² / (EPS_0*k_si)`.
Thus electrons (`rho<0`) **raise** their own energy barrier; positive donor
charge lowers it. This sign is repulsive electronic feedback, not attraction.

## Area and spin are separate from the longitudinal grid

`charge::electron_density` returns **per-spin, single-orbital linear number
density in nm⁻¹**. Its analytic half-filled-chain normalization is `1/(2a)`.
The spin-summed Landauer current already counts both spins. Physical
spin-consistent electrostatic charge therefore requires

```
rho = -2 * e * n_per_spin_nm_inv * 1e9 / effective_area_m2
```

The area is an explicitly supplied effective transverse active area, in m².
`d_ch` is a thickness, not an area; neither a width nor the number of transverse
modes can be inferred from it, `geo`, or the transport grid spacing. The
single-mode longitudinal model does not resolve the actual transverse wave
function, so dividing by area is a uniform-transverse-density approximation,
not a replacement for a transverse Poisson/subband calculation.

## Selecting the active area

```rust
use negforge_core::device::{Device, DeviceParams};

let mut device = Device::new(DeviceParams::default());
// Caller-supplied example, NOT a default or an inferred device width:
device.set_effective_area_m2(Some(20e-18))?;
// Run solve_self_consistent(&mut device, &options) with this device.
```

New devices have `effective_area_m2=None`, selecting the normalization
`rho=-e*n_per_spin*1e9`. This default mode behaves as per-spin charge
diluted over an implicit 1 m²; it is **not** a specified physical cross-section
and it is inconsistent with spin-summed current. Supplying `Some(area)` explicitly
opts into both the physical volume conversion and spin-summed charge.
`None` selects implicit unit-area mode. Zero, negative, NaN and infinite areas are rejected
by the setter and by device validation (including direct public-field edits).
The returned NEGF density remains per spin in both modes.

SCF integration must use `device.mobile_charge_density(n)` for each density
sample; changing only the stored area without using that conversion does not
create feedback. Existing `rho` values are already C/m³ and must not be
rescaled by `calc_potential`.

## Manufactured and finite-area regression

`tests/electrostatic_geometry.rs` checks:

- default normalization, explicit-area spin factor, inverse-area scaling and
  invalid input handling;
- the exact constant-source solution including Neumann boundary rows, SI/nm
  scale and both electron/donor signs;
- a complete NEGF↔Poisson solve with caller-supplied area `20e-18 m²`, requiring
  multiple iterations, a raised channel barrier, suppressed injected channel
  density, and independent reconstruction of the undamped residual.

The finite-area example has `a=0.5 nm`, `l_ch=l_ds=8 nm`, `V_g=0.5 V`,
`d_e=0.002 eV`, `eta=0.002 eV`, `mixing=0.15`, full stored energy bound and
`tolerance=1e-7 eV`. The test compares explicit-area feedback against default
unit-area feedback; it checks screening and reconstructs the residual rather
than using iteration counts as evidence of physical accuracy.
Finite absorbing eta, energy-window/quadrature error, unoccupied isolated bound
states and absent transverse/valley physics remain independent limitations.

Run:

```sh
cargo test --locked -p negforge-core --test electrostatic_geometry -- --nocapture
```
