//! Shared practical allocation limits (not a promise of available system RAM).
//! Counts are checked before casts/allocations. The conservative working-set
//! estimate includes three Rust NEGF matrices, Python float/list + NumPy result
//! copies, per-energy row metadata, and linear solver/SCF scratch space.
use crate::error::{NegForgeError, Result};

pub const MAX_SITES: usize = 100_000;
pub const MAX_ENERGIES: usize = 100_000;
pub const MAX_ENERGY_SITES: usize = 2_000_000;
pub const MAX_WORKING_BYTES: usize = 256 * 1024 * 1024;

fn invalid() -> NegForgeError {
    NegForgeError::InvalidParameter("grid exceeds practical resource budget (100000 sites, 100000 energies, 2000000 energy-sites, 256 MiB working memory)".into())
}

pub fn check_sites(sites: usize) -> Result<()> {
    if !(1..=MAX_SITES).contains(&sites) {
        return Err(invalid());
    }
    Ok(())
}

/// Conservative peak estimate, including Python result conversion copies even
/// for core callers. Checked operations also make this safe on 32-bit targets.
pub fn working_bytes(sites: usize, energies: usize) -> Result<usize> {
    sites
        .checked_mul(energies)
        .and_then(|cells| cells.checked_mul(64))
        .and_then(|bytes| {
            energies
                .checked_mul(256)
                .and_then(|rows| bytes.checked_add(rows))
        })
        .and_then(|bytes| {
            sites
                .checked_mul(512)
                .and_then(|scratch| bytes.checked_add(scratch))
        })
        .ok_or_else(invalid)
}

pub fn check_working_set(sites: usize, energies: usize) -> Result<()> {
    check_sites(sites)?;
    if energies == 0 || energies > MAX_ENERGIES {
        return Err(invalid());
    }
    let cells = sites.checked_mul(energies).ok_or_else(invalid)?;
    if cells > MAX_ENERGY_SITES || working_bytes(sites, energies)? > MAX_WORKING_BYTES {
        return Err(invalid());
    }
    Ok(())
}

/// Inclusive regularly spaced grid, requiring at least two energy samples.
/// Validate the floating count before conversion to an integer.
pub fn energy_count(min: f64, max: f64, step: f64) -> Result<usize> {
    if !min.is_finite() || !max.is_finite() || !step.is_finite() || step <= 0.0 {
        return Err(invalid());
    }
    let intervals = ((max - min) / step).floor();
    if !intervals.is_finite() || intervals < 1.0 || intervals >= MAX_ENERGIES as f64 {
        return Err(invalid());
    }
    (intervals as usize).checked_add(1).ok_or_else(invalid)
}
