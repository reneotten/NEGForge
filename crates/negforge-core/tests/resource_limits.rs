//! Pure boundary arithmetic: never allocate a limit-sized grid.
use negforge_core::resources::*;

#[test]
fn independent_grid_and_product_limits() {
    assert!(check_sites(MAX_SITES).is_ok());
    assert!(check_sites(MAX_SITES + 1).is_err());
    assert!(check_working_set(2, MAX_ENERGIES).is_ok());
    assert!(check_working_set(2, MAX_ENERGIES + 1).is_err());
    assert!(check_working_set(MAX_SITES + 1, 1).is_err());
    assert!(check_working_set(1000, MAX_ENERGY_SITES / 1000).is_ok());
    assert!(check_working_set(1000, MAX_ENERGY_SITES / 1000 + 1).is_err());
    assert!(check_working_set(usize::MAX, usize::MAX).is_err());
    assert!(check_working_set(1, 1).is_ok());
    assert!(check_sites(0).is_err());
    assert!(working_bytes(usize::MAX, 2).is_err());
    assert!(working_bytes(1, usize::MAX).is_err());
    assert_eq!(
        working_bytes(10, 20).unwrap(),
        10 * 20 * 64 + 20 * 256 + 10 * 512
    );
}

#[test]
fn energy_count_checks_before_cast_or_allocation() {
    assert_eq!(energy_count(0.0, 1.0, 0.25).unwrap(), 5);
    assert_eq!(
        energy_count(0.0, (MAX_ENERGIES - 1) as f64, 1.0).unwrap(),
        MAX_ENERGIES
    );
    assert!(energy_count(0.0, MAX_ENERGIES as f64, 1.0).is_err());
    for step in [0.0, -1.0, f64::NAN, f64::INFINITY, f64::MIN_POSITIVE] {
        assert!(energy_count(0.0, 1.0, step).is_err());
    }
    assert!(energy_count(1.0, 0.0, 0.1).is_err());
}

#[test]
fn geometry_boundaries_are_checked_without_allocating() {
    use negforge_core::{Device, DeviceParams};
    let mut params = DeviceParams {
        a: 1.0,
        l_ds: 0.0,
        l_ch: (MAX_SITES - 1) as f64,
        auto_size_contacts: false,
        ..Default::default()
    };
    assert_eq!(Device::checked_site_count(&params).unwrap(), MAX_SITES);
    params.l_ch += 1.0;
    assert!(Device::checked_site_count(&params).is_err());
    params.l_ch = f64::MAX;
    assert!(Device::checked_site_count(&params).is_err());
    params.l_ch = 10.0;
    params.a = f64::MIN_POSITIVE;
    assert!(Device::checked_site_count(&params).is_err());
    params.a = 1.0;
    params.auto_size_contacts = true;
    params.d_ox = f64::MAX;
    assert!(Device::checked_site_count(&params).is_err());
}
