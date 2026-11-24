//! Tridiagonal linear system solvers with adjacent partial pivoting.
//!
//! Both the electrostatic Poisson-like solve and the NEGF Green's-function
//! recursion operate on tridiagonal matrices, so this is the one place that
//! implements O(N) forward elimination/back substitution. Unlike unpivoted
//! Thomas elimination, adjacent partial pivoting handles nonsingular systems
//! with zero/tiny leading pivots. Empty and scalar systems are supported.
//! Invalid dimensions and exactly singular systems panic; these APIs do not
//! estimate condition numbers or return a structured numerical error.

use num_complex::Complex64;

/// Solve `A x = rhs` for a real tridiagonal `A` given as three diagonals.
///
/// `sub[i]` is `A[i+1][i]` (length N-1), `diag[i]` is `A[i][i]` (length N),
/// `sup[i]` is `A[i][i+1]` (length N-1).
pub fn solve_real(sub: &[f64], diag: &[f64], sup: &[f64], rhs: &[f64]) -> Vec<f64> {
    solve_pivoted(sub, diag, sup, rhs, f64::abs)
}

/// Complex counterpart of [`solve_real`], used by the NEGF module where the
/// matrix `(E + i*eta) I - H` has a real tridiagonal skeleton but complex
/// diagonal entries (contact self-energies) and the right-hand side may be
/// complex (unit vectors when extracting a single column of the Green's
/// function).
pub fn solve_complex(
    sub: &[f64],
    diag: &[Complex64],
    sup: &[f64],
    rhs: &[Complex64],
) -> Vec<Complex64> {
    let sub: Vec<_> = sub.iter().map(|&x| Complex64::new(x, 0.0)).collect();
    let sup: Vec<_> = sup.iter().map(|&x| Complex64::new(x, 0.0)).collect();
    solve_pivoted(&sub, diag, &sup, rhs, |z| z.norm())
}

// Adjacent partial pivoting keeps O(N) storage/work, adding at most a
// second superdiagonal (the same structure used by LAPACK's GTSV).
fn solve_pivoted<T, F>(sub: &[T], diag: &[T], sup: &[T], rhs: &[T], magnitude: F) -> Vec<T>
where
    T: Copy
        + From<f64>
        + std::ops::Sub<Output = T>
        + std::ops::Mul<Output = T>
        + std::ops::Div<Output = T>,
    F: Fn(T) -> f64,
{
    let n = diag.len();
    assert_eq!(sub.len(), n.saturating_sub(1));
    assert_eq!(sup.len(), n.saturating_sub(1));
    assert_eq!(rhs.len(), n);
    if n == 0 {
        return Vec::new();
    }
    let mut d = diag.to_vec();
    let mut u = sup.to_vec();
    let mut u2 = vec![T::from(0.0); n.saturating_sub(2)];
    let mut b = rhs.to_vec();
    for i in 0..n - 1 {
        if magnitude(d[i]) >= magnitude(sub[i]) {
            assert!(magnitude(d[i]) > 0.0, "singular tridiagonal matrix");
            let factor = sub[i] / d[i];
            d[i + 1] = d[i + 1] - factor * u[i];
            b[i + 1] = b[i + 1] - factor * b[i];
        } else {
            let factor = d[i] / sub[i];
            d[i] = sub[i];
            let next = d[i + 1];
            d[i + 1] = u[i] - factor * next;
            u[i] = next;
            if i + 2 < n {
                u2[i] = u[i + 1];
                u[i + 1] = T::from(0.0) - factor * u[i + 1];
            }
            let previous_rhs = b[i];
            b[i] = b[i + 1];
            b[i + 1] = previous_rhs - factor * b[i + 1];
        }
    }
    assert!(magnitude(d[n - 1]) > 0.0, "singular tridiagonal matrix");
    b[n - 1] = b[n - 1] / d[n - 1];
    for i in (0..n - 1).rev() {
        let mut value = b[i] - u[i] * b[i + 1];
        if i + 2 < n {
            value = value - u2[i] * b[i + 2];
        }
        b[i] = value / d[i];
    }
    b
}

#[cfg(test)]
#[allow(clippy::needless_range_loop)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    /// Dense reference solver (Gaussian elimination with partial pivoting)
    /// used only in tests to validate the O(N) Thomas-algorithm solvers.
    fn dense_solve_real(sub: &[f64], diag: &[f64], sup: &[f64], rhs: &[f64]) -> Vec<f64> {
        let n = diag.len();
        let mut a = vec![vec![0.0; n]; n];
        for i in 0..n {
            a[i][i] = diag[i];
            if i > 0 {
                a[i][i - 1] = sub[i - 1];
            }
            if i < n - 1 {
                a[i][i + 1] = sup[i];
            }
        }
        let mut b = rhs.to_vec();
        for col in 0..n {
            let mut pivot = col;
            for row in (col + 1)..n {
                if a[row][col].abs() > a[pivot][col].abs() {
                    pivot = row;
                }
            }
            a.swap(col, pivot);
            b.swap(col, pivot);
            for row in (col + 1)..n {
                let factor = a[row][col] / a[col][col];
                for k in col..n {
                    a[row][k] -= factor * a[col][k];
                }
                b[row] -= factor * b[col];
            }
        }
        let mut x = vec![0.0; n];
        for i in (0..n).rev() {
            let mut s = b[i];
            for j in (i + 1)..n {
                s -= a[i][j] * x[j];
            }
            x[i] = s / a[i][i];
        }
        x
    }

    #[test]
    fn scalar_and_empty_systems() {
        assert_eq!(solve_real(&[], &[], &[], &[]), Vec::<f64>::new());
        assert_eq!(solve_real(&[], &[2.0], &[], &[6.0]), vec![3.0]);
        let z = Complex64::new(2.0, 1.0);
        assert_eq!(
            solve_complex(&[], &[z], &[], &[z * 3.0]),
            vec![Complex64::new(3.0, 0.0)]
        );
    }

    #[test]
    fn pivoting_handles_zero_and_tiny_leading_pivots() {
        for first in [0.0, 1e-30] {
            let sub = [1.0, 2.0];
            let diag = [first, 3.0, 4.0];
            let sup = [2.0, 1.0];
            let rhs = [4.0 + first, 10.0, 16.0]; // x = [1,2,3]
            let expected = dense_solve_real(&sub, &diag, &sup, &rhs);
            let actual = solve_real(&sub, &diag, &sup, &rhs);
            let complex = solve_complex(
                &sub,
                &diag.map(|d| Complex64::new(d, 0.0)),
                &sup,
                &rhs.map(|b| Complex64::new(b, 0.0)),
            );
            for i in 0..3 {
                assert_relative_eq!(actual[i], expected[i], epsilon = 1e-12);
                assert_relative_eq!(complex[i].re, expected[i], epsilon = 1e-12);
            }
        }
    }

    #[test]
    fn complex_pivoted_solutions_have_small_residuals() {
        for case in 0..32 {
            let n = 8;
            let sub: Vec<_> = (0..n - 1).map(|i| 1.0 + 0.1 * i as f64).collect();
            let sup: Vec<_> = (0..n - 1).map(|i| -0.8 + 0.03 * i as f64).collect();
            let diag: Vec<_> = (0..n)
                .map(|i| {
                    Complex64::new(
                        ((i + case) as f64).sin() * 0.1,
                        if i == 0 { 0.0 } else { 0.2 },
                    )
                })
                .collect();
            let expected: Vec<_> = (0..n)
                .map(|i| Complex64::new(i as f64 + 1.0, (i as f64).cos()))
                .collect();
            let rhs: Vec<_> = (0..n)
                .map(|i| {
                    let mut b = diag[i] * expected[i];
                    if i > 0 {
                        b += sub[i - 1] * expected[i - 1];
                    }
                    if i + 1 < n {
                        b += sup[i] * expected[i + 1];
                    }
                    b
                })
                .collect();
            let actual = solve_complex(&sub, &diag, &sup, &rhs);
            for i in 0..n {
                assert!((actual[i] - expected[i]).norm() < 1e-12);
            }
        }
    }

    #[test]
    fn thomas_matches_dense_gaussian_elimination() {
        let n = 12;
        let sub: Vec<f64> = (0..n - 1).map(|i| 1.0 + 0.1 * i as f64).collect();
        let sup: Vec<f64> = (0..n - 1).map(|i| 0.7 + 0.05 * i as f64).collect();
        let diag: Vec<f64> = (0..n).map(|i| -3.0 - 0.2 * i as f64).collect();
        let rhs: Vec<f64> = (0..n).map(|i| (i as f64).sin() + 1.0).collect();

        let fast = solve_real(&sub, &diag, &sup, &rhs);
        let reference = dense_solve_real(&sub, &diag, &sup, &rhs);

        for (a, b) in fast.iter().zip(reference.iter()) {
            assert_relative_eq!(a, b, epsilon = 1e-9);
        }
    }

    #[test]
    fn identity_system_returns_rhs() {
        let n = 5;
        let sub = vec![0.0; n - 1];
        let sup = vec![0.0; n - 1];
        let diag = vec![1.0; n];
        let rhs = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let x = solve_real(&sub, &diag, &sup, &rhs);
        assert_eq!(x, rhs);
    }
}
