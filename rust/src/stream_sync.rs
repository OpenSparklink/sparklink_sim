//! Integer-grid energy-normalized differential screen, before the unchanged fine search.
use numpy::{Complex64, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

type Screen = (Option<f64>, usize, usize);

fn integer_screen(
    z: &[Complex64],
    sps: usize,
    reference: &[Complex64],
    min_start: f64,
) -> Result<Screen, &'static str> {
    if sps == 0 || reference.len() < 2 || !min_start.is_finite() {
        return Err("Need positive SPS, at least two sync symbols and finite lower bound");
    }
    if !z
        .iter()
        .chain(reference)
        .all(|v| v.re.is_finite() && v.im.is_finite())
    {
        return Err("Need finite filtered IQ and sync symbols");
    }
    let dr: Vec<_> = reference.windows(2).map(|r| r[1] * r[0].conj()).collect();
    // The reference uses array abs(dy)**2 (squaring), unlike scalar TED power.
    let squared_norm = |v: Complex64| {
        let magnitude = v.norm();
        magnitude * magnitude
    };
    let ref_energy: f64 = dr.iter().map(|v| squared_norm(*v)).sum();
    let mut best = None;
    let (mut phases, mut positions) = (0, 0);
    for phase in 0..sps.min(z.len()) {
        let n = (z.len() - 1 - phase) / sps + 1;
        if n < reference.len() {
            continue;
        }
        phases += 1;
        let dy: Vec<_> = (0..n - 1)
            .map(|i| z[phase + (i + 1) * sps] * z[phase + i * sps].conj())
            .collect();
        let energies: Vec<_> = dy.iter().map(|v| squared_norm(*v)).collect();
        for offset in 0..n - reference.len() + 1 {
            if ((phase + offset * sps) as f64) < min_start {
                continue;
            }
            positions += 1;
            let mut corr = Complex64::new(0., 0.);
            let mut energy = 0.;
            for k in 0..dr.len() {
                corr += dy[offset + k] * dr[k].conj();
                energy += energies[offset + k];
            }
            let score = corr.norm() / (energy * ref_energy).sqrt().max(1e-30);
            if !score.is_finite() {
                return Err("Nonfinite differential score");
            }
            if best.is_none_or(|previous| score > previous) {
                best = Some(score);
            }
        }
    }
    Ok((best, phases, positions))
}

/// No sync answer is selected here. A roundoff-borderline gate uses the NumPy reference.
#[pyfunction]
pub fn rust_integer_sync_screen(
    filtered: PyReadonlyArray1<'_, Complex64>,
    sps: usize,
    reference: PyReadonlyArray1<'_, Complex64>,
    min_start: f64,
) -> PyResult<Screen> {
    integer_screen(filtered.as_slice()?, sps, reference.as_slice()?, min_start)
        .map_err(PyValueError::new_err)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_sync_silence_bounds_and_short_input() {
        let reference: Vec<_> = (0..32)
            .map(|k| Complex64::from_polar(1., (k * k * k) as f64 * 0.127))
            .collect();
        let mut z = vec![Complex64::new(0., 0.); 256];
        let (score, phases, positions) = integer_screen(&z, 4, &reference, 1.).unwrap();
        assert_eq!((score, phases, positions), (Some(0.), 4, 131));
        for k in 0..reference.len() {
            z[8 + 4 * k] = reference[k];
        }
        assert!((integer_screen(&z, 4, &reference, 1.).unwrap().0.unwrap() - 1.).abs() < 1e-14);
        assert_eq!(
            integer_screen(&z, 4, &reference, 10000.).unwrap(),
            (None, 4, 0)
        );
        assert_eq!(
            integer_screen(&z[..4], 4, &reference, 0.).unwrap(),
            (None, 0, 0)
        );
        assert!(integer_screen(&z, 0, &reference, 0.).is_err());
        assert!(integer_screen(&z, 4, &reference[..1], 0.).is_err());
    }
}
