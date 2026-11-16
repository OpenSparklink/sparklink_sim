//! Bounded packet-local Gardner sampling. No synchronization/CRC/payload decisions.
use numpy::{Complex64, PyReadonlyArray1, PyReadwriteArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

fn cubic(z: &[Complex64], origin: f64, position: f64) -> Result<Complex64, &'static str> {
    let relative = position - origin;
    let cell = relative.floor() as isize;
    if cell < 1 || cell + 2 >= z.len() as isize || !relative.is_finite() {
        return Err("Cubic sample outside retained history/future");
    }
    let k = cell as usize;
    let u = relative - cell as f64;
    // NumPy complex division by a real scalar multiplies by its reciprocal.
    // Preserve that operation order rather than weakening backend comparisons.
    Ok(z[k - 1] * (-u * (u - 1.) * (u - 2.)) * (1. / 6.)
        + z[k] * ((u + 1.) * (u - 1.) * (u - 2.)) / 2.
        - z[k + 1] * ((u + 1.) * u * (u - 2.)) / 2.
        + z[k + 2] * ((u + 1.) * u * (u - 1.)) * (1. / 6.))
}

type State = (usize, f64, f64, f64, f64, f64, f64, usize, usize);

/// Only scalar loop state; IQ and bounded output buffers remain caller-owned.
#[pyclass]
pub struct StreamTiming {
    sps: f64,
    next: f64,
    previous: Complex64,
    previous_t: f64,
    power: f64,
    omega: f64,
    kp: f64,
    ki: f64,
    lower: f64,
    upper: f64,
    phase: f64,
    slope: f64,
    gain: f64,
    hold: usize,
    count: usize,
    rate_limit_events: usize,
    error_clip_events: usize,
}

#[pymethods]
impl StreamTiming {
    #[new]
    // Explicit names at the FFI boundary avoid an undocumented positional config vector.
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (*, sps, start, previous, power, kp, ki, lower, upper, phase, slope, gain, hold))]
    fn new(
        sps: f64,
        start: f64,
        previous: (f64, f64),
        power: f64,
        kp: f64,
        ki: f64,
        lower: f64,
        upper: f64,
        phase: f64,
        slope: f64,
        gain: f64,
        hold: usize,
    ) -> PyResult<Self> {
        let finite = [
            sps, start, previous.0, previous.1, power, kp, ki, lower, upper, phase, slope, gain,
        ]
        .iter()
        .all(|v| v.is_finite());
        if !finite
            || sps < 2.
            || power <= 0.
            || gain <= 0.
            || hold == 0
            || kp < 0.
            || ki < 0.
            || lower <= 2. * kp
            || lower > sps
            || upper < sps
        {
            return Err(PyValueError::new_err("Invalid packet timing state"));
        }
        Ok(Self {
            sps,
            next: start,
            previous: Complex64::new(previous.0, previous.1),
            previous_t: start,
            power,
            omega: sps,
            kp,
            ki,
            lower,
            upper,
            phase,
            slope,
            gain,
            hold,
            count: 0,
            rate_limit_events: 0,
            error_clip_events: 0,
        })
    }

    /// Advance only available symbols up to the next verified-parser boundary.
    #[allow(clippy::too_many_arguments)]
    fn advance(
        &mut self,
        filtered: PyReadonlyArray1<'_, Complex64>,
        origin: f64,
        latest: f64,
        target: usize,
        mut values: PyReadwriteArray1<'_, Complex64>,
        mut centers: PyReadwriteArray1<'_, f64>,
        mut rates: PyReadwriteArray1<'_, f64>,
        mut errors: PyReadwriteArray1<'_, f64>,
    ) -> PyResult<State> {
        let z = filtered.as_slice()?;
        let (v, c, r, e) = (
            values.as_slice_mut()?,
            centers.as_slice_mut()?,
            rates.as_slice_mut()?,
            errors.as_slice_mut()?,
        );
        if !origin.is_finite()
            || !latest.is_finite()
            || target < self.count
            || target > v.len()
            || target > c.len()
            || target > r.len()
            || target > e.len()
        {
            return Err(PyValueError::new_err("Invalid timing output bound"));
        }
        while self.next <= latest && self.count < target {
            let t = self.next;
            let current = cubic(z, origin, t).map_err(PyValueError::new_err)?;
            if !current.re.is_finite() || !current.im.is_finite() {
                return Err(PyValueError::new_err("Nonfinite interpolated symbol"));
            }
            // NumPy scalar **2 calls libm pow; forbid strength reduction to x*x.
            self.power = (self.power
                + (current.norm().powf(std::hint::black_box(2.)) - self.power) / 64.)
                .max(1e-30);
            let mut error = 0.;
            if self.count >= self.hold {
                let mid =
                    cubic(z, origin, (self.previous_t + t) / 2.).map_err(PyValueError::new_err)?;
                let raw = ((self.previous - current) * mid.conj()).re / self.power;
                error = raw.clamp(-2., 2.);
                self.error_clip_events += usize::from(raw.abs() > 2.);
                let proposed = self.omega + self.ki * error;
                self.rate_limit_events +=
                    usize::from(proposed < self.lower || proposed > self.upper);
                self.omega = proposed.clamp(self.lower, self.upper);
            }
            let angle = -(self.phase + self.slope * self.count as f64);
            let (sin, cos) = angle.sin_cos();
            // Same reciprocal multiplication as the Python complex ufunc.
            v[self.count] = current * Complex64::new(cos, sin) * (1. / self.gain);
            c[self.count] = t;
            r[self.count] = (self.omega / self.sps - 1.) * 1e6;
            e[self.count] = error;
            self.count += 1;
            self.previous = current;
            self.previous_t = t;
            self.next = t + (self.omega + self.kp * error);
        }
        Ok((
            self.count,
            self.next,
            self.previous.re,
            self.previous.im,
            self.previous_t,
            self.power,
            self.omega,
            self.rate_limit_events,
            self.error_clip_events,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn interpolates_cubic_and_rejects_unsafe_edges() {
        let z: Vec<_> = (0..16)
            .map(|k| {
                let t = k as f64;
                Complex64::new(t * t * t, 2. * t * t + 1.)
            })
            .collect();
        let t = 6.375;
        let y = cubic(&z, 100., 100. + t).unwrap();
        assert!((y.re - t * t * t).abs() < 1e-12);
        assert!((y.im - (2. * t * t + 1.)).abs() < 1e-12);
        // Valid call sites retain one past and two future filtered samples.
        assert!(cubic(&z, 100., 100.5).is_err());
        assert!(cubic(&z, 100., 114.).is_err());
    }
}
