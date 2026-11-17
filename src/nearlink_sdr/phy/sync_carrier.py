"""Bounded known-sync carrier initializer experiment (§6.2.1.2/6.2.3.2).

This rejected development candidate is not a default receiver backend. Data,
moments, division and phase remain floating; only optional NCO coefficients are
finite words. No expected payload, CRC or truth enters estimation.
"""

import math

import numpy as np

from nearlink_sdr.phy.sync_coherent import SyncConfirmationNCO

__all__ = ["initialize_sync_carrier", "sync_carrier_work"]


def _rotation(omega, count, fixed):
    if not fixed:
        return np.exp(-1j*omega*np.arange(count))
    node = SyncConfirmationNCO(round(float(omega)/(2*np.pi)*(1 << 20)) & ((1 << 20)-1))
    real, imag = node.advance(count)
    return (real+1j*imag)/65536


def initialize_sync_carrier(samples, reference, seed, iterations, *, fixed_nco=False):
    """One/two capped Newton steps, then sync-only phase; no phase unwrapping.

    Keep the input scale out of fourth-order moments. Component division avoids
    NumPy's reciprocal-first complex division overflowing on subnormal input.
    This floating normalization does not create ADC precision or improve SIR.
    """
    y, ref = np.asarray(samples, dtype=complex), np.asarray(reference, dtype=complex)
    if (y.ndim != 1 or y.shape != ref.shape or not 2 <= len(y) <= 128
            or not np.isfinite(y).all() or not np.isfinite(ref).all()
            or np.any(abs(ref.real) > 1+1e-12) or np.any(abs(ref.imag) > 1+1e-12)
            or not np.allclose(ref.real**2+ref.imag**2, 1., atol=2e-12, rtol=0)
            or isinstance(seed, (bool, np.bool_))
            or not isinstance(seed, (int, float, np.integer, np.floating))
            or not np.isfinite(seed) or abs(seed) > math.pi
            or type(iterations) is not int or iterations not in (1, 2)
            or type(fixed_nco) is not bool):
        raise ValueError("Need2..128 finite PSK sync symbols, principal CFO and1/2 fixed steps")
    scale = max(float(np.max(abs(y.real))), float(np.max(abs(y.imag))))
    if scale == 0:
        return dict(initialized=False, reason="zero_energy")
    y = y.real/scale+1j*(y.imag/scale)
    n = len(y)
    centered = np.arange(n)-(n-1)/2
    matched = y*ref.conj()
    energy = float(np.vdot(y, y).real*np.vdot(ref, ref).real)
    floor = 1e-12*energy*(n-1)**2
    omega, updates = float(seed), []
    for _ in range(iterations):
        z = matched*_rotation(omega, n, fixed_nco)
        a, b, c = np.sum(z), np.dot(centered, z), np.dot(centered**2, z)
        gradient = 2*float((b*a.conjugate()).imag)
        curvature = 2*float(b.real**2+b.imag**2-(c*a.conjugate()).real)
        if curvature < -floor:
            raw_step = -gradient/curvature
            step = float(np.clip(raw_step, -math.pi/n, math.pi/n))
            omega = (omega+step+math.pi) % (2*math.pi)-math.pi
            disposition = "step_clipped" if step != raw_step else "updated"
        else:
            step, disposition = 0., "curvature_hold"
        updates.append(dict(gradient=gradient, curvature=curvature, step=step,
                            disposition=disposition))
    final = np.sum(matched*_rotation(omega, n, fixed_nco))
    final_energy = float(final.real**2+final.imag**2)
    if final_energy == 0:
        return dict(initialized=False, reason="zero_coherent_sum", updates=updates)
    return dict(
        initialized=True, omega_radians_per_symbol=omega,
        phase_radians=float(np.angle(final)), seed_radians_per_symbol=float(seed),
        iterations=iterations, updates=updates, curvature_floor=floor,
        quality=final_energy/energy, input_component_normalization=scale,
        step_cap_radians=math.pi/n, fixed_nco=fixed_nco,
        method="known-sync bounded coherent Newton;no unwrap/CRC/truth/payload",
        whole_system_accepted=False,
    )


def sync_carrier_work(iterations, symbols=32):
    """Literal finite graph; no unproved CSD/reciprocal/NCO sharing credit."""
    if type(iterations) is not int or iterations not in (1, 2) or symbols not in (32, 62, 126, 128):
        raise ValueError("Need1/2 steps and an explicit sync length")
    products = dict(template=4*symbols, rotations=4*symbols*(iterations+1),
                    constant_moments=4*symbols*iterations, gradient_curvature=6*iterations,
                    input_energy=2*symbols, template_validation_squares=2*symbols,
                    energy_denominator=1)
    nco = SyncConfirmationNCO.estimate()
    return dict(
        symbols=symbols, iterations=iterations, nco_passes=iterations+1,
        literal_real_product_equivalents=sum(products.values()), products=products,
        literal_data_divisions=2*symbols, newton_divisions=iterations,
        diagnostic_quality_divisions=1, phase_atan2=1,
        old_phase_atan2_per_sync=symbols, sqrt=0,
        complex_sum_real_adds=2*(symbols-1)*(3*iterations+1),
        moment_constants="centered half-integer k and squared k;shift/add allocation unimplemented",
        comparisons_lower_bound=2*symbols+3*iterations,
        dedicated_nco_bits=nco["logical_bits"],
        nco_physical_estimate="one additional RAMB18 with two read ports;not synthesized",
        extra_symbol_and_accumulator_word_bits=None,
        scratch="bounded N<=128;input/matched/rotated vectors and six moment components",
        normalization="literal component division;reciprocal/common-exponent alternative not validated",
        data_word_formats_verified=False, whole_system_accepted=False,
    )
