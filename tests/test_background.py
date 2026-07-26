import numpy as np

from urdr import (
    EmpiricalBackgroundConfig,
    estimate_empirical_background,
    whiten_spectrum,
)


def test_legacy_background_tracks_smooth_exponential_psd() -> None:
    rng = np.random.default_rng(123)
    frequency = np.linspace(0.0, 3000.0, 30_001)
    expected = 1.0 + 40.0 / (1.0 + (frequency / 250.0) ** 2)
    power = rng.exponential(expected)

    estimated = estimate_empirical_background(frequency, power)

    select = (frequency > 50.0) & (frequency < 2800.0)
    median_fractional_error = np.median(
        np.abs(estimated[select] / expected[select] - 1.0)
    )
    assert median_fractional_error < 0.12


def test_envelope_exclusion_prevents_high_snr_overcorrection() -> None:
    rng = np.random.default_rng(321)
    frequency = np.linspace(0.0, 3000.0, 30_001)
    expected = 1.0 + 30.0 / (1.0 + (frequency / 300.0) ** 2)
    power = rng.exponential(expected)
    power += 150.0 * np.exp(-0.5 * ((frequency - 1000.0) / 120.0) ** 2)

    legacy = estimate_empirical_background(frequency, power)
    target_aware = estimate_empirical_background(
        frequency,
        power,
        EmpiricalBackgroundConfig.excluding_envelope(1000.0, 600.0),
    )
    centre = np.argmin(np.abs(frequency - 1000.0))

    legacy_error = abs(legacy[centre] - expected[centre])
    target_aware_error = abs(target_aware[centre] - expected[centre])
    assert target_aware_error < 0.25 * legacy_error


def test_whitening_preserves_fourier_phase() -> None:
    frequency = np.linspace(0.0, 2000.0, 4097)
    phase = np.linspace(-np.pi, np.pi, frequency.size)
    amplitude = np.sqrt(1.0 + 20.0 / (1.0 + (frequency / 200.0) ** 2))
    spectrum = amplitude * np.exp(1j * phase)

    whitened, background = whiten_spectrum(frequency, spectrum)

    np.testing.assert_allclose(np.angle(whitened[1:]), phase[1:], atol=1e-12)
    assert np.all(np.isfinite(background))
    assert np.all(background > 0)
