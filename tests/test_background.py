import numpy as np

from urdr import (
    EmpiricalBackgroundConfig,
    HarveyBackgroundConfig,
    RunsBackgroundConfig,
    estimate_background,
    estimate_empirical_background,
    estimate_harvey_background,
    estimate_runs_background,
    whiten_spectrum,
)


def test_runs_background_matches_idun_and_supports_dc_bin() -> None:
    """Urdr delegates positive frequencies to Idun and fills the DC bin."""
    from idun import fit_background

    rng = np.random.default_rng(918)
    frequency = np.linspace(0.0, 3000.0, 30_001)
    expected = 1.0 + 40.0 / (1.0 + (frequency / 250.0) ** 2)
    power = rng.exponential(expected)

    estimated = estimate_runs_background(frequency, power)
    direct = fit_background(frequency[1:], power[1:])

    np.testing.assert_allclose(estimated[1:], direct)
    assert estimated[0] == estimated[1]


def test_runs_background_is_default_whitening_estimator() -> None:
    """Default whitening uses the runs-informed configuration."""
    frequency = np.linspace(0.0, 2000.0, 4097)
    spectrum = np.exp(1j * np.linspace(-np.pi, np.pi, frequency.size))

    _, default = whiten_spectrum(frequency, spectrum)
    explicit = estimate_background(
        frequency,
        np.abs(spectrum) ** 2,
        RunsBackgroundConfig(),
    )

    np.testing.assert_allclose(default, explicit)


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


def test_harvey_background_tracks_lorentzian_profile() -> None:
    rng = np.random.default_rng(456)
    frequency = np.linspace(0.0, 3000.0, 30_001)
    expected = 2.0 + 50.0 / (1.0 + (frequency / 250.0) ** 2)
    power = rng.exponential(expected)

    estimated = estimate_harvey_background(
        frequency,
        power,
        HarveyBackgroundConfig(anchors=120),
    )

    select = (frequency > 50.0) & (frequency < 2800.0)
    median_fractional_error = np.median(
        np.abs(estimated[select] / expected[select] - 1.0)
    )
    assert median_fractional_error < 0.15
