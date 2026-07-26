"""Tests for coherent-contaminant simulations and diagnostics."""

import numpy as np

from urdr import (
    CoherentSignalConfig,
    SimulationConfig,
    TimeSeries,
    add_coherent_signal,
    benchmark_coherent_veto,
    coherence_diagnostics,
)


def _template(size: int = 2048) -> TimeSeries:
    """Return a short exact-window test series."""

    time = np.arange(size) * (120.0 / 86400.0)
    observed = np.ones(size, dtype=bool)
    observed[500:560] = False
    return TimeSeries(time, np.zeros(size), observed)


def test_coherent_signal_reaches_requested_rms_and_preserves_mask() -> None:
    """A coherent injection should have the requested observed RMS."""

    template = _template()
    config = CoherentSignalConfig(
        frequency_uhz=1000.0,
        amplitude=2.0,
        harmonics=2,
    )

    injected = add_coherent_signal(template, config, np.random.default_rng(4))

    np.testing.assert_allclose(
        np.std(injected.flux[injected.observed]), 2.0, rtol=0.03
    )
    assert np.all(np.isnan(injected.flux[~injected.observed]))


def test_coherent_signal_is_more_concentrated_than_mode_comb() -> None:
    """A sinusoid should occupy fewer Fourier bins than a stochastic comb."""

    template = _template(4096)
    base = SimulationConfig(
        white_noise_sigma=0.05,
        numax_uhz=1000.0,
        delta_nu_uhz=100.0,
        envelope_width_uhz=400.0,
        oscillation_amplitude=2.0,
    )
    from urdr import simulate_time_series

    modes = simulate_time_series(template.window, base, np.random.default_rng(5))
    noise = simulate_time_series(
        template.window, base, np.random.default_rng(5), False
    )
    coherent = add_coherent_signal(
        noise,
        CoherentSignalConfig(1000.0, 2.0),
        np.random.default_rng(6),
    )

    mode_diagnostics = coherence_diagnostics(modes, 1000.0, 500.0)
    coherent_diagnostics = coherence_diagnostics(coherent, 1000.0, 500.0)

    assert (
        coherent_diagnostics.maximum_bin_fraction
        > mode_diagnostics.maximum_bin_fraction
    )
    assert coherent_diagnostics.effective_bins < mode_diagnostics.effective_bins


def test_coherent_veto_benchmark_is_reproducible() -> None:
    """The paired benchmark should reproduce all thresholds and rates."""

    template = _template(512)
    simulation = SimulationConfig(
        white_noise_sigma=0.2,
        numax_uhz=1000.0,
        delta_nu_uhz=100.0,
        envelope_width_uhz=350.0,
        oscillation_amplitude=1.5,
    )
    kwargs = dict(
        window=template.window,
        simulation=simulation,
        contaminants={
            "coherent": CoherentSignalConfig(1000.0, 1.5),
        },
        centre_frequencies_uhz=np.array([900.0, 1000.0, 1100.0]),
        filter_width_uhz=500.0,
        delta_nu_grid_uhz=np.array([90.0, 100.0, 110.0]),
        realizations=8,
        target_false_positive_rate=0.25,
        target_signal_retention=0.9,
        max_lag_seconds=15_000.0,
        seed=42,
    )

    first = benchmark_coherent_veto(**kwargs)
    second = benchmark_coherent_veto(**kwargs)

    assert first == second
    assert 0 <= first.vetoed_signal_detection_rate <= 1
    assert 0 <= first.vetoed_contaminant_detection_rate["coherent"] <= 1
