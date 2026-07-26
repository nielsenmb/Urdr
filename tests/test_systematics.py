"""Tests for segment-dependent systematics and their calibrated veto."""

import numpy as np

from urdr import (
    SegmentSystematicConfig,
    SimulationConfig,
    TimeSeries,
    add_segment_systematics,
    benchmark_segment_veto,
    segment_diagnostics,
)


def _template(size: int = 1024) -> TimeSeries:
    """Return a two-segment exact-window test series."""
    time = np.arange(size) * (120.0 / 86400.0)
    observed = np.ones(size, dtype=bool)
    observed[500:524] = False
    rng = np.random.default_rng(3)
    flux = rng.normal(size=size)
    flux[~observed] = np.nan
    return TimeSeries(time, flux, observed)


def test_segment_scale_injection_is_detected_and_preserves_mask() -> None:
    """A variance jump should increase the robust segment-scale ratio."""
    series = _template()
    midpoint = float(series.time[512])
    segments = ((0.0, midpoint), (midpoint, float(series.time[-1]) + 1e-6))
    injected = add_segment_systematics(
        series,
        [SegmentSystematicConfig(midpoint, segments[-1][1], amplitude_scale=4.0)],
    )

    clean = segment_diagnostics(series, segments)
    changed = segment_diagnostics(injected, segments)

    assert changed.maximum_scale_ratio > 2.5 * clean.maximum_scale_ratio
    np.testing.assert_array_equal(injected.observed, series.observed)
    assert np.all(np.isnan(injected.flux[~injected.observed]))


def test_offset_and_drift_have_separate_diagnostics() -> None:
    """Piecewise offsets and drifts should affect their intended summaries."""
    series = _template()
    midpoint = float(series.time[512])
    stop = float(series.time[-1]) + 1e-6
    segments = ((0.0, midpoint), (midpoint, stop))
    changed = add_segment_systematics(
        series,
        [
            SegmentSystematicConfig(
                midpoint,
                stop,
                offset=5.0,
                drift_per_day=100.0,
            )
        ],
    )
    diagnostics = segment_diagnostics(changed, segments)

    assert diagnostics.maximum_location_shift > 2.0
    assert diagnostics.maximum_drift > 1.0


def test_segment_veto_benchmark_is_reproducible() -> None:
    """The paired segment benchmark should reproduce thresholds and rates."""
    template = _template(512)
    midpoint = float(template.time[256])
    stop = float(template.time[-1]) + 1e-6
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
        systematics={
            "variance_jump": [
                SegmentSystematicConfig(
                    midpoint, stop, amplitude_scale=5.0
                )
            ],
        },
        segments_days=((0.0, midpoint), (midpoint, stop)),
        centre_frequencies_uhz=np.array([900.0, 1000.0, 1100.0]),
        filter_width_uhz=500.0,
        delta_nu_grid_uhz=np.array([90.0, 100.0, 110.0]),
        realizations=8,
        target_false_positive_rate=0.25,
        target_signal_retention=0.9,
        max_lag_seconds=15_000.0,
        seed=42,
    )

    first = benchmark_segment_veto(**kwargs)
    second = benchmark_segment_veto(**kwargs)

    assert first == second
    assert 0 <= first.vetoed_signal_detection_rate <= 1
    assert 0 <= first.vetoed_systematic_detection_rate["variance_jump"] <= 1
