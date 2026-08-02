import numpy as np
import pytest

from urdr import AsteroScaleSamples, ObservingWindow, TimeSeries


def test_time_series_infers_missing_cadences() -> None:
    time = np.arange(16) / 48
    flux = np.arange(16, dtype=float)
    flux[[3, 7]] = np.nan
    series = TimeSeries.from_arrays(time, flux)
    assert series.observed.sum() == 14
    assert series.window.duty_cycle == pytest.approx(14 / 16)


def test_window_has_alias_power_for_regular_gaps() -> None:
    time = np.arange(128) / 48
    observed = np.ones(128, dtype=bool)
    observed[::4] = False
    window = ObservingWindow(time, observed)
    gap_frequency_uhz = 1e6 / (4 * (86400 / 48))
    assert window.spectral_power([gap_frequency_uhz])[0] > 0.05


def test_asteroscale_adapter_preserves_sample_correlations() -> None:
    numax = np.array([95.0, 100.0, 110.0])
    samples = AsteroScaleSamples(numax, 0.1 * numax, np.full(3, 30.0))
    region = samples.search_region(credible_mass=0.8)
    assert region.minimum_uhz < numax.min()
    assert region.maximum_uhz > numax.max()
    assert region.centre_uhz == pytest.approx(100.0)


def test_asteroscale_adapter_accepts_public_output_names_and_scalars() -> None:
    """AsteroScale's current dnu/FWHM_env names need no manual renaming."""
    samples = AsteroScaleSamples.from_mapping(
        {"numax": 1000.0, "dnu": 55.0, "FWHM_env": 300.0, "A_gran": 20.0}
    )
    assert np.array_equal(samples.numax_uhz, [1000.0])
    assert np.array_equal(samples.delta_nu_uhz, [55.0])
    assert np.array_equal(samples.envelope_width_uhz, [300.0])
    assert np.array_equal(samples.granulation_amplitude, [20.0])
