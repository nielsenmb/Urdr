"""Published Lomb--Scargle EACF workflow and exact-window calibration."""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
from mimir import power_spectrum
from numpy.typing import ArrayLike, NDArray
from scipy.fft import next_fast_len

from .background import BackgroundConfig, EmpiricalBackgroundConfig, estimate_background
from .models import TimeSeries
from .simulation import SimulationConfig, simulate_time_series

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class DeltaNuScaling:
    """Power-law physical relation between ``numax`` and ``delta_nu``.

    Parameters
    ----------
    solar_numax_uhz
        Solar reference frequency of maximum oscillation power.
    solar_delta_nu_uhz
        Solar reference large separation.
    exponent
        Exponent of the normalized power law.
    scatter_factor
        Multiplicative half-width of the accepted relation. The published
        value ``10**0.2`` reproduces the banana-shaped search region.
    """

    solar_numax_uhz: float = 3090.0
    solar_delta_nu_uhz: float = 135.1
    exponent: float = 0.791
    scatter_factor: float = 10**0.2

    def __post_init__(self) -> None:
        """Validate the scaling-relation parameters."""
        if min(self.solar_numax_uhz, self.solar_delta_nu_uhz) <= 0:
            raise ValueError("solar reference frequencies must be positive")
        if self.exponent <= 0 or self.scatter_factor <= 1:
            raise ValueError("exponent must be positive and scatter_factor exceed one")

    def predict(self, numax_uhz: ArrayLike) -> FloatArray:
        """Predict ``delta_nu`` in microhertz at each trial ``numax``."""
        numax = np.asarray(numax_uhz, dtype=float)
        if np.any(~np.isfinite(numax)) or np.any(numax <= 0):
            raise ValueError("numax values must be finite and positive")
        return np.asarray(
            self.solar_delta_nu_uhz
            * (numax / self.solar_numax_uhz) ** self.exponent,
            dtype=float,
        )


@dataclass(frozen=True)
class PublishedEACFMap:
    """Published complex-modulus EACF evaluated over frequency and lag.

    Parameters
    ----------
    centre_frequencies_uhz
        Trial filter centres, interpreted as candidate ``numax`` values.
    lags_seconds
        Non-negative lag grid.
    values
        Squared complex EACF magnitude, normalized to unity at zero lag.
    predicted_delta_nu_uhz
        Scaling-relation prediction at each filter centre.
    physical_mask
        Boolean mask selecting physically plausible ``numax``--``delta_nu``
        combinations.
    """

    centre_frequencies_uhz: FloatArray
    lags_seconds: FloatArray
    values: FloatArray
    predicted_delta_nu_uhz: FloatArray
    physical_mask: NDArray[np.bool_]

    def collapsed(self) -> FloatArray:
        """Return the mean EACF response inside the physical search band."""
        sums = np.sum(np.where(self.physical_mask, self.values, 0.0), axis=1)
        counts = np.sum(self.physical_mask, axis=1)
        if np.any(counts == 0):
            raise ValueError("physical search band contains no lag samples")
        return np.asarray(sums / counts, dtype=float)

    @property
    def global_statistic(self) -> float:
        """Return the maximum collapsed response across all trial centres."""
        return float(np.max(self.collapsed()))

    @property
    def best_numax_uhz(self) -> float:
        """Return the filter centre with the largest collapsed response."""
        return float(self.centre_frequencies_uhz[np.argmax(self.collapsed())])

    @property
    def best_delta_nu_uhz(self) -> float:
        """Return the strongest lag within the best physical search row."""
        row = int(np.argmax(self.collapsed()))
        selected = np.flatnonzero(self.physical_mask[row])
        lag_index = selected[np.argmax(self.values[row, selected])]
        return float(1e6 / self.lags_seconds[lag_index])


@dataclass(frozen=True)
class PublishedEACFDetection:
    """Observed published EACF and its exact-window null calibration."""

    observed: PublishedEACFMap
    null_statistics: FloatArray

    @property
    def false_alarm_probability(self) -> float:
        """Return the finite-sample global false-alarm probability."""
        exceedances = np.count_nonzero(
            self.null_statistics >= self.observed.global_statistic
        )
        return float((exceedances + 1) / (self.null_statistics.size + 1))

    @property
    def detection_merit(self) -> float:
        """Return one minus the globally calibrated false-alarm probability."""
        return 1.0 - self.false_alarm_probability


def envelope_width_uhz(numax_uhz: ArrayLike) -> FloatArray:
    """Return the published cool-star p-mode-envelope width relation."""
    numax = np.asarray(numax_uhz, dtype=float)
    if np.any(~np.isfinite(numax)) or np.any(numax <= 0):
        raise ValueError("numax values must be finite and positive")
    return np.asarray(0.66 * numax**0.88, dtype=float)


def compute_published_eacf_map(
    series: TimeSeries,
    centre_frequencies_uhz: ArrayLike,
    *,
    filter_widths_uhz: ArrayLike | None = None,
    max_lag_seconds: float | None = None,
    background: BackgroundConfig | None = None,
    scaling: DeltaNuScaling | None = None,
    spectrum_oversampling: int = 1,
    lag_oversampling: int = 2,
) -> PublishedEACFMap:
    """Compute the published Lomb--Scargle, Hanning-filtered EACF map.

    Mimir evaluates a critically sampled Lomb--Scargle power-density spectrum
    using only genuinely observed cadences. Urdr divides it by an empirical
    background, applies a Hanning filter at each trial centre, and computes the
    squared modulus of the complex inverse transform. The modulus removes the
    rapid carrier visible in a squared real ACF.

    Parameters
    ----------
    series
        Uniform cadence grid and exact observing mask. Only observed samples
        are supplied to Mimir.
    centre_frequencies_uhz
        Trial filter centres in microhertz.
    filter_widths_uhz
        Optional scalar or one width per centre. By default the published
        p-mode-envelope width scaling is used.
    max_lag_seconds
        Optional largest returned lag.
    background
        Background estimator. The legacy empirical estimator is the default.
    scaling
        Physical ``numax``--``delta_nu`` relation and accepted scatter.
    spectrum_oversampling
        Mimir Lomb--Scargle frequency oversampling.
    lag_oversampling
        Zero-padding factor used to refine the inverse-transform lag grid.

    Returns
    -------
    PublishedEACFMap
        Smooth complex-modulus EACF map and physical search mask.
    """
    centres = np.atleast_1d(np.asarray(centre_frequencies_uhz, dtype=float))
    if centres.ndim != 1 or centres.size == 0:
        raise ValueError("at least one filter centre is required")
    if np.any(~np.isfinite(centres)) or np.any(centres <= 0):
        raise ValueError("filter centres must be finite and positive")
    if not isinstance(lag_oversampling, int) or lag_oversampling < 1:
        raise ValueError("lag_oversampling must be a positive integer")

    if filter_widths_uhz is None:
        widths = envelope_width_uhz(centres)
    else:
        widths = np.asarray(filter_widths_uhz, dtype=float)
        if widths.ndim == 0:
            widths = np.full(centres.size, float(widths))
        if widths.shape != centres.shape:
            raise ValueError("filter widths must be scalar or match filter centres")
    if np.any(~np.isfinite(widths)) or np.any(widths <= 0):
        raise ValueError("filter widths must be finite and positive")

    observed_time = series.time[series.observed]
    observed_flux = series.flux[series.observed]
    spectrum = power_spectrum(
        time=observed_time,
        flux=observed_flux,
        oversampling=spectrum_oversampling,
        time_unit="d",
        frequency_unit="uHz",
    )
    frequency = np.asarray(spectrum.frequency, dtype=float)
    psd = np.asarray(spectrum.power_density, dtype=float)
    settings = background or EmpiricalBackgroundConfig()
    noise = estimate_background(frequency, psd, settings)
    signal_to_noise = psd / noise

    # Prefix the positive-frequency spectrum with its missing zero-frequency
    # bin. The phase of the one-sided inverse transform is intentionally kept;
    # taking its magnitude below removes the filter-centre carrier.
    snr = np.concatenate(([0.0], signal_to_noise))
    transform_size = next_fast_len(lag_oversampling * snr.size)
    spacing_hz = float(spectrum.frequency_spacing) * 1e-6
    lags = np.arange(transform_size, dtype=float) / (transform_size * spacing_hz)
    positive_lag = lags <= 0.5 / spacing_hz
    if max_lag_seconds is not None:
        if not np.isfinite(max_lag_seconds) or max_lag_seconds <= 0:
            raise ValueError("max_lag_seconds must be finite and positive")
        positive_lag &= lags <= max_lag_seconds
    lags = lags[positive_lag]

    rows = []
    frequency_with_zero = np.concatenate(([0.0], frequency))
    for centre, width in zip(centres, widths, strict=True):
        weights = _hanning_filter(frequency_with_zero, centre, width)
        acf = np.fft.ifft(snr * weights, n=transform_size)[positive_lag]
        zero_power = float(np.abs(acf[0]) ** 2)
        if zero_power <= 0:
            raise ValueError("filtered signal-to-noise spectrum has zero power")
        rows.append(np.asarray(np.abs(acf) ** 2 / zero_power, dtype=float))

    relation = scaling or DeltaNuScaling()
    predicted = relation.predict(centres)
    expected_lag = 1e6 / predicted
    lower = expected_lag / relation.scatter_factor
    upper = expected_lag * relation.scatter_factor
    physical = (lags[None, :] >= lower[:, None]) & (lags[None, :] <= upper[:, None])
    if np.any(np.sum(physical, axis=1) == 0):
        raise ValueError("lag grid does not cover the physical search relation")
    return PublishedEACFMap(
        centres,
        lags,
        np.vstack(rows),
        predicted,
        physical,
    )


def calibrate_published_eacf(
    series: TimeSeries,
    simulation: SimulationConfig,
    centre_frequencies_uhz: ArrayLike,
    *,
    simulations: int = 128,
    seed: int = 0,
    **map_kwargs: object,
) -> PublishedEACFDetection:
    """Calibrate the global published-EACF statistic for one target window.

    Each null realization uses the target's exact cadence grid and observing
    mask. Its granulation background is re-estimated and the maximum collapsed
    response over the full physical search region is retained, incorporating
    both the observing window and the look-elsewhere effect.
    """
    if simulations < 8:
        raise ValueError("at least eight null simulations are required")
    observed = compute_published_eacf_map(
        series, centre_frequencies_uhz, **map_kwargs
    )
    null = np.empty(simulations, dtype=float)
    null_config = replace(simulation, oscillation_amplitude=0.0)
    seeds = np.random.SeedSequence(seed).spawn(simulations)
    for index, child_seed in enumerate(seeds):
        candidate = simulate_time_series(
            series.window,
            null_config,
            np.random.default_rng(child_seed),
            include_oscillations=False,
        )
        null[index] = compute_published_eacf_map(
            candidate, centre_frequencies_uhz, **map_kwargs
        ).global_statistic
    return PublishedEACFDetection(observed, null)


def _hanning_filter(
    frequency_uhz: FloatArray,
    centre_uhz: float,
    width_uhz: float,
) -> FloatArray:
    """Return the Eq. 15 Hanning filter on a frequency grid."""
    offset = frequency_uhz - centre_uhz
    inside = np.abs(offset) <= width_uhz / 2.0
    if np.count_nonzero(inside) < 3:
        raise ValueError("Hanning filter contains fewer than three bins")
    weights = np.zeros_like(frequency_uhz)
    weights[inside] = 0.5 + 0.5 * np.cos(2.0 * np.pi * offset[inside] / width_uhz)
    return weights
