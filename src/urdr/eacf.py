"""Filtered time-series autocorrelation statistic."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.signal.windows import tukey

from .background import BackgroundConfig, whiten_spectrum
from .models import TimeSeries

FloatArray = NDArray[np.float64]
ComplexArray = NDArray[np.complex128]


@dataclass(frozen=True)
class EACFMap:
    """EACF values evaluated over filter centres and time lags."""

    centre_frequencies_uhz: FloatArray
    lags_seconds: FloatArray
    values: FloatArray

    def statistic(
        self,
        delta_nu_uhz: float,
        relative_half_width: float = 0.05,
    ) -> float:
        """Return maximum EACF power near the expected ``1 / delta_nu`` lag."""

        expected = 1e6 / float(delta_nu_uhz)
        select = np.abs(self.lags_seconds - expected) <= expected * relative_half_width
        if not np.any(select):
            nearest = int(np.argmin(np.abs(self.lags_seconds - expected)))
            select = np.arange(self.lags_seconds.size) == nearest
        return float(np.max(self.values[:, select]))


def compute_eacf(
    series: TimeSeries,
    centre_frequency_uhz: float,
    filter_width_uhz: float,
    max_lag_seconds: float | None = None,
    empirical_background: BackgroundConfig | None = None,
) -> tuple[FloatArray, FloatArray]:
    """Compute a normalised ACF of a frequency-filtered time series.

    A Tukey bandpass is applied in the Fourier domain. Missing cadences are
    zero-filled only after robust centring and the pair count at every lag is used
    to correct the resulting autocorrelation. If ``empirical_background`` is
    supplied, the complex spectrum is whitened before applying the bandpass.
    """

    if centre_frequency_uhz <= 0 or filter_width_uhz <= 0:
        raise ValueError("filter centre and width must be positive")
    frequencies_uhz, spectrum = _prepare_spectrum(series, empirical_background)
    return _compute_from_spectrum(
        series,
        frequencies_uhz,
        spectrum,
        centre_frequency_uhz,
        filter_width_uhz,
        max_lag_seconds,
    )


def _prepare_spectrum(
    series: TimeSeries,
    empirical_background: BackgroundConfig | None,
) -> tuple[FloatArray, ComplexArray]:
    signal = np.zeros(series.time.size, dtype=float)
    observed_flux = series.flux[series.observed]
    signal[series.observed] = observed_flux - np.median(observed_flux)

    frequencies_uhz = (
        np.fft.rfftfreq(signal.size, d=series.cadence_seconds) * 1e6
    )
    spectrum = np.fft.rfft(signal)
    if empirical_background is not None:
        spectrum, _ = whiten_spectrum(frequencies_uhz, spectrum, empirical_background)
    return frequencies_uhz, np.asarray(spectrum, dtype=complex)


def _compute_from_spectrum(
    series: TimeSeries,
    frequencies_uhz: FloatArray,
    spectrum: ComplexArray,
    centre_frequency_uhz: float,
    filter_width_uhz: float,
    max_lag_seconds: float | None,
) -> tuple[FloatArray, FloatArray]:
    taper = _frequency_taper(
        frequencies_uhz, centre_frequency_uhz, filter_width_uhz
    )
    filtered = np.fft.irfft(spectrum * taper, n=series.time.size)
    filtered[~series.observed] = 0.0

    raw = np.correlate(filtered, filtered, mode="full")[series.time.size - 1 :]
    pairs = np.correlate(
        series.observed.astype(float),
        series.observed.astype(float),
        mode="full",
    )[series.time.size - 1 :]
    valid = pairs > 0
    acovariance = np.zeros_like(raw)
    acovariance[valid] = raw[valid] / pairs[valid]
    if acovariance[0] <= 0:
        raise ValueError("filtered time series has zero variance")
    values = np.square(acovariance / acovariance[0])
    lags = np.arange(series.time.size, dtype=float) * series.cadence_seconds

    if max_lag_seconds is not None:
        keep = lags <= max_lag_seconds
        lags, values = lags[keep], values[keep]
    return lags, values


def compute_eacf_map(
    series: TimeSeries,
    centre_frequencies_uhz: ArrayLike,
    filter_width_uhz: float,
    max_lag_seconds: float | None = None,
    empirical_background: BackgroundConfig | None = None,
) -> EACFMap:
    """Evaluate the filtered ACF across trial filter-centre frequencies."""

    centres = np.atleast_1d(np.asarray(centre_frequencies_uhz, dtype=float))
    if centres.ndim != 1 or centres.size == 0:
        raise ValueError("at least one filter centre is required")
    frequencies_uhz, spectrum = _prepare_spectrum(series, empirical_background)
    rows = []
    lags = None
    for centre in centres:
        lags, values = _compute_from_spectrum(
            series,
            frequencies_uhz,
            spectrum,
            float(centre),
            filter_width_uhz,
            max_lag_seconds,
        )
        rows.append(values)
    assert lags is not None
    return EACFMap(centres, lags, np.vstack(rows))


def _frequency_taper(
    frequency_uhz: FloatArray,
    centre_uhz: float,
    width_uhz: float,
) -> FloatArray:
    lower = centre_uhz - width_uhz / 2.0
    upper = centre_uhz + width_uhz / 2.0
    inside = (frequency_uhz >= lower) & (frequency_uhz <= upper)
    weights = np.zeros_like(frequency_uhz)
    count = int(inside.sum())
    if count < 3:
        raise ValueError("filter contains fewer than three Fourier bins")
    weights[inside] = tukey(count, alpha=0.5)
    return weights
