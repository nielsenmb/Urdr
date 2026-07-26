"""Empirical spectral-background estimation and whitening."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import least_squares

FloatArray = NDArray[np.float64]
ComplexArray = NDArray[np.complex128]


@dataclass(frozen=True)
class EmpiricalBackgroundConfig:
    """Configuration for the legacy log-frequency running-median background.

    The local half-width is ``a * frequency**b`` in microhertz. Setting an
    exclusion centre and width prevents a predicted oscillation envelope from
    contributing to the running medians; the background is interpolated across
    that region.

    Parameters
    ----------
    a
        Scale coefficient for the local frequency half-width.
    b
        Power-law exponent for the local frequency half-width.
    anchors
        Number of log-spaced background anchors.
    minimum_bins
        Minimum bins required by each local estimate.
    exclude_centre_uhz
        Optional centre of an excluded oscillation envelope.
    exclude_width_uhz
        Optional full width of the excluded envelope.
    """

    a: float = 0.66
    b: float = 0.88
    anchors: int = 100
    minimum_bins: int = 5
    exclude_centre_uhz: float | None = None
    exclude_width_uhz: float | None = None

    def __post_init__(self) -> None:
        """Validate the empirical-background configuration."""
        if self.a <= 0 or self.b <= 0:
            raise ValueError("a and b must be positive")
        if self.anchors < 4:
            raise ValueError("at least four background anchors are required")
        if self.minimum_bins < 1:
            raise ValueError("minimum_bins must be positive")
        supplied = (
            self.exclude_centre_uhz is not None,
            self.exclude_width_uhz is not None,
        )
        if any(supplied) and not all(supplied):
            raise ValueError("exclusion centre and width must be supplied together")
        if all(supplied) and (
            self.exclude_centre_uhz <= 0 or self.exclude_width_uhz <= 0
        ):
            raise ValueError("exclusion centre and width must be positive")

    @classmethod
    def excluding_envelope(
        cls,
        numax_uhz: float,
        envelope_width_uhz: float,
        **kwargs: float | int,
    ) -> "EmpiricalBackgroundConfig":
        """Create a target-aware configuration from seismic predictions.

        Parameters
        ----------
        numax_uhz
            Predicted oscillation-envelope centre in microhertz.
        envelope_width_uhz
            Predicted full envelope width in microhertz.
        **kwargs
            Additional configuration values.

        Returns
        -------
        EmpiricalBackgroundConfig
            Configuration excluding the predicted envelope.
        """
        return cls(
            exclude_centre_uhz=numax_uhz,
            exclude_width_uhz=envelope_width_uhz,
            **kwargs,
        )


@dataclass(frozen=True)
class HarveyBackgroundConfig:
    """Configuration for a fitted one-component Harvey-like background.

    The model is ``white + amplitude / (1 + (frequency / knee)**exponent)``.
    It is deliberately a simple physical baseline against which the empirical
    running-median treatment can be compared.

    Parameters
    ----------
    exponent
        Fixed power-law exponent of the fitted Harvey-like profile.
    anchors
        Number of log-spaced robust background anchors.
    minimum_bins
        Minimum bins required by each anchor.
    exclude_centre_uhz
        Optional centre of an excluded oscillation envelope.
    exclude_width_uhz
        Optional full width of the excluded envelope.
    """

    exponent: float = 2.0
    anchors: int = 100
    minimum_bins: int = 5
    exclude_centre_uhz: float | None = None
    exclude_width_uhz: float | None = None

    def __post_init__(self) -> None:
        """Validate the Harvey-like background configuration."""
        if self.exponent <= 0:
            raise ValueError("exponent must be positive")
        if self.anchors < 4:
            raise ValueError("at least four background anchors are required")
        if self.minimum_bins < 1:
            raise ValueError("minimum_bins must be positive")
        supplied = (
            self.exclude_centre_uhz is not None,
            self.exclude_width_uhz is not None,
        )
        if any(supplied) and not all(supplied):
            raise ValueError("exclusion centre and width must be supplied together")
        if all(supplied) and (
            self.exclude_centre_uhz <= 0 or self.exclude_width_uhz <= 0
        ):
            raise ValueError("exclusion centre and width must be positive")

    @classmethod
    def excluding_envelope(
        cls,
        numax_uhz: float,
        envelope_width_uhz: float,
        **kwargs: float | int,
    ) -> "HarveyBackgroundConfig":
        """Create a target-aware physical-background configuration.

        Parameters
        ----------
        numax_uhz
            Predicted oscillation-envelope centre in microhertz.
        envelope_width_uhz
            Predicted full envelope width in microhertz.
        **kwargs
            Additional configuration values.

        Returns
        -------
        HarveyBackgroundConfig
            Configuration excluding the predicted envelope.
        """
        return cls(
            exclude_centre_uhz=numax_uhz,
            exclude_width_uhz=envelope_width_uhz,
            **kwargs,
        )


BackgroundConfig = EmpiricalBackgroundConfig | HarveyBackgroundConfig


def estimate_empirical_background(
    frequency_uhz: ArrayLike,
    power: ArrayLike,
    config: EmpiricalBackgroundConfig | None = None,
) -> FloatArray:
    """Estimate a smooth mean PSD using log-spaced local running medians.

    Dividing each median by ``log(2)`` converts the median of an exponentially
    distributed periodogram into an estimate of its mean. The value at zero
    frequency is copied from the lowest positive-frequency estimate.

    Parameters
    ----------
    frequency_uhz
        Strictly increasing frequency grid in microhertz.
    power
        Periodogram power on the frequency grid.
    config
        Optional empirical-background configuration.

    Returns
    -------
    numpy.ndarray
        Estimated mean background power on the input grid.
    """
    settings = config or EmpiricalBackgroundConfig()
    frequency = np.asarray(frequency_uhz, dtype=float)
    psd = np.asarray(power, dtype=float)
    if frequency.ndim != 1 or psd.ndim != 1 or frequency.size != psd.size:
        raise ValueError("frequency and power must be equal-length 1D arrays")
    if frequency.size < settings.minimum_bins:
        raise ValueError("spectrum contains too few bins")
    if np.any(~np.isfinite(frequency)) or np.any(np.diff(frequency) <= 0):
        raise ValueError("frequency must be finite and strictly increasing")
    if np.any(~np.isfinite(psd)) or np.any(psd < 0):
        raise ValueError("power must be finite and non-negative")

    positive = frequency > 0
    if np.count_nonzero(positive) < settings.minimum_bins:
        raise ValueError("spectrum contains too few positive-frequency bins")
    eligible = positive.copy()
    if settings.exclude_centre_uhz is not None:
        half_width = settings.exclude_width_uhz / 2.0
        eligible &= (
            np.abs(frequency - settings.exclude_centre_uhz) > half_width
        )
    if np.count_nonzero(eligible) < settings.minimum_bins:
        raise ValueError("background exclusion leaves too few spectral bins")

    positive_frequency = frequency[positive]
    anchors = np.geomspace(
        positive_frequency[0], positive_frequency[-1], settings.anchors
    )
    medians = np.full(anchors.size, np.nan)
    for index, anchor in enumerate(anchors):
        local_half_width = settings.a * anchor**settings.b
        select = eligible & (np.abs(frequency - anchor) < local_half_width)
        if np.count_nonzero(select) >= settings.minimum_bins:
            medians[index] = np.median(psd[select]) / np.log(2.0)

    valid = np.isfinite(medians) & (medians > 0)
    if np.count_nonzero(valid) < 2:
        raise ValueError("could not estimate the empirical background")
    background = np.interp(frequency, anchors[valid], medians[valid])
    floor = np.finfo(float).tiny
    return np.maximum(background, floor)


def estimate_harvey_background(
    frequency_uhz: ArrayLike,
    power: ArrayLike,
    config: HarveyBackgroundConfig | None = None,
) -> FloatArray:
    """Fit a one-component Harvey-like profile to robust PSD anchors.

    Parameters
    ----------
    frequency_uhz
        Strictly increasing frequency grid in microhertz.
    power
        Periodogram power on the frequency grid.
    config
        Optional Harvey-like background configuration.

    Returns
    -------
    numpy.ndarray
        Fitted background power on the input grid.
    """
    settings = config or HarveyBackgroundConfig()
    frequency, psd = _validate_spectrum(
        frequency_uhz, power, settings.minimum_bins
    )
    anchors, medians = _log_median_anchors(
        frequency,
        psd,
        settings.anchors,
        settings.minimum_bins,
        settings.exclude_centre_uhz,
        settings.exclude_width_uhz,
    )
    valid = np.isfinite(medians) & (medians > 0)
    if np.count_nonzero(valid) < 4:
        raise ValueError("could not estimate the Harvey-like background")
    fit_frequency = anchors[valid]
    fit_power = medians[valid]
    floor = np.finfo(float).tiny

    high = fit_power[fit_frequency >= np.quantile(fit_frequency, 0.8)]
    white_initial = max(float(np.median(high)), floor)
    amplitude_initial = max(float(np.max(fit_power) - white_initial), white_initial)
    knee_initial = float(np.median(fit_frequency))

    def residual(log_parameters: FloatArray) -> FloatArray:
        white, amplitude, knee = np.exp(log_parameters)
        model = white + amplitude / (
            1.0 + (fit_frequency / knee) ** settings.exponent
        )
        return np.log(np.maximum(model, floor)) - np.log(fit_power)

    lower = np.log(
        [
            max(float(np.min(fit_power)) * 1e-4, floor),
            max(float(np.min(fit_power)) * 1e-4, floor),
            max(float(fit_frequency[0]) * 0.1, floor),
        ]
    )
    upper = np.log(
        [
            float(np.max(fit_power)) * 10.0,
            float(np.max(fit_power)) * 100.0,
            float(fit_frequency[-1]) * 10.0,
        ]
    )
    result = least_squares(
        residual,
        np.log([white_initial, amplitude_initial, knee_initial]),
        bounds=(lower, upper),
    )
    white, amplitude, knee = np.exp(result.x)
    background = white + amplitude / (
        1.0 + (frequency / knee) ** settings.exponent
    )
    return np.maximum(background, floor)


def estimate_background(
    frequency_uhz: ArrayLike,
    power: ArrayLike,
    config: BackgroundConfig,
) -> FloatArray:
    """Estimate a PSD background using the requested comparison arm.

    Parameters
    ----------
    frequency_uhz
        Strictly increasing frequency grid in microhertz.
    power
        Periodogram power on the frequency grid.
    config
        Empirical or Harvey-like background configuration.

    Returns
    -------
    numpy.ndarray
        Estimated background power on the input grid.
    """
    if isinstance(config, EmpiricalBackgroundConfig):
        return estimate_empirical_background(frequency_uhz, power, config)
    if isinstance(config, HarveyBackgroundConfig):
        return estimate_harvey_background(frequency_uhz, power, config)
    raise TypeError(f"unsupported background configuration: {type(config)!r}")


def whiten_spectrum(
    frequency_uhz: ArrayLike,
    spectrum: ArrayLike,
    config: BackgroundConfig | None = None,
) -> tuple[ComplexArray, FloatArray]:
    """Flatten a complex Fourier spectrum while retaining its phases.

    Parameters
    ----------
    frequency_uhz
        Strictly increasing frequency grid in microhertz.
    spectrum
        Complex Fourier spectrum.
    config
        Optional empirical or Harvey-like background configuration.

    Returns
    -------
    whitened
        Whitened complex Fourier spectrum.
    background
        Estimated background power on the input grid.
    """
    complex_spectrum = np.asarray(spectrum, dtype=complex)
    settings = config or EmpiricalBackgroundConfig()
    background = estimate_background(
        frequency_uhz, np.abs(complex_spectrum) ** 2, settings
    )
    whitened = complex_spectrum / np.sqrt(background)
    return np.asarray(whitened, dtype=complex), background


def _validate_spectrum(
    frequency_uhz: ArrayLike,
    power: ArrayLike,
    minimum_bins: int,
) -> tuple[FloatArray, FloatArray]:
    frequency = np.asarray(frequency_uhz, dtype=float)
    psd = np.asarray(power, dtype=float)
    if frequency.ndim != 1 or psd.ndim != 1 or frequency.size != psd.size:
        raise ValueError("frequency and power must be equal-length 1D arrays")
    if frequency.size < minimum_bins:
        raise ValueError("spectrum contains too few bins")
    if np.any(~np.isfinite(frequency)) or np.any(np.diff(frequency) <= 0):
        raise ValueError("frequency must be finite and strictly increasing")
    if np.any(~np.isfinite(psd)) or np.any(psd < 0):
        raise ValueError("power must be finite and non-negative")
    if np.count_nonzero(frequency > 0) < minimum_bins:
        raise ValueError("spectrum contains too few positive-frequency bins")
    return frequency, psd


def _log_median_anchors(
    frequency: FloatArray,
    psd: FloatArray,
    anchors_count: int,
    minimum_bins: int,
    exclude_centre_uhz: float | None,
    exclude_width_uhz: float | None,
) -> tuple[FloatArray, FloatArray]:
    positive = frequency > 0
    eligible = positive.copy()
    if exclude_centre_uhz is not None:
        eligible &= (
            np.abs(frequency - exclude_centre_uhz)
            > float(exclude_width_uhz) / 2.0
        )
    if np.count_nonzero(eligible) < minimum_bins:
        raise ValueError("background exclusion leaves too few spectral bins")
    positive_frequency = frequency[positive]
    anchors = np.geomspace(
        positive_frequency[0], positive_frequency[-1], anchors_count
    )
    edges = np.geomspace(
        positive_frequency[0],
        positive_frequency[-1],
        anchors_count + 1,
    )
    medians = np.full(anchors.size, np.nan)
    for index in range(anchors.size):
        select = (
            eligible
            & (frequency >= edges[index])
            & (frequency <= edges[index + 1])
        )
        if np.count_nonzero(select) >= minimum_bins:
            medians[index] = np.median(psd[select]) / np.log(2.0)
    return anchors, medians
