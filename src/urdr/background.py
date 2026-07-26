"""Empirical spectral-background estimation and whitening."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]
ComplexArray = NDArray[np.complex128]


@dataclass(frozen=True)
class EmpiricalBackgroundConfig:
    """Configuration for the legacy log-frequency running-median background.

    The local half-width is ``a * frequency**b`` in microhertz. Setting an
    exclusion centre and width prevents a predicted oscillation envelope from
    contributing to the running medians; the background is interpolated across
    that region.
    """

    a: float = 0.66
    b: float = 0.88
    anchors: int = 100
    minimum_bins: int = 5
    exclude_centre_uhz: float | None = None
    exclude_width_uhz: float | None = None

    def __post_init__(self) -> None:
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
        """Create a target-aware configuration from seismic predictions."""

        return cls(
            exclude_centre_uhz=numax_uhz,
            exclude_width_uhz=envelope_width_uhz,
            **kwargs,
        )


def estimate_empirical_background(
    frequency_uhz: ArrayLike,
    power: ArrayLike,
    config: EmpiricalBackgroundConfig | None = None,
) -> FloatArray:
    """Estimate a smooth mean PSD using log-spaced local running medians.

    Dividing each median by ``log(2)`` converts the median of an exponentially
    distributed periodogram into an estimate of its mean. The value at zero
    frequency is copied from the lowest positive-frequency estimate.
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


def whiten_spectrum(
    frequency_uhz: ArrayLike,
    spectrum: ArrayLike,
    config: EmpiricalBackgroundConfig | None = None,
) -> tuple[ComplexArray, FloatArray]:
    """Flatten a complex Fourier spectrum while retaining its phases."""

    complex_spectrum = np.asarray(spectrum, dtype=complex)
    background = estimate_empirical_background(
        frequency_uhz, np.abs(complex_spectrum) ** 2, config
    )
    whitened = complex_spectrum / np.sqrt(background)
    return np.asarray(whitened, dtype=complex), background
