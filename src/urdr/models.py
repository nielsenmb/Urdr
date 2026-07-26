"""Core data models shared by the EACF and simulation layers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class TimeSeries:
    """Uniformly sampled flux series with an explicit observing mask.

    Missing cadences remain on the time grid. Their flux values are ignored wherever
    ``observed`` is false, which lets simulations reproduce the exact target window.
    Time is measured in days.

    Parameters
    ----------
    time
        Uniform cadence grid in days.
    flux
        Flux values on the cadence grid.
    observed
        Boolean mask selecting valid observations.
    """

    time: FloatArray
    flux: FloatArray
    observed: NDArray[np.bool_]

    def __post_init__(self) -> None:
        """Validate and normalise the time-series arrays."""
        time = np.asarray(self.time, dtype=float)
        flux = np.asarray(self.flux, dtype=float)
        observed = np.asarray(self.observed, dtype=bool)
        if time.ndim != 1 or flux.ndim != 1 or observed.ndim != 1:
            raise ValueError("time, flux, and observed must be one-dimensional")
        if not (time.size == flux.size == observed.size):
            raise ValueError("time, flux, and observed must have equal lengths")
        if time.size < 8:
            raise ValueError("a time series needs at least eight cadences")
        if not np.all(np.isfinite(time)) or np.any(np.diff(time) <= 0):
            raise ValueError("time must be finite and strictly increasing")
        if observed.sum() < 4 or not np.all(np.isfinite(flux[observed])):
            raise ValueError("at least four observed flux values must be finite")
        cadence = np.median(np.diff(time))
        tolerance = max(1e-10, abs(cadence) * 1e-5)
        if not np.allclose(np.diff(time), cadence, rtol=1e-5, atol=tolerance):
            raise ValueError("time must describe a uniform cadence grid")
        object.__setattr__(self, "time", time)
        object.__setattr__(self, "flux", flux)
        object.__setattr__(self, "observed", observed)

    @classmethod
    def from_arrays(
        cls,
        time: ArrayLike,
        flux: ArrayLike,
        observed: ArrayLike | None = None,
    ) -> "TimeSeries":
        """Construct a time series from array-like inputs.

        Parameters
        ----------
        time
            Uniform cadence grid in days.
        flux
            Flux values on the cadence grid.
        observed
            Optional explicit observing mask. Non-finite flux values define the
            mask when this argument is omitted.

        Returns
        -------
        TimeSeries
            Validated time-series object.
        """
        time_array = np.asarray(time, dtype=float)
        flux_array = np.asarray(flux, dtype=float)
        mask = np.isfinite(flux_array) if observed is None else np.asarray(observed, bool)
        return cls(time_array, flux_array, mask)

    @property
    def cadence_days(self) -> float:
        """Return the median cadence in days."""
        return float(np.median(np.diff(self.time)))

    @property
    def cadence_seconds(self) -> float:
        """Return the median cadence in seconds."""
        return self.cadence_days * 86400.0

    @property
    def window(self) -> "ObservingWindow":
        """Return the observing mask and cadence grid."""
        return ObservingWindow(self.time, self.observed)


@dataclass(frozen=True)
class ObservingWindow:
    """Cadence grid and boolean mask describing an observation.

    Parameters
    ----------
    time
        Cadence grid in days.
    observed
        Boolean mask selecting observed cadences.
    """

    time: FloatArray
    observed: NDArray[np.bool_]

    def __post_init__(self) -> None:
        """Validate and normalise the window arrays."""
        time = np.asarray(self.time, dtype=float)
        observed = np.asarray(self.observed, dtype=bool)
        if time.ndim != 1 or observed.ndim != 1 or time.size != observed.size:
            raise ValueError("time and observed must be equal-length 1D arrays")
        if time.size < 8 or not np.all(np.isfinite(time)):
            raise ValueError("window needs at least eight finite time samples")
        if np.any(np.diff(time) <= 0):
            raise ValueError("time must be strictly increasing")
        object.__setattr__(self, "time", time)
        object.__setattr__(self, "observed", observed)

    @property
    def duty_cycle(self) -> float:
        """Return the fraction of observed cadences."""
        return float(np.mean(self.observed))

    def spectral_power(self, frequency_uhz: ArrayLike) -> FloatArray:
        """Return zero-frequency-normalised spectral-window power.

        Parameters
        ----------
        frequency_uhz
            Frequencies at which to evaluate the spectral window.

        Returns
        -------
        numpy.ndarray
            Spectral-window power at each input frequency.
        """
        frequency = np.atleast_1d(np.asarray(frequency_uhz, dtype=float))
        phase = 2j * np.pi * np.outer(frequency * 1e-6, self.time * 86400.0)
        amplitude = np.exp(-phase) @ self.observed.astype(float)
        normalisation = max(float(self.observed.sum()) ** 2, 1.0)
        return np.asarray(np.abs(amplitude) ** 2 / normalisation, dtype=float)

    def diagnostics(self, delta_nu_uhz: float | None = None) -> dict[str, float]:
        """Summarise duty cycle and window power at seismic spacings.

        Parameters
        ----------
        delta_nu_uhz
            Optional large separation used to evaluate common aliases.

        Returns
        -------
        dict
            Named observing-window diagnostics.
        """
        result = {"duty_cycle": self.duty_cycle}
        if delta_nu_uhz is not None:
            frequencies = np.array([0.5, 1.0, 2.0]) * delta_nu_uhz
            powers = self.spectral_power(frequencies)
            result.update(
                {
                    "power_at_half_delta_nu": float(powers[0]),
                    "power_at_delta_nu": float(powers[1]),
                    "power_at_twice_delta_nu": float(powers[2]),
                }
            )
        return result


@dataclass(frozen=True)
class SearchRegion:
    """Frequency region used by the EACF search.

    Parameters
    ----------
    minimum_uhz
        Lower search bound in microhertz.
    maximum_uhz
        Upper search bound in microhertz.
    centre_uhz
        Central search frequency in microhertz.
    """

    minimum_uhz: float
    maximum_uhz: float
    centre_uhz: float

    def __post_init__(self) -> None:
        """Validate the ordered positive search bounds."""
        if not (0 < self.minimum_uhz < self.centre_uhz < self.maximum_uhz):
            raise ValueError("search bounds must be positive and contain the centre")


@dataclass(frozen=True)
class AsteroScaleSamples:
    """Minimal adapter for correlated samples produced by AsteroScale.

    Parameters
    ----------
    numax_uhz
        Correlated posterior samples of ``numax`` in microhertz.
    delta_nu_uhz
        Correlated posterior samples of ``delta_nu`` in microhertz.
    envelope_width_uhz
        Correlated posterior samples of envelope width in microhertz.
    granulation_amplitude
        Optional correlated samples of granulation amplitude.
    granulation_timescale_days
        Optional correlated samples of granulation timescale in days.
    """

    numax_uhz: FloatArray
    delta_nu_uhz: FloatArray
    envelope_width_uhz: FloatArray
    granulation_amplitude: FloatArray | None = None
    granulation_timescale_days: FloatArray | None = None

    def __post_init__(self) -> None:
        """Validate and normalise the correlated posterior samples."""
        required = [
            np.asarray(self.numax_uhz, dtype=float),
            np.asarray(self.delta_nu_uhz, dtype=float),
            np.asarray(self.envelope_width_uhz, dtype=float),
        ]
        size = required[0].size
        if size == 0 or any(item.ndim != 1 or item.size != size for item in required):
            raise ValueError("required sample arrays must be non-empty and equal length")
        if any(np.any(~np.isfinite(item)) or np.any(item <= 0) for item in required):
            raise ValueError("required AsteroScale samples must be finite and positive")
        object.__setattr__(self, "numax_uhz", required[0])
        object.__setattr__(self, "delta_nu_uhz", required[1])
        object.__setattr__(self, "envelope_width_uhz", required[2])
        for name in ("granulation_amplitude", "granulation_timescale_days"):
            value = getattr(self, name)
            if value is not None:
                array = np.asarray(value, dtype=float)
                if array.ndim != 1 or array.size != size:
                    raise ValueError(f"{name} must match required sample arrays")
                object.__setattr__(self, name, array)

    @classmethod
    def from_mapping(cls, samples: Mapping[str, ArrayLike]) -> "AsteroScaleSamples":
        """Build samples from a dictionary-like posterior collection.

        Parameters
        ----------
        samples
            Mapping containing ``numax``, ``delta_nu``, and
            ``envelope_width`` arrays, with optional granulation arrays.

        Returns
        -------
        AsteroScaleSamples
            Validated adapter preserving the input sample pairing.
        """
        return cls(
            numax_uhz=np.asarray(samples["numax"], dtype=float),
            delta_nu_uhz=np.asarray(samples["delta_nu"], dtype=float),
            envelope_width_uhz=np.asarray(samples["envelope_width"], dtype=float),
            granulation_amplitude=_optional_array(samples, "granulation_amplitude"),
            granulation_timescale_days=_optional_array(
                samples, "granulation_timescale_days"
            ),
        )

    def search_region(self, credible_mass: float = 0.99) -> SearchRegion:
        """Return a posterior-derived ``numax`` search interval.

        Parameters
        ----------
        credible_mass
            Posterior probability mass enclosed by the base interval.

        Returns
        -------
        SearchRegion
            Interval expanded by half the median envelope width.
        """
        if not 0 < credible_mass < 1:
            raise ValueError("credible_mass must lie between zero and one")
        tail = 0.5 * (1.0 - credible_mass)
        lower, centre, upper = np.quantile(
            self.numax_uhz, [tail, 0.5, 1.0 - tail]
        )
        width = float(np.median(self.envelope_width_uhz))
        return SearchRegion(
            max(float(lower - 0.5 * width), np.finfo(float).eps),
            float(upper + 0.5 * width),
            float(centre),
        )

    def median_parameters(self) -> dict[str, float]:
        """Return median parameters suitable for baseline simulations.

        Returns
        -------
        dict
            Median seismic parameters and any available granulation values.
        """
        output = {
            "numax_uhz": float(np.median(self.numax_uhz)),
            "delta_nu_uhz": float(np.median(self.delta_nu_uhz)),
            "envelope_width_uhz": float(np.median(self.envelope_width_uhz)),
        }
        if self.granulation_amplitude is not None:
            output["granulation_amplitude"] = float(
                np.median(self.granulation_amplitude)
            )
        if self.granulation_timescale_days is not None:
            output["granulation_timescale_days"] = float(
                np.median(self.granulation_timescale_days)
            )
        return output


def _optional_array(samples: Mapping[str, ArrayLike], key: str) -> FloatArray | None:
    value = samples.get(key)
    return None if value is None else np.asarray(value, dtype=float)
