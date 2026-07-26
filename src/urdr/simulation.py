"""Deterministic, exact-window time-series simulations."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.random import Generator

from .models import ObservingWindow, TimeSeries


@dataclass(frozen=True)
class SimulationConfig:
    """Parameters for a simple seismic time-series forward model.

    Parameters
    ----------
    white_noise_sigma
        Standard deviation of Gaussian measurement noise.
    granulation_amplitude
        Stationary standard deviation of the Ornstein-Uhlenbeck component.
    granulation_timescale_days
        Correlation timescale of the granulation component in days.
    numax_uhz
        Oscillation-envelope centre in microhertz.
    delta_nu_uhz
        Injected large separation in microhertz.
    envelope_width_uhz
        Gaussian envelope width in microhertz.
    oscillation_amplitude
        Root-mean-square amplitude of the stochastic mode comb.
    mode_linewidth_uhz
        Full mode linewidth in microhertz.
    """

    white_noise_sigma: float = 1.0
    granulation_amplitude: float = 0.0
    granulation_timescale_days: float = 0.25
    numax_uhz: float = 1000.0
    delta_nu_uhz: float = 55.0
    envelope_width_uhz: float = 300.0
    oscillation_amplitude: float = 0.0
    mode_linewidth_uhz: float = 2.0

    def __post_init__(self) -> None:
        """Validate simulation amplitudes and frequency scales."""
        positive = (
            self.white_noise_sigma,
            self.granulation_timescale_days,
            self.numax_uhz,
            self.delta_nu_uhz,
            self.envelope_width_uhz,
            self.mode_linewidth_uhz,
        )
        if any(value <= 0 for value in positive):
            raise ValueError("noise and frequency-scale parameters must be positive")
        if self.granulation_amplitude < 0 or self.oscillation_amplitude < 0:
            raise ValueError("component amplitudes cannot be negative")


def simulate_time_series(
    window: ObservingWindow,
    config: SimulationConfig,
    rng: Generator,
    include_oscillations: bool = True,
) -> TimeSeries:
    """Simulate noise, granulation, and a stochastic p-mode comb.

    Parameters
    ----------
    window
        Exact cadence grid and observing mask.
    config
        Forward-model parameters.
    rng
        NumPy random number generator.
    include_oscillations
        Whether to include the configured stochastic mode comb.

    Returns
    -------
    TimeSeries
        Simulated time series with missing cadences represented as NaN.
    """
    size = window.time.size
    flux = rng.normal(0.0, config.white_noise_sigma, size)
    if config.granulation_amplitude > 0:
        flux += _simulate_ou(
            size,
            float(np.median(np.diff(window.time))),
            config.granulation_timescale_days,
            config.granulation_amplitude,
            rng,
        )
    if include_oscillations and config.oscillation_amplitude > 0:
        flux += _simulate_mode_comb(
            size,
            float(np.median(np.diff(window.time))) * 86400.0,
            config,
            rng,
        )
    flux = np.asarray(flux, dtype=float)
    flux[~window.observed] = np.nan
    return TimeSeries(window.time, flux, window.observed)


def _simulate_ou(
    size: int,
    cadence_days: float,
    timescale_days: float,
    amplitude: float,
    rng: Generator,
) -> np.ndarray:
    coefficient = np.exp(-cadence_days / timescale_days)
    innovation = amplitude * np.sqrt(1.0 - coefficient**2)
    output = np.empty(size, dtype=float)
    output[0] = rng.normal(0.0, amplitude)
    for index in range(1, size):
        output[index] = coefficient * output[index - 1] + rng.normal(
            0.0, innovation
        )
    return output


def _simulate_mode_comb(
    size: int,
    cadence_seconds: float,
    config: SimulationConfig,
    rng: Generator,
) -> np.ndarray:
    frequency_uhz = np.fft.rfftfreq(size, cadence_seconds) * 1e6
    power = np.zeros(frequency_uhz.size, dtype=float)
    radial_orders = np.arange(
        np.floor(
            (config.numax_uhz - 2.5 * config.envelope_width_uhz)
            / config.delta_nu_uhz
        ),
        np.ceil(
            (config.numax_uhz + 2.5 * config.envelope_width_uhz)
            / config.delta_nu_uhz
        )
        + 1,
    )
    mode_frequencies = radial_orders * config.delta_nu_uhz
    for mode_frequency in mode_frequencies:
        envelope = np.exp(
            -0.5
            * ((mode_frequency - config.numax_uhz) / config.envelope_width_uhz)
            ** 2
        )
        half_width = config.mode_linewidth_uhz / 2.0
        power += envelope / (
            1.0 + ((frequency_uhz - mode_frequency) / half_width) ** 2
        )
    phases = rng.normal(size=frequency_uhz.size) + 1j * rng.normal(
        size=frequency_uhz.size
    )
    spectrum = phases * np.sqrt(np.maximum(power, 0.0))
    signal = np.fft.irfft(spectrum, n=size)
    standard_deviation = np.std(signal)
    if standard_deviation > 0:
        signal *= config.oscillation_amplitude / standard_deviation
    return signal
