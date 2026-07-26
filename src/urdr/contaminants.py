"""Coherent-contaminant simulations and diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
from numpy.random import Generator
from numpy.typing import NDArray

from .background import BackgroundConfig, whiten_spectrum
from .benchmark import estimate_delta_nu
from .eacf import compute_eacf_map
from .models import ObservingWindow, TimeSeries
from .simulation import SimulationConfig, simulate_time_series

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class CoherentSignalConfig:
    """Configuration for a coherent harmonic contaminant.

    Parameters
    ----------
    frequency_uhz
        Fundamental frequency in microhertz.
    amplitude
        Root-mean-square amplitude of the complete harmonic signal.
    harmonics
        Number of harmonics, including the fundamental.
    harmonic_decay
        Multiplicative amplitude decay between successive harmonics.
    """

    frequency_uhz: float
    amplitude: float
    harmonics: int = 1
    harmonic_decay: float = 0.5

    def __post_init__(self) -> None:
        """Validate the coherent-signal parameters."""
        if self.frequency_uhz <= 0:
            raise ValueError("frequency_uhz must be positive")
        if self.amplitude <= 0:
            raise ValueError("amplitude must be positive")
        if self.harmonics < 1:
            raise ValueError("harmonics must be at least one")
        if not 0 < self.harmonic_decay <= 1:
            raise ValueError("harmonic_decay must lie in (0, 1]")


@dataclass(frozen=True)
class CoherenceDiagnostics:
    """Spectral-concentration diagnostics inside one frequency band.

    Parameters
    ----------
    maximum_bin_fraction
        Fraction of band power in the strongest Fourier bin.
    effective_bins
        Inverse participation ratio of the normalised band power.
    spectral_entropy
        Shannon entropy divided by the maximum entropy for the number of bins.
    """

    maximum_bin_fraction: float
    effective_bins: float
    spectral_entropy: float


@dataclass(frozen=True)
class CoherentVetoMetrics:
    """Performance of a coherent-signal veto.

    Parameters
    ----------
    eacf_threshold
        Clean-null threshold for the EACF statistic.
    concentration_threshold
        Maximum-bin fraction above which a detection is vetoed.
    raw_signal_detection_rate
        Signal detection rate before applying the veto.
    vetoed_signal_detection_rate
        Signal detection rate after applying the veto.
    raw_contaminant_detection_rate
        Detection rate for each contaminant before applying the veto.
    vetoed_contaminant_detection_rate
        Detection rate for each contaminant after applying the veto.
    target_false_positive_rate
        Requested clean-null false-positive rate.
    target_signal_retention
        Requested fraction of EACF-detected signals retained by the veto.
    realizations
        Number of paired realisations per simulation class.
    """

    eacf_threshold: float
    concentration_threshold: float
    raw_signal_detection_rate: float
    vetoed_signal_detection_rate: float
    raw_contaminant_detection_rate: Mapping[str, float]
    vetoed_contaminant_detection_rate: Mapping[str, float]
    target_false_positive_rate: float
    target_signal_retention: float
    realizations: int


def add_coherent_signal(
    series: TimeSeries,
    config: CoherentSignalConfig,
    rng: Generator,
) -> TimeSeries:
    """Add a coherent harmonic signal to an existing time series.

    Independent random phases are used for the harmonics. The complete signal
    is rescaled to the requested root-mean-square amplitude before the observing
    mask is applied.

    Parameters
    ----------
    series
        Base uniformly sampled time series.
    config
        Coherent-signal parameters.
    rng
        NumPy random number generator used for harmonic phases.

    Returns
    -------
    TimeSeries
        Copy of ``series`` containing the added coherent signal.
    """
    time_seconds = (series.time - series.time[0]) * 86400.0
    signal = np.zeros(series.time.size, dtype=float)
    for harmonic in range(1, config.harmonics + 1):
        phase = rng.uniform(0.0, 2.0 * np.pi)
        weight = config.harmonic_decay ** (harmonic - 1)
        signal += weight * np.sin(
            2.0
            * np.pi
            * harmonic
            * config.frequency_uhz
            * 1e-6
            * time_seconds
            + phase
        )
    rms = float(np.std(signal))
    if rms > 0:
        signal *= config.amplitude / rms

    flux = np.asarray(series.flux, dtype=float).copy()
    flux[series.observed] += signal[series.observed]
    flux[~series.observed] = np.nan
    return TimeSeries(series.time, flux, series.observed)


def coherence_diagnostics(
    series: TimeSeries,
    centre_frequency_uhz: float,
    filter_width_uhz: float,
    background: BackgroundConfig | None = None,
) -> CoherenceDiagnostics:
    """Measure spectral concentration in a candidate oscillation band.

    Parameters
    ----------
    series
        Uniformly sampled time series with an explicit observing mask.
    centre_frequency_uhz
        Centre of the candidate band in microhertz.
    filter_width_uhz
        Full width of the candidate band in microhertz.
    background
        Optional background model used to whiten the complex spectrum.

    Returns
    -------
    CoherenceDiagnostics
        Maximum-bin fraction, effective bin count, and normalised entropy.

    Notes
    -----
    A coherent sinusoid concentrates power into relatively few Fourier bins,
    whereas a stochastic mode envelope distributes power across many bins.
    Window aliases can broaden either signal, so the veto threshold should be
    calibrated with simulations using the exact observing mask.
    """
    if centre_frequency_uhz <= 0 or filter_width_uhz <= 0:
        raise ValueError("band centre and width must be positive")
    signal = np.zeros(series.time.size, dtype=float)
    observed_flux = series.flux[series.observed]
    signal[series.observed] = observed_flux - np.median(observed_flux)
    frequency = (
        np.fft.rfftfreq(signal.size, d=series.cadence_seconds) * 1e6
    )
    spectrum = np.fft.rfft(signal)
    if background is not None:
        spectrum, _ = whiten_spectrum(frequency, spectrum, background)

    half_width = filter_width_uhz / 2.0
    select = np.abs(frequency - centre_frequency_uhz) <= half_width
    if np.count_nonzero(select) < 3:
        raise ValueError("candidate band contains fewer than three Fourier bins")
    power = np.abs(spectrum[select]) ** 2
    total = float(np.sum(power))
    if not np.isfinite(total) or total <= 0:
        raise ValueError("candidate band has zero or non-finite power")

    probability = power / total
    nonzero = probability > 0
    entropy = -float(np.sum(probability[nonzero] * np.log(probability[nonzero])))
    maximum_entropy = np.log(probability.size)
    return CoherenceDiagnostics(
        maximum_bin_fraction=float(np.max(probability)),
        effective_bins=float(1.0 / np.sum(probability**2)),
        spectral_entropy=float(entropy / maximum_entropy),
    )


def benchmark_coherent_veto(
    window: ObservingWindow,
    simulation: SimulationConfig,
    contaminants: Mapping[str, CoherentSignalConfig],
    centre_frequencies_uhz: FloatArray,
    filter_width_uhz: float,
    delta_nu_grid_uhz: FloatArray,
    *,
    background: BackgroundConfig | None = None,
    realizations: int = 128,
    target_false_positive_rate: float = 0.01,
    target_signal_retention: float = 0.95,
    max_lag_seconds: float | None = None,
    seed: int = 0,
) -> CoherentVetoMetrics:
    """Benchmark a target-calibrated coherent-signal veto.

    The EACF threshold is calibrated on noise-plus-granulation simulations. The
    concentration threshold is then calibrated on injected oscillators that
    pass the EACF threshold, retaining the requested fraction of those
    detections. Contaminants share the same base realisations and observing
    window.

    Parameters
    ----------
    window
        Exact observing window used for every simulation.
    simulation
        Noise, granulation, and injected-oscillation parameters.
    contaminants
        Named coherent contaminants to benchmark.
    centre_frequencies_uhz
        Trial EACF filter centres in microhertz.
    filter_width_uhz
        Full EACF filter width in microhertz.
    delta_nu_grid_uhz
        Trial large separations in microhertz.
    background
        Optional spectral-background treatment.
    realizations
        Number of paired realisations per simulation class.
    target_false_positive_rate
        Requested false-positive rate for the clean null.
    target_signal_retention
        Fraction of EACF-detected signal injections the veto should retain.
    max_lag_seconds
        Optional maximum EACF lag.
    seed
        Root seed for deterministic paired simulations.

    Returns
    -------
    CoherentVetoMetrics
        Raw and vetoed detection rates and the calibrated thresholds.
    """
    if realizations < 8:
        raise ValueError("at least eight realizations are required")
    if not contaminants:
        raise ValueError("at least one contaminant is required")
    if not 0 < target_false_positive_rate < 1:
        raise ValueError("target_false_positive_rate must lie between zero and one")
    if not 0 < target_signal_retention <= 1:
        raise ValueError("target_signal_retention must lie in (0, 1]")

    centres = np.atleast_1d(np.asarray(centre_frequencies_uhz, dtype=float))
    delta_nu_grid = np.atleast_1d(np.asarray(delta_nu_grid_uhz, dtype=float))
    seeds = np.random.SeedSequence(seed).spawn(realizations)

    null_scores = np.empty(realizations)
    signal_scores = np.empty(realizations)
    signal_concentration = np.empty(realizations)
    contaminant_scores = {
        name: np.empty(realizations) for name in contaminants
    }
    contaminant_concentration = {
        name: np.empty(realizations) for name in contaminants
    }

    for index, child_seed in enumerate(seeds):
        streams = child_seed.spawn(1 + len(contaminants))
        null_series = simulate_time_series(
            window,
            simulation,
            np.random.default_rng(streams[0]),
            include_oscillations=False,
        )
        signal_series = simulate_time_series(
            window,
            simulation,
            np.random.default_rng(streams[0]),
            include_oscillations=True,
        )
        null_scores[index] = _eacf_score(
            null_series,
            centres,
            filter_width_uhz,
            delta_nu_grid,
            background,
            max_lag_seconds,
        )
        signal_scores[index] = _eacf_score(
            signal_series,
            centres,
            filter_width_uhz,
            delta_nu_grid,
            background,
            max_lag_seconds,
        )
        signal_concentration[index] = coherence_diagnostics(
            signal_series,
            simulation.numax_uhz,
            filter_width_uhz,
            background,
        ).maximum_bin_fraction

        for offset, (name, contaminant) in enumerate(contaminants.items(), start=1):
            contaminated = add_coherent_signal(
                null_series,
                contaminant,
                np.random.default_rng(streams[offset]),
            )
            contaminant_scores[name][index] = _eacf_score(
                contaminated,
                centres,
                filter_width_uhz,
                delta_nu_grid,
                background,
                max_lag_seconds,
            )
            contaminant_concentration[name][index] = coherence_diagnostics(
                contaminated,
                simulation.numax_uhz,
                filter_width_uhz,
                background,
            ).maximum_bin_fraction

    eacf_threshold = _higher_quantile(
        null_scores, 1.0 - target_false_positive_rate
    )
    signal_detected = signal_scores >= eacf_threshold
    detected_concentration = signal_concentration[signal_detected]
    calibration_values = (
        detected_concentration
        if detected_concentration.size
        else signal_concentration
    )
    concentration_threshold = _higher_quantile(
        calibration_values, target_signal_retention
    )

    raw_signal = float(np.mean(signal_detected))
    vetoed_signal = float(
        np.mean(signal_detected & (signal_concentration <= concentration_threshold))
    )
    raw_contaminant = {}
    vetoed_contaminant = {}
    for name in contaminants:
        detected = contaminant_scores[name] >= eacf_threshold
        raw_contaminant[name] = float(np.mean(detected))
        vetoed_contaminant[name] = float(
            np.mean(
                detected
                & (
                    contaminant_concentration[name]
                    <= concentration_threshold
                )
            )
        )

    return CoherentVetoMetrics(
        eacf_threshold=eacf_threshold,
        concentration_threshold=concentration_threshold,
        raw_signal_detection_rate=raw_signal,
        vetoed_signal_detection_rate=vetoed_signal,
        raw_contaminant_detection_rate=raw_contaminant,
        vetoed_contaminant_detection_rate=vetoed_contaminant,
        target_false_positive_rate=target_false_positive_rate,
        target_signal_retention=target_signal_retention,
        realizations=realizations,
    )


def _eacf_score(
    series: TimeSeries,
    centres: FloatArray,
    filter_width_uhz: float,
    delta_nu_grid: FloatArray,
    background: BackgroundConfig | None,
    max_lag_seconds: float | None,
) -> float:
    """Return the maximum EACF score over a large-separation grid."""
    result = compute_eacf_map(
        series,
        centres,
        filter_width_uhz,
        max_lag_seconds=max_lag_seconds,
        empirical_background=background,
    )
    return estimate_delta_nu(result, delta_nu_grid)[1]


def _higher_quantile(values: FloatArray, probability: float) -> float:
    """Return a conservative empirical quantile."""
    try:
        return float(np.quantile(values, probability, method="higher"))
    except TypeError:  # NumPy < 1.22 compatibility
        return float(np.quantile(values, probability, interpolation="higher"))
