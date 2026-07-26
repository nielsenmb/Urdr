# Urdr

Urdr is a window-aware evolution of the time-series envelope autocorrelation
function (EACF) method used in Atropos and described by
[Nielsen et al. (2022)](https://ui.adsabs.harvard.edu/abs/2022MNRAS.515.1239N).
It is intended to detect solar-like oscillations while making the effects of
gaps, low-frequency variability, and target-specific noise explicit.

The current implementation provides:

- a frequency-filtered time-series ACF and frequency-lag map;
- explicit `TimeSeries` and `ObservingWindow` models;
- exact-window white-noise, granulation, and stochastic mode-comb simulations;
- deterministic paired simulations for target-specific null calibration;
- optional empirical running-median background removal, including an
  AsteroScale-informed high-S/N safeguard;
- a paired benchmark for no removal, legacy empirical whitening,
  target-aware empirical whitening, and a fitted Harvey-like baseline;
- harmonic spot/planet-like contaminant simulations and a target-calibrated
  spectral-concentration veto;
- a lightweight adapter for correlated AsteroScale posterior samples.

Skuld is deliberately not a dependency. This allows an AsteroScale-only Urdr
baseline to be compared fairly with an optional experiment in which Skuld
localises the oscillation envelope.

## Installation

```bash
python -m pip install -e ".[test]"
pytest
```

## Minimal example

```python
import numpy as np

from urdr import SimulationCalibrator, SimulationConfig, TimeSeries, compute_eacf_map

series = TimeSeries.from_arrays(time_days, flux)
centres = np.linspace(800.0, 1200.0, 21)

def statistic(candidate):
    result = compute_eacf_map(
        candidate,
        centres,
        filter_width_uhz=400.0,
        max_lag_seconds=30_000.0,
    )
    return result.statistic(delta_nu_uhz=55.0)

config = SimulationConfig(
    white_noise_sigma=np.nanstd(flux),
    numax_uhz=1000.0,
    delta_nu_uhz=55.0,
    envelope_width_uhz=400.0,
    oscillation_amplitude=0.2,
)
result = SimulationCalibrator(simulations=128, seed=42).calibrate(
    series, config, statistic
)
print(result.false_alarm_probability)
```

## Empirical background experiment

The background treatment used in the earlier implementation is available as an
explicit comparison arm. It estimates the local mean PSD from running medians at
log-spaced frequencies, using the original defaults
`half_width = 0.66 * frequency**0.88`, and whitens the complex Fourier spectrum
without changing its phases:

```python
from urdr import EmpiricalBackgroundConfig

legacy_background = EmpiricalBackgroundConfig()
target_aware_background = EmpiricalBackgroundConfig.excluding_envelope(
    numax_uhz=1000.0,
    envelope_width_uhz=400.0,
)

result = compute_eacf_map(
    series,
    centres,
    filter_width_uhz=400.0,
    empirical_background=target_aware_background,
)
```

The target-aware form excludes the predicted oscillation envelope when
estimating the running medians, then interpolates the background across it. This
is intended to prevent high-S/N modes from being mistaken for background. Both
forms should be calibrated through the same exact-window simulations; `a` and
`b` remain configurable so they can later be fitted on held-out injections.

Fresh random simulations are not generated inside individual likelihood
evaluations. Fixed seeds and common random numbers keep the synthetic calibration
reproducible and avoid turning Monte Carlo noise into likelihood noise.

## AsteroScale boundary

`AsteroScaleSamples` accepts dictionary-like posterior output containing
`numax`, `delta_nu`, and `envelope_width`. Correlated samples remain paired:

```python
from urdr import AsteroScaleSamples

samples = AsteroScaleSamples.from_mapping(asteroscale_output)
search_region = samples.search_region()
simulation_parameters = samples.median_parameters()
```

## Background-treatment benchmark

The background arms can be compared at a common false-positive rate while
reporting detection and `delta_nu` recovery separately:

```python
from urdr import benchmark_background_treatments

benchmark = benchmark_background_treatments(
    window=series.window,
    simulation=config,
    centre_frequencies_uhz=centres,
    filter_width_uhz=400.0,
    delta_nu_grid_uhz=np.linspace(45.0, 65.0, 41),
    oscillation_amplitudes=[0.05, 0.1, 0.2, 0.4],
    realizations=128,
    target_false_positive_rate=0.01,
    seed=42,
)
```

Every arm uses the same simulated realizations. Its detection threshold is
derived independently from its own exact-window null distribution, avoiding an
unfair comparison caused by different statistic scales. The Harvey-like arm is
a deliberately simple fitted reference model, not a claim that a single Harvey
component describes real granulation perfectly.

## Coherent-signal diagnostics

Spot modulation, planet-like periodic variability, and narrow instrumental
signals can produce strong autocorrelation while concentrating their Fourier
power into relatively few bins. Urdr represents these hard negatives with
`CoherentSignalConfig` and measures three complementary diagnostics in the
candidate oscillation band:

- maximum single-bin power fraction;
- effective number of occupied Fourier bins;
- normalised spectral entropy.

```python
from urdr import (
    CoherentSignalConfig,
    add_coherent_signal,
    coherence_diagnostics,
)

contaminated = add_coherent_signal(
    noise_series,
    CoherentSignalConfig(
        frequency_uhz=1000.0,
        amplitude=0.5,
        harmonics=3,
    ),
    np.random.default_rng(12),
)
diagnostics = coherence_diagnostics(
    contaminated,
    centre_frequency_uhz=1000.0,
    filter_width_uhz=400.0,
    background=target_aware_background,
)
```

The veto threshold is not a universal hard-coded cut. It is calibrated from
window-aware signal injections at a requested signal-retention rate, while the
EACF threshold remains independently calibrated from the clean null:

```python
from urdr import benchmark_coherent_veto

veto = benchmark_coherent_veto(
    window=series.window,
    simulation=config,
    contaminants={
        "single_line": CoherentSignalConfig(1000.0, 0.5),
        "three_harmonics": CoherentSignalConfig(333.3, 0.5, harmonics=3),
    },
    centre_frequencies_uhz=centres,
    filter_width_uhz=400.0,
    delta_nu_grid_uhz=np.linspace(45.0, 65.0, 41),
    background=target_aware_background,
    realizations=128,
    target_false_positive_rate=0.01,
    target_signal_retention=0.95,
    seed=42,
)
```

This reports raw and vetoed detection rates separately for the signal
injections and every contaminant class. The exact observing mask is used in all
arms, because aliases can broaden a coherent peak and change its apparent
concentration.

## Notebooks

- [`notebooks/coherent_contaminants.ipynb`](notebooks/coherent_contaminants.ipynb)
  introduces harmonic contaminants, inspects the concentration diagnostics, and
  runs a small reproducible veto benchmark.

All public APIs use NumPy-style docstrings. The documentation check can be run
with:

```bash
pydocstyle src/urdr
```
