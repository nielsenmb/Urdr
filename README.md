# Urdr

Urdr is a window-aware evolution of the time-series envelope autocorrelation
function (EACF) method used in Atropos and described by
[Nielsen et al. (2022)](https://ui.adsabs.harvard.edu/abs/2022MNRAS.515.1239N).
It is intended to detect solar-like oscillations while making the effects of
gaps, low-frequency variability, and target-specific noise explicit.

This first milestone provides:

- a frequency-filtered time-series ACF and frequency-lag map;
- explicit `TimeSeries` and `ObservingWindow` models;
- exact-window white-noise, granulation, and stochastic mode-comb simulations;
- deterministic paired simulations for target-specific null calibration;
- optional empirical running-median background removal, including an
  AsteroScale-informed high-S/N safeguard;
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

The next milestone will benchmark the unchanged published decision statistic
against the AsteroScale-restricted and window-calibrated configurations before
adding richer EACF morphology or coherent-signal vetoes.
