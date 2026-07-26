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

