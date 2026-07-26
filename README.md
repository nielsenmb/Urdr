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
- reproducible multi-target grids for calibrating the empirical smoothing law
  across S/N, seismic regime, and observing-window patterns;
- harmonic spot/planet-like contaminant simulations and a target-calibrated
  spectral-concentration veto;
- piecewise variance, offset, and drift simulations with a calibrated
  campaign/sector-instability veto;
- frequency-lag ridge morphology diagnostics with a signal-calibrated joint
  check against coherent and segment-dependent hard negatives;
- a single held-out-calibrated detector combining the EACF, spectral
  concentration, segment stability, and ridge morphology without sequential
  vetoes;
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

## Calibrating the empirical smoothing law

The legacy \(a=0.66,\ b=0.88\) values can be compared with alternative
frequency-dependent smoothing widths over multiple target and window
conditions. Cases carry an explicit `train` or `validation` label so parameter
choices can be checked on held-out simulations:

```python
from urdr import (
    BackgroundBenchmarkCase,
    EmpiricalGridPoint,
    benchmark_empirical_grid,
    make_observing_window,
)

window = make_observing_window(
    duration_days=27.4,
    cadence_seconds=120.0,
    gaps_days=((13.6, 13.9),),
    random_missing_fraction=0.02,
    seed=5,
)
case = BackgroundBenchmarkCase(
    name="mid_numax_sector",
    split="train",
    window=window,
    simulation=config,
    centre_frequencies_uhz=centres,
    filter_width_uhz=400.0,
    delta_nu_grid_uhz=np.linspace(45.0, 65.0, 41),
    oscillation_amplitudes=np.array([0.05, 0.1, 0.2]),
)
grid = benchmark_empirical_grid(
    cases=[case, held_out_case],
    candidates=[
        EmpiricalGridPoint(0.40, 0.80),
        EmpiricalGridPoint(0.66, 0.88),
        EmpiricalGridPoint(0.90, 0.95),
    ],
    realizations=128,
    target_false_positive_rate=0.01,
    seed=42,
)

training_front = grid.pareto_front("train")
validation_metrics = grid.summaries("validation")
records = grid.to_records()
```

Detection and \(\Delta\nu\) recovery remain separate metrics. `pareto_front`
returns the candidates that are not worse on both, avoiding an undocumented
weighted score. `to_records()` provides flat rows that can be written to CSV or
loaded into pandas without making pandas a package dependency.

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

## Campaign and sector systematics

Variance changes, offsets, and slow drifts between observing segments are
represented explicitly rather than folded into a generic noise penalty:

```python
from urdr import (
    SegmentSystematicConfig,
    add_segment_systematics,
    benchmark_segment_veto,
    segment_diagnostics,
)

segments = ((0.0, 13.7), (13.7, 27.4))
variance_jump = [
    SegmentSystematicConfig(13.7, 27.4, amplitude_scale=3.0)
]
contaminated = add_segment_systematics(noise_series, variance_jump)
diagnostics = segment_diagnostics(contaminated, segments)

benchmark = benchmark_segment_veto(
    window=series.window,
    simulation=config,
    systematics={"variance_jump": variance_jump},
    segments_days=segments,
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

The diagnostic combines robust scale ratios, median shifts, and within-segment
drifts. These quantities have different natural scales, so each is converted to
an empirical percentile using exact-window signal injections. The maximum
percentile is then calibrated to the requested signal-retention rate. This
keeps the veto target-specific and avoids assuming that the three diagnostics
are independent.

## EACF ridge morphology

The peak EACF statistic discards most of the frequency-lag map. Urdr can also
measure whether a candidate forms a localised, continuous ridge near the
AsteroScale-predicted envelope:

```python
from urdr import eacf_morphology

eacf_map = compute_eacf_map(
    series,
    centres,
    filter_width_uhz=400.0,
    max_lag_seconds=2e6 / expected_delta_nu,
)
diagnostics = eacf_morphology(
    eacf_map,
    delta_nu_uhz=expected_delta_nu,
    expected_numax_uhz=expected_numax,
    envelope_width_uhz=expected_envelope_width,
)
```

The diagnostics describe envelope localisation, connected ridge width and
fill, contrast against neighbouring filter centres, roughness, and support near
the second ACF peak. Their useful ranges depend on the target, observing
window, filter grid, and S/N, so none is used as a universal cut.

`benchmark_morphology_veto` learns a joint empirical outlier threshold from
exact-window oscillator injections, then applies it to both coherent and
segment-systematic hard negatives. The raw and accepted detection rates remain
separate, making the signal cost of the morphology check explicit.

## Joint inference

The individual diagnostics are useful for understanding failure modes, but
their vetoes should not be applied sequentially. For example, three separate
95 per cent retention cuts need not retain 95 per cent of signals overall.
`calibrate_joint_detector` instead learns one decision boundary from the EACF
statistic and all twelve contaminant/morphology diagnostics:

```python
from urdr import (
    CoherentSignalConfig,
    SegmentSystematicConfig,
    calibrate_joint_detector,
)

detector = calibrate_joint_detector(
    window=series.window,
    simulation=config,
    centre_frequencies_uhz=centres,
    filter_width_uhz=400.0,
    delta_nu_grid_uhz=np.linspace(45.0, 65.0, 41),
    segments_days=((0.0, 13.7), (13.7, 27.4)),
    coherent_contaminants={
        "single_line": CoherentSignalConfig(1000.0, 0.5),
        "harmonic_comb": CoherentSignalConfig(333.3, 0.5, harmonics=3),
    },
    segment_systematics={
        "variance_jump": [
            SegmentSystematicConfig(13.7, 27.4, amplitude_scale=3.0)
        ],
    },
    background=target_aware_background,
    realizations=512,
    validation_fraction=0.25,
    target_false_positive_rate=0.01,
    seed=42,
)
result = detector.detect(series)
print(
    result.detection_probability,
    result.false_alarm_probability,
    result.false_alarm_interval,
    result.delta_nu_uhz,
)
```

Paired realisations are split before fitting. The training subset learns a
regularised linear discriminant, while the held-out subset calibrates its
probability scale, false-alarm rate, uncertainty interval, and explanatory
flags. Clean nulls and every represented contaminant class form one
hard-negative hypothesis. The false-alarm calculation uses the most
signal-like hard negative in each paired realisation, so adding contaminant
classes does not artificially inflate the effective number of simulations.

`diagnostic_flags` identify unusual spectral concentration, segment
instability, or morphology. They explain a result but do not act as additional
independent vetoes. `detector.validation` reports the held-out true-positive
rate, family-wise false-positive rate, and Brier probability score.
The calibration rejects a simulation count too small to resolve the requested
false-positive rate; for example, a 1 per cent rate needs at least 100 held-out
paired realisations.

## Notebooks

- [`notebooks/coherent_contaminants.ipynb`](notebooks/coherent_contaminants.ipynb)
  introduces harmonic contaminants, inspects the concentration diagnostics, and
  runs a small reproducible veto benchmark.
- [`notebooks/empirical_background_grid.ipynb`](notebooks/empirical_background_grid.ipynb)
  calibrates \(a,b\) on paired simulations and evaluates the Pareto candidates
  on a held-out observing window.
- [`notebooks/segment_systematics.ipynb`](notebooks/segment_systematics.ipynb)
  injects campaign/sector variance, offset, and drift changes and benchmarks the
  calibrated instability veto.
- [`notebooks/eacf_morphology.ipynb`](notebooks/eacf_morphology.ipynb)
  visualises a frequency-lag ridge and benchmarks the joint morphology check
  against coherent and segment-dependent hard negatives.
- [`notebooks/joint_inference.ipynb`](notebooks/joint_inference.ipynb)
  fits the unified detector, runs it on a fresh injection, and inspects its
  held-out probability and false-alarm calibration.

All public APIs use NumPy-style docstrings. The documentation check can be run
with:

```bash
pydocstyle src/urdr
```
