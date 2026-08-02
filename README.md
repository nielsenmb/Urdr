# Urdr

Urdr is a window-aware evolution of the time-series envelope autocorrelation
function (EACF) method used in Atropos and described by
[Nielsen et al. (2022)](https://ui.adsabs.harvard.edu/abs/2022MNRAS.515.1239N).
It is intended to detect solar-like oscillations while making the effects of
gaps, low-frequency variability, and target-specific noise explicit.

The current implementation provides:

- a faithful published Lomb--Scargle EACF using Mimir, Hanning filters, and
  complex-modulus autocorrelations;
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
- pre-registered, checkpointed synthetic experiments with deterministic
  multiprocessing and machine-readable result tables;
- a frozen targeted TESS v1 grid with held-out regimes, matching empirical
  background cases, and post-run calibration assessments;
- a lightweight adapter for correlated AsteroScale posterior samples.

Skuld is deliberately not a dependency. This allows an AsteroScale-only Urdr
baseline to be compared fairly with an optional experiment in which Skuld
localises the oscillation envelope.

## Published EACF workflow

The original Nielsen et al. (2022) repeating-pattern workflow is available
separately from Urdr's experimental time-domain estimator. Mimir calculates a
critically sampled Lomb--Scargle power-density spectrum from only the observed
timestamps. Urdr divides out the empirical background, applies the published
Hanning filter bank, and evaluates the squared complex magnitude of the
inverse transform. Taking the complex magnitude removes the rapid filter-centre
carrier that remains in a squared real ACF.

```python
import numpy as np

from urdr import (
    AsteroScaleSamples,
    EmpiricalBackgroundConfig,
    PublishedEACFSearch,
    SimulationConfig,
    TimeSeries,
    calibrate_published_eacf,
    compute_published_eacf_map,
)

series = TimeSeries.from_arrays(time_grid_days, flux_with_nans)
centres = np.linspace(500.0, 1500.0, 101)

eacf = compute_published_eacf_map(
    series,
    centres,
    background=EmpiricalBackgroundConfig(),
)
print(eacf.best_numax_uhz, eacf.best_delta_nu_uhz)

simulation = SimulationConfig(
    white_noise_sigma=white_noise_sigma,
    granulation_amplitude=granulation_amplitude,
    granulation_timescale_days=granulation_timescale_days,
)
detection = calibrate_published_eacf(
    series,
    simulation,
    centres,
    simulations=1024,
    seed=42,
)
print(detection.detection_merit, detection.false_alarm_probability)
```

The default physical mask follows

\[
\Delta\nu = \Delta\nu_\odot
\left(\frac{\nu_{\max}}{\nu_{\max,\odot}}\right)^{0.791}
\]

with the published multiplicative range of \(10^{\pm0.2}\). This is the
banana-shaped region in the frequency--lag map. The headline statistic is the
largest collapsed response inside that complete region. Each null realization
uses the target's exact cadence grid and observing mask, re-estimates the
background, and records the same global maximum. The resulting false-alarm
probability therefore accounts for the real window and the look-elsewhere
effect, rather than transforming a Gamma density into a probability.

AsteroScale can define a second, target-informed search without replacing the
broad result:

```python
import asteroscale as ast

prediction = ast.solve(
    given={
        "M": (1.0, 0.05),
        "R": (2.6, 0.10),
        "Teff": (5600.0, 80.0),
        "FeH": (0.0, 0.10),
    },
    want=["numax", "dnu", "FWHM_env"],
    preset="fast",
    seed=42,
)
samples = AsteroScaleSamples.from_mapping(prediction)
informed_search = PublishedEACFSearch.from_asteroscale(samples)
informed = calibrate_published_eacf(
    series,
    simulation,
    search=informed_search,
    simulations=1024,
    batch_size=16,
    fft_workers=4,
    seed=42,
)
```

The paired AsteroScale samples define local conditional
\(\Delta\nu\) intervals and Hanning widths across the predicted
\(\nu_{\max}\) range. Point predictions fall back to the broad published
scatter. The same informed region is applied to the observed map and every
null realization, so its false-alarm probability includes its smaller
look-elsewhere volume. It should be labelled as conditional on the external
stellar information and reported alongside the broad scaling-relation result.

Null calibration is batched because `nifty-ls`, Mimir's Lomb--Scargle backend,
can evaluate several light curves on one shared timestamp/frequency grid.
Granulation is generated with a compiled AR(1) filter and Hanning transforms
reuse a static search plan. `batch_size` trades memory for throughput, while
`fft_workers` controls SciPy FFT parallelism (`-1` means all cores). On a
180-day, 120-second benchmark with 41 broad filters, 64 nulls fell from about
15.2 s to 3.2 s using batches of 16 and four FFT workers; a 15-centre
AsteroScale search took about 2.2 s. Exact timings depend strongly on hardware,
cadence, filter count, and gap pattern.

`notebooks/published_eacf.ipynb` walks through the spectrum, empirical
background, smooth EACF map, physical mask, collapsed response, and calibrated
detection. The older `compute_eacf_map` API remains available unchanged for
direct comparison with the time-domain, pair-count-corrected estimator.

## Scientific-scale synthetic experiments

`SyntheticExperimentPlan` freezes the complete validation design before the
first simulation is run. The saved manifest includes every target and
contaminant setting plus SHA-256 hashes of the exact cadence grids and observing
masks. Reusing an output directory with a changed design raises an error rather
than mixing incompatible results.

```python
from urdr import SyntheticExperimentPlan, run_synthetic_experiment

plan = SyntheticExperimentPlan(
    name="tess-synthetic-v1",
    cases=tuple(preregistered_cases),
    calibration_realizations=4096,
    evaluation_realizations=2048,
    target_false_positive_rate=0.01,
    seed=20260726,
)
run = run_synthetic_experiment(
    plan,
    "results/tess-synthetic-v1",
    workers=8,
    resume=True,
)
print(run.metrics_path, run.reliability_path)
```

Each validation case is an independent job with a deterministic seed derived
from the root seed and case name. Results are therefore unchanged by worker
count or completion order. Successful jobs are atomically checkpointed and
reused after interruption. The output directory contains:

- `manifest.json`: the immutable pre-registration and its fingerprint;
- `checkpoints/*.json`: one complete result per regime cell;
- `metrics.csv`: class-resolved detection, false-positive, \(\Delta\nu\)
  recovery, and Brier metrics;
- `reliability.csv`: probability calibration bins.

The experiment grid should explicitly cross evolutionary regime, oscillation
S/N, observing duration, duty cycle or gap pattern, and contaminant strength.
Training and validation cells remain labelled through `ValidationCase.split`.
See `notebooks/scientific_experiment.ipynb` for a small executable example.

### Frozen TESS v1 grid

`make_tess_scientific_grid` supplies the first production design. It contains
nine targeted cells rather than a full Cartesian product:

| Split | Regime | Signal/window combinations |
|---|---|---|
| Training | Red giant | low/one sector, high/three sectors |
| Training | Subgiant | low/three sectors, high/one sector |
| Training | Main sequence | low/one sector |
| Validation | Red giant | low/sparse sector |
| Validation | Subgiant | high/sparse three sectors |
| Validation | Main sequence | low/three sectors, high/CVZ-like |

The exact random masks, sector gaps, simulation settings, contaminant
strengths, and split labels are deterministic and included in the experiment
fingerprint. The CVZ-like cell is held out because it is both scientifically
important and the most computationally expensive.

The production manifest includes enough lag coverage for the second ACF peak
at every registered trial \(\Delta\nu\). Use a new output directory if you
created a dry-run manifest before this full-grid lag correction; the
fingerprint mismatch is intentional and prevents incompatible checkpoints from
being combined.

```python
from urdr import make_tess_scientific_grid, run_synthetic_experiment

grid = make_tess_scientific_grid()
run = run_synthetic_experiment(
    grid.validation_plan,
    "results/tess-synthetic-v1",
    workers=8,
    resume=True,
)
```

The equivalent command-line entry point is convenient on a compute node:

```bash
urdr-tess-grid --dry-run
urdr-tess-grid results/tess-synthetic-v1 --workers 8
```

EACF autocorrelations use zero-padded FFTs and reuse the observing-window pair
counts across filter centres. This keeps long multi-sector and CVZ-like cases
at approximately \(N\log N\), rather than quadratic, scaling with cadence
count. Production runs remain simulation-heavy and should still use the
checkpointed command on a compute node.

The same object provides nine matching `BackgroundBenchmarkCase` objects and
the pre-registered \(3\times3\) candidate set around the legacy
\(a=0.66,\ b=0.88\) law. After that benchmark, use
`assess_background_calibration` to test whether any training-frontier
candidate also survives the held-out and regime-specific Pareto frontiers.
`assess_validation` separately reports signal detection, \(\Delta\nu\)
recovery, worst hard-negative rate, Brier score, and expected calibration
error. No single score combines discrimination and probability calibration.

## Installation

```bash
python -m pip install -e ".[test]"
pytest
```

Install the optional AsteroScale integration for the informed-search example:

```bash
python -m pip install -e ".[test,asteroscale]"
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

`AsteroScaleSamples` accepts AsteroScale's public `numax`, `dnu`, and
`FWHM_env` output names as well as the legacy Urdr aliases. Correlated samples
remain paired:

```python
from urdr import AsteroScaleSamples, PublishedEACFSearch

samples = AsteroScaleSamples.from_mapping(asteroscale_output)
search_region = samples.search_region()
simulation_parameters = samples.median_parameters()
published_search = PublishedEACFSearch.from_asteroscale(samples)
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

## Broad synthetic validation

The target-specific detector should be evaluated on regimes that were not used
to choose its background law or diagnostic design. `ValidationCase` records a
pre-registered cell in \(\nu_{\max}\), signal-to-noise ratio, observing window,
and contaminant strength. `benchmark_synthetic_validation` compares three
methods using paired simulations:

1. the published-style EACF with its broad search and legacy background;
2. an AsteroScale-restricted EACF;
3. the jointly calibrated Urdr detector.

```python
from urdr import ValidationCase, benchmark_synthetic_validation

case = ValidationCase(
    name="held_out_low_duty_cycle",
    split="validation",
    window=window,
    simulation=simulation,
    published_centres_uhz=np.linspace(400.0, 1600.0, 25),
    restricted_centres_uhz=np.linspace(800.0, 1200.0, 9),
    filter_width_uhz=500.0,
    delta_nu_grid_uhz=np.linspace(85.0, 115.0, 31),
    segments_days=((0.0, 13.7), (13.7, 27.4)),
    coherent_contaminants=coherent_contaminants,
    segment_systematics=segment_systematics,
    background=target_aware_background,
)
validation = benchmark_synthetic_validation(
    cases=[training_case, case],
    calibration_realizations=512,
    evaluation_realizations=256,
    target_false_positive_rate=0.01,
    seed=42,
)
records = validation.to_records()
reliability = validation.reliability
```

Calibration and evaluation use independent random streams. Detection rate,
false-positive rate, \(\Delta\nu\) recovery, fractional error, Brier score, and
reliability remain available separately by method, case, split, and simulation
class. The package returns flat records but does not require pandas or a plotting
library.

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
  plots filtered ACFs at individual trial frequencies, visualises the complete
  frequency-lag ridge, and benchmarks the joint morphology check against
  coherent and segment-dependent hard negatives.
- [`notebooks/joint_inference.ipynb`](notebooks/joint_inference.ipynb)
  follows a fresh injection through the Fourier-amplitude taper, filtered time
  series, lag-domain ACF, frequency-lag map, diagnostic vector, calibration,
  and final unified detection result.
- [`notebooks/synthetic_validation.ipynb`](notebooks/synthetic_validation.ipynb)
  compares all three detection methods on a small pre-registered matrix and
  inspects reliability without conflating it with detection performance.
- [`notebooks/tess_scientific_grid.ipynb`](notebooks/tess_scientific_grid.ipynb)
  inspects the frozen production grid, its exact manifest, and the post-run
  decision APIs before launching an expensive experiment.

All public APIs use NumPy-style docstrings. The documentation check can be run
with:

```bash
pydocstyle src/urdr
```
