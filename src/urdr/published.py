"""Published Lomb--Scargle EACF workflow and exact-window calibration."""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
from mimir import power_spectrum
from nifty_ls import lombscargle
from numpy.typing import ArrayLike, NDArray
from scipy.fft import ifft, next_fast_len

from .background import BackgroundConfig, EmpiricalBackgroundConfig, estimate_background
from .models import AsteroScaleSamples, TimeSeries
from .simulation import SimulationConfig, simulate_time_series

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class DeltaNuScaling:
    """Power-law physical relation between ``numax`` and ``delta_nu``.

    Parameters
    ----------
    solar_numax_uhz
        Solar reference frequency of maximum oscillation power.
    solar_delta_nu_uhz
        Solar reference large separation.
    exponent
        Exponent of the normalized power law.
    scatter_factor
        Multiplicative half-width of the accepted relation. The published
        value ``10**0.2`` reproduces the banana-shaped search region.
    """

    solar_numax_uhz: float = 3090.0
    solar_delta_nu_uhz: float = 135.1
    exponent: float = 0.791
    scatter_factor: float = 10**0.2

    def __post_init__(self) -> None:
        """Validate the scaling-relation parameters."""
        if min(self.solar_numax_uhz, self.solar_delta_nu_uhz) <= 0:
            raise ValueError("solar reference frequencies must be positive")
        if self.exponent <= 0 or self.scatter_factor <= 1:
            raise ValueError("exponent must be positive and scatter_factor exceed one")

    def predict(self, numax_uhz: ArrayLike) -> FloatArray:
        """Predict ``delta_nu`` in microhertz at each trial ``numax``."""
        numax = np.asarray(numax_uhz, dtype=float)
        if np.any(~np.isfinite(numax)) or np.any(numax <= 0):
            raise ValueError("numax values must be finite and positive")
        return np.asarray(
            self.solar_delta_nu_uhz * (numax / self.solar_numax_uhz) ** self.exponent,
            dtype=float,
        )


@dataclass(frozen=True)
class PublishedEACFSearch:
    """Filter grid and physical lag band for a published-EACF search.

    Parameters
    ----------
    centre_frequencies_uhz
        Trial oscillation-envelope centres in microhertz.
    filter_widths_uhz
        Hanning-filter widths at the trial centres.
    predicted_delta_nu_uhz
        Central large-separation prediction at each trial centre.
    lower_delta_nu_uhz, upper_delta_nu_uhz
        Accepted large-separation interval at each trial centre.
    source
        Short label describing how the search was constructed.
    """

    centre_frequencies_uhz: FloatArray
    filter_widths_uhz: FloatArray
    predicted_delta_nu_uhz: FloatArray
    lower_delta_nu_uhz: FloatArray
    upper_delta_nu_uhz: FloatArray
    source: str = "scaling"

    def __post_init__(self) -> None:
        """Validate and normalize the one-dimensional search arrays."""
        names = (
            "centre_frequencies_uhz",
            "filter_widths_uhz",
            "predicted_delta_nu_uhz",
            "lower_delta_nu_uhz",
            "upper_delta_nu_uhz",
        )
        arrays = [np.asarray(getattr(self, name), dtype=float) for name in names]
        size = arrays[0].size
        if size == 0 or any(array.ndim != 1 or array.size != size for array in arrays):
            raise ValueError(
                "search arrays must be non-empty, one-dimensional, and equal length"
            )
        if any(np.any(~np.isfinite(array)) or np.any(array <= 0) for array in arrays):
            raise ValueError("search values must be finite and positive")
        if np.any(np.diff(arrays[0]) <= 0):
            raise ValueError("filter centres must be strictly increasing")
        if np.any(arrays[3] >= arrays[4]):
            raise ValueError("lower delta_nu bounds must be below upper bounds")
        for name, array in zip(names, arrays, strict=True):
            object.__setattr__(self, name, array)

    @classmethod
    def from_scaling(
        cls,
        centre_frequencies_uhz: ArrayLike,
        *,
        filter_widths_uhz: ArrayLike | None = None,
        scaling: DeltaNuScaling | None = None,
    ) -> PublishedEACFSearch:
        """Construct the broad published scaling-relation search."""
        centres = _centres_array(centre_frequencies_uhz)
        widths = _widths_array(centres, filter_widths_uhz)
        relation = scaling or DeltaNuScaling()
        predicted = relation.predict(centres)
        return cls(
            centres,
            widths,
            predicted,
            predicted / relation.scatter_factor,
            predicted * relation.scatter_factor,
            "scaling",
        )

    @classmethod
    def from_asteroscale(
        cls,
        samples: AsteroScaleSamples,
        *,
        centre_count: int = 21,
        credible_mass: float = 0.99,
        conditional_neighbours: int | None = None,
        point_numax_fraction: float = 0.25,
        point_delta_nu_factor: float = 10**0.2,
    ) -> PublishedEACFSearch:
        """Construct a target-informed search from paired AsteroScale output.

        Multiple posterior samples define local conditional intervals in the
        paired ``numax``--``delta_nu`` distribution. A scalar AsteroScale point
        prediction falls back to a deliberately broad fractional search and
        the published ``delta_nu`` scatter factor.
        """
        if not isinstance(centre_count, int) or centre_count < 3:
            raise ValueError("centre_count must be an integer of at least three")
        if not 0 < credible_mass < 1:
            raise ValueError("credible_mass must lie between zero and one")
        if not 0 < point_numax_fraction < 1:
            raise ValueError("point_numax_fraction must lie between zero and one")
        if point_delta_nu_factor <= 1:
            raise ValueError("point_delta_nu_factor must exceed one")

        numax = samples.numax_uhz
        delta_nu = samples.delta_nu_uhz
        widths = samples.envelope_width_uhz
        tail = 0.5 * (1.0 - credible_mass)
        lower_numax, upper_numax = np.quantile(numax, [tail, 1.0 - tail])
        if numax.size == 1 or np.isclose(lower_numax, upper_numax):
            centre = float(np.median(numax))
            centres = np.linspace(
                centre * (1.0 - point_numax_fraction),
                centre * (1.0 + point_numax_fraction),
                centre_count,
            )
            prediction = float(np.median(delta_nu)) * (centres / centre) ** 0.791
            filter_widths = float(np.median(widths)) * (centres / centre) ** 0.88
            lower = prediction / point_delta_nu_factor
            upper = prediction * point_delta_nu_factor
        else:
            centres = np.linspace(float(lower_numax), float(upper_numax), centre_count)
            if conditional_neighbours is None:
                neighbours = max(64, int(np.ceil(0.1 * numax.size)))
            else:
                neighbours = int(conditional_neighbours)
            if neighbours < 8:
                raise ValueError("conditional_neighbours must be at least eight")
            neighbours = min(neighbours, numax.size)
            prediction = np.empty(centre_count, dtype=float)
            filter_widths = np.empty(centre_count, dtype=float)
            lower = np.empty(centre_count, dtype=float)
            upper = np.empty(centre_count, dtype=float)
            log_numax = np.log(numax)
            for index, centre in enumerate(centres):
                nearest = np.argpartition(
                    np.abs(log_numax - np.log(centre)), neighbours - 1
                )[:neighbours]
                prediction[index] = np.median(delta_nu[nearest])
                filter_widths[index] = np.median(widths[nearest])
                lower[index], upper[index] = np.quantile(
                    delta_nu[nearest], [tail, 1.0 - tail]
                )
            too_narrow = np.isclose(lower, upper)
            lower[too_narrow] = prediction[too_narrow] / point_delta_nu_factor
            upper[too_narrow] = prediction[too_narrow] * point_delta_nu_factor
        return cls(
            centres,
            filter_widths,
            prediction,
            lower,
            upper,
            "asteroscale",
        )

    def physical_mask(self, lags_seconds: ArrayLike) -> NDArray[np.bool_]:
        """Return the accepted frequency--lag region on a lag grid."""
        lags = np.asarray(lags_seconds, dtype=float)
        lower_lag = 1e6 / self.upper_delta_nu_uhz
        upper_lag = 1e6 / self.lower_delta_nu_uhz
        return (lags[None, :] >= lower_lag[:, None]) & (
            lags[None, :] <= upper_lag[:, None]
        )


@dataclass(frozen=True)
class PublishedEACFMap:
    """Published complex-modulus EACF evaluated over frequency and lag.

    Parameters
    ----------
    centre_frequencies_uhz
        Trial filter centres, interpreted as candidate ``numax`` values.
    lags_seconds
        Non-negative lag grid.
    values
        Squared complex EACF magnitude, normalized to unity at zero lag.
    predicted_delta_nu_uhz
        Scaling-relation prediction at each filter centre.
    physical_mask
        Boolean mask selecting physically plausible ``numax``--``delta_nu``
        combinations.
    search_source
        Label identifying the broad scaling or AsteroScale-informed search.
    """

    centre_frequencies_uhz: FloatArray
    lags_seconds: FloatArray
    values: FloatArray
    predicted_delta_nu_uhz: FloatArray
    physical_mask: NDArray[np.bool_]
    search_source: str = "scaling"

    def collapsed(self) -> FloatArray:
        """Return the mean EACF response inside the physical search band."""
        sums = np.sum(np.where(self.physical_mask, self.values, 0.0), axis=1)
        counts = np.sum(self.physical_mask, axis=1)
        if np.any(counts == 0):
            raise ValueError("physical search band contains no lag samples")
        return np.asarray(sums / counts, dtype=float)

    @property
    def global_statistic(self) -> float:
        """Return the maximum collapsed response across all trial centres."""
        return float(np.max(self.collapsed()))

    @property
    def best_numax_uhz(self) -> float:
        """Return the filter centre with the largest collapsed response."""
        return float(self.centre_frequencies_uhz[np.argmax(self.collapsed())])

    @property
    def best_delta_nu_uhz(self) -> float:
        """Return the strongest lag within the best physical search row."""
        row = int(np.argmax(self.collapsed()))
        selected = np.flatnonzero(self.physical_mask[row])
        lag_index = selected[np.argmax(self.values[row, selected])]
        return float(1e6 / self.lags_seconds[lag_index])


@dataclass(frozen=True)
class PublishedEACFDetection:
    """Observed published EACF and its exact-window null calibration."""

    observed: PublishedEACFMap
    null_statistics: FloatArray
    pointwise_exceedances: NDArray[np.int64]

    def __post_init__(self) -> None:
        """Validate the pointwise exact-window calibration."""
        counts = np.asarray(self.pointwise_exceedances, dtype=np.int64)
        if counts.shape != self.observed.values.shape:
            raise ValueError("pointwise exceedances must match the observed EACF map")
        if np.any(counts < 0) or np.any(counts > self.null_statistics.size):
            raise ValueError("pointwise exceedances are outside the simulation count")
        object.__setattr__(self, "pointwise_exceedances", counts)

    @property
    def false_alarm_probability(self) -> float:
        """Return the finite-sample global false-alarm probability."""
        exceedances = np.count_nonzero(
            self.null_statistics >= self.observed.global_statistic
        )
        return float((exceedances + 1) / (self.null_statistics.size + 1))

    @property
    def detection_merit(self) -> float:
        """Return one minus the globally calibrated false-alarm probability."""
        return 1.0 - self.false_alarm_probability

    @property
    def pointwise_false_alarm_probability(self) -> FloatArray:
        """Return empirical tail probabilities at every frequency--lag pixel.

        These probabilities account for the target's sampling window and the
        complete background-estimation pipeline, but not for searching across
        pixels. Use :attr:`false_alarm_probability` for global significance.
        """
        return np.asarray(
            (self.pointwise_exceedances + 1) / (self.null_statistics.size + 1),
            dtype=float,
        )

    @property
    def window_aware_score(self) -> FloatArray:
        """Return the pointwise ``-log10(p)`` exact-window significance map."""
        return np.asarray(
            -np.log10(self.pointwise_false_alarm_probability), dtype=float
        )

    def window_aware_collapsed(self) -> FloatArray:
        """Mean the pointwise significance inside each physical lag band."""
        score = self.window_aware_score
        sums = np.sum(np.where(self.observed.physical_mask, score, 0.0), axis=1)
        counts = np.sum(self.observed.physical_mask, axis=1)
        if np.any(counts == 0):
            raise ValueError("physical search band contains no lag samples")
        return np.asarray(sums / counts, dtype=float)

    @property
    def window_aware_best_numax_uhz(self) -> float:
        """Return the trial centre with greatest mean pointwise significance."""
        row = int(np.argmax(self.window_aware_collapsed()))
        return float(self.observed.centre_frequencies_uhz[row])

    @property
    def window_aware_best_delta_nu_uhz(self) -> float:
        """Return the raw-EACF peak lag in the best window-aware row.

        The window-aware collapse selects the filter row. Within that row the
        raw EACF peak is used because selecting the smallest pointwise tail
        probability would introduce another uncalibrated lag-wise maximum.
        """
        row = int(np.argmax(self.window_aware_collapsed()))
        selected = np.flatnonzero(self.observed.physical_mask[row])
        lag_index = selected[np.argmax(self.observed.values[row, selected])]
        return float(1e6 / self.observed.lags_seconds[lag_index])


def envelope_width_uhz(numax_uhz: ArrayLike) -> FloatArray:
    """Return the published cool-star p-mode-envelope width relation."""
    numax = np.asarray(numax_uhz, dtype=float)
    if np.any(~np.isfinite(numax)) or np.any(numax <= 0):
        raise ValueError("numax values must be finite and positive")
    return np.asarray(0.66 * numax**0.88, dtype=float)


def compute_published_eacf_map(
    series: TimeSeries,
    centre_frequencies_uhz: ArrayLike | None = None,
    *,
    search: PublishedEACFSearch | None = None,
    filter_widths_uhz: ArrayLike | None = None,
    max_lag_seconds: float | None = None,
    background: BackgroundConfig | None = None,
    scaling: DeltaNuScaling | None = None,
    spectrum_oversampling: int = 1,
    lag_oversampling: int = 2,
    fft_workers: int | None = None,
) -> PublishedEACFMap:
    """Compute the published Lomb--Scargle, Hanning-filtered EACF map.

    Mimir evaluates a critically sampled Lomb--Scargle power-density spectrum
    using only genuinely observed cadences. Urdr divides it by an empirical
    background, applies a Hanning filter at each trial centre, and computes the
    squared modulus of the complex inverse transform. The modulus removes the
    rapid carrier visible in a squared real ACF.

    Parameters
    ----------
    series
        Uniform cadence grid and exact observing mask. Only observed samples
        are supplied to Mimir.
    centre_frequencies_uhz
        Trial filter centres in microhertz. Omit when ``search`` is supplied.
    search
        Precomputed broad-scaling or AsteroScale-informed search definition.
    filter_widths_uhz
        Optional scalar or one width per centre. By default the published
        p-mode-envelope width scaling is used.
    max_lag_seconds
        Optional largest returned lag.
    background
        Background estimator. The legacy empirical estimator is the default.
    scaling
        Physical ``numax``--``delta_nu`` relation and accepted scatter.
    spectrum_oversampling
        Mimir Lomb--Scargle frequency oversampling.
    lag_oversampling
        Zero-padding factor used to refine the inverse-transform lag grid.
    fft_workers
        Worker count passed to SciPy's inverse FFT. Use ``-1`` for all cores.

    Returns
    -------
    PublishedEACFMap
        Smooth complex-modulus EACF map and physical search mask.
    """
    if not isinstance(lag_oversampling, int) or lag_oversampling < 1:
        raise ValueError("lag_oversampling must be a positive integer")
    resolved_search = _resolve_search(
        centre_frequencies_uhz,
        search,
        filter_widths_uhz,
        scaling,
    )
    spectrum, snr = _spectrum_and_snr(
        series,
        background or EmpiricalBackgroundConfig(),
        oversampling=spectrum_oversampling,
    )
    plan = _prepare_plan(
        np.asarray(spectrum.frequency, dtype=float),
        float(spectrum.frequency_spacing),
        resolved_search,
        max_lag_seconds,
        lag_oversampling,
    )
    return _map_from_snr(snr, plan, fft_workers)


def calibrate_published_eacf(
    series: TimeSeries,
    simulation: SimulationConfig,
    centre_frequencies_uhz: ArrayLike | None = None,
    *,
    search: PublishedEACFSearch | None = None,
    simulations: int = 128,
    seed: int = 0,
    batch_size: int = 16,
    fft_workers: int | None = None,
    **map_kwargs: object,
) -> PublishedEACFDetection:
    """Calibrate the global published-EACF statistic for one target window.

    Each null realization uses the target's exact cadence grid and observing
    mask. Null periodograms are evaluated in bounded batches using the same
    ``nifty-ls`` backend and Parseval normalization as Mimir. Granulation
    backgrounds are re-estimated independently. Urdr retains both the maximum
    collapsed response for global calibration and the number of null
    exceedances at every frequency--lag pixel for window-aware localisation.
    """
    if simulations < 8:
        raise ValueError("at least eight null simulations are required")
    if not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")

    allowed = {
        "filter_widths_uhz",
        "max_lag_seconds",
        "background",
        "scaling",
        "spectrum_oversampling",
        "lag_oversampling",
    }
    unexpected = set(map_kwargs) - allowed
    if unexpected:
        raise TypeError(f"unexpected map keyword arguments: {sorted(unexpected)}")
    filter_widths = map_kwargs.get("filter_widths_uhz")
    scaling = map_kwargs.get("scaling")
    if scaling is not None and not isinstance(scaling, DeltaNuScaling):
        raise TypeError("scaling must be a DeltaNuScaling")
    resolved_search = _resolve_search(
        centre_frequencies_uhz,
        search,
        filter_widths,
        scaling,
    )
    settings = map_kwargs.get("background") or EmpiricalBackgroundConfig()
    spectrum_oversampling = int(map_kwargs.get("spectrum_oversampling", 1))
    lag_oversampling = int(map_kwargs.get("lag_oversampling", 2))
    spectrum, observed_snr = _spectrum_and_snr(
        series,
        settings,
        oversampling=spectrum_oversampling,
    )
    plan = _prepare_plan(
        np.asarray(spectrum.frequency, dtype=float),
        float(spectrum.frequency_spacing),
        resolved_search,
        map_kwargs.get("max_lag_seconds"),
        lag_oversampling,
    )
    observed = _map_from_snr(observed_snr, plan, fft_workers)
    null = np.empty(simulations, dtype=float)
    pointwise_exceedances = np.zeros(observed.values.shape, dtype=np.int64)
    null_config = replace(simulation, oscillation_amplitude=0.0)
    seeds = np.random.SeedSequence(seed).spawn(simulations)
    for start in range(0, simulations, batch_size):
        stop = min(start + batch_size, simulations)
        candidates = [
            simulate_time_series(
                series.window,
                null_config,
                np.random.default_rng(child_seed),
                include_oscillations=False,
            )
            for child_seed in seeds[start:stop]
        ]
        power_density = _batched_null_power_density(
            candidates,
            np.asarray(spectrum.frequency, dtype=float),
            float(spectrum.frequency_spacing),
            spectrum_oversampling,
        )
        snr_batch = np.empty((stop - start, power_density.shape[1] + 1), dtype=float)
        snr_batch[:, 0] = 0.0
        for index, density in enumerate(power_density):
            noise = estimate_background(plan.frequency_uhz[1:], density, settings)
            snr_batch[index, 1:] = density / noise
        statistics, exceedances = _calibration_from_snr_batch(
            snr_batch,
            plan,
            fft_workers,
            observed.values,
        )
        null[start:stop] = statistics
        pointwise_exceedances += exceedances
    return PublishedEACFDetection(observed, null, pointwise_exceedances)


@dataclass(frozen=True)
class _FilterBand:
    """Compact non-zero section of one Hanning filter."""

    start: int
    stop: int
    weights: FloatArray


@dataclass(frozen=True)
class _PublishedPlan:
    """Static frequency, filter, lag, and physical-mask calibration plan."""

    frequency_uhz: FloatArray
    lags_seconds: FloatArray
    lag_selection: NDArray[np.bool_]
    filters: tuple[_FilterBand, ...]
    physical_mask: NDArray[np.bool_]
    transform_size: int
    search: PublishedEACFSearch


def _resolve_search(
    centre_frequencies_uhz: ArrayLike | None,
    search: PublishedEACFSearch | None,
    filter_widths_uhz: ArrayLike | None,
    scaling: DeltaNuScaling | None,
) -> PublishedEACFSearch:
    if search is not None:
        if (
            centre_frequencies_uhz is not None
            or filter_widths_uhz is not None
            or scaling is not None
        ):
            raise ValueError(
                "search cannot be combined with centre frequencies, filter widths, or scaling"
            )
        return search
    if centre_frequencies_uhz is None:
        raise ValueError("filter centres or a search definition are required")
    return PublishedEACFSearch.from_scaling(
        centre_frequencies_uhz,
        filter_widths_uhz=filter_widths_uhz,
        scaling=scaling,
    )


def _spectrum_and_snr(
    series: TimeSeries,
    background: BackgroundConfig,
    *,
    oversampling: int,
) -> tuple[object, FloatArray]:
    observed = series.observed
    spectrum = power_spectrum(
        time=series.time[observed],
        flux=series.flux[observed],
        oversampling=oversampling,
        time_unit="d",
        frequency_unit="uHz",
    )
    frequency = np.asarray(spectrum.frequency, dtype=float)
    density = np.asarray(spectrum.power_density, dtype=float)
    noise = estimate_background(frequency, density, background)
    return spectrum, np.concatenate(([0.0], density / noise))


def _prepare_plan(
    frequency_uhz: FloatArray,
    frequency_spacing_uhz: float,
    search: PublishedEACFSearch,
    max_lag_seconds: object,
    lag_oversampling: int,
) -> _PublishedPlan:
    if not isinstance(lag_oversampling, int) or lag_oversampling < 1:
        raise ValueError("lag_oversampling must be a positive integer")
    frequency = np.concatenate(([0.0], frequency_uhz))
    transform_size = next_fast_len(lag_oversampling * frequency.size)
    spacing_hz = frequency_spacing_uhz * 1e-6
    all_lags = np.arange(transform_size, dtype=float) / (transform_size * spacing_hz)
    selected = all_lags <= 0.5 / spacing_hz
    if max_lag_seconds is not None:
        maximum = float(max_lag_seconds)
        if not np.isfinite(maximum) or maximum <= 0:
            raise ValueError("max_lag_seconds must be finite and positive")
        selected &= all_lags <= maximum
    lags = all_lags[selected]
    physical = search.physical_mask(lags)
    if np.any(np.sum(physical, axis=1) == 0):
        raise ValueError("lag grid does not cover the physical search relation")
    filters = []
    for centre, width in zip(
        search.centre_frequencies_uhz,
        search.filter_widths_uhz,
        strict=True,
    ):
        weights = _hanning_filter(frequency, centre, width)
        nonzero = np.flatnonzero(weights > 0)
        start, stop = int(nonzero[0]), int(nonzero[-1] + 1)
        filters.append(_FilterBand(start, stop, weights[start:stop]))
    return _PublishedPlan(
        frequency,
        lags,
        selected,
        tuple(filters),
        physical,
        transform_size,
        search,
    )


def _map_from_snr(
    snr: FloatArray,
    plan: _PublishedPlan,
    fft_workers: int | None,
) -> PublishedEACFMap:
    rows = []
    for band in plan.filters:
        acf = ifft(
            snr[band.start : band.stop] * band.weights,
            n=plan.transform_size,
            workers=fft_workers,
        )[plan.lag_selection]
        zero_power = float(np.abs(acf[0]) ** 2)
        if zero_power <= 0:
            raise ValueError("filtered signal-to-noise spectrum has zero power")
        rows.append(np.asarray(np.abs(acf) ** 2 / zero_power, dtype=float))
    return PublishedEACFMap(
        plan.search.centre_frequencies_uhz,
        plan.lags_seconds,
        np.vstack(rows),
        plan.search.predicted_delta_nu_uhz,
        plan.physical_mask,
        plan.search.source,
    )


def _batched_null_power_density(
    candidates: list[TimeSeries],
    frequency_uhz: FloatArray,
    frequency_spacing_uhz: float,
    oversampling: int,
) -> FloatArray:
    observed = candidates[0].observed
    time = candidates[0].time[observed]
    flux = np.stack([candidate.flux[observed] for candidate in candidates])
    spacing_native = 1.0 / ((time[-1] - time[0]) * oversampling)
    result = lombscargle(
        time,
        flux,
        fmin=spacing_native,
        fmax=frequency_uhz.size * spacing_native,
        Nf=frequency_uhz.size,
        center_data=True,
        fit_mean=True,
        normalization="psd",
        assume_sorted_t=True,
        backend="auto",
    )
    raw = np.atleast_2d(np.asarray(result.power, dtype=float))
    variance = np.mean((flux - np.mean(flux, axis=1, keepdims=True)) ** 2, axis=1)
    totals = np.sum(raw, axis=1)
    if np.any(totals <= 0) or np.any(~np.isfinite(totals)):
        raise RuntimeError("nifty-ls returned no positive finite null power")
    power = raw * (variance / totals)[:, None]
    return np.asarray(power / frequency_spacing_uhz, dtype=float)


def _calibration_from_snr_batch(
    snr: FloatArray,
    plan: _PublishedPlan,
    fft_workers: int | None,
    observed_values: FloatArray,
) -> tuple[FloatArray, NDArray[np.int64]]:
    maxima = np.full(snr.shape[0], -np.inf, dtype=float)
    exceedances = np.zeros(observed_values.shape, dtype=np.int64)
    for row, band in enumerate(plan.filters):
        acf = ifft(
            snr[:, band.start : band.stop] * band.weights,
            n=plan.transform_size,
            axis=1,
            workers=fft_workers,
        )
        zero_power = np.abs(acf[:, 0]) ** 2
        if np.any(zero_power <= 0):
            raise ValueError("filtered signal-to-noise spectrum has zero power")
        values = np.asarray(
            np.abs(acf[:, plan.lag_selection]) ** 2 / zero_power[:, None],
            dtype=float,
        )
        exceedances[row] = np.count_nonzero(
            values >= observed_values[row][None, :], axis=0
        )
        collapsed = np.mean(values[:, plan.physical_mask[row]], axis=1)
        maxima = np.maximum(maxima, collapsed)
    return maxima, exceedances


def _centres_array(values: ArrayLike) -> FloatArray:
    centres = np.atleast_1d(np.asarray(values, dtype=float))
    if centres.ndim != 1 or centres.size == 0:
        raise ValueError("at least one filter centre is required")
    if np.any(~np.isfinite(centres)) or np.any(centres <= 0):
        raise ValueError("filter centres must be finite and positive")
    if np.any(np.diff(centres) <= 0):
        raise ValueError("filter centres must be strictly increasing")
    return centres


def _widths_array(centres: FloatArray, values: ArrayLike | None) -> FloatArray:
    if values is None:
        return envelope_width_uhz(centres)
    widths = np.asarray(values, dtype=float)
    if widths.ndim == 0:
        widths = np.full(centres.size, float(widths))
    if widths.shape != centres.shape:
        raise ValueError("filter widths must be scalar or match filter centres")
    if np.any(~np.isfinite(widths)) or np.any(widths <= 0):
        raise ValueError("filter widths must be finite and positive")
    return widths


def _hanning_filter(
    frequency_uhz: FloatArray,
    centre_uhz: float,
    width_uhz: float,
) -> FloatArray:
    """Return the Eq. 15 Hanning filter on a frequency grid."""
    offset = frequency_uhz - centre_uhz
    inside = np.abs(offset) <= width_uhz / 2.0
    if np.count_nonzero(inside) < 3:
        raise ValueError("Hanning filter contains fewer than three bins")
    weights = np.zeros_like(frequency_uhz)
    weights[inside] = 0.5 + 0.5 * np.cos(2.0 * np.pi * offset[inside] / width_uhz)
    return weights
