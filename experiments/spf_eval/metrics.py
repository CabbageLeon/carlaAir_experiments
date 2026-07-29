"""Metric definitions from Section B.4 of experiment.md."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from statistics import mean, pstdev
from typing import Iterable, Mapping, Sequence


TRACKING_SECONDS = 3.0
STABILITY_SECONDS = 2.0
RECOVERY_IOU = 0.15
RECOVERY_SECONDS = 0.5
RECOVERY_LIMIT_SECONDS = 15.0


@dataclass(frozen=True)
class LandingEpisode:
    seed: int
    visible_seconds: float
    landed_on_bed: bool
    stable_after_touchdown: bool

    @property
    def tracked(self) -> bool:
        return self.visible_seconds >= TRACKING_SECONDS

    @property
    def landed(self) -> bool:
        return self.landed_on_bed and self.stable_after_touchdown


@dataclass(frozen=True)
class EscortEpisode:
    seed: int
    recovered_events: int
    events: int
    reacquisition_times: tuple[float, ...]


def recovery_time(samples: Sequence[tuple[float, float]], onset: float) -> float:
    """First sustained >=0.15-IoU recovery after an occlusion; failures cap at 15 s."""
    sustained_from: float | None = None
    for timestamp, iou in samples:
        if timestamp < onset:
            continue
        if timestamp - onset > RECOVERY_LIMIT_SECONDS:
            break
        if iou >= RECOVERY_IOU:
            sustained_from = timestamp if sustained_from is None else sustained_from
            if timestamp - sustained_from >= RECOVERY_SECONDS:
                return sustained_from - onset
        else:
            sustained_from = None
    return RECOVERY_LIMIT_SECONDS


def _per_seed_rates(episodes: Iterable[LandingEpisode]) -> dict[int, dict[str, float]]:
    grouped: dict[int, list[LandingEpisode]] = defaultdict(list)
    for episode in episodes:
        grouped[episode.seed].append(episode)
    result = {}
    for seed, values in grouped.items():
        tsr = mean(float(value.tracked) for value in values)
        lsr = mean(float(value.landed) for value in values)
        result[seed] = {"TSR": tsr, "LSR": lsr, "CCR": lsr / max(tsr, 0.05)}
    return result


def landing_summary(episodes: Iterable[LandingEpisode], c0_episodes: Iterable[LandingEpisode] | None = None) -> dict[str, float]:
    """Mean/std over seeds, with CG paired against C0 as required by B.4."""
    per_seed = _per_seed_rates(episodes)
    if not per_seed:
        raise ValueError("no landing episodes")
    c0 = _per_seed_rates(c0_episodes or episodes)
    values: dict[str, list[float]] = defaultdict(list)
    for seed, result in per_seed.items():
        for name in ("TSR", "LSR", "CCR"):
            values[name].append(result[name])
        values["CG"].append(result["LSR"] - c0[seed]["LSR"])
    summary: dict[str, float] = {}
    for name, numbers in values.items():
        summary[name] = mean(numbers)
        summary[f"{name}_std"] = pstdev(numbers)
    return summary


def escort_summary(episodes: Iterable[EscortEpisode]) -> dict[str, float]:
    """RSR and capped RAT, reported as mean/std over seeds."""
    grouped: dict[int, list[EscortEpisode]] = defaultdict(list)
    for episode in episodes:
        grouped[episode.seed].append(episode)
    if not grouped:
        raise ValueError("no escort episodes")
    rsr_values, rat_values = [], []
    for values in grouped.values():
        events = sum(value.events for value in values)
        if events == 0:
            raise ValueError("escort episode contains no verified occlusion events")
        rsr_values.append(sum(value.recovered_events for value in values) / events)
        rat_values.append(mean(time for value in values for time in value.reacquisition_times))
    return {
        "RSR": mean(rsr_values),
        "RSR_std": pstdev(rsr_values),
        "RAT": mean(rat_values),
        "RAT_std": pstdev(rat_values),
    }


def _percentile(values: Sequence[float], percentile: float) -> float:
    if not values:
        raise ValueError("no values")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * percentile / 100.0
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def timing_summary(events: Iterable[Mapping[str, object]]) -> dict[str, float]:
    """DF and ECL timing statistics from decision events."""
    decision_events = [event for event in events if event.get("type") == "decision"]
    end_times = [float(event["t"]) for event in events if isinstance(event.get("t"), (int, float))]
    duration = max(end_times) if end_times else 0.0
    ecls = [
        float(event["ecl_seconds"]) * 1000.0
        for event in decision_events
        if isinstance(event.get("ecl_seconds"), (int, float))
    ]
    return {
        "DF": len(decision_events) / duration if duration > 0.0 else 0.0,
        "ECL_mean_ms": mean(ecls) if ecls else 0.0,
        "ECL_p95_ms": _percentile(ecls, 95.0) if ecls else 0.0,
        "ECL_iqr_low_ms": _percentile(ecls, 25.0) if ecls else 0.0,
        "ECL_iqr_high_ms": _percentile(ecls, 75.0) if ecls else 0.0,
    }
