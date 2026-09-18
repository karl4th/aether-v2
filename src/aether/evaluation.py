"""Versioned dialogue evaluation plans and reports; never infer unmeasured quality."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Annotated, Literal, Self

from pydantic import Field, model_validator

from aether.config import StrictConfig

Text = Annotated[str, Field(min_length=1)]
Time = Annotated[float, Field(ge=0, allow_inf_nan=False)]
Category = Literal[
    "question",
    "pause",
    "correction",
    "interruption",
    "backchannel",
    "overlap",
    "silence",
    "noise",
    "context",
    "reconnect",
]


class Event(StrictConfig):
    at_seconds: Time
    kind: Literal[
        "user",
        "pause",
        "interrupt",
        "backchannel",
        "overlap",
        "silence",
        "noise",
        "disconnect",
        "reconnect",
    ]
    instruction: Text


class Scenario(StrictConfig):
    id: Text
    category: Category
    duration_seconds: Annotated[float, Field(gt=0)]
    events: tuple[Event, ...]
    expected: tuple[Text, ...]

    @model_validator(mode="after")
    def timeline(self) -> Self:
        times = [event.at_seconds for event in self.events]
        if not times or times != sorted(times) or times[-1] >= self.duration_seconds:
            raise ValueError("events must be ordered and within scenario duration")
        if not self.expected:
            raise ValueError("expected behavior is required")
        return self


class ScenarioSuite(StrictConfig):
    suite_version: Text
    language: Literal["en"] = "en"
    seeds: tuple[Annotated[int, Field(ge=0)], ...] = (11, 29, 47)
    scenarios: tuple[Scenario, ...]

    @model_validator(mode="after")
    def coverage(self) -> Self:
        if len(self.seeds) < 3 or len(set(self.seeds)) != len(self.seeds):
            raise ValueError("at least three distinct seeds required")
        ids = [scenario.id for scenario in self.scenarios]
        if len(ids) < 100 or len(set(ids)) != len(ids):
            raise ValueError("at least 100 uniquely identified scenarios required")
        categories = {scenario.category for scenario in self.scenarios}
        if len(categories) != 10:
            raise ValueError("all ten categories required")
        return self


def load_scenarios(path: Path) -> ScenarioSuite:
    return ScenarioSuite.model_validate_json(path.read_text(encoding="utf-8"))


class HumanReview(StrictConfig):
    reviewer: Text
    content_correct: bool | None = None
    constraints_followed: bool | None = None
    intelligible: bool | None = None
    audio_artifacts: bool | None = None
    natural_pauses: bool | None = None
    notes: str = ""


class SessionMeasurement(StrictConfig):
    session_id: Text
    scenario_id: Text
    seed: Annotated[int, Field(ge=0)]
    recording_uri: Text | None = None
    clock_basis: Literal["server_monotonic", "external_recording", "gpu_timer"]
    compute_ms: tuple[Time, ...] = ()
    pipeline_ms: tuple[Time, ...] = ()
    response_gap_ms: tuple[Annotated[float, Field(allow_inf_nan=False)], ...] = ()
    yield_ms: tuple[Time, ...] = ()
    cold_start_ms: Time | None = None
    failures: tuple[Text, ...] = ()
    human_review: HumanReview | None = None


class EvaluationRun(StrictConfig):
    run_id: Text
    suite_version: Text
    source_revision: Text
    artifact_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    environment: dict[str, Text]
    sessions: tuple[SessionMeasurement, ...]

    @model_validator(mode="after")
    def identities(self) -> Self:
        ids = [session.session_id for session in self.sessions]
        pairs = [(session.scenario_id, session.seed) for session in self.sessions]
        if len(ids) != len(set(ids)) or len(pairs) != len(set(pairs)):
            raise ValueError("duplicate session or scenario/seed")
        required = {
            "gpu",
            "cpu",
            "ram",
            "driver",
            "versions",
            "dtype",
            "sessions",
            "network",
            "packet_samples",
            "duration_seconds",
        }
        if not required <= self.environment.keys():
            raise ValueError("missing measurement environment fields")
        return self


def quantiles(values: tuple[float, ...]) -> dict[str, int | float | None]:
    """Linear interpolation at (n-1)*p; empty samples remain missing, not zero."""
    if any(not math.isfinite(value) for value in values):
        raise ValueError("nonfinite measurement")
    ordered = sorted(values)
    result: dict[str, int | float | None] = {"count": len(values)}
    for name, p in (("p50", 0.5), ("p95", 0.95), ("p99", 0.99)):
        if not ordered:
            result[name] = None
            continue
        position = (len(ordered) - 1) * p
        lower = math.floor(position)
        upper = math.ceil(position)
        result[name] = ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)
    return result


def compare_runs(
    suite: ScenarioSuite, baseline: EvaluationRun, candidate: EvaluationRun
) -> dict[str, object]:
    """Require paired trials; report descriptive data without claiming quality gains."""
    expected = {(scenario.id, seed) for scenario in suite.scenarios for seed in suite.seeds}
    pairs = [{(s.scenario_id, s.seed) for s in run.sessions} for run in (baseline, candidate)]
    if any(run.suite_version != suite.suite_version for run in (baseline, candidate)):
        raise ValueError("suite version mismatch")
    if pairs[0] != pairs[1] or not pairs[0] or not pairs[0] <= expected:
        raise ValueError("runs must contain identical nonempty scenario/seed pairs from suite")
    runs: dict[str, object] = {}
    for label, run in (("baseline", baseline), ("candidate", candidate)):
        per_session = []
        for session in run.sessions:
            metrics = {
                name: quantiles(getattr(session, name))
                for name in ("compute_ms", "pipeline_ms", "response_gap_ms", "yield_ms")
            }
            per_session.append({**session.model_dump(mode="json"), "metrics": metrics})
        runs[label] = {"run": run.model_dump(mode="json"), "per_session": per_session}
    return {
        "schema_version": 1,
        "suite_version": suite.suite_version,
        "paired_trials": len(pairs[0]),
        "planned_trials": len(expected),
        "complete": pairs[0] == expected,
        "runs": runs,
        "decision": "Human review and remote acceptance required; no automatic verdict.",
    }


def write_comparison(report: dict[str, object], output: Path) -> None:
    """Write machine-readable report and Markdown with actual per-session quantiles."""
    output.mkdir(parents=True, exist_ok=True)
    (output / "comparison.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# aether evaluation",
        "",
        f"Paired trials: {report['paired_trials']} / {report['planned_trials']}",
        "",
        str(report["decision"]),
        "",
        "Measured per-session distributions are preserved in comparison.json.",
        "",
        "No missing measurement is represented as zero.",
    ]
    runs = report["runs"]
    if not isinstance(runs, dict):
        raise ValueError("invalid report runs")
    for label, data in runs.items():
        # data["run"] is our own model_dump(mode="json") output: tuples became lists,
        # so re-parsing this trusted round-trip needs lax (non-strict) validation.
        run = EvaluationRun.model_validate(data["run"], strict=False)
        lines += [
            "",
            f"## {label}: {run.run_id}",
            "",
            "| Session | Metric | n | p50 | p95 | p99 |",
            "|---|---|---:|---:|---:|---:|",
        ]
        for session in run.sessions:
            for name in ("compute_ms", "pipeline_ms", "response_gap_ms", "yield_ms"):
                stats = quantiles(getattr(session, name))
                cells = [
                    "missing" if stats[key] is None else str(stats[key])
                    for key in ("count", "p50", "p95", "p99")
                ]
                safe_id = session.session_id.replace("|", "\\|").replace("\n", " ")
                lines.append(f"| {safe_id} | {name} | " + " | ".join(cells) + " |")
            reviewed = "yes" if session.human_review else "no"
            lines += [
                "",
                f"Human review present: {reviewed}; failures: {len(session.failures)}.",
                "",
            ]
    (output / "comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
