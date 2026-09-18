import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from aether.evaluation import (
    EvaluationRun,
    Event,
    Scenario,
    SessionMeasurement,
    compare_runs,
    load_scenarios,
    quantiles,
    write_comparison,
)

SUITE = Path(__file__).resolve().parents[2] / "configs/evaluation/scenarios.json"


def run_payload():
    return {
        "run_id": "baseline",
        "suite_version": "english-dialogue-v1",
        "source_revision": "abc",
        "artifact_sha256": "a" * 64,
        "environment": dict.fromkeys(
            [
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
            ],
            "fixture only",
        ),
        "sessions": [
            {
                "session_id": "one",
                "scenario_id": "question-01",
                "seed": 11,
                "clock_basis": "server_monotonic",
                "compute_ms": [0.0, 10.0],
            }
        ],
    }


def test_suite_has_100_distinct_instructed_scenarios():
    suite = load_scenarios(SUITE)
    assert len(suite.scenarios) == 100
    assert len({s.category for s in suite.scenarios}) == 10
    assert all(
        len([s for s in suite.scenarios if s.category == c]) == 10
        for c in {s.category for s in suite.scenarios}
    )
    assert len({tuple(e.instruction for e in s.events) for s in suite.scenarios}) == 100


def test_quantiles_missing_zero_and_interpolation():
    assert quantiles(()) == {"count": 0, "p50": None, "p95": None, "p99": None}
    assert quantiles((0.0,))["p95"] == 0
    assert quantiles((100.0, 0.0))["p95"] == 95.0
    assert quantiles((5.0, 1.0, 3.0))["p50"] == 3.0
    with pytest.raises(ValueError):
        quantiles((float("nan"),))


@pytest.mark.parametrize(
    "patch",
    [
        {"at_seconds": -1.0},
        {"at_seconds": "1"},
        {"at_seconds": float("inf")},
        {"schema_version": True},
        {"unexpected": 1},
    ],
)
def test_strict_events(patch):
    with pytest.raises(ValidationError):
        Event.model_validate({"at_seconds": 1.0, "kind": "user", "instruction": "hello", **patch})


def test_scenario_out_of_order_and_out_of_bounds():
    base = dict(id="x", category="pause", duration_seconds=3.0, expected=("wait",))

    def event(at: float) -> Event:
        return Event(at_seconds=at, kind="pause", instruction="wait")

    for times in [(2.0, 1.0), (3.0,)]:
        with pytest.raises(ValidationError):
            Scenario(**base, events=tuple(event(t) for t in times))


def test_comparison_preserves_missing_and_reports_partial(tmp_path):
    run = EvaluationRun.model_validate_json(json.dumps(run_payload()))
    report = compare_runs(load_scenarios(SUITE), run, run)
    assert report["complete"] is False
    assert report["paired_trials"] == 1
    assert report["planned_trials"] == 300
    write_comparison(report, tmp_path)
    document = (tmp_path / "comparison.md").read_text()
    assert "5.0" in document and "9.5" in document and "missing" in document
    assert json.loads((tmp_path / "comparison.json").read_text())["complete"] is False


def test_comparison_rejects_unpaired_or_unknown_trials():
    baseline = EvaluationRun.model_validate_json(json.dumps(run_payload()))
    payload = run_payload()
    payload["sessions"][0]["seed"] = 29
    other = EvaluationRun.model_validate_json(json.dumps(payload))
    with pytest.raises(ValueError, match="identical"):
        compare_runs(load_scenarios(SUITE), baseline, other)
    payload["sessions"][0]["scenario_id"] = "not-real"
    other = EvaluationRun.model_validate_json(json.dumps(payload))
    with pytest.raises(ValueError, match="identical"):
        compare_runs(load_scenarios(SUITE), other, other)


def test_duplicate_session_and_incomplete_environment_rejected():
    payload = run_payload()
    payload["sessions"] *= 2
    with pytest.raises(ValidationError):
        EvaluationRun.model_validate_json(json.dumps(payload))
    payload = run_payload()
    del payload["environment"]["gpu"]
    with pytest.raises(ValidationError):
        EvaluationRun.model_validate_json(json.dumps(payload))


def test_negative_response_gap_is_real_overlap_but_negative_compute_is_invalid():
    SessionMeasurement(
        session_id="x",
        scenario_id="x",
        seed=1,
        clock_basis="external_recording",
        response_gap_ms=(-20.0,),
    )
    with pytest.raises(ValidationError):
        SessionMeasurement(
            session_id="x",
            scenario_id="x",
            seed=1,
            clock_basis="server_monotonic",
            compute_ms=(-1.0,),
        )
