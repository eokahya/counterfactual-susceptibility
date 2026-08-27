from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any, ClassVar

from cfsus.stage1c_v3.execution_journal import CanonicalExecutionJournal
from cfsus.stage1c_v3.intervention import applied_plan_record
from cfsus.stage1c_v3.prediction import build_prospective_pair
from cfsus.stage1c_v3.quantization_audit import bf16_round
from cfsus.stage1f import (
    EXPERIMENT_CLASS,
    RUNTIME_FINGERPRINT,
    build_protocol_manifest,
    build_records,
    classify_terminal,
    execute_frozen_pairs,
    first_probe_mapping,
    load_config,
    publish_records,
    select_q_panel,
    validate_bundle,
    validate_prediction,
)
from cfsus.types import (
    FeatureActivity,
    FeatureRef,
    MeasuredFeatureState,
    NearThresholdCandidate,
)


class FakeTensor:
    def __init__(self, value: float, dtype: object) -> None:
        self.value = bf16_round(value)
        self.dtype = dtype
        self.device = SimpleNamespace(type="mps")

    def reshape(self, _: object) -> FakeTensor:
        return self

    def item(self) -> float:
        return self.value

    def numel(self) -> int:
        return 1


class FakeMPS:
    @staticmethod
    def empty_cache() -> None:
        return None


class FakeTorch:
    Tensor = FakeTensor
    bfloat16 = object()
    mps = FakeMPS()

    @classmethod
    def tensor(cls, value: float, *, device: str, dtype: object) -> FakeTensor:
        assert device == "mps"
        assert dtype is cls.bfloat16
        return FakeTensor(value, dtype)


class FakeSampler:
    @contextmanager
    def stage(self, _: str) -> Iterator[None]:
        yield


class FakeBackend:
    rows: ClassVar[dict[str, dict[str, Any]]] = {}

    def __init__(
        self,
        _: Any,
        *,
        prompt: str,
        prompt_id: str,
        torch: Any,
        token_count: int,
        attempt_recorder: Any,
        call_index_offset: int,
    ) -> None:
        del prompt, torch, token_count
        self.prompt_id = prompt_id
        self.attempt_recorder = attempt_recorder
        self.source_suppression_api_calls = call_index_offset

    def measure_states(
        self, features: tuple[FeatureRef, FeatureRef]
    ) -> dict[FeatureRef, MeasuredFeatureState]:
        row = next(
            value
            for value in self.rows.values()
            if value["prompt_id"] == self.prompt_id
            and FeatureRef(**value["source"]) == features[0]
            and FeatureRef(**value["target"]) == features[1]
        )
        return {
            features[0]: MeasuredFeatureState(
                features[0],
                row["source_activation"],
                row["source_activation"],
                0.5,
                FeatureActivity.ACTIVE,
                "mps:0",
                "torch.bfloat16",
            ),
            features[1]: MeasuredFeatureState(
                features[1],
                row["target_preactivation"],
                0.0,
                row["target_threshold"],
                FeatureActivity.INACTIVE,
                "mps:0",
                "torch.bfloat16",
            ),
        }

    def measure_point(
        self, pair: dict[str, Any], plan: Any, **_: Any
    ) -> dict[str, Any]:
        self.source_suppression_api_calls += 1
        self.attempt_recorder(pair, self.source_suppression_api_calls)
        true_drive = 0.8 * float(pair["q"])
        z = float(pair["target_preactivation"]) + plan.realized_suppression * true_drive
        active = z > float(pair["target_threshold"])
        record = applied_plan_record(plan)
        record.update(
            {
                "source_suppression_api_call_index": self.source_suppression_api_calls,
                "target_preactivation": z,
                "target_threshold": pair["target_threshold"],
                "target_activation": z if active else 0.0,
                "target_active": active,
            }
        )
        return record


def _synthetic_prediction(
    config: dict[str, Any], repository: Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    protocol = build_protocol_manifest(repository, protocol_commit="0" * 40)
    prompts: list[dict[str, Any]] = []
    for prompt_index, frozen in enumerate(config["prompts"]):
        candidates = []
        for index, alpha in enumerate((0.05, 0.20, 0.60)):
            q = 0.20 + prompt_index / 1000 + index / 10000
            preactivation = 1.0 - alpha * q
            margin = 1.0 - preactivation
            source = FeatureRef(index, 1, prompt_index * 100 + index)
            target = FeatureRef(10 + index, 1, prompt_index * 100 + 50 + index)
            candidates.append(
                build_prospective_pair(
                    source=MeasuredFeatureState(
                        source,
                        1.0,
                        1.0,
                        0.5,
                        FeatureActivity.ACTIVE,
                        "mps:0",
                        "torch.bfloat16",
                    ),
                    target=NearThresholdCandidate(
                        target,
                        preactivation,
                        0.0,
                        1.0,
                        margin,
                        "mps:0",
                        "torch.bfloat16",
                    ),
                    targeted_response=-q,
                    seed=config["scoring"]["pair_seed"],
                    prompt_id=frozen["id"],
                    runtime_fingerprint=RUNTIME_FINGERPRINT,
                    epsilon=config["scoring"]["epsilon"],
                    tolerance=config["scoring"]["crossing_tolerance"],
                    experiment_class=EXPERIMENT_CLASS,
                )
            )
        selected, audit = select_q_panel(
            candidates, prompt_id=frozen["id"], config=config
        )
        prompts.append(
            {
                "id": frozen["id"],
                "text": frozen["text"],
                "token_ids": frozen["token_ids"],
                "selected_positions": frozen["selected_positions"],
                "baseline_pools": {},
                "panel_audit": audit,
                "execution_pairs": selected,
            }
        )
    totals = {
        name: sum(
            prompt["panel_audit"]["selected_count_by_stratum"][name]
            for prompt in prompts
        )
        for name in ("B1", "B2", "B3")
    }
    prediction = {
        "schema_version": 1,
        "artifact_type": "stage1f_prediction_manifest",
        "status": "prediction_frozen_ready_for_commit",
        "experiment_class": EXPERIMENT_CLASS,
        "branch": config["branch"],
        "base_commit": config["base_commit"],
        "protocol_commit": protocol["protocol_commit"],
        "protocol_map_sha256": protocol["protocol_map_sha256"],
        "prediction_execution_commit": "0" * 40,
        "runtime_identity": config["runtime"],
        "prompt_provenance": config["prompt_provenance"],
        "prompt_order": [row["id"] for row in config["prompts"]],
        "prompts": prompts,
        "selection_totals": {
            "prompt_count": 10,
            "selected_pair_count": 30,
            "selected_count_by_stratum": totals,
        },
        "prediction_only_guards": {
            "fresh_source_suppression_api_calls": 0,
            "fresh_target_responses_inspected": False,
            "historical_intervention_outcomes_used": False,
            "graph_edge_input_used_for_inactive_predictions": False,
            "network_accessed": False,
            "discovery_ranker": "q",
        },
        "claim_boundary": config["claim_boundary"],
    }
    return protocol, prediction


def test_exact_e1_probe_and_frozen_classifier() -> None:
    mapping = first_probe_mapping(1.0, (0.125, 0.1875, 0.25))
    assert mapping is not None
    assert mapping["nominal_requested_alpha"] == 0.125
    common = {
        "paired_count": 30,
        "coverage": 1.0,
        "e0_median": 0.2,
        "e1_median": 0.1,
        "improvement_lower": 0.01,
        "e0_spearman": 0.8,
        "e1_spearman": 0.8,
        "e0_accuracy": 1.0,
        "e1_accuracy": 1.0,
        "rules": load_config()["decision_rule"],
    }
    assert classify_terminal(**common)[0] == "completed_stage1f_e1_confirmed"
    assert (
        classify_terminal(**(common | {"paired_count": 23}))[0]
        == "completed_stage1f_underpowered"
    )
    assert (
        classify_terminal(**(common | {"e1_median": 0.17}))[0]
        == "completed_stage1f_e1_mixed"
    )
    assert (
        classify_terminal(**(common | {"e1_median": 0.19}))[0]
        == "completed_stage1f_e1_not_supported"
    )


def test_synthetic_worker_journal_assembler_standalone_validator(
    tmp_path: Path,
) -> None:
    repository = Path(__file__).resolve().parents[1]
    config = load_config()
    protocol, prediction = _synthetic_prediction(config, repository)
    validate_prediction(prediction, config, protocol)
    FakeBackend.rows = {
        pair["pair_id"]: pair
        for prompt in prediction["prompts"]
        for pair in prompt["execution_pairs"]
    }
    journal_path = tmp_path / "stage1f.jsonl"
    with CanonicalExecutionJournal(
        journal_path,
        None,
        frozen_pair_ids=tuple(FakeBackend.rows),
        pre_intervention_commit="0" * 40,
        prediction_manifest_sha256="0" * 64,
        experiment_class=EXPERIMENT_CLASS,
    ) as journal:
        sweeps, calls = execute_frozen_pairs(
            model=object(),
            torch=FakeTorch,
            sampler=FakeSampler(),
            prompts=prediction["prompts"],
            journal=journal,
            backend_factory=FakeBackend,
            maximum_bisection_steps=6,
            stop_bracket_width=1.0 / 64.0,
        )
    points = [point for sweep in sweeps for point in sweep["points"]]
    assert len(points) == calls
    worker = {
        "instrumented_source_suppression_api_calls": calls,
        "prediction_freeze_commit": "0" * 40,
        "pre_run_commit": "0" * 40,
        "environment": {"device": "mps:0", "dtype": "torch.bfloat16"},
        "runtime_evidence": {"synthetic": True},
        "telemetry": {
            "started_at_unix": 1.0,
            "finished_at_unix": 2.0,
            "sample_count": 2,
            "sampling_interval_seconds": 1.0,
            "attempt_peaks": {
                "mps_current_bytes": 1,
                "mps_driver_bytes": 1,
                "process_rss_bytes": 1,
                "swap_used_bytes": 1,
                "swap_growth_bytes": 0,
                "minimum_available_memory_bytes": 10,
            },
            "thermal_states": ["nominal"],
            "violations": [],
            "telemetry_failures": 0,
        },
    }
    records = build_records(
        protocol=protocol,
        prediction=prediction,
        worker=worker,
        points=points,
        config=config,
    )
    output = tmp_path / "bundle"
    publish_records(output, records)
    result = validate_bundle(repository, output)
    assert result["status"] == "passed"
    assert result["terminal_class"] == "completed_stage1f_e1_confirmed"
    assert result["instrumented_api_call_count"] == calls
    assert result["journal_completed_point_count"] == calls
    assert result["serialized_unique_point_count"] == calls
