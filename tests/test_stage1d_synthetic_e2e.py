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
from cfsus.stage1d.artifacts import build_records, publish_records
from cfsus.stage1d.benchmark import select_prompt_panels
from cfsus.stage1d.config import EXPERIMENT_CLASS, load_stage1d_config
from cfsus.stage1d.execution import execute_frozen_pairs
from cfsus.stage1d.prediction_runtime import RUNTIME_FINGERPRINT
from cfsus.stage1d.protocol import build_protocol_manifest
from cfsus.stage1d.validation import validate_bundle
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

    def reshape(self, _: tuple[()] | tuple[object, ...]) -> FakeTensor:
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


def _synthetic_prediction(config: dict[str, Any], repository: Path) -> dict[str, Any]:
    prompts: list[dict[str, Any]] = []
    for prompt_index, frozen in enumerate(config["prompts"]):
        candidates = []
        alphas = [0.05, 0.20, 0.60, 1.50] + [0.8] * 16
        for index, alpha in enumerate(alphas):
            source = FeatureRef(index % 8, 1, prompt_index * 1000 + index)
            target = FeatureRef(10 + index % 8, 1, prompt_index * 1000 + 100 + index)
            source_state = MeasuredFeatureState(
                source, 1.0, 1.0, 0.5, FeatureActivity.ACTIVE, "mps:0", "torch.bfloat16"
            )
            requested_margin = 0.01 + index / 1000
            z = 1.0 - requested_margin
            target_state = NearThresholdCandidate(
                target,
                z,
                0.0,
                1.0,
                1.0 - z,
                "mps:0",
                "torch.bfloat16",
            )
            q = target_state.margin / alpha
            candidates.append(
                build_prospective_pair(
                    source=source_state,
                    target=target_state,
                    targeted_response=-q,
                    seed=config["scoring"]["pair_seed"],
                    prompt_id=frozen["id"],
                    runtime_fingerprint=RUNTIME_FINGERPRINT,
                    epsilon=config["scoring"]["epsilon"],
                    tolerance=config["scoring"]["crossing_tolerance"],
                    experiment_class=EXPERIMENT_CLASS,
                )
            )
        for index, q in enumerate((-0.3, -0.2), start=20):
            source = FeatureRef(index % 8, 1, prompt_index * 1000 + index)
            target = FeatureRef(10 + index % 8, 1, prompt_index * 1000 + 100 + index)
            source_state = MeasuredFeatureState(
                source, 1.0, 1.0, 0.5, FeatureActivity.ACTIVE, "mps:0", "torch.bfloat16"
            )
            z = 0.9
            candidates.append(
                build_prospective_pair(
                    source=source_state,
                    target=NearThresholdCandidate(
                        target, z, 0.0, 1.0, 1.0 - z, "mps:0", "torch.bfloat16"
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
        selected = select_prompt_panels(
            candidates, prompt_id=frozen["id"], config=config
        )
        prompts.append(
            {
                "id": frozen["id"],
                "text": frozen["text"],
                "token_ids": frozen["token_ids"],
                "selected_positions": frozen["selected_positions"],
                "baseline_pools": {},
                "method_pair_ids": selected.method_pair_ids,
                "directional_pair_ids": selected.directional_pair_ids,
                "detailed_pair_ids": selected.detailed_pair_ids,
                "execution_pairs": selected.execution_pairs,
                "quantization_audit": selected.quantization_audit,
                "missing_strata": selected.missing_strata,
            }
        )
    protocol = build_protocol_manifest(repository, protocol_commit="0" * 40)
    return {
        "schema_version": 1,
        "artifact_type": "stage1d_prediction_manifest",
        "status": "prediction_frozen_ready_for_commit",
        "experiment_class": EXPERIMENT_CLASS,
        "branch": config["branch"],
        "base_commit": config["base_commit"],
        "protocol_commit": protocol["protocol_commit"],
        "protocol_map_sha256": protocol["protocol_map_sha256"],
        "prediction_execution_commit": "0" * 40,
        "runtime_identity": config["runtime"],
        "protocol": {
            key: config[key]
            for key in (
                "scanner",
                "source_pool",
                "responses",
                "scoring",
                "quantization_resolvability",
                "full_ablation_panel",
                "detailed_panel",
                "schedules",
                "metrics",
                "decision_rule",
                "intervention",
            )
        },
        "prompt_order": [item["id"] for item in prompts],
        "prompts": prompts,
        "selection_totals": {},
        "prediction_only_guards": {
            "evaluation_source_suppression_api_calls": 0,
            "historical_intervention_outcomes_read": False,
            "norway_development_outcome_used": False,
            "graph_edge_input_used_for_inactive_predictions": False,
            "network_accessed": False,
        },
        "claim_boundary": config["claim_boundary"],
    }


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
            item
            for item in self.rows.values()
            if FeatureRef(**item["source"]) == features[0]
            and FeatureRef(**item["target"]) == features[1]
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
        z = pair["target_preactivation"] + plan.realized_suppression * pair["q"]
        active = z > pair["target_threshold"]
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


def test_synthetic_worker_journal_assembler_validator(tmp_path: Path) -> None:
    repository = Path(__file__).resolve().parents[1]
    config = load_stage1d_config()
    protocol = build_protocol_manifest(repository, protocol_commit="0" * 40)
    prediction = _synthetic_prediction(config, repository)
    FakeBackend.rows = {
        row["pair_id"]: row
        for prompt in prediction["prompts"]
        for row in prompt["execution_pairs"]
    }
    pair_ids = tuple(FakeBackend.rows)
    journal_path = tmp_path / "points.jsonl"
    with CanonicalExecutionJournal(
        journal_path,
        None,
        frozen_pair_ids=pair_ids,
        pre_intervention_commit="0" * 40,
        prediction_manifest_sha256="0" * 64,
        experiment_class="stage1d_synthetic_rehearsal",
    ) as journal:
        sweeps, calls = execute_frozen_pairs(
            model=object(),
            torch=FakeTorch,
            sampler=FakeSampler(),
            prompts=prediction["prompts"],
            journal=journal,
            backend_factory=FakeBackend,
            maximum_bisection_steps=6,
        )
    assert calls == sum(item["point_count"] for item in sweeps)
    telemetry = {
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
    }
    worker = {
        "instrumented_evaluation_api_calls": calls,
        "prediction_freeze_commit": "0" * 40,
        "pre_run_commit": "0" * 40,
        "environment": {"device": "mps:0", "dtype": "torch.bfloat16"},
        "telemetry": telemetry,
    }
    records = build_records(
        protocol=protocol,
        prediction=prediction,
        worker=worker,
        points=[point for sweep in sweeps for point in sweep["points"]],
        config=config,
    )
    output = tmp_path / "bundle"
    publish_records(output, records)
    result = validate_bundle(repository, output)
    assert result["status"] == "passed"
    assert result["evaluation_call_count"] == calls
