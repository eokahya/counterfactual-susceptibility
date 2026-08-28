from __future__ import annotations

import json
import subprocess
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any, ClassVar

from cfsus.stage1c_v3.execution_journal import CanonicalExecutionJournal
from cfsus.stage1c_v3.prediction import canonical_v3_pair_id
from cfsus.stage1c_v3.quantization_audit import bf16_round
from cfsus.stage1g import (
    EXPERIMENT_CLASS,
    RUNTIME_FINGERPRINT,
    behavior_resolution_audit,
    bf16_logit_resolution,
    build_protocol_manifest,
    build_records,
    classify_terminal,
    execute_frozen_pairs,
    load_config,
    ordered_prompts,
    prompt_digest,
    publish_records,
    read_completed_journal,
    select_panels,
    validate_bundle,
    validate_prediction,
)
from cfsus.types import FeatureActivity, FeatureRef, MeasuredFeatureState


class FakeTensor:
    def __init__(self, value: float, dtype: object) -> None:
        self.value = bf16_round(value)
        self.dtype = dtype
        self.device = SimpleNamespace(type="mps")

    def reshape(self, _: object) -> FakeTensor:
        return self

    def item(self) -> float:
        return self.value


class FakeMPS:
    @staticmethod
    def empty_cache() -> None:
        return None


class FakeTorch:
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
        answer_token_id: int,
        contrast_token_id: int,
        token_count: int,
        attempt_recorder: Any,
        call_index_offset: int,
    ) -> None:
        del prompt, torch, token_count
        self.prompt_id = prompt_id
        self.answer_token_id = answer_token_id
        self.contrast_token_id = contrast_token_id
        self.attempt_recorder = attempt_recorder
        self.source_suppression_api_calls = call_index_offset

    def measure_states(
        self, features: tuple[FeatureRef, FeatureRef]
    ) -> dict[FeatureRef, MeasuredFeatureState]:
        row = self.rows[
            next(
                pair_id
                for pair_id, value in self.rows.items()
                if value["prompt_id"] == self.prompt_id
                and FeatureRef(**value["source"]) == features[0]
                and FeatureRef(**value["target"]) == features[1]
            )
        ]
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

    def measure_condition(
        self,
        pair: dict[str, Any],
        *,
        condition: str,
        desired_source_activation: float | None,
        desired_target_activation: float | None,
        stage: str,
    ) -> dict[str, Any]:
        self.source_suppression_api_calls += 1
        self.attempt_recorder(pair, self.source_suppression_api_calls)
        positive = float(pair["q"]) > 0.0
        if condition in {"source_full_ablation", "source_ablation_target_clamp"}:
            z_value = 1.125 if positive else 0.75
        else:
            z_value = float(pair["target_preactivation"])
        threshold = float(pair["target_threshold"])
        active = z_value > threshold
        natural = z_value if active else 0.0
        applied_source = (
            None
            if desired_source_activation is None
            else bf16_round(desired_source_activation)
        )
        applied_target = (
            None
            if desired_target_activation is None
            else bf16_round(desired_target_activation)
        )
        mediation = float(pair["g_i"]) * 1.125 if positive else 0.0
        behavior = 2.0
        if condition in {"source_full_ablation", "target_only_injection"}:
            behavior += mediation
        return {
            "source_suppression_api_call_index": self.source_suppression_api_calls,
            "condition": condition,
            "stage": stage,
            "point_elapsed_seconds": 0.01,
            "desired_source_activation": desired_source_activation,
            "actual_bf16_source_activation": applied_source,
            "desired_target_activation": desired_target_activation,
            "actual_bf16_target_activation": applied_target,
            "target_preactivation": z_value,
            "target_threshold": threshold,
            "target_natural_activation": natural,
            "target_effective_activation": natural
            if applied_target is None
            else applied_target,
            "target_active": active,
            "strict_crossing": active,
            "loaded_gate": "a=z*1[z>tau]",
            "threshold_equality_activity": "inactive",
            "answer_token_id": self.answer_token_id,
            "contrast_token_id": self.contrast_token_id,
            "answer_logit": behavior + 1.0,
            "contrast_logit": 1.0,
            "behavior_T": behavior,
            "freeze_attention": True,
            "constrained_layers": None,
            "target_clamped": condition == "source_ablation_target_clamp",
            "source_value_device": None if applied_source is None else "mps:0",
            "source_value_dtype": None if applied_source is None else "torch.bfloat16",
            "target_value_device": None if applied_target is None else "mps:0",
            "target_value_dtype": None if applied_target is None else "torch.bfloat16",
            "logits_finite": True,
            "logits_shape": [1, 3, 128],
            "preactivation_cache_persisted": False,
            "full_logits_persisted": False,
        }


def _sensitivity() -> dict[str, Any]:
    derivative_rows = [
        {
            "feature": {"layer": index + 1, "position": 2, "feature_id": index},
            "baseline_activation": 1.0,
            "batched_vjp_g_i": float(index + 1),
            "raw_answer_logit_edge": float(index + 1),
            "raw_contrast_logit_edge": 0.0,
            "raw_edge_activation_denominator": 1.0,
            "independent_reference_g_i": float(index + 1),
            "sign_agreement": True,
            "symmetric_normalized_error": 0.0,
        }
        for index in range(8)
    ]
    amplitudes = [0.0625, 0.125, 0.25, 0.5, 1.0, 2.0]
    resolution_rows = []
    for derivative in derivative_rows:
        ladder = []
        for amplitude in amplitudes:
            answer = bf16_round(3.0 + amplitude)
            behavior = answer - 1.0
            ladder.append(
                {
                    "relative_activation_change": amplitude,
                    "requested_target_activation": 1.0 + amplitude,
                    "actual_bf16_target_activation": bf16_round(1.0 + amplitude),
                    "answer_logit": answer,
                    "contrast_logit": 1.0,
                    "behavior_T": behavior,
                    "answer_logit_change": answer - 3.0,
                    "contrast_logit_change": 0.0,
                    "behavior_response": behavior - 2.0,
                    "bf16_distinct_nonzero_behavior_response": behavior != 2.0,
                }
            )
        resolution_rows.append(
            {
                "feature": derivative["feature"],
                "baseline_activation": 1.0,
                "first_resolvable_relative_activation_change": 0.0625,
                "ladder": ladder,
            }
        )
    baseline = {
        "answer_logit": 3.0,
        "contrast_logit": 1.0,
        "behavior_T": 2.0,
        "answer_in_top64": True,
        "logits_finite": True,
        "logits_shape": [1, 3, 128],
    }
    return {
        "schema_version": 2,
        "artifact_type": "stage1g_v2_output_sensitivity_and_resolution_validation",
        "status": "passed",
        "prompt_id": "development_norway_stage1g_output_sensitivity",
        "prompt": "The capital of Norway is",
        "token_ids": [2, 3, 4],
        "answer_token_id": 10,
        "contrast_token_id": 11,
        "baseline_behavior": baseline,
        "baseline_repeat_behavior": baseline,
        "selected_feature_count": 8,
        "selection_rule": (
            "descending_absolute_batched_vjp_then_feature_coordinate_with_"
            "unique_layer_preference"
        ),
        "independent_reference": {
            "method": (
                "raw_unnormalized_feature_to_answer_and_contrast_logit_edges_"
                "divided_by_positive_activation"
            ),
            "source_api_rationale": (
                "pinned_circuit_tracer_0_5_2_exposes_signed_raw_unnormalized_"
                "output_edge_rows_without_consuming_"
                "OutputSensitivityVJPContext_compute"
            ),
            "selected_before_runtime_result": True,
            "batched_vjp_result_consumed": False,
            "raw_edges_are_signed_and_unnormalized": True,
        },
        "derivative_rows": derivative_rows,
        "derivative_metrics": {
            "all_finite": True,
            "sign_agreement": 1.0,
            "spearman": 1.0,
            "median_symmetric_normalized_error": 0.0,
            "p95_symmetric_normalized_error": 0.0,
        },
        "frozen_derivative_tolerances": {
            "sign_agreement_minimum": 1.0,
            "spearman_minimum": 0.99,
            "median_symmetric_normalized_error_maximum": 0.01,
            "p95_symmetric_normalized_error_maximum": 0.03,
        },
        "bf16_resolution_audit": {
            "status": "passed",
            "derivative_truth_or_validation_used": False,
            "scientific_selection_ranking_or_threshold_tuning_used": False,
            "baseline_repeat_behavior_error": 0.0,
            "max_abs_baseline_repeat_behavior_error": 0.0,
            "answer_logit_bf16_resolution": bf16_logit_resolution(3.0),
            "contrast_logit_bf16_resolution": bf16_logit_resolution(1.0),
            "behavior_resolution_floor": 0.09375,
            "floor_formula": (
                "max(1e-4,5*prompt_max_abs_baseline_repeat_behavior_error,"
                "4*(answer_logit_BF16_ULP+contrast_logit_BF16_ULP))"
            ),
            "relative_activation_changes": amplitudes,
            "features_with_resolvable_response": 8,
            "any_finite_nonzero_bf16_behavior_response_by_maximum_ladder": True,
            "rows": resolution_rows,
        },
        "instrumented_target_injection_calls": 48,
        "scientific_attempt_consumed": False,
        "scientific_pair_overlap": False,
        "gradient_tensor_persisted": False,
        "full_logits_persisted": False,
        "finite_response_presented_as_derivative_validation": False,
    }


def _prediction(
    repository: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    config = load_config(repository / "configs/stage1g_behavioral_mediation_pilot.json")
    protocol = build_protocol_manifest(repository, protocol_commit="0" * 40)
    sensitivity = _sensitivity()
    prompts: list[dict[str, Any]] = []
    eligibility: list[dict[str, Any]] = []
    chosen = ordered_prompts(config)[:8]
    chosen_ids = {row["id"] for row in chosen}
    for frozen in ordered_prompts(config):
        common = {
            "id": frozen["id"],
            "order_sha256": prompt_digest(frozen["id"], config),
        }
        if frozen["id"] not in chosen_ids:
            eligibility.append(
                common
                | {
                    "status": "not_evaluated_after_quota",
                    "eligible": None,
                    "model_calls": 0,
                }
            )
            continue
        token_ids = [2, 3, 4]
        baseline = {
            "answer_logit": 3.0,
            "contrast_logit": 1.0,
            "behavior_T": 2.0,
            "answer_in_top64": True,
            "logits_finite": True,
            "logits_shape": [1, 3, 128],
        }
        resolution = behavior_resolution_audit(baseline, baseline, config)
        eligibility.append(
            common
            | {
                "status": "eligible",
                "eligible": True,
                "reasons": [],
                "token_ids": token_ids,
                "answer_token_id": 10,
                "contrast_token_id": 11,
                "baseline_behavior": baseline,
                "baseline_repeat_behavior": baseline,
                "behavior_resolution_audit": resolution,
                "model_calls": 2,
            }
        )
        candidates: list[dict[str, Any]] = []
        prompt_index = len(prompts)
        for index in range(6):
            source = FeatureRef(index, 1, prompt_index * 100 + index)
            target = FeatureRef(index + 10, 2, prompt_index * 100 + 50 + index)
            q_value = 0.2 + index * 0.05
            g_value = 1.0 + index
            predicted = g_value * (0.9 + q_value)
            floor = float(resolution["behavior_resolution_floor"])
            candidates.append(
                {
                    "pair_id": canonical_v3_pair_id(
                        source=source,
                        target=target,
                        runtime_fingerprint=RUNTIME_FINGERPRINT,
                        prompt_id=frozen["id"],
                        seed=config["scoring"]["pair_seed"],
                        experiment_class=EXPERIMENT_CLASS,
                    ),
                    "source": {
                        "layer": source.layer,
                        "position": source.position,
                        "feature_id": source.feature_id,
                    },
                    "target": {
                        "layer": target.layer,
                        "position": target.position,
                        "feature_id": target.feature_id,
                    },
                    "source_activation": 1.0,
                    "target_preactivation": 0.9,
                    "target_threshold": 1.0,
                    "margin": 1.0 - 0.9,
                    "targeted_response": -q_value,
                    "q": q_value,
                    "g_i": g_value,
                    "predicted_full_ablation_crossing": True,
                    "predicted_target_activation": 0.9 + q_value,
                    "predicted_signed_mediation": predicted,
                    "predicted_abs_mediation_to_bf16_resolution_floor_ratio": (
                        abs(predicted) / floor
                    ),
                    "predicted_abs_mediation_exceeds_resolution_floor": {
                        "1x": abs(predicted) > floor,
                        "2x": abs(predicted) > 2.0 * floor,
                        "4x": abs(predicted) > 4.0 * floor,
                        "8x": abs(predicted) > 8.0 * floor,
                    },
                    (
                        "bf16_resolution_metadata_used_for_panel_ranking_or_exclusion"
                    ): False,
                    "q_over_margin_computed_or_used": False,
                    "intervention_outcome_used": False,
                }
            )
        for index in range(2):
            source = FeatureRef(index + 6, 1, prompt_index * 100 + 20 + index)
            target = FeatureRef(index + 15, 2, prompt_index * 100 + 80 + index)
            candidates.append(
                {
                    "pair_id": canonical_v3_pair_id(
                        source=source,
                        target=target,
                        runtime_fingerprint=RUNTIME_FINGERPRINT,
                        prompt_id=frozen["id"],
                        seed=config["scoring"]["pair_seed"],
                        experiment_class=EXPERIMENT_CLASS,
                    ),
                    "source": {
                        "layer": source.layer,
                        "position": source.position,
                        "feature_id": source.feature_id,
                    },
                    "target": {
                        "layer": target.layer,
                        "position": target.position,
                        "feature_id": target.feature_id,
                    },
                    "source_activation": 1.0,
                    "target_preactivation": 0.9,
                    "target_threshold": 1.0,
                    "margin": 1.0 - 0.9,
                    "targeted_response": 0.1,
                    "q": -0.1,
                    "g_i": 20.0 - index,
                    "predicted_full_ablation_crossing": False,
                    "predicted_target_activation": 0.0,
                    "predicted_signed_mediation": 0.0,
                    "predicted_abs_mediation_to_bf16_resolution_floor_ratio": 0.0,
                    "predicted_abs_mediation_exceeds_resolution_floor": {
                        "1x": False,
                        "2x": False,
                        "4x": False,
                        "8x": False,
                    },
                    (
                        "bf16_resolution_metadata_used_for_panel_ranking_or_exclusion"
                    ): False,
                    "q_over_margin_computed_or_used": False,
                    "intervention_outcome_used": False,
                }
            )
        selected, audit = select_panels(
            candidates, prompt_id=frozen["id"], config=config
        )
        for row in selected:
            row["answer_token_id"] = 10
            row["contrast_token_id"] = 11
        prompts.append(
            {
                "id": frozen["id"],
                "text": frozen["text"],
                "answer": frozen["answer"],
                "contrast": frozen["contrast"],
                "token_ids": token_ids,
                "selected_positions": [1, 2],
                "answer_token_id": 10,
                "contrast_token_id": 11,
                "baseline_behavior": baseline,
                "baseline_repeat_behavior": baseline,
                "behavior_resolution_audit": resolution,
                "baseline_pools": {},
                "panel_audit": audit,
                "execution_pairs": selected,
            }
        )
    totals = {
        "eligible_prompt_count": 8,
        "scientific_baseline_model_calls": 16,
        "execution_pair_count": sum(
            len(prompt["execution_pairs"]) for prompt in prompts
        ),
        "membership_count_by_panel": {
            panel: sum(
                prompt["panel_audit"]["selected_membership_count_by_panel"][panel]
                for prompt in prompts
            )
            for panel in ("B", "Q", "G", "D")
        },
        "shortfall_by_panel": {
            panel: sum(
                prompt["panel_audit"]["shortfall_by_panel"][panel] for prompt in prompts
            )
            for panel in ("B", "Q", "G", "D")
        },
    }
    prediction = {
        "schema_version": 2,
        "artifact_type": "stage1g_v2_prediction_manifest",
        "status": "prediction_frozen_ready_for_commit",
        "experiment_class": EXPERIMENT_CLASS,
        "branch": config["branch"],
        "base_commit": config["base_commit"],
        "protocol_commit": protocol["protocol_commit"],
        "protocol_map_sha256": protocol["protocol_map_sha256"],
        "prediction_execution_commit": "0" * 40,
        "runtime_identity": config["runtime"],
        "output_sensitivity_validation": sensitivity,
        "ordered_prompt_eligibility": eligibility,
        "eligible_prompt_order": [prompt["id"] for prompt in prompts],
        "prompts": prompts,
        "selection_totals": totals,
        "prediction_only_guards": {
            "fresh_scientific_intervention_api_calls": 0,
            "historical_intervention_outcomes_used": False,
            "independent_reference_raw_edges_used": True,
            "graph_edge_input_used_for_candidate_discovery_or_batched_vjp": False,
            "finite_ladder_used_for_derivative_validation_selection_or_tuning": False,
            "q_over_margin_discovery_used": False,
            "E1_or_E2_computed": False,
            "diagnostic_fp32_shadow_T_used": False,
            "network_accessed": False,
        },
        "claim_boundary": config["claim_boundary"],
    }
    return protocol, sensitivity, prediction


def test_frozen_classifier_classes() -> None:
    rules = load_config()["decision_rule"]
    common = {
        "eligible_prompt_count": 8,
        "B_crossing_count": 32,
        "B_crossing_prompt_count": 8,
        "B_sign_accuracy": 0.8,
        "B_sign_bootstrap_lower": 0.6,
        "B_minus_Q_mean_abs_M": 0.1,
        "B_minus_Q_bootstrap_lower": 0.01,
        "B_above_floor_fraction": 0.8,
        "B_injection_sign_agreement": 0.8,
        "directional_violation_fraction": 0.0,
        "rules": rules,
    }
    assert classify_terminal(**common)[0] == "completed_stage1g_v2_supported"
    assert (
        classify_terminal(**(common | {"B_crossing_count": 20}))[0]
        == "completed_stage1g_v2_underpowered"
    )
    negative = common | {
        "B_sign_accuracy": 0.0,
        "B_sign_bootstrap_lower": 0.0,
        "B_minus_Q_mean_abs_M": -0.1,
        "B_minus_Q_bootstrap_lower": -0.2,
        "B_above_floor_fraction": 0.0,
        "B_injection_sign_agreement": 0.0,
        "directional_violation_fraction": 1.0,
    }
    assert classify_terminal(**negative)[0] == "completed_stage1g_v2_not_supported"


def test_synthetic_worker_journal_assembler_validator(tmp_path: Path) -> None:
    repository = Path(__file__).resolve().parents[1]
    config = load_config(repository / "configs/stage1g_behavioral_mediation_pilot.json")
    protocol, sensitivity, prediction = _prediction(repository)
    pairs = validate_prediction(prediction, config, protocol)
    FakeBackend.rows = pairs
    journal_path = tmp_path / "synthetic.journal"
    lock_path = tmp_path / "attempt.lock"
    with CanonicalExecutionJournal(
        journal_path,
        lock_path,
        frozen_pair_ids=tuple(pairs),
        pre_intervention_commit="0" * 40,
        prediction_manifest_sha256="1" * 64,
        experiment_class=EXPERIMENT_CLASS,
        attempt_boundary=config["intervention"]["attempt_boundary"],
        attempt_lock_artifact_type="stage1g_synthetic_attempt_lock",
    ) as journal:
        sweeps, calls = execute_frozen_pairs(
            model=object(),
            torch=FakeTorch,
            sampler=FakeSampler(),
            prompts=prediction["prompts"],
            journal=journal,
            backend_factory=FakeBackend,
        )
    points = read_completed_journal(journal_path)
    assert calls == len(points)
    assert sum(sweep["point_count"] for sweep in sweeps) == calls
    telemetry = {
        "started_at_unix": 1.0,
        "finished_at_unix": 2.0,
        "sample_count": 2,
        "sampling_interval_seconds": 1.0,
        "attempt_peaks": {},
        "thermal_states": ["nominal"],
        "violations": [],
        "telemetry_failures": 0,
    }
    worker = {
        "prediction_freeze_commit": "0" * 40,
        "pre_run_commit": "0" * 40,
        "telemetry": telemetry,
        "environment": {},
        "runtime_evidence": {},
    }
    records = build_records(
        protocol=protocol,
        sensitivity=sensitivity,
        prediction=prediction,
        worker=worker,
        points=points,
        config=config,
    )
    output = tmp_path / "bundle"
    publish_records(output, records)
    result = validate_bundle(repository, output)
    assert result["status"] == "passed"
    assert result["instrumented_intervention_api_calls"] == calls
    assert result["serialized_points"] == calls
    standalone = subprocess.run(
        [
            sys.executable,
            str(repository / "scripts/validate_stage1g_artifacts.py"),
            str(output),
        ],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    )
    standalone_result = json.loads(standalone.stdout)
    assert standalone_result["hostile_input_checks"] is True
    assert standalone_result["fresh_process"] is True
