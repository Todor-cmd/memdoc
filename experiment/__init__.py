"""Evaluation harness: variant prepare specs, store lifecycle hooks, runner."""

from .data import (
    load_evidence_id_to_url,
    load_persona_background_session_ids,
    load_reasonable_questions,
    sort_questions_by_persona,
)
from .design_runner import (
    default_design_output_path,
    filter_design_for_agent,
    load_design_matrix,
    run_design_experiment,
)
from .io import (
    VARIANT_PLACEHOLDER,
    IncrementalInferenceWriter,
    default_variant_output_path,
    format_output_path_for_variant,
    write_inference_rows,
)
from .prepare_spec import PrepareSpec, build_prepare_spec
from .runner import run_experiment
from .store_hooks import NoOpStoreHooks, StoreHooksProtocol
from .variants import DatasetVariant

__all__ = [
    "DatasetVariant",
    "PrepareSpec",
    "build_prepare_spec",
    "NoOpStoreHooks",
    "StoreHooksProtocol",
    "run_experiment",
    "run_design_experiment",
    "load_design_matrix",
    "filter_design_for_agent",
    "default_design_output_path",
    "IncrementalInferenceWriter",
    "load_reasonable_questions",
    "sort_questions_by_persona",
    "load_persona_background_session_ids",
    "load_evidence_id_to_url",
    "write_inference_rows",
    "default_variant_output_path",
    "format_output_path_for_variant",
    "VARIANT_PLACEHOLDER",
]
