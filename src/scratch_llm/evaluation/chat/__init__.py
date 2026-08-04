"""Chat-model task evaluation contracts."""

from scratch_llm.evaluation.chat.categorical import (
    CHAT_CATEGORICAL_PROTOCOL_ID,
    CHAT_CATEGORICAL_PROTOCOL_VERSION,
    CategoricalEvaluationError,
    CategoricalExample,
    CategoricalTask,
    CategoricalTaskResult,
    evaluate_categorical_task,
    render_multiple_choice_prompt,
)
from scratch_llm.evaluation.chat.diagnostics import (
    CodeFenceCategory,
    CodePromptDiagnostic,
    FixedSFTDiagnostics,
    JSONPromptDiagnostic,
    compute_fixed_sft_diagnostics,
)
from scratch_llm.evaluation.chat.generative import (
    CHAT_GENERATIVE_PROTOCOL_ID,
    CHAT_GENERATIVE_PROTOCOL_VERSION,
    GenerativeEvaluationConfig,
    GenerativeEvaluationError,
    GenerativeProblem,
    GenerativeTask,
    GenerativeTaskResult,
    derive_generative_sample_seed,
    evaluate_generative_task,
)
from scratch_llm.evaluation.chat.gsm8k import (
    GSM8KProblem,
    build_gsm8k_task,
    extract_gsm8k_answer,
    get_gsm8k_dataset_spec,
    load_gsm8k_task,
    score_gsm8k_completion,
)
from scratch_llm.evaluation.chat.protocol import CHAT_EVAL_REFERENCE_COMMIT


__all__ = [
    "CHAT_CATEGORICAL_PROTOCOL_ID",
    "CHAT_CATEGORICAL_PROTOCOL_VERSION",
    "CHAT_EVAL_REFERENCE_COMMIT",
    "CHAT_GENERATIVE_PROTOCOL_ID",
    "CHAT_GENERATIVE_PROTOCOL_VERSION",
    "CategoricalEvaluationError",
    "CategoricalExample",
    "CategoricalTask",
    "CategoricalTaskResult",
    "CodeFenceCategory",
    "CodePromptDiagnostic",
    "FixedSFTDiagnostics",
    "GenerativeEvaluationConfig",
    "GenerativeEvaluationError",
    "GenerativeProblem",
    "GenerativeTask",
    "GenerativeTaskResult",
    "GSM8KProblem",
    "JSONPromptDiagnostic",
    "build_gsm8k_task",
    "compute_fixed_sft_diagnostics",
    "derive_generative_sample_seed",
    "evaluate_categorical_task",
    "evaluate_generative_task",
    "extract_gsm8k_answer",
    "get_gsm8k_dataset_spec",
    "load_gsm8k_task",
    "render_multiple_choice_prompt",
    "score_gsm8k_completion",
]
