"""Chat-model task evaluation contracts."""

from scratch_llm.evaluation.chat.categorical import (
    CHAT_CATEGORICAL_PROTOCOL_ID,
    CHAT_CATEGORICAL_PROTOCOL_VERSION,
    CHAT_EVAL_REFERENCE_COMMIT,
    CategoricalEvaluationError,
    CategoricalExample,
    CategoricalTask,
    CategoricalTaskResult,
    evaluate_categorical_task,
    render_multiple_choice_prompt,
)


__all__ = [
    "CHAT_CATEGORICAL_PROTOCOL_ID",
    "CHAT_CATEGORICAL_PROTOCOL_VERSION",
    "CHAT_EVAL_REFERENCE_COMMIT",
    "CategoricalEvaluationError",
    "CategoricalExample",
    "CategoricalTask",
    "CategoricalTaskResult",
    "evaluate_categorical_task",
    "render_multiple_choice_prompt",
]
