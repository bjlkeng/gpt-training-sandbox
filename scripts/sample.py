"""Generate text from a pretrained checkpoint."""

from __future__ import annotations

import argparse
from collections.abc import Sequence

import torch

from scratch_llm.training.checkpoint import CheckpointError, load_model_checkpoint
from scratch_llm.generation import generate_sequences
from scripts._common import (
    add_generation_arguments,
    checkpoint_parser,
    resolve_generation_arguments,
)


COMMAND = "sample"


def build_parser() -> argparse.ArgumentParser:
    """Return the checkpoint-driven base sampling command parser."""

    parser = checkpoint_parser(COMMAND, "Sample text from a base-model checkpoint.")
    parser.add_argument(
        "-p",
        "--prompt",
        action="append",
        help="Optional prompt; repeat to sample from multiple prompts.",
    )
    add_generation_arguments(parser)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the base sampling command."""

    parser = build_parser()
    arguments = parser.parse_args(argv)
    try:
        checkpoint = load_model_checkpoint(
            arguments.checkpoint,
            device=arguments.device,
        )
        settings = resolve_generation_arguments(
            checkpoint.config.generation,
            arguments,
        )
        if settings.top_p is not None:
            raise ValueError(
                "top_p sampling is not implemented in the shared generator"
            )
        device = next(checkpoint.model.parameters()).device

        for prompt in arguments.prompt or [""]:
            prompt_ids = checkpoint.tokenizer.encode(prompt)
            used_synthetic_bos = not prompt_ids
            if used_synthetic_bos:
                prompt_ids = [checkpoint.tokenizer.get_bos_token_id()]
            token_ids = torch.tensor(
                [prompt_ids],
                dtype=torch.long,
                device=device,
            )
            bos_token_id = checkpoint.tokenizer.get_bos_token_id()
            generated = generate_sequences(
                checkpoint.model,
                token_ids,
                max_new_tokens=settings.max_new_tokens,
                temperature=settings.temperature,
                top_k=settings.top_k,
                seed=settings.seed,
                stop_token_ids={bos_token_id},
            )
            first_visible_token = 1 if used_synthetic_bos else 0
            decoded_ids = generated.sequences[0].token_ids[first_visible_token:]
            print(checkpoint.tokenizer.decode(decoded_ids))
    except (CheckpointError, OSError, RuntimeError, TypeError, ValueError) as error:
        parser.error(str(error))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
