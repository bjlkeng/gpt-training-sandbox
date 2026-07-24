"""Generate text from a pretrained checkpoint."""

from __future__ import annotations

import argparse
from collections.abc import Sequence

import torch

from scratch_llm.checkpoint import CheckpointError, load_model_checkpoint
from scratch_llm.generation import generate
from scripts._common import checkpoint_parser


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
    parser.add_argument(
        "--device",
        default="cpu",
        help="Torch device for checkpoint loading and generation (default: cpu).",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        help="Override the checkpoint's generation.max_new_tokens.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        help="Override the checkpoint's generation.temperature; zero is greedy.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        help="Override the checkpoint's generation.top_k.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="Override the checkpoint's generation.seed.",
    )
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
        settings = checkpoint.config.generation
        if settings.top_p is not None:
            raise ValueError("top_p sampling is not implemented in the naive generator")

        max_new_tokens = (
            settings.max_new_tokens
            if arguments.max_new_tokens is None
            else arguments.max_new_tokens
        )
        temperature = (
            settings.temperature
            if arguments.temperature is None
            else arguments.temperature
        )
        top_k = settings.top_k if arguments.top_k is None else arguments.top_k
        seed = settings.seed if arguments.seed is None else arguments.seed
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
            generated = generate(
                checkpoint.model,
                token_ids,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                seed=seed,
            )
            first_visible_token = 1 if used_synthetic_bos else 0
            decoded_ids = generated[0, first_visible_token:].cpu().tolist()
            print(checkpoint.tokenizer.decode(decoded_ids))
    except (CheckpointError, OSError, RuntimeError, TypeError, ValueError) as error:
        parser.error(str(error))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
