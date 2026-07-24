"""Educational single-device training loops and optimizer-step mechanics."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn.utils import clip_grad_norm_
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader

from scratch_llm._validation import require_positive_integer, require_positive_real
from scratch_llm.config import ProjectConfig
from scratch_llm.data import NextTokenDataset
from scratch_llm.model import GPT
from scratch_llm.optim import build_lr_scheduler, build_optimizer
from scratch_llm.tokenizer import SPECIAL_TOKENS, ByteTokenizer
from scratch_llm.utils import get_device, set_seed


def derive_grad_accum_steps(
    *,
    device_batch_size: int,
    seq_len: int,
    total_batch_size_tokens: int,
) -> int:
    """Return the exact number of microbatches in one optimizer step."""

    device_batch_size = require_positive_integer(
        device_batch_size,
        name="device_batch_size",
    )
    seq_len = require_positive_integer(seq_len, name="seq_len")
    total_batch_size_tokens = require_positive_integer(
        total_batch_size_tokens,
        name="total_batch_size_tokens",
    )

    tokens_per_microbatch = device_batch_size * seq_len
    grad_accum_steps, remainder = divmod(
        total_batch_size_tokens,
        tokens_per_microbatch,
    )
    if remainder:
        raise ValueError(
            "total_batch_size_tokens must be divisible by "
            "device_batch_size * seq_len "
            f"({tokens_per_microbatch}); got {total_batch_size_tokens}"
        )
    return grad_accum_steps


@dataclass(frozen=True)
class OptimizerStepResult:
    """Metrics produced by one completed gradient-accumulation window."""

    loss: float
    grad_norm: float


@dataclass(frozen=True)
class TinyTextTrainingResult:
    """Reusable state and step history from the phase-one text training path."""

    model: GPT
    tokenizer: ByteTokenizer
    optimizer: Optimizer
    scheduler: LRScheduler
    steps: tuple[OptimizerStepResult, ...]


def run_optimizer_step(
    optimizer: Optimizer,
    micro_losses: Iterable[Tensor],
    *,
    grad_accum_steps: int,
    grad_clip: float,
) -> OptimizerStepResult:
    """Accumulate scaled losses, clip gradients, update once, and clear gradients.

    ``micro_losses`` may be a lazy iterable so each forward pass can release its
    graph after backward. Exactly ``grad_accum_steps`` losses are consumed.
    """

    if not isinstance(optimizer, Optimizer):
        raise TypeError(
            f"optimizer must be an Optimizer, got {type(optimizer).__name__}"
        )
    grad_accum_steps = require_positive_integer(
        grad_accum_steps,
        name="grad_accum_steps",
    )
    grad_clip = require_positive_real(grad_clip, name="grad_clip")

    loss_iterator = iter(micro_losses)
    loss_sum: Tensor | None = None
    try:
        for microstep in range(grad_accum_steps):
            try:
                micro_loss = next(loss_iterator)
            except StopIteration as error:
                raise ValueError(
                    "micro_losses ended after "
                    f"{microstep} of {grad_accum_steps} required losses"
                ) from error
            if not isinstance(micro_loss, Tensor):
                raise TypeError(
                    f"micro loss {microstep} must be a Tensor, "
                    f"got {type(micro_loss).__name__}"
                )
            if micro_loss.ndim != 0:
                raise ValueError(
                    f"micro loss {microstep} must be scalar, "
                    f"got shape {tuple(micro_loss.shape)}"
                )

            detached_loss = micro_loss.detach()
            loss_sum = detached_loss if loss_sum is None else loss_sum + detached_loss
            (micro_loss / grad_accum_steps).backward()
    except Exception:
        optimizer.zero_grad(set_to_none=True)
        raise

    parameters = [
        parameter for group in optimizer.param_groups for parameter in group["params"]
    ]
    grad_norm = clip_grad_norm_(parameters, grad_clip)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

    if loss_sum is None:  # pragma: no cover - positive steps make this unreachable.
        raise RuntimeError("optimizer step completed without a loss")
    return OptimizerStepResult(
        loss=float((loss_sum / grad_accum_steps).item()),
        grad_norm=float(grad_norm.item()),
    )


def _repeat_batches(
    batches: Iterable[tuple[Tensor, Tensor]],
) -> Iterator[tuple[Tensor, Tensor]]:
    """Repeat a re-iterable batch source and reject an empty training set."""

    while True:
        yielded_batch = False
        for batch in batches:
            yielded_batch = True
            if not isinstance(batch, (tuple, list)) or len(batch) != 2:
                raise TypeError(
                    "each training batch must contain exactly inputs and targets"
                )
            inputs, targets = batch
            if not isinstance(inputs, Tensor) or not isinstance(targets, Tensor):
                raise TypeError("training batch inputs and targets must be Tensors")
            yield inputs, targets
        if not yielded_batch:
            raise ValueError("batches must yield at least one training batch")


def run_training_steps(
    model: nn.Module,
    batches: Iterable[tuple[Tensor, Tensor]],
    optimizer: Optimizer,
    scheduler: LRScheduler,
    *,
    max_steps: int,
    grad_accum_steps: int,
    grad_clip: float,
    device: str | torch.device,
) -> list[OptimizerStepResult]:
    """Train a model for a bounded number of single-device optimizer steps."""

    if not isinstance(model, nn.Module):
        raise TypeError(f"model must be an nn.Module, got {type(model).__name__}")
    if not isinstance(optimizer, Optimizer):
        raise TypeError(
            f"optimizer must be an Optimizer, got {type(optimizer).__name__}"
        )
    if not isinstance(scheduler, LRScheduler):
        raise TypeError(
            f"scheduler must be an LRScheduler, got {type(scheduler).__name__}"
        )
    max_steps = require_positive_integer(max_steps, name="max_steps")
    grad_accum_steps = require_positive_integer(
        grad_accum_steps,
        name="grad_accum_steps",
    )
    grad_clip = require_positive_real(grad_clip, name="grad_clip")
    resolved_device = get_device(device)

    model.to(resolved_device)
    model.train()
    batch_iterator = iter(_repeat_batches(batches))
    results: list[OptimizerStepResult] = []

    for _ in range(max_steps):

        def micro_losses() -> Iterator[Tensor]:
            for _ in range(grad_accum_steps):
                inputs, targets = next(batch_iterator)
                yield model(
                    inputs.to(resolved_device),
                    targets.to(resolved_device),
                )

        result = run_optimizer_step(
            optimizer,
            micro_losses(),
            grad_accum_steps=grad_accum_steps,
            grad_clip=grad_clip,
        )
        scheduler.step()
        results.append(result)

    return results


def train_tiny_text(
    text: str,
    config: ProjectConfig,
) -> TinyTextTrainingResult:
    """Compose the byte tokenizer, tiny dataset, GPT, AdamW, and train loop."""

    if not isinstance(config, ProjectConfig):
        raise TypeError(f"config must be a ProjectConfig, got {type(config).__name__}")
    config.validate()
    if config.tokenizer.type != "byte":
        raise ValueError(
            "tiny-text training requires tokenizer.type='byte', "
            f"got {config.tokenizer.type!r}"
        )

    set_seed(config.run.seed)
    device = get_device(config.run.device)
    tokenizer = ByteTokenizer()
    if config.tokenizer.vocab_size != tokenizer.get_vocab_size():
        raise ValueError(
            "tiny-text training requires tokenizer.vocab_size="
            f"{tokenizer.get_vocab_size()}, got {config.tokenizer.vocab_size}"
        )
    if tuple(config.tokenizer.special_tokens) != SPECIAL_TOKENS:
        raise ValueError(
            "tiny-text training requires the ByteTokenizer special-token order"
        )

    token_ids = tokenizer.encode(text)
    dataset = NextTokenDataset(
        token_ids,
        config.model.seq_len,
        vocab_size=tokenizer.get_vocab_size(),
    )
    if len(dataset) < config.train.device_batch_size:
        raise ValueError(
            "tiny text must produce at least one complete device batch; "
            f"found {len(dataset)} examples for batch size "
            f"{config.train.device_batch_size}"
        )

    data_generator = torch.Generator().manual_seed(config.run.seed)
    batches = DataLoader(
        dataset,
        batch_size=config.train.device_batch_size,
        shuffle=True,
        drop_last=True,
        generator=data_generator,
    )
    model = GPT(config.model).to(device)
    optimizer = build_optimizer(model, config.train)
    scheduler = build_lr_scheduler(optimizer, config.train)
    grad_accum_steps = derive_grad_accum_steps(
        device_batch_size=config.train.device_batch_size,
        seq_len=config.model.seq_len,
        total_batch_size_tokens=config.train.total_batch_size_tokens,
    )
    steps = run_training_steps(
        model,
        batches,
        optimizer,
        scheduler,
        max_steps=config.train.max_steps,
        grad_accum_steps=grad_accum_steps,
        grad_clip=config.train.grad_clip,
        device=device,
    )
    return TinyTextTrainingResult(
        model=model,
        tokenizer=tokenizer,
        optimizer=optimizer,
        scheduler=scheduler,
        steps=tuple(steps),
    )


__all__ = [
    "OptimizerStepResult",
    "TinyTextTrainingResult",
    "derive_grad_accum_steps",
    "run_optimizer_step",
    "run_training_steps",
    "train_tiny_text",
]
