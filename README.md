# gpt-training-sandbox

A from-scratch PyTorch sandbox for pretraining, supervised finetuning, evaluating, and post-training small GPT-style chat models.

The repository is being built in small vertical slices. The byte tokenizer,
tiny decoder-only GPT, typed configuration, run layout, and local metrics
foundations are present. The tiny-text pretraining path and checkpoint-backed
sampling are executable; evaluation and chat commands have stable interfaces
whose non-dry-run implementations land in later slices.

## Roadmap status

Milestone 1 — Hello Tiny GPT: complete. The first vertical slice runs local
text through the byte tokenizer and tiny decoder-only GPT, trains to a
checkpoint with decreasing loss and local JSONL metrics, samples non-empty
text from that checkpoint, and resumes training from a saved step.

Run its bounded CPU command-level acceptance check with:

```bash
uv run --extra dev pytest -q tests/test_pretrain_integration.py
```

The full test suite also protects the deterministic fixed-batch overfit
threshold. Milestone 2 extends the local tracking foundation.

## Setup

Install [uv](https://docs.astral.sh/uv/), clone the repository, and create the
locked development environment:

```bash
uv sync --extra dev
```

The core install deliberately excludes W&B and web/demo frameworks. Install an
optional group only when working on it, for example `uv sync --extra tracking`
or `uv sync --extra web`. Every command's `--help` path works with the core
dependencies alone.

Ruff is pinned in the development extra because formatter output is
version-dependent. Update that pin and `uv.lock` together when intentionally
adopting a new formatter version.

## Tests

Run the full test suite from the repository root:

```bash
uv run --extra dev pytest
```

The repository-wide formatting check is:

```bash
uv run --extra dev ruff format --check .
```

## Smoke dry-run

Resolve the CPU-safe smoke configuration and prepare its run paths without
starting training:

```bash
uv run python -m scripts.pretrain --config configs/smoke.yaml --dry-run
```

The command prints the run directory, the resolved config path, and all
resolved values. It creates `runs/smoke/config.yaml` plus empty `metrics/` and
`checkpoints/` directories; it does not train or write a checkpoint.

Apply dotted overrides by repeating `--override`. Later values win:

```bash
uv run python -m scripts.pretrain \
  --config configs/smoke.yaml \
  --override run.name=smoke-two \
  --override train.max_steps=2 \
  --dry-run
```

The same `--config`, repeated `--override`, and `--dry-run` convention is
available on `train_tokenizer`, `eval_tokenizer`, `pretrain`, `eval_base`,
`train_sft`, and `eval_chat`.

## Training and sampling interfaces

The first-sprint executable path is:

```bash
uv run python -m scripts.pretrain --config configs/smoke.yaml
uv run python -m scripts.sample --checkpoint runs/smoke/checkpoints/last.pt
```

Pretraining reads the repository's deterministic `data/fixtures/tiny.txt`
corpus, writes the complete resolved config and JSONL metrics under
`runs/smoke/`, saves periodic `step_*.pt` checkpoints at `train.save_every`,
and atomically updates `checkpoints/last.pt`. A fresh run refuses to overwrite
existing training outputs.

Resume a periodic checkpoint into a new named run while keeping every other
resolved setting unchanged:

```bash
uv run python -m scripts.pretrain \
  --config configs/smoke.yaml \
  --override run.name=smoke-resumed \
  --resume runs/smoke/checkpoints/step_000075.pt
```

This first-sprint resume contract restores the model, optimizer, and scheduler
and advances from the saved step. Exact RNG and dataloader-position continuity
remain later roadmap work.

The sampling command loads the model, byte tokenizer, and generation defaults
from a versioned checkpoint. Pass `--prompt` more than once to sample multiple
prompts, or override checkpoint settings with `--device`, `--max-new-tokens`,
`--temperature`, `--top-k`, and `--seed`. The remaining command skeletons fail
explicitly until their slices land. Inspect any interface without optional
dependencies by running, for example, `uv run python -m scripts.web_chat
--help`.
