# gpt-training-sandbox

A from-scratch PyTorch sandbox for pretraining, supervised finetuning, evaluating, and post-training small GPT-style chat models.

The repository is being built in small vertical slices. The byte tokenizer,
tiny decoder-only GPT, typed configuration, run layout, and local metrics
foundations are present. Training, sampling, evaluation, and chat commands have
stable interfaces, but their non-dry-run implementations land in later slices.

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

Those two commands currently exit with an explicit `not implemented` error;
the interfaces are reserved for the training-loop and generation slices. The
same explicit behavior applies to the remaining command skeletons. Inspect any
interface without optional dependencies by running, for example,
`uv run python -m scripts.web_chat --help`.
