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
resolved values. It creates `runs/smoke/config.yaml`, a config-only
`metrics/metrics.jsonl`, a completed `metrics/summary.json`, and an empty
`checkpoints/` directory; it does not train or write a checkpoint.

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

## Run-local tracking outputs

Every config-driven command creates an always-local event stream at
`<run.output_dir>/<run.name>/metrics/metrics.jsonl`. Each line is one complete
UTF-8 JSON object with a `record_type` of `config`, `metrics`, or `artifact`. A
new run writes its fully resolved configuration once as the first record.
Reopening the same run validates that record instead of appending a duplicate
or a conflicting configuration.

`<run.output_dir>/<run.name>/metrics/summary.json` is an atomically replaced
view of the same lifecycle. Its stable schema is:

```json
{
  "schema_version": 1,
  "run": {
    "name": "smoke",
    "output_dir": "runs/smoke",
    "stage": "pretrain"
  },
  "status": "completed",
  "latest_step": 200,
  "latest_metrics": {
    "train/loss": 0.25
  }
}
```

`status` is `running`, `completed`, or `failed`. `latest_metrics` retains only
the latest JSON scalar for each metric name; nested diagnostic values remain
in the append-only JSONL audit trail. Resuming the same run identity preserves
its latest step and scalar metrics while updating the summary atomically.

## Optional W&B tracking

W&B is an optional fan-out from the always-on local JSONL tracker. The base
development install exercises every local workflow without importing W&B:

```bash
uv sync --extra dev
```

Install the tracking extra only for online or offline W&B runs:

```bash
uv sync --extra dev --extra tracking
```

The complete tracking section can be set in YAML. JSONL cannot be disabled:

```yaml
tracking:
  jsonl:
    enabled: true
    path: metrics/metrics.jsonl
  wandb:
    enabled: true
    project: scratch-llm
    entity: null
    group: 3090-pretrain
    name: null
    tags: [base, 3090]
    mode: offline
    dir: runs/wandb
    log_code: false
    log_model_artifacts: false
    log_dataset_artifacts: false
    log_tokenizer_artifacts: true
    log_prompts: false
    log_responses: false
```

Tracking values resolve in this order, with later sources winning:

1. YAML values override project defaults.
2. `WANDB_MODE`, `WANDB_PROJECT`, `WANDB_ENTITY`, and `WANDB_RUN_GROUP`
   override YAML.
3. Repeated dotted `--override tracking.wandb.<field>=<value>` options override
   the environment.
4. `--wandb` or `--no-wandb` and `--wandb-mode` are the final dedicated
   overrides.

`WANDB_MODE` selects a mode but does not by itself change
`tracking.wandb.enabled`; use `--wandb` or set `enabled: true` in YAML. The
factory passes an explicit W&B name through unchanged, or defaults it to the
resolved `run.name`. The configured group passes through, configured tags
preserve first-occurrence order, and each command adds a stable
`pipeline-stage:<command>` tag such as `pipeline-stage:pretrain`.

### Disabled: local-only smoke

This credential-free smoke command uses the base install, never imports W&B,
and still writes `config.yaml`, `metrics/metrics.jsonl`, and
`metrics/summary.json` below `runs/tracking-disabled-smoke/`:

```bash
uv run python -m scripts.pretrain \
  --config configs/smoke.yaml \
  --override run.name=tracking-disabled-smoke \
  --no-wandb \
  --wandb-mode disabled \
  --dry-run
```

Either `tracking.wandb.enabled: false` or `mode: disabled` keeps the remote
backend off. Local JSONL remains enabled in both cases.

### Offline: credential-free W&B smoke

Offline mode initializes the optional W&B SDK but performs no cloud sync and
does not require credentials:

```bash
WANDB_MODE=offline \
WANDB_PROJECT=scratch-llm \
WANDB_RUN_GROUP=3090-offline-smoke \
uv run --extra tracking python -m scripts.pretrain \
  --config configs/smoke.yaml \
  --override run.name=tracking-offline-smoke \
  --override tracking.wandb.dir=runs/wandb \
  --wandb \
  --dry-run
```

The command prints the exact offline run directory, normally below
`runs/wandb/wandb/offline-run-<timestamp>-<id>/`. When the user later has
credentials and network access, upload that directory with:

```bash
uv run --extra tracking wandb sync \
  runs/wandb/wandb/offline-run-<timestamp>-<id>
```

See the official [W&B sync command
reference](https://docs.wandb.ai/models/ref/cli/wandb-sync) for account and
bulk-sync options.

### Online: authenticated W&B

Online mode requires the user to authenticate W&B and have network access. The
following dry run demonstrates environment identity, an explicit name, and
configured tags; replace the entity before running it:

```bash
WANDB_PROJECT=scratch-llm \
WANDB_ENTITY=your-wandb-entity \
WANDB_RUN_GROUP=3090-pretrain \
uv run --extra tracking python -m scripts.pretrain \
  --config configs/smoke.yaml \
  --override run.name=tracking-online \
  --override tracking.wandb.name=tracking-online-explicit \
  --override tracking.wandb.tags=[pretrain,online] \
  --wandb \
  --wandb-mode online \
  --dry-run
```

The supported variables follow the official [W&B environment-variable
reference](https://docs.wandb.ai/models/track/environment-variables), but the
project's precedence rules above remain authoritative for these commands.

All three modes keep run configuration and metrics locally. Model checkpoints
are not uploaded by default because `log_model_artifacts` defaults to `false`;
large dataset artifacts also default off. Raw prompts and responses remain
independently opt-in through `log_prompts` and `log_responses`, both of which
default to `false`.

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
