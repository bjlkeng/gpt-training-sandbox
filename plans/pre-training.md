# Build a Chat LLM From Scratch in PyTorch — nanochat-Aligned Project Plan

**Audience:** engineering implementation planning  

**Target use:** break this document into GitHub/Jira tickets  

**Primary hardware target:** single RTX 3090, single process, no data parallelism initially  

**Implementation style:** build core components from scratch using PyTorch, NumPy, Pandas, and small supporting libraries  

**Reference inspiration:** Andrej Karpathy's `nanochat`, especially its pipeline shape, tokenizer behavior, metrics, and evaluation names

---

## 0. Locked Decisions

These are the project decisions already made.

| Area | Decision |

|---|---|

| Overall shape | Roughly follow `karpathy/nanochat`: tokenizer training, base pretraining, base eval, SFT, chat eval, inference, and local UI. |

| Core implementation | Build core educational pieces from scratch: tokenizer, transformer, dataloading, training loop, eval harness. Use PyTorch as the tensor/autograd primitive. |

| Tokenizer behavior | Match nanochat's GPT-4-style regex byte-BPE behavior, but implement the BPE training/encoding logic ourselves. |

| Data behavior | Follow nanochat's ClimbMix-style parquet-shard flow: train shards plus final validation shard convention. |

| Model size | Start in the lower band: roughly **10M–50M parameters**. |

| Model architecture | First ship a plain GPT decoder. Add nanochat-style architecture features later, one at a time. |

| Optimizer | Start with AdamW. Add Muon/custom optimizers later. |

| Hardware | Single RTX 3090. No DDP/FSDP/tensor parallel/data parallel in early phases. |

| SFT data | Use both a tiny local JSONL fixture and nanochat-style SFT data. |

| Metrics | Compute nanochat-compatible metrics so runs can be compared roughly: tokenizer compression, validation BPB, CORE, ChatCORE, training throughput, MFU, FLOPs, VRAM, and inference latency. |

| Experiment tracking | Always write local JSONL metrics. Add optional W&B across tokenizer, data prep, pretraining, eval, SFT, chat eval, inference, and UI sessions. |

| Local UI | Add a simple local web UI for checkpoint testing. Keep it as a testing harness, not a product. |

---

## 1. Reference Alignment Notes

This project should not blindly copy nanochat's implementation. The goal is to preserve the comparable external behavior while keeping the internals educational and staged.

### 1.1 nanochat Concepts to Match Closely

Match these because they affect comparability:

- Tokenizer special-token set.

- GPT-4-style regex split pattern.

- `token_bytes.pt` concept for BPB.

- Validation BPB calculation.

- CORE metric naming and output format.

- ChatCORE task list and centered-score formula.

- Fixed base-model sample prompts.

- Training metric names like `val_bpb`, `core_metric`, `train/mfu`, `train/tok_per_sec`, `total_training_time`, and `total_training_flops`.

- Inference metrics like prefill latency, decode throughput, VRAM, KV cache size, MFU, and MBU.

### 1.2 nanochat Concepts to Delay

Delay these until the baseline works:

- FlashAttention.

- FP8.

- MuonAdamW.

- GQA.

- Sliding-window attention.

- KV-cache inference.

- Explicit nanochat dtype system.

- Tool execution.

- Multi-GPU request serving.

- Distributed training.

### 1.3 Important Current nanochat Details

As of this plan, nanochat's public README describes the project as a minimal experimental harness covering tokenization, pretraining, finetuning, evaluation, and inference. It also says current development focuses on reducing time-to-GPT-2 capability using DCLM CORE and tracks `val_bpb`, `core_metric`, VRAM utilization, `train/mfu`, and `train/tok_per_sec`.

The current nanochat repo structure includes:

```text

nanochat/

  checkpoint_[manager.py](http://manager.py)

  [common.py](http://common.py)

  core_[eval.py](http://eval.py)

  [dataloader.py](http://dataloader.py)

  [dataset.py](http://dataset.py)

  [engine.py](http://engine.py)

  [execution.py](http://execution.py)

  [gpt.py](http://gpt.py)

  loss_[eval.py](http://eval.py)

  [optim.py](http://optim.py)

  [tokenizer.py](http://tokenizer.py)

scripts/

  base_[eval.py](http://eval.py)

  base_[train.py](http://train.py)

  chat_[cli.py](http://cli.py)

  chat_[eval.py](http://eval.py)

  chat_[rl.py](http://rl.py)

  chat_[sft.py](http://sft.py)

  infer_[bench.py](http://bench.py)

  tok_[eval.py](http://eval.py)

  tok_[train.py](http://train.py)

  chat_[web.py](http://web.py)

```

This project should mirror that conceptual split while using clearer educational staging.

---

## 2. Guiding Engineering Principles

1. **Vertical slice first.** Get from raw text to generated text as early as possible.

2. **Correctness before speed.** The first attention implementation should be boring and testable.

3. **Simple before clever.** Use AdamW, manual causal attention, and learned positional embeddings first.

4. **Comparable metrics early.** BPB and metric logging should not be bolted on at the end.

5. **Single source of inference truth.** CLI and web UI must use the same `ChatEngine`.

6. **Single source of metric truth.** Local JSONL metrics are always written; W&B is optional.

7. **Parameterize from day one.** Config should expose model size, batch size, sequence length, gradient accumulation, learning rate, tokenizer size, eval intervals, and logging.

8. **Avoid framework drift.** Do not accidentally build a general LLM framework. Build one understandable end-to-end training harness.

9. **Prefer explicit tickets over vague epics.** Each ticket should have acceptance criteria and a test or artifact.

10. **Keep 3090 constraints real.** Start small. Make OOM recovery easy. Scale only when the baseline is stable.

---

## 3. High-Level Phase Outline

| Phase | Name | Goal | Primary Output |

|---:|---|---|---|

| 0 | Project foundation | Create repo, config, logging, test scaffolding | Runnable skeleton |

| 1 | Tiny vertical slice | Byte tokenizer + tiny GPT + train + sample | First text generation |

| 2 | Data pipeline | nanochat-style parquet ingestion and token shard prep | Reproducible train/val data |

| 3 | Tokenizer | From-scratch regex byte-BPE matching nanochat behavior | 32K tokenizer + `token_bytes.pt` |

| 4 | Transformer | Simple GPT decoder from scratch | 10M–50M model presets |

| 5 | Pretraining | Single-GPU training loop | Checkpointed base model |

| 6 | Metrics and base eval | BPB, samples, CORE, throughput, memory | nanochat-comparable base metrics |

| 7 | SFT | Chat rendering, assistant-only masking, SFT trainer | First chat checkpoint |

| 8 | Chat CLI | Terminal chat testing | Interactive local CLI |

| 9 | Local web UI | Browser-based testing harness | Streaming local chat UI |

| 10 | W&B tracking | Optional experiment tracking everywhere | W&B + JSONL metrics pipeline |

| 11 | 3090 scaling pass | Stable longer single-GPU runs | Practical 20M/45M configs |

| 12 | Performance optimization | AMP, SDPA, FlashAttention, compile, KV cache | Faster training/inference |

| 13 | Architecture parity | Add nanochat-style features | Better model architecture variants |

| 14 | Quality and eval | Better data, evals, comparisons | Run comparison reports |

| 15 | Optional advanced chat | RL, tools, preference tuning | Experimental extensions |

---

## 4. Proposed Repository Structure

```text

llm-from-scratch/

  [README.md](http://README.md)

  pyproject.toml

  uv.lock                         # optional if using uv

  configs/

    smoke.yaml

    tiny_20m_3090.yaml

    small_45m_3090.yaml

    tokenizer_32k.yaml

    sft_smoke.yaml

    sft_3090.yaml

    web_local.yaml

  data/

    raw/

    parquet/

    processed/

    tokenized/

    fixtures/

  metrics/

    [README.md](http://README.md)

  runs/

    smoke_[pretrain.sh](http://pretrain.sh)

    tiny_[pretrain.sh](http://pretrain.sh)

    small_[pretrain.sh](http://pretrain.sh)

    [sft.sh](http://sft.sh)

    chat_[cli.sh](http://cli.sh)

    web_[chat.sh](http://chat.sh)

  src/

    scratch_llm/

      **init**.py

      [common.py](http://common.py)

      [config.py](http://config.py)

      [tracking.py](http://tracking.py)

      [checkpoint.py](http://checkpoint.py)

      [tokenizer.py](http://tokenizer.py)

      [data.py](http://data.py)

      [dataloader.py](http://dataloader.py)

      [model.py](http://model.py)

      [attention.py](http://attention.py)

      [optim.py](http://optim.py)

      [schedule.py](http://schedule.py)

      [train.py](http://train.py)

      loss_[eval.py](http://eval.py)

      base_[eval.py](http://eval.py)

      core_[eval.py](http://eval.py)

      chat_[format.py](http://format.py)

      [sft.py](http://sft.py)

      chat_[eval.py](http://eval.py)

      [generation.py](http://generation.py)

      [engine.py](http://engine.py)

      infer_[bench.py](http://bench.py)

      web/

        **init**.py

        [app.py](http://app.py)

        [schemas.py](http://schemas.py)

        [server.py](http://server.py)

        static/

          index.html

          app.js

          styles.css

  scripts/

    prepare_[data.py](http://data.py)

    download_[climbmix.py](http://climbmix.py)

    train_[tokenizer.py](http://tokenizer.py)

    eval_[tokenizer.py](http://tokenizer.py)

    [pretrain.py](http://pretrain.py)

    eval_[base.py](http://base.py)

    [sample.py](http://sample.py)

    train_[sft.py](http://sft.py)

    eval_[chat.py](http://chat.py)

    [chat.py](http://chat.py)

    web_[chat.py](http://chat.py)

    infer_[bench.py](http://bench.py)

    compare_[runs.py](http://runs.py)

  tasks/

    [common.py](http://common.py)

    [arc.py](http://arc.py)

    [gsm8k.py](http://gsm8k.py)

    [humaneval.py](http://humaneval.py)

    [mmlu.py](http://mmlu.py)

    [smoltalk.py](http://smoltalk.py)

  tests/

    test_[config.py](http://config.py)

    test_[tokenizer.py](http://tokenizer.py)

    test_[data.py](http://data.py)

    test_[dataloader.py](http://dataloader.py)

    test_[attention.py](http://attention.py)

    test_[model.py](http://model.py)

    test_[training.py](http://training.py)

    test_[checkpoint.py](http://checkpoint.py)

    test_loss_[eval.py](http://eval.py)

    test_chat_[format.py](http://format.py)

    test_chat_[engine.py](http://engine.py)

    test_[tracking.py](http://tracking.py)

    test_[web.py](http://web.py)

```

---

## 5. Configuration System

Use YAML plus typed dataclasses.

### 5.1 Example Config

```yaml

run:

  name: smoke

  seed: 1337

  device: cuda

  output_dir: runs/out

tracking:

  jsonl:

    enabled: true

    path: metrics/metrics.jsonl

  wandb:

    enabled: false

    project: scratch-llm

    entity: null

    group: null

    name: null

    tags: []

    mode: online      # online | offline | disabled

    dir: runs/wandb

    log_code: false

    log_model_artifacts: false

    log_dataset_artifacts: false

    log_tokenizer_artifacts: true

    log_prompts: false

    log_responses: false

data:

  profile: nanochat_climbmix

  base_dir: data

  parquet_dir: data/parquet/base_data_climbmix

  tokenized_dir: data/tokenized

  text_column: text

  num_tokenizer_train_shards: 8

  num_pretrain_train_shards: 16

  always_use_final_shard_for_val: true

  max_shard: 6542

  doc_cap_chars: 10000

tokenizer:

  type: regex_byte_bpe

  vocab_size: 32768

  max_chars: 2000000000

  doc_cap: 10000

  special_tokens:

    - "<|bos|>"

    - "<|user_start|>"

    - "<|user_end|>"

    - "<|assistant_start|>"

    - "<|assistant_end|>"

    - "<|python_start|>"

    - "<|python_end|>"

    - "<|output_start|>"

    - "<|output_end|>"

model:

  profile: simple_gpt

  vocab_size: 32768

  seq_len: 512

  n_layer: 6

  n_head: 6

  n_embd: 384

  mlp_ratio: 4

  dropout: 0.0

  bias: false

  tie_weights: true

  norm: layernorm

  activation: gelu

  use_rope: false

  use_rmsnorm: false

  use_qk_norm: false

  use_gqa: false

  use_flash_attention: false

  use_kv_cache: false

train:

  device_batch_size: 4

  total_batch_size_tokens: 65536

  grad_accum_steps: auto

  max_steps: 20000

  learning_rate: 0.0003

  min_lr: 0.000015

  weight_decay: 0.1

  beta1: 0.9

  beta2: 0.95

  grad_clip: 1.0

  warmup_steps: 40

  warmdown_ratio: 0.65

  final_lr_frac: 0.05

  eval_every: 250

  eval_tokens: 1048576

  sample_every: 1000

  save_every: 1000

  log_every: 10

  dtype: float32      # start fp32; later float16 AMP on 3090

  compile: false

  activation_checkpointing: false

generation:

  temperature: 0.8

  top_k: 50

  top_p: null

  max_new_tokens: 256

  seed: null

web:

  enabled: true

  host: 127.0.0.1

  port: 8000

  checkpoint_dir: runs/out

  allow_remote_bind: false

```

### 5.2 Config Acceptance Criteria

- Config loads into typed dataclasses.

- Config validates obvious contradictions early.

- Resolved config is written to every run directory.

- CLI args can override config fields.

- Every training/eval script prints the resolved config path and run directory.

---

# Phase 0 — Project Foundation

## Goal

Create a clean repo skeleton that can support the full training pipeline without turning into a large framework.

## Build

### 0.1 Dependencies

Required early dependencies:

```text

python

pytorch

numpy

pandas

pyarrow

pyyaml or tomllib/tomli

tqdm

pytest

regex

```

Optional dependency groups:

```toml

[project.optional-dependencies]

tracking = ["wandb"]

web = ["fastapi", "uvicorn", "websockets", "pydantic"]

demo = ["gradio"]

dev = ["pytest", "matplotlib", "ruff", "mypy"]

```

Avoid early:

```text

transformers as core dependency

tokenizers as core dependency

datasets as core dependency

accelerate

lightning

trl

deepseed

```

Optional eval comparison dependencies can come later.

### 0.2 Common Utilities

Implement:

```text

set_seed()

get_device()

autodetect_device_type()

count_parameters()

format_num()

format_bytes()

get_run_dir()

save_json()

load_json()

atomic_write()

timer utilities

basic logger

GPU memory stats helper

OOM helper message

```

### 0.3 Script Skeletons

Every script should parse config and exit cleanly even before implementation is done.

```bash

python -m scripts.train_tokenizer --config configs/tokenizer_32k.yaml

python -m scripts.eval_tokenizer --config configs/tokenizer_32k.yaml

python -m scripts.pretrain --config configs/smoke.yaml

python -m scripts.eval_base --config configs/smoke.yaml --eval bpb,sample

python -m scripts.train_sft --config configs/sft_smoke.yaml

python -m scripts.eval_chat --config configs/sft_smoke.yaml

python -m [scripts.chat](http://scripts.chat) --checkpoint runs/.../[best.pt](http://best.pt)

python -m scripts.web_chat --checkpoint runs/.../[best.pt](http://best.pt)

```

## Acceptance Criteria

- `pytest` runs.

- Config loading works.

- All script entrypoints exist.

- `python -m scripts.pretrain --config configs/smoke.yaml --dry-run` prints a resolved config and exits.

- Repo has a README with setup and smoke commands.

## Tickets

```text

FOUND-001: Create repo skeleton

FOUND-002: Add pyproject and optional dependency groups

FOUND-003: Add config dataclasses

FOUND-004: Add YAML config loader

FOUND-005: Add CLI override mechanism

FOUND-006: Add common utility module

FOUND-007: Add run directory manager

FOUND-008: Add logging helper

FOUND-009: Add GPU memory helper

FOUND-010: Add OOM advice helper

FOUND-011: Add script stubs

FOUND-012: Add smoke configs

FOUND-013: Add pytest setup

FOUND-014: Add README setup instructions

```

---

# Phase 1 — Tiny Vertical Slice

## Goal

Get a full path working before building the real tokenizer and data pipeline:

```text

local text file -> byte tokenizer -> tiny GPT -> train -> sample

```

## Build

### 1.1 Toy Dataset

Use:

```text

data/fixtures/tiny.txt

```

This can be public-domain text or a small synthetic file. The point is not quality. The point is proving every component can talk to every other component.

### 1.2 Byte Tokenizer

Start with a byte tokenizer:

```text

UTF-8 bytes 0..255 + special tokens

```

Advantages:

- No unknown tokens.

- Unicode round trips.

- Easy bridge to byte-level BPE.

- Tiny implementation.

### 1.3 Tiny Model

Example:

```yaml

model:

  vocab_size: 256 + special_tokens

  seq_len: 128

  n_layer: 2

  n_head: 2

  n_embd: 128

  mlp_ratio: 4

  tie_weights: true

train:

  device_batch_size: 16

  grad_accum_steps: 1

  max_steps: 500

```

### 1.4 Naive Generation

Implement simple autoregressive generation:

```python

for  *in range(max*new_tokens):

    idx_cond = idx[:, -seq_len:]

    logits = model(idx_cond)

    logits = logits[:, -1, :]

    next_id = sample(logits, temperature, top_k)

    idx = [torch.cat](http://torch.cat)([idx, next_id], dim=1)

```

No KV cache yet.

## Acceptance Criteria

- Training loss decreases on tiny text.

- Model can overfit one fixed batch.

- `sample.py` prints generated text.

- Tiny run completes quickly on CPU and CUDA.

- All core shapes are tested.

## Tickets

```text

SMOKE-001: Add byte tokenizer

SMOKE-002: Add tiny text fixture

SMOKE-003: Add simple token dataset

SMOKE-004: Add tiny GPT model config

SMOKE-005: Add naive train loop

SMOKE-006: Add naive sample script

SMOKE-007: Add overfit-one-batch test

SMOKE-008: Add generation smoke test

```

---

# Phase 2 — nanochat-Style Data Pipeline

## Goal

Create a reliable data pipeline that follows nanochat's pretraining data shape closely enough for rough comparison.

## 2.1 Data Layout

Use:

```text

data/parquet/base_data_climbmix/

  shard_00000.parquet

  shard_00001.parquet

  ...

  shard_06542.parquet

```

Each shard should expose a `text` column.

Split convention:

```text

train = all selected shards except final validation shard

val   = final shard

```

For 3090 work, support partial downloads:

```yaml

data:

  num_tokenizer_train_shards: 8

  num_pretrain_train_shards: 16

  always_use_final_shard_for_val: true

```

## 2.2 Raw Data Iterator

Implement:

```python

def list_parquet_files(data_dir: str) -> list[Path]: ...

def parquets_iter_batched(split: str, start: int = 0, step: int = 1) -> Iterator[list[str]]: ...

```

The `startstep` parameters are useful later if distributed training is added, but early single-GPU mode can use `start=0, step=1`.

## 2.3 Download Utility

Implement:

```bash

python -m [scripts.download](http://scripts.download)_climbmix --num-train-shards 16 --include-val

```

Requirements:

- Resume partial downloads.

- Download to temporary file, then atomic rename.

- Retry on network errors.

- Verify nonzero file size.

- Print shard count and total size.

## 2.4 Local Fixtures

Add tiny parquet fixtures under:

```text

data/fixtures/parquet/

```

Use these for tests so CI or local test runs do not need internet or large files.

## 2.5 Tokenized Shards

After tokenizer exists, write tokenized output as:

```text

data/tokenized/base/

  train_000000.bin

  train_000001.bin

  val_000000.bin

  manifest.json

```

Suggested binary format:

- `uint16` when `vocab_size <= 65535`.

- `uint32` otherwise.

- Sidecar manifest contains dtype, token count, source shards, tokenizer hash, and special token ids.

## 2.6 Dataloader

First version:

```python

x: LongTensor[batch_size, seq_len]

y: LongTensor[batch_size, seq_len]

y = x shifted by one token

```

Implement random contiguous chunks from a flat memmap token array.

Later version:

- BOS-aware best-fit packing.

- Document boundary handling.

- Resumeable dataloader state.

## Acceptance Criteria

- Can download N train shards plus the fixed validation shard.

- Can iterate train and val splits deterministically.

- Can read local parquet fixtures.

- Can write tokenized binary shards and manifest.

- Dataloader returns correct `x`, `y` shapes.

- `y[:, :-1] == x[:, 1:]` for simple contiguous batches.

- Dataloader works without raw text once tokenized shards exist.

## Tickets

```text

DATA-001: Implement ClimbMix-style shard downloader

DATA-002: Add resumable temporary-file download flow

DATA-003: Implement parquet file lister

DATA-004: Implement parquet batched iterator

DATA-005: Implement nanochat train/val split convention

DATA-006: Add local parquet fixture

DATA-007: Add data stats command

DATA-008: Implement tokenized shard manifest

DATA-009: Implement tokenized shard writer

DATA-010: Implement tokenized shard reader

DATA-011: Implement random-offset token dataloader

DATA-012: Add dataloader state serialization

DATA-013: Add document-boundary-aware packing later

DATA-014: Add data pipeline tests

```

---

# Phase 3 — Tokenizer From Scratch, nanochat-Compatible Behavior

## Goal

Build a regex byte-BPE tokenizer from scratch that matches nanochat's behavior closely enough for data and metric comparability.

nanochat currently uses a Rust BPE trainer plus tiktoken for inference. This project should implement the BPE itself in Python for educational value, even if slower.

## 3.1 Special Tokens

Use nanochat's special tokens exactly:

```text

<|bos|>

<|user_start|>

<|user_end|>

<|assistant_start|>

<|assistant_end|>

<|python_start|>

<|python_end|>

<|output_start|>

<|output_end|>

```

Do not add `<|pad|>` initially. For padding in categorical chat eval, use BOS as pad and ignore those positions.

## 3.2 Regex Split Pattern

Use nanochat's split pattern:

```python

SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,2}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""

```

Implementation note: use the third-party `regex` package, not Python's built-in `re`, because the pattern uses Unicode property classes like `\p{L}` and `\p{N}`.

## 3.3 Tokenizer API

```python

class Tokenizer:

    def train(self, texts: Iterable[str]) -> None: ...

    def encode(self, text: str, prepend=None, append=None) -> list[int]: ...

    def decode(self, ids: list[int]) -> str: ...

    def encode_special(self, token: str) -> int: ...

    def decode_single_token_bytes(self, token_id: int) -> bytes: ...

    def get_vocab_size(self) -> int: ...

    def get_bos_token_id(self) -> int: ...

    def get_special_tokens(self) -> set[str]: ...

    def save(self, path: str) -> None: ...

    @classmethod

    def load(cls, path: str) -> "Tokenizer": ...

```

## 3.4 BPE Training Algorithm

Plain first implementation:

```text

1. Regex-split each document into chunks.

2. Encode each chunk to UTF-8 bytes.

3. Represent bytes as token ids 0..255.

4. Count adjacent token pairs within each regex chunk.

5. Pick the most frequent pair.

6. Assign a new token id.

7. Replace all occurrences of that pair.

8. Repeat until mergeable vocab size is vocab_size - len(special_tokens).

9. Add special tokens after mergeable tokens.

```

Do not merge across regex-chunk boundaries.

## 3.5 Save Artifacts

Save:

```text

tokenizer.json

merges.json

vocab.json

special_tokens.json

token_[bytes.pt](http://bytes.pt)

```

`token_bytes.pt` is mandatory for BPB.

Definition:

```python

token_bytes[token_id] = len(raw_token_bytes)

token_bytes[special_token_id] = 0

```

Do not compute token byte lengths by decoding to string first. Some token byte sequences are not valid standalone UTF-8.

## 3.6 Tokenizer Eval

Implement a tokenizer eval script similar to nanochat's `tok_eval.py`.

Report:

```text

vocab_size

bytes

tokens

bytes_per_token

relative token-count difference vs GPT-2 tokenizer

relative token-count difference vs GPT-4/cl100k tokenizer

round-trip decode pass/fail

encode tokens/sec

decode tokens/sec

```

Evaluate on categories:

```text

news

korean

code

math

science

climbmix-train sample

climbmix-val sample

```

## Acceptance Criteria

- `decode(encode(text)) == text` for ASCII, Unicode, whitespace, code, math, Korean, and emoji fixtures.

- Special tokens encode as single tokens.

- Special tokens are never counted in BPB bytes.

- Tokenizer can save/load with identical output.

- Tokenizer reaches requested vocab size.

- `token_bytes.pt` has shape `(vocab_size,)`.

- Regex split tests pass for fixed strings.

- Tokenizer eval writes JSON and Markdown summaries.

## Tickets

```text

TOK-001: Implement tokenizer interface

TOK-002: Implement byte tokenizer

TOK-003: Add nanochat special token constants

TOK-004: Implement nanochat regex splitter

TOK-005: Implement BPE pair counting

TOK-006: Implement merge application

TOK-007: Implement regex-chunk-local BPE training

TOK-008: Implement BPE encoder

TOK-009: Implement BPE decoder from raw bytes

TOK-010: Implement special token encode/decode

TOK-011: Implement tokenizer save/load

TOK-012: Implement token_[bytes.pt](http://bytes.pt) writer

TOK-013: Implement tokenizer eval script

TOK-014: Add optional GPT-2/GPT-4 tokenizer comparison

TOK-015: Add tokenizer speed benchmark

TOK-016: Add Unicode round-trip tests

TOK-017: Add regex split parity tests

TOK-018: Add special-token BPB masking tests

TOK-019: Optimize pair counting later

```

---

# Phase 4 — Transformer From Scratch

## Goal

Implement a simple decoder-only GPT model in PyTorch.

Start plain:

```text

token embeddings

learned positional embeddings

causal self-attention

MLP

LayerNorm

residual connections

LM head

cross-entropy loss

```

Delay:

```text

RoPE

RMSNorm

QK norm

GQA

FlashAttention

KV cache

sliding windows

value embeddings

smear/backout experiments

```

## 4.1 Core Classes

```python

@dataclass

class GPTConfig:

    vocab_size: int

    seq_len: int

    n_layer: int

    n_head: int

    n_embd: int

    mlp_ratio: int = 4

    dropout: float = 0.0

    bias: bool = False

    tie_weights: bool = True

```

```python

class CausalSelfAttention(nn.Module): ...

class MLP(nn.Module): ...

class Block(nn.Module): ...

class GPT(nn.Module): ...

```

`n_embd` is the residual-stream width and remains unchanged between blocks.
The MLP temporarily expands each token to `mlp_ratio * n_embd`, then projects
back to `n_embd` before the residual addition.

## 4.2 Manual Attention First

```python

q, k, v = self.qkv(x).split(...)

scores = q @ k.transpose(-2, -1) / math.sqrt(head_dim)

scores = scores.masked_fill(causal_mask == 0, float("-inf"))

att = F.softmax(scores, dim=-1)

y = att @ v

```

Causal-mask correctness deserves a dedicated test. A silent mask bug makes every later result meaningless.

## 4.3 Forward API

```python

def forward(self, idx, targets=None, loss_reduction="mean"):

    logits = ...

    if targets is None:

        return logits

    loss = F.cross_entropy(

        logits.view(-1, logits.size(-1)),

        targets.view(-1),

        ignore_index=-1,

        reduction=loss_reduction,

    )

    if loss_reduction == "none":

        loss = loss.view(targets.shape)

    return loss

```

Use `ignore_index=-1` so SFT masking matches nanochat.

## 4.4 Starter Model Presets

With a 32K vocab, embeddings dominate small models. Use tied embeddings initially.

| Preset | Layers | Embedding | Heads | Seq Len | Vocab | Tie Weights | Approx Params |

|---|---:|---:|---:|---:|---:|---|---:|

| `smoke` | 2 | 128 | 2 | 256 | 32K | yes | ~5M |

| `tiny_20m` | 6 | 384 | 6 | 512 | 32K | yes | ~23M |

| `small_45m` | 8 | 512 | 8 | 1024 | 32K | yes | ~42M |

Do not train 100M+ until dataloading, checkpointing, BPB, and mixed precision are stable.

## 4.5 nanochat-Style Depth Profile Later

Add optional config:

```yaml

model:

  profile: nanochat_depth

  depth: 4

  aspect_ratio: 64

  head_dim: 128

  max_seq_len: 1024

```

Formula:

```python

base_dim = depth * aspect_ratio

model_dim = ceil_to_multiple(base_dim, head_dim)

num_heads = model_dim // head_dim

```

Keep this behind a profile flag until the simple GPT is stable.

## Acceptance Criteria

- Forward pass shape is correct.

- Loss is scalar for mean reduction.

- Loss is `(B, T)` for unreduced reduction.

- Model can overfit one batch.

- Causal mask test passes.

- Generation works.

- Parameter count is printed.

- Model can save/load through checkpoint system.

## Tickets

```text

MODEL-001: Implement GPTConfig

MODEL-002: Implement token embeddings

MODEL-003: Implement learned positional embeddings

MODEL-004: Implement manual causal attention

MODEL-005: Implement MLP

MODEL-006: Implement transformer block

MODEL-007: Implement GPT wrapper

MODEL-008: Add tied embedding option

MODEL-009: Add ignore_index=-1 loss

MODEL-010: Add unreduced loss path

MODEL-011: Add generation helper

MODEL-012: Add parameter counting

MODEL-013: Add causal mask unit test

MODEL-014: Add overfit-one-batch test

MODEL-015: Add 20M and 45M configs

MODEL-016: Add nanochat depth profile later

```

---

# Phase 5 — Base-Model Pretraining Loop

## Goal

Train the base model on next-token prediction with a single-GPU training loop.

## 5.1 Core Loop

```text

load config

set seed

load tokenizer

load tokenized dataset

create dataloader

create model

create optimizer

create tracker

for step:

  for microstep in grad_accum_steps:

    x, y = next(train_loader)

    loss = model(x, y)

    loss = loss / grad_accum_steps

    backward

  clip gradients

  optimizer step

  scheduler step

  zero grads

  log metrics

  eval periodically

  sample periodically

  save checkpoint periodically

```

## 5.2 Batch Size Terms

Use nanochat-like terminology:

```yaml

train:

  device_batch_size: 4

  total_batch_size_tokens: 65536

  grad_accum_steps: auto

```

Compute:

```python

tokens_per_microbatch = device_batch_size * seq_len

grad_accum_steps = total_batch_size_tokens // tokens_per_microbatch

```

For single GPU, world size is 1.

## 5.3 Optimizer

Initial optimizer:

```text

PyTorch AdamW

```

Example:

```yaml

learning_rate: 3e-4

weight_decay: 0.1

beta1: 0.9

beta2: 0.95

grad_clip: 1.0

```

Later:

```text

custom AdamW

fused AdamW

MuonAdamW

separate LR groups

```

## 5.4 LR Schedule

Use nanochat-like warmup/constant/warmdown:

```text

linear warmup

constant middle

linear warmdown to final_lr_frac

```

Config:

```yaml

warmup_steps: 40

warmdown_ratio: 0.65

final_lr_frac: 0.05

```

## 5.5 Checkpointing

Save:

```text

model state_dict

optimizer state_dict

scheduler state

config

step

best val bpb

rng states

dataloader state

tokenizer path/hash

metric summary

```

Files:

```text

[last.pt](http://last.pt)

[best.pt](http://best.pt)

step_[000100.pt](http://000100.pt)

```

## 5.6 Logging

Log every `log_every` steps:

```text

step

train/loss

train/lrm

train/dt

train/tok_per_sec

train/mfu

train/grad_norm

train/epoch

peak_memory_mib

total_training_time

total_training_flops

```

## Acceptance Criteria

- Loss decreases on smoke data.

- Checkpoint resume works.

- Dataloader state resumes without repeating from the beginning.

- Validation runs without gradient updates.

- Training writes JSONL metrics.

- OOM gives useful advice.

- `tiny_20m` can run on RTX 3090 with conservative batch size.

## Tickets

```text

TRAIN-001: Implement AdamW optimizer setup

TRAIN-002: Implement grad accumulation from total_batch_size_tokens

TRAIN-003: Implement warmup/constant/warmdown LR schedule

TRAIN-004: Implement gradient clipping

TRAIN-005: Implement training metrics

TRAIN-006: Implement validation loop hook

TRAIN-007: Implement checkpoint save

TRAIN-008: Implement checkpoint resume

TRAIN-009: Save dataloader state

TRAIN-010: Save RNG state

TRAIN-011: Add best-checkpoint tracking

TRAIN-012: Add OOM handling advice

TRAIN-013: Add tokens/sec logging

TRAIN-014: Add FLOPs estimate logging

TRAIN-015: Add MFU estimate logging

TRAIN-016: Add peak VRAM logging

```

---

# Phase 6 — nanochat-Compatible Base Metrics and Evaluation

## Goal

Produce the same broad metrics nanochat uses so runs are roughly comparable.

## 6.1 BPB: Bits Per Byte

Do not use only mean token loss. Implement BPB.

Formula:

```python

bpb = total_nats / math.log(2) / total_bytes

```

Where:

```text

total_nats  = sum of unreduced cross-entropy over counted target tokens

total_bytes = sum of raw byte lengths for counted target tokens

```

Ignore:

```text

special tokens, where token_bytes[id] == 0

masked targets, where y < 0

padding targets

```

## 6.2 Base Eval Modes

CLI:

```bash

python -m scripts.eval_base --checkpoint runs/.../[best.pt](http://best.pt) --eval bpb,sample,core

python -m scripts.eval_base --checkpoint runs/.../[best.pt](http://best.pt) --eval bpb

python -m scripts.eval_base --checkpoint runs/.../[best.pt](http://best.pt) --eval sample

python -m scripts.eval_base --checkpoint runs/.../[best.pt](http://best.pt) --eval core --max-per-task 100

```

## 6.3 Fixed Base Sampling Prompts

Use nanochat's base-training sample prompts:

```text

The capital of France is

The chemical symbol of gold is

If yesterday was Friday, then tomorrow will be

The opposite of hot is

The planets of the solar system are:

My favorite color is

If 5*x + 3 = 13, then x is

```

## 6.4 CORE Metric

Implement the DCLM CORE-style task harness.

High-level flow:

```text

load eval bundle

load core.yaml

for each ICL task:

  load examples

  evaluate accuracy

  load random baseline

  compute centered score

CORE = mean centered score across tasks

```

Centered formula:

```python

centered = (accuracy - 0.01  *random_baseline) / (1.0 - 0.01*  random_baseline)

core_metric = mean(centered_results.values())

```

Start with small `--max-per-task` for 3090 iteration.

## 6.5 Required Output Files

```text

metrics/metrics.jsonl

metrics/base_eval.json

metrics/base_[samples.md](http://samples.md)

metrics/summary.json

```

## Acceptance Criteria

- `evaluate_bpb` agrees with a hand-computed toy example.

- Special tokens are ignored in BPB.

- Masked targets are ignored in BPB.

- Base eval can run `bpb`, `sample`, and `core` independently.

- Samples are saved to Markdown.

- Eval metrics are logged to JSONL and optional W&B.

## Tickets

```text

EVAL-001: Implement token_bytes-based BPB

EVAL-002: Add BPB toy unit test

EVAL-003: Implement base_eval CLI

EVAL-004: Implement eval mode selection

EVAL-005: Implement fixed sample prompt generation

EVAL-006: Save [samples.md](http://samples.md)

EVAL-007: Implement CORE task loader

EVAL-008: Implement CORE centered formula

EVAL-009: Implement max-per-task option

EVAL-010: Write base_eval.json

EVAL-011: Log eval metrics through Tracker

EVAL-012: Add run comparison report

```

---

# Phase 7 — Chat Formatting and Supervised Finetuning

## Goal

Turn the base model into a chat model using supervised finetuning.

Blunt constraint: SFT teaches response format and instruction-following behavior. It does not magically create strong knowledge or reasoning if the base model is weak.

## 7.1 Chat Data Schema

Tiny fixture format:

```json

{"messages":[{"role":"user","content":"Hello"},{"role":"assistant","content":"Hi! How can I help?"}]}

```

Support optional system messages by merging them into the first user message, matching nanochat's current renderer behavior.

## 7.2 Conversation Rendering

Base template:

```text

<|bos|>

<|user_start|>

{user content}

<|user_end|>

<|assistant_start|>

{assistant content}

<|assistant_end|>

```

Multi-turn:

```text

<|bos|>

<|user_start|>...<|user_end|>

<|assistant_start|>...<|assistant_end|>

<|user_start|>...<|user_end|>

<|assistant_start|>...<|assistant_end|>

```

Tool-part support later:

```text

<|python_start|> code <|python_end|>

<|output_start|> output <|output_end|>

```

## 7.3 Assistant-Only Loss Masking

Train only on assistant response tokens and assistant control tokens that the model is expected to emit.

Use:

```python

ignore_index = -1

targets[loss_mask == 0] = -1

```

Mask out:

```text

BOS

user_start/user content/user_end

assistant_start prompt token

python output tokens

padding/cropping fill

```

Train on:

```text

assistant content

assistant_end

python tool-call text when generated by assistant

python_start/python_end around assistant tool calls

```

## 7.4 SFT Data Sources

Use both:

```text

1. Tiny local JSONL fixture for correctness.

2. Larger nanochat-style SFT mixture.

```

Mixture target:

```text

SmolTalk-style chat data

MMLU auxiliary train data

GSM8K train data

small identity/personality fixture if desired

spelling/counting tasks later

```

## 7.5 SFT Training

SFT should resume from base checkpoint.

Example config:

```yaml

sft:

  base_checkpoint: runs/base/tiny_20m/[best.pt](http://best.pt)

  learning_rate: 0.00002

  weight_decay: 0.0

  device_batch_size: 2

  total_batch_size_tokens: 32768

  max_steps: 5000

  eval_every: 250

  max_seq_len: 1024

```

## Acceptance Criteria

- Conversation renderer produces exact expected tokens and mask.

- SFT can overfit tiny local fixture.

- SFT dataloader packs conversations without losing labels.

- SFT validation BPB works.

- Chat checkpoint samples with chat template.

- Model stops on `<|assistant_end|>`.

## Tickets

```text

SFT-001: Define chat JSONL schema

SFT-002: Add tiny chat fixture

SFT-003: Implement conversation renderer

SFT-004: Implement system-message merge behavior

SFT-005: Implement assistant-only loss mask

SFT-006: Use ignore_index=-1 in SFT labels

SFT-007: Implement SFT dataloader

SFT-008: Implement best-fit conversation packing

SFT-009: Implement SmolTalk loader

SFT-010: Implement MMLU auxiliary loader

SFT-011: Implement GSM8K loader

SFT-012: Implement SFT trainer

SFT-013: Implement SFT validation BPB

SFT-014: Add SFT overfit test

SFT-015: Add stop-token generation test

```

---

# Phase 8 — Chat CLI

## Goal

Build a simple terminal interface for testing SFT checkpoints.

## 8.1 Command

```bash

python -m [scripts.chat](http://scripts.chat) --checkpoint runs/sft/tiny_20m/[best.pt](http://best.pt)

python -m [scripts.chat](http://scripts.chat) --checkpoint runs/sft/tiny_20m/[best.pt](http://best.pt) -p "Explain gradient descent simply."

```

## 8.2 Behavior

Features:

```text

interactive user input

one-shot prompt mode

conversation history

reset command

exit command

temperature

top_k

max_new_tokens

stop at assistant_end

transcript logging

```

## 8.3 Shared ChatEngine

Do not put generation logic directly in CLI.

Create:

```python

class ChatEngine:

    def **init**(self, checkpoint_path: str, device: str): ...

    def reset(self) -> None: ...

    def append_user_message(self, text: str) -> None: ...

    def generate_stream(self, config: GenerationConfig) -> Iterator[TokenEvent]: ...

    def get_state(self) -> ChatState: ...

    def save_transcript(self, path: str) -> None: ...

```

CLI and web UI both call `ChatEngine`.

## Acceptance Criteria

- Can chat interactively.

- Can run one-shot prompt mode.

- Can reset conversation.

- Stops on assistant end token.

- Transcript can be saved.

- Context length never exceeds model sequence length.

## Tickets

```text

CLI-001: Implement GenerationConfig dataclass

CLI-002: Implement ChatState dataclass

CLI-003: Implement TokenEvent dataclass

CLI-004: Implement ChatEngine

CLI-005: Implement one-shot chat CLI

CLI-006: Implement interactive chat CLI

CLI-007: Implement reset/exit commands

CLI-008: Implement transcript logging

CLI-009: Implement stop-token handling

CLI-010: Implement context cropping

CLI-011: Add ChatEngine tests

```

---

# Phase 9 — Simple Local Web UI

## Goal

Build a local browser UI for testing checkpoints. Keep it intentionally simple.

Current nanochat has a FastAPI `scripts/chat_web.py` that can serve a UI and streaming API and supports multiple GPU workers. This project should start simpler:

```text

single process

single GPU

single loaded checkpoint

local-only bind by default

same ChatEngine as CLI

token streaming to browser

```

## 9.1 Recommended Stack

First implementation:

```text

FastAPI backend

plain HTML/CSS/JavaScript frontend

WebSocket or server-sent events for streaming

```

A Gradio prototype is acceptable as an optional demo, but not as the main implementation. The main implementation should expose the mechanics we want to understand.

## 9.2 Web Command

```bash

python -m scripts.web_chat \

  --checkpoint runs/sft/tiny_20m/[best.pt](http://best.pt) \

  --host 127.0.0.1 \

  --port 8000

```

Default host must be:

```text

127.0.0.1

```

Do not bind to `0.0.0.0` unless explicitly requested.

## 9.3 Must-Have Features

```text

Load checkpoint

Infer tokenizer path from checkpoint metadata

Enter user message

Stream assistant response token-by-token

Stop generation

Reset conversation

Show current context length

Set temperature

Set top_k

Set max_new_tokens

Stop on <|assistant_end|>

Save transcript locally

Show generated tokens/sec

Show peak VRAM if CUDA

```

## 9.4 Nice-to-Have Features

```text

Checkpoint dropdown

Prompt template selector

Raw token/debug view

Token IDs for current conversation

Prefill latency

Decode latency per token

Side-by-side checkpoint comparison

Export conversation as JSONL SFT fixture

```

## 9.5 Explicitly Out of Scope Initially

```text

user accounts

authentication

database

remote hosting

multi-user concurrency

RAG

file uploads

tool execution

public internet exposure

```

This is a testing harness. Do not make it a product.

## 9.6 Backend API Sketch

HTTP:

```text

GET  /

GET  /api/health

GET  /api/config

GET  /api/checkpoints

POST /api/load_checkpoint

POST /api/reset

POST /api/save_transcript

POST /api/tokenize

POST /api/detokenize

```

WebSocket:

```text

WS /ws/generate

```

Client sends:

```json

{

  "message": "Explain backpropagation simply.",

  "temperature": 0.8,

  "top_k": 50,

  "max_new_tokens": 256

}

```

Server streams:

```json

{"type": "token", "text": "Back"}

{"type": "token", "text": "prop"}

{"type": "metrics", "generated_tokens": 128, "tokens_per_sec": 41.2}

{"type": "done"}

```

Error shape:

```json

{"type": "error", "message": "Checkpoint not loaded"}

```

## 9.7 Concurrency Rule

For a single 3090, use a single generation lock.

```text

one generation request at a time

stop button can cancel current generation

new request during active generation receives busy error

```

## Acceptance Criteria

- Web UI launches locally from one command.

- UI loads checkpoint and tokenizer.

- UI streams tokens visibly.

- Stop button interrupts generation cleanly.

- Reset button clears conversation state.

- Generation settings are passed to ChatEngine.

- UI shows context length and tokens/sec.

- Transcript can be exported as JSONL.

- Server defaults to local-only host.

- Browser smoke test works with tiny checkpoint.

## Tickets

```text

WEB-001: Refactor CLI generation into reusable ChatEngine

WEB-002: Add web optional dependency group

WEB-003: Implement FastAPI app factory

WEB-004: Add /api/health endpoint

WEB-005: Add /api/config endpoint

WEB-006: Add /api/checkpoints endpoint

WEB-007: Add /api/load_checkpoint endpoint

WEB-008: Add /api/reset endpoint

WEB-009: Add /api/tokenize endpoint

WEB-010: Add /api/detokenize endpoint

WEB-011: Add WebSocket /ws/generate endpoint

WEB-012: Add streamed TokenEvent serialization

WEB-013: Add generation cancellation support

WEB-014: Add single-GPU request lock

WEB-015: Add local-only host default

WEB-016: Add plain HTML chat page

WEB-017: Add frontend generation controls

WEB-018: Add streamed token rendering

WEB-019: Add stop/reset buttons

WEB-020: Add context-length display

WEB-021: Add tokens/sec and latency display

WEB-022: Add transcript export as JSONL

WEB-023: Add checkpoint dropdown

WEB-024: Add raw token debug panel

WEB-025: Add prompt-template selector

WEB-026: Add browser smoke test

WEB-027: Add FastAPI TestClient test

WEB-028: Add optional Gradio prototype

WEB-029: Add side-by-side checkpoint comparison later

WEB-030: Add UI screenshots to README

```

---

# Phase 10 — Experiment Tracking With Local JSONL and Optional W&B

## Goal

Add experiment tracking across the entire project without hard-coupling the codebase to W&B.

Design rule:

```text

JSONL metrics are always available.

W&B is optional.

No direct wandb.log calls scattered through training code.

```

## 10.1 Tracker Abstraction

Create:

```python

class Tracker:

    def log(self, metrics: dict, step: int | None = None) -> None: ...

    def log_config(self, config: dict) -> None: ...

    def log_artifact(self, path: str, name: str, type: str) -> None: ...

    def finish(self) -> None: ...

```

Implement:

```text

NullTracker

JsonlTracker

WandbTracker

CompositeTracker

```

Default:

```text

CompositeTracker(JsonlTracker, maybe WandbTracker)

```

If W&B is not installed and disabled, everything still works.

## 10.2 Config

```yaml

tracking:

  jsonl:

    enabled: true

    path: metrics/metrics.jsonl

  wandb:

    enabled: false

    project: scratch-llm

    entity: null

    group: null

    name: null

    tags: []

    mode: online        # online | offline | disabled

    dir: runs/wandb

    log_code: false

    log_model_artifacts: false

    log_dataset_artifacts: false

    log_tokenizer_artifacts: true

    log_prompts: false

    log_responses: false

```

Environment override support:

```bash

WANDB_MODE=offline

WANDB_PROJECT=scratch-llm

WANDB_ENTITY=your-entity

WANDB_RUN_GROUP=3090-pretrain

```

## 10.3 What to Track

### Tokenizer Training

```text

tokenizer/vocab_size

tokenizer/max_chars

tokenizer/doc_cap

tokenizer/num_docs

tokenizer/num_chars

tokenizer/train_seconds

tokenizer/bytes_per_token

tokenizer/encode_tokens_per_sec

tokenizer/decode_tokens_per_sec

```

Artifacts:

```text

tokenizer.json

merges.json

vocab.json

special_tokens.json

token_[bytes.pt](http://bytes.pt)

tokenizer_eval.json

```

### Data Preparation

```text

data/train_shards

data/val_shards

data/train_docs

data/val_docs

data/train_chars

data/val_chars

data/tokenized_train_tokens

data/tokenized_val_tokens

data/shard_write_seconds

```

Artifacts:

```text

data_stats.json

tokenized_shard_manifest.json

```

Do not upload huge raw/tokenized datasets by default.

### Base Pretraining

```text

step

train/loss

train/lrm

train/dt

train/tok_per_sec

train/mfu

train/epoch

train/grad_norm

train/peak_memory_mib

val_bpb

min_val_bpb

core_metric

total_training_flops

total_training_time

```

Artifacts:

```text

config.yaml

metrics.jsonl

base_eval.json

base_[samples.md](http://samples.md)

[best.pt](http://best.pt) optional

[last.pt](http://last.pt) optional

```

Checkpoint artifact logging should be opt-in.

### Base Eval

```text

eval/val_bpb

eval/core_metric

eval/core/*

eval/sample_tokens_per_sec

```

Artifacts:

```text

base_eval.json

base_[samples.md](http://samples.md)

```

### SFT

```text

sft/train_loss

sft/val_bpb

sft/chatcore_metric

sft/chatcore_cat

sft/chatcore/ARC-Easy

sft/chatcore/ARC-Challenge

sft/chatcore/MMLU

sft/chatcore/GSM8K

sft/chatcore/HumanEval

sft/tok_per_sec

sft/mfu

sft/peak_memory_mib

```

Artifacts:

```text

sft_eval.json

chat_eval.json

sft_[samples.md](http://samples.md)

```

### Inference Benchmarks

```text

inference/prompt_tokens

inference/generated_tokens

inference/prefill_ms

inference/decode_ms_per_token

inference/tokens_per_sec

inference/peak_memory_mib

inference/kv_cache_enabled

inference/temperature

inference/top_k

inference/mbu

inference/mfu

```

Artifacts:

```text

inference_bench.json

sample_transcripts.jsonl

```

### Local Web UI Sessions

Default: no raw prompt/response logging.

Track only aggregate session metrics when explicitly enabled:

```text

web/session_id

web/turn_count

web/generated_tokens

web/tokens_per_sec

web/avg_decode_ms_per_token

web/peak_memory_mib

```

Raw prompt/response logging must be opt-in:

```yaml

tracking:

  wandb:

    log_prompts: false

    log_responses: false

```

## Acceptance Criteria

- Code runs with W&B uninstalled.

- JSONL metrics are always written.

- W&B can be enabled from config or CLI.

- W&B offline mode works.

- W&B disabled mode works.

- Full resolved config is logged.

- Tokenizer metrics and artifacts are logged.

- Data-prep stats are logged.

- Base training logs nanochat-compatible metrics.

- Eval logs BPB, CORE, and samples.

- SFT logs loss, BPB, and ChatCORE.

- Inference benchmark logs latency and throughput.

- Checkpoint artifact logging is opt-in.

- Prompt/response logging is opt-in.

## Tickets

```text

TRACK-001: Add optional wandb dependency group

TRACK-002: Define Tracker interface

TRACK-003: Implement NullTracker

TRACK-004: Implement JsonlTracker

TRACK-005: Implement WandbTracker

TRACK-006: Implement CompositeTracker

TRACK-007: Add tracking config schema

TRACK-008: Add CLI flags for tracking enable/disable

TRACK-009: Add WANDB_MODE support

TRACK-010: Add offline W&B mode support

TRACK-011: Add disabled W&B mode support

TRACK-012: Log full resolved config

TRACK-013: Log tokenizer training metrics

TRACK-014: Log tokenizer artifacts

TRACK-015: Log data-prep metrics

TRACK-016: Log data manifest artifact

TRACK-017: Log base pretraining metrics

TRACK-018: Log BPB metrics

TRACK-019: Log CORE metrics

TRACK-020: Log training throughput and MFU

TRACK-021: Log GPU memory metrics

TRACK-022: Log base sample outputs as artifact

TRACK-023: Log SFT training metrics

TRACK-024: Log ChatCORE metrics

TRACK-025: Log SFT samples as artifact

TRACK-026: Log inference benchmark metrics

TRACK-027: Add opt-in checkpoint artifact logging

TRACK-028: Add opt-in prompt/response logging

TRACK-029: Add run group/name/tag conventions

TRACK-030: Add run summary writer

TRACK-031: Add W&B resume support

TRACK-032: Add tests proving code works when wandb is absent

TRACK-033: Add tests proving tracker writes JSONL correctly

TRACK-034: Add README section for W&B online/offline/disabled modes

```

---

# Phase 11 — 3090 Scaling Pass

## Goal

Move from toy correctness to practical single-GPU training.

## 11.1 Presets

### `smoke.yaml`

```yaml

seq_len: 128

n_layer: 2

n_head: 2

n_embd: 128

mlp_ratio: 4

vocab_size: 32768

device_batch_size: 4

total_batch_size_tokens: 8192

max_steps: 500

dtype: float32

```

### `tiny_20m_3090.yaml`

```yaml

seq_len: 512

n_layer: 6

n_head: 6

n_embd: 384

mlp_ratio: 4

vocab_size: 32768

device_batch_size: 4

total_batch_size_tokens: 65536

max_steps: 20000

dtype: float16

amp: true

```

### `small_45m_3090.yaml`

```yaml

seq_len: 1024

n_layer: 8

n_head: 8

n_embd: 512

mlp_ratio: 4

vocab_size: 32768

device_batch_size: 1-4

total_batch_size_tokens: 65536-262144

max_steps: 50000+

dtype: float16

amp: true

activation_checkpointing: optional

```

## 11.2 3090 Rules

- Start with fp32 smoke tests.

- Use float16 AMP with GradScaler for longer 3090 runs.

- Reduce `device_batch_size` first when OOM occurs.

- Then reduce `seq_len`.

- Then reduce `n_embd`.

- Then reduce `n_layer`.

- Use gradient accumulation to recover effective batch size.

- Track peak VRAM every eval interval.

- Do not start architecture experiments until `tiny_20m` has sane BPB curves.

## Acceptance Criteria

- Presets run without code edits.

- OOM messages recommend exact config reductions.

- Training logs tokens/sec and peak VRAM.

- Longer runs resume correctly.

- Model samples visibly improve over time.

## Tickets

```text

SCALE-001: Add smoke config

SCALE-002: Add tiny_20m_3090 config

SCALE-003: Add small_45m_3090 config

SCALE-004: Add model size estimator

SCALE-005: Add tokens-per-step calculator

SCALE-006: Add VRAM estimate helper

SCALE-007: Add OOM advice text

SCALE-008: Add run comparison script

SCALE-009: Add throughput benchmark

SCALE-010: Add 3090 tuning README

```

---

# Phase 12 — Performance Optimizations

## Goal

Improve speed and memory after the simple implementation is correct.

## 12.1 Mixed Precision

Add PyTorch AMP:

```text

torch.autocast

torch.amp.GradScaler for fp16

```

Keep it off for the earliest smoke tests. Turn it on for practical 3090 runs.

Acceptance criteria:

```text

AMP smoke run is numerically sane.

No NaNs on standard configs.

Training is faster or uses less memory.

GradScaler state is checkpointed.

```

## 12.2 SDPA Attention Backend

Add PyTorch scaled-dot-product attention as an optional backend:

```yaml

attention_backend: manual | sdpa

```

Acceptance criteria:

```text

manual and SDPA outputs are close on small tensors.

SDPA trains successfully.

Benchmark reports before/after tokens/sec.

```

## 12.3 FlashAttention

Add later:

```yaml

attention_backend: flash

```

Requirements:

```text

clean fallback when unavailable

small-tensor correctness test

long-sequence speed/memory benchmark

```

## 12.4 torch.compile

Add optional:

```yaml

compile: true

```

Keep it off by default while debugging.

## 12.5 Activation Checkpointing

Add:

```yaml

activation_checkpointing: true

```

Use for larger 3090 configs.

## 12.6 KV Cache

Implement for inference:

```text

KVCache class

prefill path

single-token decode path

cache reset

max cache length

```

Acceptance criteria:

```text

cached greedy generation matches naive greedy generation.

cached generation is faster for long outputs.

CLI and web UI can use cache.

```

## Tickets

```text

PERF-001: Add AMP autocast

PERF-002: Add GradScaler

PERF-003: Save/load GradScaler state

PERF-004: Add SDPA attention backend

PERF-005: Add SDPA correctness tests

PERF-006: Add FlashAttention backend

PERF-007: Add FlashAttention fallback

PERF-008: Add torch.compile option

PERF-009: Add activation checkpointing

PERF-010: Add KVCache class

PERF-011: Add cached generation path

PERF-012: Add cached vs naive generation test

PERF-013: Add inference latency benchmark

PERF-014: Add training throughput benchmark

```

---

# Phase 13 — nanochat Architecture Parity Features

## Goal

Add modern architecture pieces one at a time and measure their effect.

## Recommended Order

```text

1. RMSNorm

2. RoPE

3. bias-free Linear

4. ReLU² MLP

5. untied embeddings

6. QK norm

7. GQA

8. sliding-window attention

9. value embeddings

10. residual scalar experiments

11. x0/smear/backout experiments

```

Do not add multiple architecture changes in one ticket unless the ticket is explicitly an integration ticket. If BPB changes, you need to know why.

## Acceptance Criteria

For every feature:

```text

unit test

shape test

tiny overfit test

BPB comparison run

throughput comparison

config flag

README note

```

## Tickets

```text

ARCH-001: Add RMSNorm

ARCH-002: Add RoPE

ARCH-003: Add bias-free Linear option

ARCH-004: Add ReLU² MLP

ARCH-005: Add untied embedding option

ARCH-006: Add QK norm

ARCH-007: Add GQA

ARCH-008: Add sliding-window attention

ARCH-009: Add value embeddings

ARCH-010: Add residual scalar experiments

ARCH-011: Add smear/backout experiments later

ARCH-012: Add architecture ablation report template

```

---

# Phase 14 — Chat Evaluation and Quality Improvements

## Goal

Make chat model quality measurable beyond vibes.

## 14.1 ChatCORE

Implement nanochat-style ChatCORE over:

```text

ARC-Easy

ARC-Challenge

MMLU

GSM8K

HumanEval-style simple coding task

```

Evaluation style:

```text

ARC/MMLU: categorical, score answer-letter logits

GSM8K/HumanEval: generative, sample and evaluate

```

Centered baselines:

```python

baseline_accuracies = {

    "ARC-Easy": 0.25,

    "ARC-Challenge": 0.25,

    "MMLU": 0.25,

    "GSM8K": 0.0,

    "HumanEval": 0.0,

}

```

Formula:

```python

centered_acc = (acc - baseline_acc) / (1.0 - baseline_acc)

chatcore_metric = mean(centered_acc over all tasks)

chatcore_cat = mean(centered_acc over ARC-Easy, ARC-Challenge, MMLU)

```

## 14.2 SFT Prompt Suite

Maintain fixed prompts:

```text

Explain gradient descent in simple terms.

Write a Python function to reverse a string.

Give me three project ideas for learning PyTorch.

What is 17 * 23? Show your work.

Return a JSON object with keys name, age, and city.

```

Save outputs per run.

## 14.3 Format Metrics

Track:

```text

assistant_end stop rate

average response length

empty response rate

format violation rate

JSON validity on JSON prompts

code fence behavior on code prompts

```

## Tickets

```text

CHAT-EVAL-001: Implement ARC-Easy categorical eval

CHAT-EVAL-002: Implement ARC-Challenge categorical eval

CHAT-EVAL-003: Implement MMLU categorical eval

CHAT-EVAL-004: Implement GSM8K generative eval

CHAT-EVAL-005: Implement HumanEval-style code eval

CHAT-EVAL-006: Implement ChatCORE

CHAT-EVAL-007: Implement ChatCORE_cat

CHAT-EVAL-008: Add SFT fixed prompt suite

CHAT-EVAL-009: Add stop-rate metric

CHAT-EVAL-010: Add response-length stats

CHAT-EVAL-011: Add format violation checks

CHAT-EVAL-012: Write chat_eval.json

CHAT-EVAL-013: Log ChatCORE to Tracker

```

---

# Phase 15 — Optional Advanced Chat Extensions

## Goal

Add advanced capabilities only after the base/SFT/eval pipeline is stable.

## 15.1 Preference Tuning

Possible methods:

```text

DPO

IPO

simple pairwise preference loss

```

Data schema:

```json

{

  "prompt": "...",

  "chosen": "...",

  "rejected": "..."

}

```

## 15.2 RL-Style Finetuning

Toy reward domains:

```text

math answer correctness

JSON format validity

unit-test pass/fail for simple code

string-counting tasks

```

## 15.3 Tool Use

Special tokens already exist:

```text

<|python_start|>

<|python_end|>

<|output_start|>

<|output_end|>

```

Do not enable tool execution without sandboxing. Tool execution is a separate safety and security project.

## Tickets

```text

ADV-001: Add preference dataset schema

ADV-002: Implement DPO loss

ADV-003: Implement preference trainer

ADV-004: Add toy reward eval harness

ADV-005: Add tool-call parser

ADV-006: Add sandboxed Python execution later

ADV-007: Add tool-use chat eval later

```

---

# Cross-Cutting Testing Plan

## Tokenizer Tests

```text

ASCII round trip

Unicode round trip

emoji round trip

Korean round trip

code round trip

math/LaTeX round trip

whitespace preservation

special token encode/decode

save/load equivalence

BPE merge correctness

unknown-free encoding

token_bytes special token zeroing

```

## Data Tests

```text

parquet fixture read

deterministic train/val split

token shard save/load

manifest validation

batch shape

x/y shift correctness

no invalid token ids

dataloader state resume

```

## Model Tests

```text

forward shape

mean loss scalar

unreduced loss shape

ignore_index=-1 behavior

causal mask correctness

parameter count sanity

overfit one batch

generation smoke test

save/load equivalence

```

## Training Tests

```text

loss decreases on toy data

checkpoint resume produces valid next step

gradient accumulation approximately matches larger batch

eval does not update weights

seed reproducibility for smoke config

AMP smoke test later

```

## Chat Tests

```text

conversation rendering

system-message merge

assistant-only loss mask

stop token handling

context cropping

multi-turn state

transcript export

```

## Tracking Tests

```text

JSONL tracker writes valid JSON lines

CompositeTracker calls all children

WandbTracker no-ops when disabled

Code works when wandb is absent

Offline mode config is respected

Artifact logging is opt-in

Prompt logging is opt-in

```

## Web Tests

```text

health endpoint returns ok

config endpoint returns sanitized config

checkpoint list endpoint works

websocket generation streams token events

stop cancels generation

single-GPU lock blocks simultaneous generation

server defaults to 127.0.0.1

transcript export works

```

---

# Metrics Output Contract

Every run should produce:

```text

runs/<run_name>/

  config.yaml

  metrics/

    metrics.jsonl

    summary.json

    tokenizer_eval.json

    base_eval.json

    chat_eval.json

    [samples.md](http://samples.md)

  checkpoints/

    [last.pt](http://last.pt)

    [best.pt](http://best.pt)

```

## Tokenizer Metrics

```text

tokenizer/vocab_size

tokenizer/bytes

tokenizer/tokens

tokenizer/bytes_per_token

tokenizer/relative_diff_vs_gpt2

tokenizer/relative_diff_vs_gpt4

tokenizer/roundtrip_pass

tokenizer/encode_tokens_per_sec

tokenizer/decode_tokens_per_sec

```

## Base Training Metrics

```text

step

train/loss

train/lrm

train/dt

train/tok_per_sec

train/mfu

train/epoch

train/grad_norm

train/peak_memory_mib

val_bpb

min_val_bpb

core_metric

core/task_accuracy/*

core/centered/*

total_training_flops

total_training_time

```

## SFT Metrics

```text

step

sft/train_loss

sft/val_bpb

sft/chatcore_metric

sft/chatcore_cat

sft/chatcore/ARC-Easy

sft/chatcore/ARC-Challenge

sft/chatcore/MMLU

sft/chatcore/GSM8K

sft/chatcore/HumanEval

sft/tok_per_sec

sft/mfu

sft/peak_memory_mib

```

## Inference Metrics

```text

inference/prompt_tokens

inference/generated_tokens

inference/prefill_ms

inference/decode_ms_per_token

inference/tokens_per_second

inference/kv_cache_bytes_per_token

inference/peak_memory_mib

inference/mbu

inference/mfu

inference/temperature

inference/top_k

```

---

# Recommended Milestones

## Milestone 1 — Hello Tiny GPT

Deliverable:

```text

byte tokenizer

tiny GPT

tiny local dataset

train loop

sample script

loss decreases

```

## Milestone 2 — Tracking Foundation

Deliverable:

```text

Tracker abstraction

JSONL tracker

optional W&B tracker

resolved config logging

script-wide tracking integration skeleton

```

## Milestone 3 — nanochat Tokenizer/Data Compatibility

Deliverable:

```text

ClimbMix-style parquet downloader

train/val split convention

from-scratch regex byte-BPE

32K vocab

token_[bytes.pt](http://bytes.pt)

tokenizer eval table

```

## Milestone 4 — 20M Base Model on 3090

Deliverable:

```text

tiny_20m config

AdamW baseline

checkpoint/resume

BPB eval

fixed samples

tok/sec, MFU, FLOPs, VRAM logging

```

## Milestone 5 — Base CORE Eval

Deliverable:

```text

CORE task runner

centered CORE calculation

base_eval --eval core,bpb,sample

rough nanochat comparison report

```

## Milestone 6 — First Chat Model

Deliverable:

```text

chat special-token rendering

assistant-only SFT masking

tiny local fixture overfit

SmolTalk/MMLU/GSM8K data path

SFT val BPB

chat checkpoint

```

## Milestone 7 — CLI and Local Web UI

Deliverable:

```text

ChatEngine

terminal chat CLI

FastAPI local web UI

streamed browser chat

stop/reset controls

transcript export

```

## Milestone 8 — ChatCORE

Deliverable:

```text

ARC-Easy

ARC-Challenge

MMLU

GSM8K

HumanEval-style eval

ChatCORE

ChatCORE_cat

```

## Milestone 9 — Make It Fast Enough

Deliverable:

```text

AMP

SDPA

activation checkpointing

inference benchmark

KV cache

optional torch.compile

optional FlashAttention

```

## Milestone 10 — Architecture and Optimizer Experiments

Deliverable:

```text

RMSNorm

RoPE

ReLU²

QK norm

GQA

MuonAdamW

ablation reports

```

---

# First Sprint Recommendation

Do not start with BPE or the web UI. Start with the vertical slice.

## Sprint 1 Scope

```text

FOUND-001 to FOUND-008

SMOKE-001 to SMOKE-008

MODEL-001 to MODEL-014

TRAIN-001 to TRAIN-006

TRACK-002 to TRACK-004

```

## Sprint 1 Exit Criteria

```text

python -m scripts.pretrain --config configs/smoke.yaml

python -m scripts.sample --checkpoint runs/smoke/checkpoints/[last.pt](http://last.pt)

pytest

```

Expected result:

```text

loss decreases

generated text is non-random-ish

metrics.jsonl exists

checkpoint can resume

```

---

# Major Risks and Direct Mitigations

## Risk: Tokenizer becomes too slow

Mitigation:

```text

Start with correctness on small data.

Add pair-count optimization later.

Allow training on capped docs.

Keep tokenizer artifacts stable.

```

## Risk: BPB is implemented incorrectly

Mitigation:

```text

Write hand-computed toy tests.

Use token_bytes, not decoded string lengths.

Zero special-token byte counts.

Ignore masked labels.

```

## Risk: Model appears to train but causal mask is wrong

Mitigation:

```text

Dedicated causal mask unit test.

Gradient or logits dependency test proving future tokens are invisible.

```

## Risk: Web UI forks inference behavior

Mitigation:

```text

ChatEngine owns generation.

CLI and web UI call the same ChatEngine.

No generation logic directly inside frontend/server handlers.

```

## Risk: W&B pollutes the codebase

Mitigation:

```text

Tracker abstraction.

JsonlTracker always available.

WandbTracker optional.

No raw wandb calls in training/eval code.

```

## Risk: 3090 OOM churn wastes time

Mitigation:

```text

Start with 20M config.

Log peak VRAM.

Use gradient accumulation.

Add clear OOM suggestions.

Add AMP only after fp32 smoke correctness.

```

## Risk: Chasing nanochat sophistication too early

Mitigation:

```text

Simple GPT first.

AdamW first.

Manual attention first.

Add nanochat features only after BPB and checkpointing are stable.

```

---

# Source References

These links are included to explain what this plan is aligning with. The implementation should still be written from scratch where noted.

- nanochat repository: [https://github.com/karpathy/nanochat](https://github.com/karpathy/nanochat)

- nanochat README and repo structure: [https://github.com/karpathy/nanochat/blob/master/README.md](https://github.com/karpathy/nanochat/blob/master/README.md)

- nanochat tokenizer source: [https://raw.githubusercontent.com/karpathy/nanochat/master/nanochat/tokenizer.py](https://raw.githubusercontent.com/karpathy/nanochat/master/nanochat/tokenizer.py)

- nanochat tokenizer training: [https://raw.githubusercontent.com/karpathy/nanochat/master/scripts/tok_train.py](https://raw.githubusercontent.com/karpathy/nanochat/master/scripts/tok_train.py)

- nanochat tokenizer eval: [https://raw.githubusercontent.com/karpathy/nanochat/master/scripts/tok_eval.py](https://raw.githubusercontent.com/karpathy/nanochat/master/scripts/tok_eval.py)

- nanochat dataset utilities: [https://raw.githubusercontent.com/karpathy/nanochat/master/nanochat/dataset.py](https://raw.githubusercontent.com/karpathy/nanochat/master/nanochat/dataset.py)

- nanochat base training: [https://raw.githubusercontent.com/karpathy/nanochat/master/scripts/base_train.py](https://raw.githubusercontent.com/karpathy/nanochat/master/scripts/base_train.py)

- nanochat BPB evaluator: [https://raw.githubusercontent.com/karpathy/nanochat/master/nanochat/loss_eval.py](https://raw.githubusercontent.com/karpathy/nanochat/master/nanochat/loss_eval.py)

- nanochat chat eval: [https://raw.githubusercontent.com/karpathy/nanochat/master/scripts/chat_eval.py](https://raw.githubusercontent.com/karpathy/nanochat/master/scripts/chat_eval.py)

- nanochat web server: [https://raw.githubusercontent.com/karpathy/nanochat/master/scripts/chat_web.py](https://raw.githubusercontent.com/karpathy/nanochat/master/scripts/chat_web.py)

- nanochat inference benchmark: [https://raw.githubusercontent.com/karpathy/nanochat/master/scripts/infer_bench.py](https://raw.githubusercontent.com/karpathy/nanochat/master/scripts/infer_bench.py)

- PyTorch AMP docs: [https://docs.pytorch.org/docs/stable/amp.html](https://docs.pytorch.org/docs/stable/amp.html)

- PyTorch SDPA docs: [https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html)

- FastAPI WebSockets docs: [https://fastapi.tiangolo.com/advanced/websockets/](https://fastapi.tiangolo.com/advanced/websockets/)

- W&B `wandb.init` docs: [https://docs.wandb.ai/models/ref/python/functions/init](https://docs.wandb.ai/models/ref/python/functions/init)

- W&B offline docs: [https://docs.wandb.ai/models/ref/cli/wandb-offline](https://docs.wandb.ai/models/ref/cli/wandb-offline)

- Gradio ChatInterface docs: [https://gradio.app/docs/gradio/chatinterface](https://gradio.app/docs/gradio/chatinterface)
