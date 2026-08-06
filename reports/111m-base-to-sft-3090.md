# 111M Base-to-SFT Experiment on an RTX 3090

> Status: complete. Base pretraining, base evaluation, weighted SFT,
> post-SFT regression evaluation, report/W&B audits, and repository quality
> gates all passed.

## Executive summary

This experiment trains the repository's 110,906,112-parameter decoder-only GPT
from random initialization on one RTX 3090, evaluates the selected base
checkpoint, applies a weighted supervised-fine-tuning mixture, and evaluates
the resulting chat checkpoint. Local JSONL is the authoritative telemetry.
W&B receives scalar metrics and evaluation reports; model/checkpoint, dataset,
and tokenizer artifacts are not uploaded.

The base checkpoint reached nanochat-compatible BPB `1.087918` and CORE
`0.044687`. SFT reduced assistant-only held-out BPB to `0.772646`; afterward,
base-text BPB regressed 3.94% to `1.130756` while CORE improved 27.78% to
`0.057101`. Despite the CORE gain, every frozen chat-native and post-SFT
base-style prompt still failed its requested task. The closest official
nanochat d11 point remains better at BPB `1.0096` and CORE `0.0918` under
substantially different training conditions.

The final code/report audit passed all 1,138 repository tests in 203.28 seconds,
including pretraining and SFT integration suites. Ruff lint and the
repository-wide formatting check also passed.

## Reproducible setup

| Item | Value |
| --- | --- |
| Git branch | `codex/dnc-111m-base-to-sft` |
| Experiment config commit | `5919caf` |
| Exact-resume metrics fix | `8561942` |
| Post-SFT data-identity fix | `ba40be4` |
| Post-SFT BPB-protocol fix | `686a561` |
| Base config | `configs/base_111m_3090.yaml` |
| SFT config | `configs/sft_111m_3090.yaml` |
| GPU | NVIDIA GeForce RTX 3090 |
| Model | 12 layers, width 768, 12 heads, learned positions, LayerNorm, GELU |
| Parameter count | 110,906,112 |
| Parameter breakdown | token embedding 25,165,824; positions 786,432; blocks 84,953,088; final norm 768 |
| Vocabulary | 32,768-token regex byte-BPE |
| Context length | 1,024 tokens |
| Precision | FP32 |
| Weight tying | Token embedding and LM head tied |
| Base optimizer batch | 8 device batch x 8 accumulation x 1,024 = 65,536 tokens |
| Base schedule | 30,000 steps; 1,966,080,000 scheduled tokens |
| Base token exposure | 17.7274 processed model tokens/parameter |
| Logged token-normalized epoch at step 30,000 | 0.974593 (not unique-source coverage) |
| Runtime FLOP estimate | 773,849,088/token; 5.0715e13/step; 1.52145e18 scheduled |
| Base optimizer | AdamW beta=(0.9, 0.95), weight decay 0.1, clip 1.0 |
| Base LR schedule | applied warmup 1.5e-6 at update 1 to 3e-4 at update 200; full LR through update 15,001; warmdown from update 15,002; final applied LR 1.5019e-5, saved scheduler endpoint 1.5e-5 |
| SFT schedule | 2,000 steps; 32,768 tokens per optimizer step; peak LR 1e-5 |
| SFT mixture | SmolTalk 1, MMLU 3, GSM8K 4 |

## Data and identity

The pretraining dataset was generated into a separate `data/tokenized_37`
directory and independently reopened with `TokenizedShardReader` after atomic
publication.

| Split | Shards | Documents | Tokens |
| --- | ---: | ---: | ---: |
| Train | 37 | 3,136,512 | 2,017,333,584 |
| Validation | 1 | 84,992 | 54,719,729 |

The tokenized manifest identity is
`sha256:4685d249f48463ce6d4d2b75e9b4cd8adbb23b3eff9c4a6b0247e7340a8ee883`.
Data preparation took 8,937.638 seconds (2h 28m 58s).

The deterministic epoch-0 packing plan contains 3,136,512 documents,
3,721,436 document pieces, and 2,029,510 populated rows; batch alignment adds
two empty padding rows. At 64 packed rows per optimizer step, 30,000 steps
consume 1,920,000 of 2,029,512 rows, or 94.6040% of the first shuffled plan,
without repeating a row. The logged `train/epoch=0.974593` instead divides all
processed model slots by manifest source tokens. It includes masked/padding and
carried-context positions and must not be interpreted as exact unique-source
coverage. Accordingly, this run uses a large subset of the corpus rather than
one complete source-data epoch.

The local SFT caches were independently validated before launch:

| Split/source | Rows | SHA-256 |
| --- | ---: | --- |
| Train SmolTalk | 460,341 | `70dc230ab3e8ddc30a9526cf61e26f4896076618d2191908e110d2d5ab750db6` |
| Train MMLU | 99,842 | `2dd01ef838d0a8da511024f74913a6b95610252457a6a4f8d42d07fe77e75307` |
| Train GSM8K | 7,473 | `bcfa1105293def3b0e8a1691c3621b78ece85a023ff83da959edcbf3e86e1f85` |
| Validation SmolTalk | 24,229 | `1ae4b2feca1db6db50cfa07e0806f3fef665323cf94a1e559a3a437d05ae62be` |
| Validation MMLU | 14,042 | `66abad5e2d8814568fc627498133016a33323e3444796b4d55f9b888f9a4051a` |
| Validation GSM8K | 1,319 | `4b0138a3e94fdb4f99a37fcd10a3a5fcfe141509b690f2e5a932461cb635978a` |

A read-only launch preflight constructed the exact weighted training loader and
finite validation loader from these caches without creating the SFT run path.
Both emitted contiguous `8 x 1,024` batches with assistant-supervised targets;
the inspected training and validation batches contained 7,486 and 6,331
supervised tokens respectively. Restoring the JSON-safe loader state after two
batches reproduced the next input and label tensors bit-for-bit, confirming
the configured mixture can support exact interruption recovery.

The pinned CORE v1 archive was checksum-validated and every task file was
parsed before the training run was left unattended:

| CORE item | Value |
| --- | --- |
| Bundle identity | `sha256:90a7c19e28ee7a52b4f6e1f87658deb9fde7f63deba2379045bdb1fe9ea5d200` |
| Config identity | `sha256:463f6ae0aa127fbd41fa655f8919e136858f4849eef176922a3be96e102ce7e7` |
| Metadata identity | `sha256:84e279cdc0d836c9ea3a6e55a99102f8ee18f72809fe7aa6ac5a1fe630aa492f` |
| Tasks | 22 |
| Total examples | 91,037 |
| Pinned reference tables | 3 |

The implementation evaluates CORE examples sequentially and batches only the
candidate rows within each example. Two prior full 45M evaluations in this
repository took 1,823.833 and 1,849.122 seconds (30.4 and 30.8 minutes). The
measured 111M base pass took 3,262.402 seconds (54m 22s), slightly below the
original 60--75 minute planning allowance. The post-SFT pass then took
3,267.781 seconds (54m 28s), only 5.378 seconds longer.

## Tracking and artifact policy

| Stage | W&B run |
| --- | --- |
| Data preparation | [s3iw1948](https://wandb.ai/bjlkeng/gpt-training-sandbox/runs/s3iw1948) |
| Base pretraining | [ngbuyhxj](https://wandb.ai/bjlkeng/gpt-training-sandbox/runs/ngbuyhxj) |
| Interrupted SFT attempt (steps 10--100) | [61lxq6dl](https://wandb.ai/bjlkeng/gpt-training-sandbox/runs/61lxq6dl) |
| Failed detached launch (no optimizer steps) | [ahxld1qg](https://wandb.ai/bjlkeng/gpt-training-sandbox/runs/ahxld1qg) |
| Final SFT attempt | [4uitqomj](https://wandb.ai/bjlkeng/gpt-training-sandbox/runs/4uitqomj) |

The resolved W&B configuration sets `log_model_artifacts`,
`log_dataset_artifacts`, and `log_tokenizer_artifacts` to `false`. Evaluation
reports are permitted. Application-level tracking has uploaded no model,
dataset, or tokenizer payload. After the data-preparation run finished, W&B
materialized one internal `wandb-events` artifact containing a single
80,257-byte `events_0000.parquet` telemetry file; it contains no project data
shard, tokenizer, or checkpoint. During training, the base run's remote
artifact list remained empty through the audited step 23,020 and through its
terminal training row; only the explicitly permitted evaluation reports were
added afterward.

The post-training evaluation handoff was preflighted against that checkpoint:
`eval_base` resolved the persisted tracking state back to W&B run `ngbuyhxj`.
The installed W&B 0.28.1 client receives that ID with `resume="must"`; W&B's
[current resume contract](https://docs.wandb.ai/models/runs/resuming) says an
existing ID resumes the same run from its last step. Standalone evaluation
metrics intentionally carry no explicit historical step, so they advance from
the remote cursor rather than being backdated to the selected checkpoint step
if `best.pt` predates the terminal checkpoint.
Its JSON/Markdown reports use the distinct `evaluation` artifact type, which is
permitted by policy; it does not fork a run or enable any of the three blocked
artifact classes.

Fresh SFT intentionally receives no W&B resume state and therefore creates a
separate run under group `111m-3090-base-to-sft`. The selected base checkpoint
is a pretraining-stage checkpoint whose model and tokenizer configurations
match the SFT config exactly. The SFT output path was absent before launch, and
its three artifact gates are also false. SFT checkpoints persisted their new
run ID, allowing post-SFT evaluation to resume that run instead of the base
run.

An early telemetry audit found exactly one local config record, monotonically
non-decreasing metric steps, and unique artifact event IDs. W&B's merged
step-500 history row exactly matched local training loss and both BPB values.
The local JSONL correctly contains a model-artifact registration for recovery,
while the remote run contains zero uploaded artifacts.

An independent final training-log audit parsed all 3,181 complete
newline-delimited records: every 10-step training row from 10 through 30,000,
all 60 validation rows from 500 through 30,000, finite numeric metrics, and 120
unique checkpoint event IDs with no duplicates. Metric steps were monotonically
non-decreasing
and the file ended with a complete newline. W&B contained the exact step-30,000
validation values and zero uploaded artifacts.

Standalone base evaluation then uploaded exactly three permitted, small
`evaluation` reports (`base_eval`, `base_samples`, and `core_comparison`) to
the same run. The finished run contains no model/checkpoint, dataset, or
tokenizer artifact. The report uploads do not change any of the three blocked
configuration gates.

A separate W&B run-file audit at step 13,190 also found no indirect binary
uploads. The live base run contained only its small config, output log,
requirements, metadata, and summary files; the largest was the 6,460-byte
output log. The completed data-preparation run contained only its small config,
summary, requirements, metadata, and the zero-byte manifest pointer associated
with W&B's internal event artifact. Neither remote run contained a checkpoint,
tokenizer file, or data shard.

## Base pretraining

### Initialization caveat

`simple_gpt` does not install a custom GPT-style parameter initializer. A fresh
seed-1337 reconstruction measured standard deviations of 1.00033 for token
embeddings and 0.99959 for position embeddings, versus about 0.02084 for the
first attention and MLP linear weights. Because the LM head is tied to the
unit-scale token embedding, initial logits and cross-entropy are unusually
large. The early trajectory was:

| Step | Training loss (nats/token) |
| ---: | ---: |
| 10 | 428.0405 |
| 100 | 37.7255 |
| 200 | 26.6165 |
| 300 | 17.6466 |
| 400 | 12.7246 |
| 500 | 9.1962 |
| 700 | 7.9953 |
| 1,000 | 7.1914 |

The loss recovered rapidly without a sustained reversal and the optimizer
remained finite, so this was not a run failure. It does consume part of the
token budget learning out of a nonstandard initialization and is a material
difference from nanochat; final metric gaps must not be attributed only to data
or parameter count.

A pre-clipping gradient audit through step 30,000 found no non-finite values.
Across the 2,101 logged rows from step 9,000 onward, the median norm was
`0.505869`, p95 was `0.859413`, and p99 was `1.157787`. Six rows exceeded
`2.0`: `3.182` at
step 10,150, `3.697` at step 10,850, `2.503` at step 11,110, and `2.820` at
step 12,440, plus `4.797` at step 16,840 and `2.924` at step 22,710. The
configured `1.0` clip handled each; the immediately following norms were
`0.948`, `0.838`, `0.910`, `0.607`, `1.046`, and `0.541`, respectively, and
neither loss nor throughput showed a sustained disturbance. The step-16,840
outlier coincided with an unusually easy
loss-`0.415052` batch; loss returned to `3.976934` on the next logged row and
gradient norm returned below `1.0` one row later.
The step-22,710 outlier likewise coincided with an easy loss-`2.293702` batch;
the next logged row returned to loss `3.827365` and gradient norm `0.540726`.

Windowed loss shows the underlying trend more clearly than individual rows.
Mean training loss fell monotonically from `4.6864` over steps 8,010--8,500,
to `4.5603` over 8,510--9,000, `4.4913` over 9,010--9,500, and `4.4423`
over 9,510--10,000. Subsequent complete windows continued downward: `4.3569`
over 10,010--10,500, `4.2856` over 10,510--11,000, `4.2709` over
11,010--11,500, and `4.2339` over 11,510--12,000.
The next complete window, 12,010--12,500, declined again to `4.1750`.
Steps 12,510--13,000 declined further to `4.1391`.
Steps 13,010--13,500 continued the monotonic decline to `4.1185`.
Steps 13,510--14,000 declined again to `4.1153`.
Steps 14,010--14,500 declined once more to `4.0750`.
Steps 14,510--15,000 continued downward to `4.0307`.
The first warmdown window, steps 15,010--15,500, declined again to `3.9962`.
Steps 15,510--16,000 declined further to `3.9835`.
Steps 16,010--16,500 declined again to `3.9503`.
Steps 16,510--17,000 had arithmetic mean `3.8485`, median `3.9343`, and mean
`3.9185` in a sensitivity calculation excluding only the isolated
loss-`0.4151` batch; no training metric was actually filtered.
Steps 17,010--17,500 had mean `3.8898` and median `3.8684`, improving on that
prior outlier-excluded sensitivity mean while remaining, appropriately, above
the prior window's outlier-depressed raw mean.
Steps 17,510--18,000 had mean `3.8866` and median `3.8696`: a slight mean
improvement with an effectively flat median.
Steps 18,010--18,500 improved more clearly to mean `3.8479` and median
`3.8342`.
Steps 18,510--19,000 continued the mean decline to `3.8361`; median increased
slightly to `3.8378`.
Steps 19,010--19,500 improved both mean and median to `3.7811` and `3.7956`.
Steps 19,510--20,000 continued that decline to mean `3.7722` and median
`3.7813`.
Steps 20,010--20,500 improved both again to mean `3.7582` and median `3.7627`.
Steps 20,510--21,000 continued that decline to mean `3.7420` and median
`3.7440`.
Steps 21,010--21,500 had mean `3.7411`, a marginal improvement, while median
rose slightly to `3.7469`.
Steps 21,510--22,000 improved both mean and median to `3.7209` and `3.7371`.
Steps 22,010--22,500 had mean `3.7411`, an increase of `0.0203`, while median
improved slightly to `3.7361`.
Steps 22,510--23,000 then improved both mean and median to `3.7016` and
`3.7041`, despite the isolated easy batch at step 22,710.
Steps 23,010--23,500 improved slightly again to mean `3.6989` and median
`3.7017`.
Steps 23,510--24,000 improved both to mean `3.6883` and median `3.6899`.
Steps 24,010--24,500 held the mean essentially flat but slightly lower at
`3.6878`, while median improved to `3.6767`.
Steps 24,510--25,000 then improved strongly to mean `3.6481` and median
`3.6402`.
Steps 25,010--25,500 held mean essentially flat but slightly lower at `3.6475`,
while median was also effectively flat at `3.6409`.
Steps 25,510--26,000 improved slightly again to mean `3.6447` and median
`3.6330`.
Steps 26,010--26,500 improved mean to `3.6294`, while median rose modestly to
`3.6385`.
Steps 26,510--27,000 rose modestly to mean `3.6431` and median `3.6454`.
Steps 27,010--27,500 held mean nearly flat at `3.6450`, while median improved
to `3.6367`.
Steps 27,510--28,000 improved both mean and median clearly to `3.6072` and
`3.6066`.
Steps 28,010--28,500 rose slightly to mean `3.6148` and median `3.6075`.
Steps 28,510--29,000 were nearly flat, with mean `3.6165` and median `3.6133`.
Steps 29,010--29,500 were also nearly flat, with mean `3.6209` and median
`3.6378`.
Steps 29,510--30,000 remained nearly flat in mean at `3.6232`, while median
improved to `3.6175`.

### Measured performance and validation gates

| Metric | Value |
| --- | ---: |
| Mature optimizer throughput, steps 5,010--13,610 | mean 16,610 tokens/s; p5--p95 16,594--16,626 |
| Mature optimizer step time | mean 3.9455s; p5--p95 3.9417--3.9493s |
| MFU against 35.58 TFLOP/s FP32 peak | mean 36.126%; p5--p95 36.092--36.162% |
| Conservative preflight memory estimate | 21,120 MiB |
| Peak allocated CUDA memory | 14,073 MiB |
| Observed full-load board memory | 15,811--15,812 MiB |
| Observed full-load power / temperature | approximately 345--350 W / 71--73 C |
| Step-500 training loss | 9.19617 nats/token |
| Step-500 nanochat-compatible BPB | 2.842049 |
| Step-500 full-document BPB | 2.949778 |
| Step-500 dual-protocol validation time | approximately 21m 51s |
| Step-1,000 training loss | 7.19144 nats/token |
| Step-1,000 nanochat-compatible BPB | 2.176092 |
| Step-1,000 full-document BPB | 2.240685 |
| Step-1,000 gate-to-durable-checkpoint time | approximately 21m 52s |
| Step-1,500 training loss | 6.91362 nats/token |
| Step-1,500 nanochat-compatible BPB | 2.110908 |
| Step-1,500 full-document BPB | 2.169972 |
| Step-1,500 gate-to-durable-checkpoint time | approximately 21m 54s |
| Step-2,000 training loss | 6.68842 nats/token |
| Step-2,000 nanochat-compatible BPB | 2.053895 |
| Step-2,000 full-document BPB | 2.112036 |
| Step-2,000 gate-to-durable-checkpoint time | approximately 21m 54s |
| Step-2,500 training loss | 6.64828 nats/token |
| Step-2,500 nanochat-compatible BPB | 1.992870 |
| Step-2,500 full-document BPB | 2.044620 |
| Step-2,500 gate-to-durable-checkpoint time | approximately 21m 54s |
| Step-3,000 training loss | 6.39657 nats/token |
| Step-3,000 nanochat-compatible BPB | 1.957112 |
| Step-3,000 full-document BPB | 2.009505 |
| Step-3,000 gate-to-durable-checkpoint time | approximately 22m 01s |
| Step-3,500 training loss | 5.91360 nats/token |
| Step-3,500 nanochat-compatible BPB | 1.872970 |
| Step-3,500 full-document BPB | 1.922690 |
| Step-3,500 gate-to-durable-checkpoint time | approximately 22m 02s |
| Step-4,000 training loss | 5.99299 nats/token |
| Step-4,000 nanochat-compatible BPB | 1.810129 |
| Step-4,000 full-document BPB | 1.856630 |
| Step-4,000 gate-to-durable-checkpoint time | approximately 22m 15s |
| Step-4,500 training loss | 5.70133 nats/token |
| Step-4,500 nanochat-compatible BPB | 1.761399 |
| Step-4,500 full-document BPB | 1.806538 |
| Step-4,500 gate-to-durable-best time | approximately 21m 56s |
| Step-5,000 training loss | 5.73017 nats/token |
| Step-5,000 nanochat-compatible BPB | 1.712214 |
| Step-5,000 full-document BPB | 1.756513 |
| Step-5,000 gate-to-durable-checkpoint time | approximately 22m 23s |
| Step-5,500 training loss | 5.34415 nats/token |
| Step-5,500 nanochat-compatible BPB | 1.669486 |
| Step-5,500 full-document BPB | 1.717575 |
| Step-5,500 gate-to-durable-best time | approximately 21m 56s |
| Step-6,000 training loss | 5.19748 nats/token |
| Step-6,000 nanochat-compatible BPB | 1.620833 |
| Step-6,000 full-document BPB | 1.666910 |
| Step-6,000 gate-to-durable-checkpoint time | approximately 22m 14s |
| Step-6,500 training loss | 5.19373 nats/token |
| Step-6,500 nanochat-compatible BPB | 1.583498 |
| Step-6,500 full-document BPB | 1.625537 |
| Step-6,500 gate-to-durable-best time | approximately 21m 56s |
| Step-7,000 training loss | 5.06497 nats/token |
| Step-7,000 nanochat-compatible BPB | 1.533449 |
| Step-7,000 full-document BPB | 1.573916 |
| Step-7,000 gate-to-durable-checkpoint time | approximately 22m 18s |
| Step-7,500 training loss | 4.82470 nats/token |
| Step-7,500 nanochat-compatible BPB | 1.483938 |
| Step-7,500 full-document BPB | 1.528027 |
| Step-7,500 gate-to-durable-best time | approximately 21m 57s |
| Step-8,000 training loss | 4.53106 nats/token |
| Step-8,000 nanochat-compatible BPB | 1.433670 |
| Step-8,000 full-document BPB | 1.473420 |
| Step-8,000 gate-to-durable-checkpoint time | approximately 22m 17s |
| Step-8,500 training loss | 4.67865 nats/token |
| Step-8,500 nanochat-compatible BPB | 1.401768 |
| Step-8,500 full-document BPB | 1.440529 |
| Step-8,500 gate-to-durable-best time | approximately 21m 58s |
| Step-9,000 training loss | 4.62877 nats/token |
| Step-9,000 nanochat-compatible BPB | 1.376264 |
| Step-9,000 full-document BPB | 1.415232 |
| Step-9,000 gate-to-durable-checkpoint time | approximately 22m 14s |
| Step-9,500 training loss | 4.53294 nats/token |
| Step-9,500 nanochat-compatible BPB | 1.352190 |
| Step-9,500 full-document BPB | 1.390306 |
| Step-9,500 gate-to-durable-best time | approximately 21m 57s |
| Step-10,000 training loss | 4.32392 nats/token |
| Step-10,000 nanochat-compatible BPB | 1.332459 |
| Step-10,000 full-document BPB | 1.373700 |
| Step-10,000 gate-to-durable-checkpoint time | approximately 22m 18s |
| Step-10,500 training loss | 4.34647 nats/token |
| Step-10,500 nanochat-compatible BPB | 1.315947 |
| Step-10,500 full-document BPB | 1.356959 |
| Step-10,500 gate-to-durable-best time | approximately 21m 57s |
| Step-11,000 training loss | 4.14707 nats/token |
| Step-11,000 nanochat-compatible BPB | 1.300720 |
| Step-11,000 full-document BPB | 1.338830 |
| Step-11,000 gate-to-durable-checkpoint time | approximately 22m 17s |
| Step-11,500 training loss | 4.31763 nats/token |
| Step-11,500 nanochat-compatible BPB | 1.285029 |
| Step-11,500 full-document BPB | 1.322168 |
| Step-11,500 gate-to-durable-best time | approximately 21m 57s |
| Step-12,000 training loss | 4.01776 nats/token |
| Step-12,000 nanochat-compatible BPB | 1.272637 |
| Step-12,000 full-document BPB | 1.309881 |
| Step-12,000 gate-to-durable-checkpoint time | approximately 22m 21s |
| Step-12,500 training loss | 4.35628 nats/token |
| Step-12,500 nanochat-compatible BPB | 1.259785 |
| Step-12,500 full-document BPB | 1.296730 |
| Step-12,500 gate-to-durable-best time | approximately 21m 57s |
| Step-13,000 training loss | 4.14341 nats/token |
| Step-13,000 nanochat-compatible BPB | 1.251262 |
| Step-13,000 full-document BPB | 1.290878 |
| Step-13,000 gate-to-durable-checkpoint time | approximately 22m 20s |
| Step-13,500 training loss | 4.26398 nats/token |
| Step-13,500 nanochat-compatible BPB | 1.240769 |
| Step-13,500 full-document BPB | 1.277918 |
| Step-13,500 gate-to-durable-best time | approximately 21m 57s |
| Step-14,000 training loss | 4.01927 nats/token |
| Step-14,000 validation entry | 2026-08-04 15:39:42 EDT |
| Step-14,000 nanochat-compatible BPB | 1.232974 |
| Step-14,000 full-document BPB | 1.269248 |
| Step-14,000 observed gate-to-durable-last time | approximately 21m 22s |
| Step-14,500 training loss | 4.04672 nats/token |
| Step-14,500 validation entry | 2026-08-04 16:34:52 EDT |
| Step-14,500 nanochat-compatible BPB | 1.222491 |
| Step-14,500 full-document BPB | 1.259026 |
| Step-14,500 observed gate-to-durable-best time | approximately 21m 13s |
| Step-15,000 training loss | 4.02107 nats/token |
| Step-15,000 validation entry | 2026-08-04 17:29:14 EDT |
| Step-15,000 nanochat-compatible BPB | 1.213596 |
| Step-15,000 full-document BPB | 1.249528 |
| Step-15,000 observed gate-to-durable-last time | approximately 22m 10s |
| Step-15,500 training loss | 3.95515 nats/token |
| Step-15,500 validation entry | 2026-08-04 18:24:44 EDT |
| Step-15,500 nanochat-compatible BPB | 1.207040 |
| Step-15,500 full-document BPB | 1.243338 |
| Step-15,500 observed gate-to-durable-best time | approximately 21m 40s |
| Step-16,000 training loss | 4.41687 nats/token |
| Step-16,000 validation entry | 2026-08-04 19:19:40 EDT |
| Step-16,000 nanochat-compatible BPB | 1.199832 |
| Step-16,000 full-document BPB | 1.236416 |
| Step-16,000 observed gate-to-durable-last time | approximately 22m 08s |
| Step-16,500 training loss | 3.87120 nats/token |
| Step-16,500 validation entry | 2026-08-04 20:14:54 EDT |
| Step-16,500 nanochat-compatible BPB | 1.191039 |
| Step-16,500 full-document BPB | 1.227025 |
| Step-16,500 observed gate-to-durable-best time | approximately 21m 54s |
| Step-17,000 training loss | 3.97400 nats/token |
| Step-17,000 validation entry | 2026-08-04 21:09:53 EDT |
| Step-17,000 nanochat-compatible BPB | 1.183895 |
| Step-17,000 full-document BPB | 1.221383 |
| Step-17,000 observed gate-to-durable-last time | approximately 22m 17s |
| Step-17,500 training loss | 3.92962 nats/token |
| Step-17,500 validation entry | 2026-08-04 22:05:14 EDT |
| Step-17,500 nanochat-compatible BPB | 1.178044 |
| Step-17,500 full-document BPB | 1.213249 |
| Step-17,500 observed gate-to-durable-best time | approximately 21m 57s |
| Step-18,000 training loss | 4.10793 nats/token |
| Step-18,000 validation entry | 2026-08-04 23:00:15 EDT |
| Step-18,000 nanochat-compatible BPB | 1.171197 |
| Step-18,000 full-document BPB | 1.207094 |
| Step-18,000 observed gate-to-durable-last time | approximately 22m 18s |
| Step-18,500 training loss | 3.74510 nats/token |
| Step-18,500 validation entry | 2026-08-04 23:55:37 EDT |
| Step-18,500 nanochat-compatible BPB | 1.165518 |
| Step-18,500 full-document BPB | 1.201678 |
| Step-18,500 observed gate-to-durable-best time | approximately 21m 57s |
| Step-19,000 training loss | 3.85173 nats/token |
| Step-19,000 validation entry | 2026-08-05 00:50:40 EDT |
| Step-19,000 nanochat-compatible BPB | 1.157966 |
| Step-19,000 full-document BPB | 1.193100 |
| Step-19,000 observed gate-to-durable-last time | approximately 22m 22s |
| Step-19,500 training loss | 3.80207 nats/token |
| Step-19,500 validation entry | 2026-08-05 01:46:15 EDT |
| Step-19,500 nanochat-compatible BPB | 1.153893 |
| Step-19,500 full-document BPB | 1.189530 |
| Step-19,500 observed gate-to-durable-best time | approximately 21m 55s |
| Step-20,000 training loss | 4.20172 nats/token |
| Step-20,000 validation entry | 2026-08-05 02:41:16 EDT |
| Step-20,000 nanochat-compatible BPB | 1.151338 |
| Step-20,000 full-document BPB | 1.186410 |
| Step-20,000 observed gate-to-durable-last time | approximately 22m 18s |
| Step-20,500 training loss | 3.76948 nats/token |
| Step-20,500 validation entry | 2026-08-05 03:36:41 EDT |
| Step-20,500 nanochat-compatible BPB | 1.146989 |
| Step-20,500 full-document BPB | 1.182177 |
| Step-20,500 observed gate-to-durable-best time | approximately 21m 57s |
| Step-21,000 training loss | 3.96840 nats/token |
| Step-21,000 validation entry | 2026-08-05 04:31:47 EDT |
| Step-21,000 nanochat-compatible BPB | 1.139862 |
| Step-21,000 full-document BPB | 1.177376 |
| Step-21,000 observed gate-to-durable-last time | approximately 22m 21s |
| Step-21,500 training loss | 3.66824 nats/token |
| Step-21,500 validation entry | 2026-08-05 05:27:16 EDT |
| Step-21,500 nanochat-compatible BPB | 1.136773 |
| Step-21,500 full-document BPB | 1.172265 |
| Step-21,500 observed gate-to-durable-best time | approximately 21m 57s |
| Step-22,000 training loss | 3.68250 nats/token |
| Step-22,000 validation entry | 2026-08-05 06:22:18 EDT |
| Step-22,000 nanochat-compatible BPB | 1.131092 |
| Step-22,000 full-document BPB | 1.165991 |
| Step-22,000 observed gate-to-durable-last time | approximately 22m 18s |
| Step-22,500 training loss | 3.69650 nats/token |
| Step-22,500 validation entry | 2026-08-05 07:17:42 EDT |
| Step-22,500 nanochat-compatible BPB | 1.129480 |
| Step-22,500 full-document BPB | 1.166171 |
| Full-document minimum through step 22,500 | 1.165991 at step 22,000 |
| Step-22,500 observed gate-to-durable-best time | approximately 21m 56s |
| Step-23,000 training loss | 3.53974 nats/token |
| Step-23,000 validation entry | 2026-08-05 08:12:49 EDT |
| Step-23,000 nanochat-compatible BPB | 1.124950 |
| Step-23,000 full-document BPB | 1.159525 |
| Step-23,000 observed gate-to-durable-last time | approximately 22m 14s |
| Step-23,500 training loss | 3.56805 nats/token |
| Step-23,500 validation entry | 2026-08-05 09:08:10 EDT |
| Step-23,500 nanochat-compatible BPB | 1.120800 |
| Step-23,500 full-document BPB | 1.154783 |
| Step-23,500 observed gate-to-durable-best time | approximately 21m 55s |
| Step-24,000 training loss | 3.57112 nats/token |
| Step-24,000 validation entry | 2026-08-05 10:03:17 EDT |
| Step-24,000 nanochat-compatible BPB | 1.117621 |
| Step-24,000 full-document BPB | 1.153179 |
| Step-24,000 observed gate-to-durable-last time | approximately 22m 14s |
| Step-24,500 training loss | 3.85112 nats/token |
| Step-24,500 validation entry | 2026-08-05 10:58:38 EDT |
| Step-24,500 nanochat-compatible BPB | 1.113883 |
| Step-24,500 full-document BPB | 1.147832 |
| Step-24,500 observed gate-to-durable-best time | approximately 21m 58s |
| Step-25,000 training loss | 3.55041 nats/token |
| Step-25,000 validation entry | 2026-08-05 11:53:41 EDT |
| Step-25,000 nanochat-compatible BPB | 1.111288 |
| Step-25,000 full-document BPB | 1.148621 |
| Full-document minimum through step 25,000 | 1.147832 at step 24,500 |
| Step-25,000 observed gate-to-durable-last time | approximately 22m 16s |
| Step-25,500 training loss | 3.70767 nats/token |
| Step-25,500 validation entry | 2026-08-05 12:49:01 EDT |
| Step-25,500 nanochat-compatible BPB | 1.107898 |
| Step-25,500 full-document BPB | 1.142000 |
| Step-25,500 observed gate-to-durable-best time | approximately 21m 57s |
| Step-26,000 training loss | 3.63198 nats/token |
| Step-26,000 validation entry | 2026-08-05 13:44:05 EDT |
| Step-26,000 nanochat-compatible BPB | 1.105340 |
| Step-26,000 full-document BPB | 1.139533 |
| Step-26,000 observed gate-to-durable-last time | approximately 22m 23s |
| Step-26,500 training loss | 3.63583 nats/token |
| Step-26,500 validation entry | 2026-08-05 14:39:46 EDT |
| Step-26,500 nanochat-compatible BPB | 1.102590 |
| Step-26,500 full-document BPB | 1.136765 |
| Step-26,500 observed gate-to-durable-best time | approximately 21m 48s |
| Step-27,000 training loss | 3.47720 nats/token |
| Step-27,000 validation entry | 2026-08-05 15:34:43 EDT |
| Step-27,000 nanochat-compatible BPB | 1.100628 |
| Step-27,000 full-document BPB | 1.134693 |
| Step-27,000 observed gate-to-durable-last time | approximately 22m 12s |
| Step-27,500 training loss | 3.68657 nats/token |
| Step-27,500 validation entry | 2026-08-05 16:30:04 EDT |
| Step-27,500 nanochat-compatible BPB | 1.097482 |
| Step-27,500 full-document BPB | 1.131346 |
| Step-27,500 observed gate-to-durable-best time | approximately 21m 51s |
| Step-28,000 training loss | 3.57041 nats/token |
| Step-28,000 validation entry | 2026-08-05 17:25:03 EDT |
| Step-28,000 nanochat-compatible BPB | 1.095372 |
| Step-28,000 full-document BPB | 1.129247 |
| Step-28,000 observed gate-to-durable-last time | approximately 22m 17s |
| Step-28,500 training loss | 3.55109 nats/token |
| Step-28,500 validation entry | 2026-08-05 18:20:25 EDT |
| Step-28,500 nanochat-compatible BPB | 1.093877 |
| Step-28,500 full-document BPB | 1.127747 |
| Step-28,500 observed gate-to-durable-best time | approximately 21m 57s |
| Step-29,000 training loss | 3.65651 nats/token |
| Step-29,000 validation entry | 2026-08-05 19:15:38 EDT |
| Step-29,000 nanochat-compatible BPB | 1.091155 |
| Step-29,000 full-document BPB | 1.124971 |
| Step-29,000 observed gate-to-durable-last time | approximately 22m 12s |
| Step-29,500 training loss | 3.82417 nats/token |
| Step-29,500 validation entry | 2026-08-05 20:10:58 EDT |
| Step-29,500 nanochat-compatible BPB | 1.089663 |
| Step-29,500 full-document BPB | 1.123466 |
| Step-29,500 observed gate-to-durable-best time | approximately 21m 56s |
| Step-30,000 training loss | 3.60802 nats/token |
| Step-30,000 validation entry | 2026-08-05 21:06:04 EDT |
| Step-30,000 nanochat-compatible BPB | 1.087918 |
| Step-30,000 full-document BPB | 1.121593 |
| Step-30,000 observed gate-to-final-durable-last time | approximately 22m 32s |
| Free local storage after first checkpoint | 993 GB |
| Free local storage at step 18,010 | 970 GB |
| Retained base checkpoints at step 18,010 | 20 files, approximately 24.8 GiB |
| Free local storage at step 22,530 | 965 GB |
| Retained base checkpoints at step 22,530 | 24 files, approximately 29.8 GiB |
| Free local storage at step 23,530 | 964 GB |
| Retained base checkpoints at step 23,530 | 25 files, 33,275,058,575 bytes (approximately 31.0 GiB) |
| Free local storage at step 24,020 | 963 GB |
| Retained base checkpoints at step 24,020 | 26 files, 34,606,060,918 bytes (approximately 32.2 GiB) |
| Free local storage at step 24,520 | 963 GB |
| Retained base checkpoints at step 24,520 | 26 files, 34,606,060,918 bytes (approximately 32.2 GiB) |
| Free local storage at step 25,020 | 962 GB |
| Retained base checkpoints at step 25,020 | 27 files, 35,937,063,261 bytes (approximately 33.5 GiB) |
| Free local storage at step 25,520 | 962 GB |
| Retained base checkpoints at step 25,520 | 27 files, 35,937,063,261 bytes (approximately 33.5 GiB) |
| Free local storage at step 26,040 | 960 GB |
| Retained base checkpoints at step 26,040 | 28 files, 37,268,065,604 bytes (approximately 34.7 GiB) |
| Free local storage at step 26,520 | 960 GB |
| Retained base checkpoints at step 26,520 | 28 files, 37,268,065,604 bytes (approximately 34.7 GiB) |
| Free local storage at step 27,020 | 959 GB |
| Retained base checkpoints at step 27,020 | 29 files, 38,599,067,947 bytes (approximately 35.9 GiB) |
| Free local storage at step 27,520 | 959 GB |
| Retained base checkpoints at step 27,520 | 29 files, 38,599,067,947 bytes (approximately 35.9 GiB) |
| Free local storage at step 28,020 | 958 GB |
| Retained base checkpoints at step 28,020 | 30 files, 39,930,070,290 bytes (approximately 37.2 GiB) |
| Free local storage at step 28,520 | 958 GB |
| Retained base checkpoints at step 28,520 | 30 files, 39,930,070,290 bytes (approximately 37.2 GiB) |
| Free local storage at step 29,020 | 957 GB |
| Retained base checkpoints at step 29,020 | 31 files, 41,261,072,633 bytes (approximately 38.4 GiB) |
| Free local storage at step 29,520 | 957 GB |
| Retained base checkpoints at step 29,520 | 31 files, 41,261,072,633 bytes (approximately 38.4 GiB) |
| Free local storage after step 30,000 publication | 955 GB |
| Final retained base checkpoints | 32 files, 42,592,074,976 bytes (approximately 39.7 GiB) |
| Final retained SFT checkpoints | 10 files, 13,310,073,222 bytes (approximately 12.4 GiB) |

The compatibility protocol scores a fixed 1,048,576-token budget. The
full-document protocol additionally scores every token in the 54,719,729-token
validation split. This exact protocol explains why periodic validation is a
material portion of total wall-clock time.

Across all 60 gates from step 500 through step 30,000, compatibility BPB
decreased strictly at every gate, falling from `2.842049` to `1.087918`, an
absolute improvement of `1.754131` (61.72%). Full-document BPB set another new
minimum of `1.121593`. Relative to step 500, full-document BPB improved by
`1.828185` (61.98%). The latest six compatibility improvements remained
positive at `0.003146`, `0.002110`, `0.001495`, `0.002722`, `0.001492`, and
`0.001745`.
The diminishing size is expected late in training; the terminal checkpoint
and standalone evaluation reproduce the final step-30,000 values exactly.

The LR transition was audited before warmdown. Telemetry shows multipliers
`0.05`, `0.95`, and `1.0` at one-based updates 10, 190, and 200. The
step-15,000 checkpoint independently stores scheduler step 15,000 and matching
optimizer/scheduler rates of `3e-4`. Because scheduler positions are zero-based
while reported optimizer updates are one-based, update 15,001 still applies
the full rate and update 15,002 is the first reduced-rate update. The first
post-gate telemetry row at step 15,010 reports multiplier `0.99943`, matching
that warmdown trajectory. Every logged multiplier from step 15,000 through
30,000 matches the configured linear schedule's closed-form value within
`1.11e-16`. Likewise,
update 30,000 applies `1.5019e-5`, after which the saved scheduler endpoint is
exactly `1.5e-5`. This is the implementation's tested schedule, not a mid-run
change.

Through step 30,000, the trainer's cumulative optimizer timer recorded
118,307.260 seconds (32h 51m 47s). The 60 recorded validation-to-durable-
checkpoint intervals sum to approximately 22h 03m 35s, for about 54h 55m 22s of
combined optimizer plus gate time. Validation and checkpoint publication
therefore account for approximately 40.16% of that measured subtotal. The gate
sum is approximate because individual intervals are recorded to whole seconds;
it excludes the user-requested pause and other out-of-process downtime.

The resource estimator deliberately includes conservative activation,
workspace, and allocator headroom. Its estimate exceeded observed board use by
about 5.2 GiB, providing enough margin for batch 8 without activation
checkpointing while avoiding a false claim that the estimate was measured
usage.

At the first measured gate, 500 optimizer steps took approximately 1,967
seconds and dual-protocol validation took approximately 1,311 seconds. Thus,
validation accounted for about 40.0% of gate-to-gate wall time. Extrapolating
those measurements across 30,000 steps gives 32.78 hours of optimizer work and
21.85 hours of periodic validation, or 54.63 hours before checkpoint I/O and
final-evaluation overhead. This projection is retained separately from the
eventual measured duration. The recovered step-1,000 gate took approximately
the same 1,312 seconds through durable checkpoint publication, corroborating
the estimate.

A step-13,650 projection snapshot, after 45.5% of optimizer steps, measured
24.88 hours of cumulative optimizer plus periodic-gate time. The mature
3.94555-second optimizer-step mean and 1,324.2-second mean gate imply another
17.92 optimizer hours and 12.14 gate hours, or 30.06 hours to step 30,000.
That revises projected base runtime to 54.94 hours, only 0.31 hours above the
original 54.63-hour estimate. If uninterrupted, the snapshot projected base
completion around August 5 at 21:20 EDT; standalone base evaluation, SFT, and
post-SFT evaluation are outside that timestamp.

Prior local full-CORE runs over this exact 91,037-example bundle took
1,823.8 and 1,849.1 seconds (30.4 and 30.8 minutes) for the 45M model on the
same RTX 3090. Mature training throughput is 34.15K tokens/s for that 45M run
versus 16.61K tokens/s here, a measured ratio of about 2.06. Training throughput
is only a proxy for CORE's heterogeneous forward passes, but scaling the two
observations by that ratio suggests roughly 62--63 minutes for each 111M full
CORE pass; a conservative operational allowance is 60--75 minutes. Base and
post-SFT CORE together should therefore add about 2--2.5 hours, while each
standalone dual-BPB pass has a directly observed current-model cost near 21.4
minutes. These downstream allowances remain separate from the base ETA and will
be replaced by measured durations in the final report.

The prior completed 45M SFT is a second local timing anchor. Its 5,000 steps at
the same 32,768-token optimizer batch took 4,630.3 seconds of optimizer time and
4,918.1 seconds wall-clock, averaging 35.38K tokens/s across 20 validation and
checkpoint gates. Scaling optimizer work by the 35.38K-to-16.61K measured
throughput ratio projects about 3,945 seconds (65.8 minutes) for this run's
2,000 SFT steps. The architectures and workloads differ and SFT packing can
shift realized throughput, so the operational allowance is 70--90 minutes,
including its eight gates, checkpoint writes, and final chat sampling. The
actual final run took 66m 59s of optimizer time and 72m 35s through publication
of its final evaluation reports, landing inside that allowance.

Combining the measured and proxy bounds gives an expected 80--100 minutes for
each full 111M evaluation (dual BPB, fixed samples, and CORE), plus 70--90
minutes for SFT: about 3.8--4.8 hours after base training finishes. If the
step-13,650 base projection continues to hold, the no-retry end-to-end window is
approximately August 6 at 01:10--02:10 EDT. This is an operational forecast,
not a result; checkpoint recovery, a failed evaluation attempt, or runtime
variance would move it later. Final measured timestamps supersede every
projection in this section.

The config contains `train.sample_every=1000`, but the current pretraining
lifecycle does not invoke a periodic sampler. Accordingly, this projection has
no periodic-sampling allowance; only checkpoint I/O and final evaluation remain
outside it. The frozen base prompt suite runs once during standalone evaluation
of the selected checkpoint.

The local checkpoint retention plan fit comfortably without deleting recovery
points: final base plus SFT checkpoint storage is about 52.1 GiB, versus 993 GB
free after the first checkpoint.

### Metric interpretation

Training loss is mean cross-entropy in natural-log units per supervised
token; tokenizer-level perplexity is `exp(loss)` and can be read as an
effective next-token branching factor. BPB instead divides accumulated nats by
raw source bytes and `ln(2)`, making it much less sensitive to tokenizer
segmentation. Its analogous effective byte-level branching factor is `2^BPB`.
The two denominators differ, so token perplexity and BPB must not be compared as
if they were the same scale.

For orientation only, step-500 training loss 9.1962 gives token PPL about
9,859, while compatibility BPB 2.8420 gives an effective byte branching factor
about 7.17. At step 30,000, compatibility BPB 1.087918 corresponds to about
2.13 effective byte choices; full-document BPB 1.121593 corresponds to about
2.18. The compatibility value remains `0.0783` BPB above nanochat d11's 1.0096
and `0.1054` above d12's 0.9825. Those nanochat values correspond to
about 2.01 and 1.98 byte choices, respectively. These are interim orientation
figures, not controlled architecture comparisons: the context length,
tokenizer, data, optimizer, implementation, and evaluation shapes differ.
The standalone held-out BPB and full CORE results below determine the base
run's outcome.

### BPB protocol provenance

The ranking metric is `nanochat_compat_v1` version 1, pinned to nanochat commit
[`41865401f73ff1c5321ae53297bceb2b78d4c8b4`](https://github.com/karpathy/nanochat/commit/41865401f73ff1c5321ae53297bceb2b78d4c8b4).
The protocol also embeds SHA-256 identities for the reference data loader, loss
evaluator, and base-training script. Its resolved local settings are batch 8,
context 1,024, 128 evaluation steps, buffer size 1,000, tokenizer batch 128,
four tokenizer threads, and exactly 1,048,576 processed model tokens. Packing,
BOS insertion, cropping, refill, and validation-shard restart semantics follow
the pinned source. The miniseries rows used 2,048 context, however, so this is a
source-compatible protocol at different resolved shape settings rather than an
identical miniseries evaluation.

Checkpoint selection is likewise protocol-pinned. A validation gate is accepted
only after both compatibility and full-document BPB complete, but `best.pt`
advances only when compatibility BPB strictly improves its prior minimum.
Full-document BPB keeps an independent minimum for reporting; it is not a
second ranking key. `last.pt` separately preserves the exact terminal training
state. Standalone base evaluation and SFT therefore consume the selected
`best.pt`, whose step may differ from 30,000 if a later gate does not improve the
ranking metric. The final section records both the selected and terminal steps.

### Final base metrics

The compatibility-ranked checkpoint was also the terminal step-30,000
checkpoint. Standalone evaluation started at 21:29:07 EDT and published its
complete reports at 22:46:10 EDT, a wall time of approximately 77m 03s.

| Base-evaluation item | Result |
| --- | ---: |
| Selected / terminal step | 30,000 / 30,000 |
| Checkpoint identity | `sha256:d8668d558cbcf382c24eeb7ea4d2445d43d5b6fb9ff9a84da9d1c5c5e770672a` |
| Nanochat-compatible BPB | 1.087918 |
| Full-document BPB | 1.121593 |
| Effective byte choices (`2^BPB`) | 2.126 compatible / 2.176 full |
| CORE v1 | 0.044687 |
| CORE scope | all 22 tasks / 91,037 examples |
| CORE elapsed | 3,262.402s (54m 22s) |
| Fixed samples | 7 prompts / 1,720 sampled tokens |
| Aggregate sample throughput | 204.361 sampled tokens/s |

The compatibility pass processed exactly 1,048,576 model tokens and counted
1,047,054 target tokens over 4,904,886 retained source bytes. The
full-document pass counted all 54,719,729 validation tokens and all
252,081,736 source bytes while processing 56,459,264 padded model tokens. Its
source-byte retention was therefore 100%, versus 1.9458% for the deliberately
bounded compatibility protocol.

The full CORE score is `0.047113` below nanochat d11 and `0.061213` below d12.
Against the reference aggregates bundled with the pinned evaluator it is
`0.069204` below GPT-2, `0.140216` below GPT-2 Medium, and `0.169954` below
GPT-2 Large. Several individual multiple-choice tasks were near or below their
random baselines; stronger relative results included PIQA centered `0.174102`,
ARC Easy `0.146465`, and CS Algorithms `0.378030`.

All seven frozen samples were syntactically fluent but unreliable. The model
failed the France-capital, gold-symbol, weekday, planet-list, and linear-equation
prompts, often drifting into repetitive topical prose. It mentioned cold in
the hot-opposite continuation but did not answer directly. This is consistent
with the quantitative result: the checkpoint learned local English form, but
not dependable factual recall, reasoning, or instruction following.

## Supervised fine-tuning

The active SFT configuration resolves to:

| SFT item | Value |
| --- | ---: |
| Device batch | 8 |
| Gradient accumulation | 4 |
| Tokens per optimizer step | 32,768 |
| Optimizer steps | 2,000 |
| Scheduled tokens | 65,536,000 |
| Scheduled tokens per parameter | 0.590914 |
| Warmup | 50 steps |
| Warmdown | final 50% of steps |
| LR schedule | peak 1e-5 at step 49; hold through 1,000; decay to 5e-7 |
| Optimizer | AdamW beta=(0.9, 0.95), weight decay 0, clip 1.0 |
| Assistant-only validation cadence | every 250 steps |
| Assistant-only validation budget | 65,536 tokens per gate |
| Validation/checkpoint gates | 8 |
| Post-SFT base-regression corpus | `data/tokenized_37`, same manifest as base |
| Post-SFT base-regression BPB shape | batch 8, 1,048,576 compatibility tokens |

Repeat weights 1:3:4 correspond to nominal draw proportions of 12.5% SmolTalk,
37.5% MMLU auxiliary-train, and 50.0% GSM8K train. The final run completed all
2,000 steps, all eight validation/checkpoint gates, and the frozen five-prompt
sample suite. The SFT config explicitly pins the base run's packed-data
directory, 37-shard count, and fixed final validation shard so post-SFT BPB is
measured against the same validation identity rather than an inherited default.
It also pins evaluation batch 8 and the 1,048,576-token compatibility budget,
which are read from `train` by the shared base evaluator even for an SFT
checkpoint. A CPU-only downstream preflight passed 86 tests spanning the
base-evaluation CLI, pipeline and reporting, SFT CLI and training lifecycle,
assistant-only BPB, loaders, checkpointing, exact resume, tracking, and the
bounded overfit integration.

### SFT trajectory

| Step | Train loss | Assistant-only validation BPB | Throughput (tokens/s) |
| ---: | ---: | ---: | ---: |
| 250 | 2.648106 | 0.839210 | 16,322.99 |
| 500 | 2.448669 | 0.817651 | 16,337.16 |
| 750 | 2.524045 | 0.802807 | 16,347.84 |
| 1,000 | 2.466818 | 0.791659 | 16,269.06 |
| 1,250 | 2.588515 | 0.783238 | 16,225.12 |
| 1,500 | 2.501292 | 0.777536 | 16,222.42 |
| 1,750 | 2.435317 | 0.774023 | 16,321.06 |
| 2,000 | 2.395646 | 0.772646 | 16,274.14 |

The first gate wrote `best.pt`, `step_000250.pt`, and `last.pt`, each
1,331,007,271 bytes, then resumed at step 260. Full CPU reconstruction of
`last.pt` verified SFT stage and step 250, scheduler step 250, tracker step
250, 110,906,112 parameters, 75 optimizer states, W&B run `4uitqomj`, and the
selected base-checkpoint identity
`sha256:d8668d558cbcf382c24eeb7ea4d2445d43d5b6fb9ff9a84da9d1c5c5e770672a`.
Its exact loader continuation was at 1,000 device batches with mixture repeats
`[1, 3, 4]`; it had seen 15,993 conversations, packed 15,742, cropped 3,819,
skipped 152 with zero assistant supervision, and used 64,632 padding tokens
across 8,000 emitted rows with no padding rows. Local model-artifact events
were present for recoverability, while the remote W&B artifact count remained
zero. Assistant BPB uses a different chat-rendered, assistant-masked corpus
than base BPB and therefore must not be compared numerically to the base or
nanochat BPB values.

At step 500, assistant BPB improved by `0.021559` (2.57%) and promoted a new
`best.pt`; the periodic and latest files were each 1,331,007,399 bytes and
training resumed at step 510. Steps 260--500 had mean/median loss
`2.582853`/`2.599312`, mean throughput 16,313.31 tokens/s, and maximum logged
pre-clipping gradient norm `1.132069` at step 420. CPU reconstruction verified
step/scheduler/tracker 500, loader device-batch step 2,000, validation
current/minimum `0.817651`, the expected W&B and base-checkpoint identities,
110,906,112 parameters, and 75 optimizer states. W&B again matched the local
BPB and retained zero uploaded artifacts.

At step 750, assistant BPB improved another `0.014844` (1.82%) to `0.802807`
and promoted `best.pt`; all three newly written checkpoint roles were
1,331,007,271 bytes, and training resumed at step 760. Steps 510--750 had
mean/median loss `2.550071`/`2.544886`, mean throughput 16,331.34 tokens/s,
and maximum logged gradient norm `1.269459` at step 570. The following logged
norms returned below 1.0, so this was an isolated, clipped event. Full CPU
reconstruction verified SFT step/scheduler/tracker 750, loader device-batch
step 3,000, the expected identities, and validation current/minimum
`0.802807`; W&B again matched with zero remote artifacts.

At step 1,000, assistant BPB improved another `0.011148` (1.39%) to
`0.791659` and promoted `best.pt`; the new checkpoint files were each
1,331,007,399 bytes, and training resumed at step 1,010. Steps 760--1,000 had
mean/median loss `2.509405`/`2.515053`, mean throughput 16,317.07 tokens/s,
and maximum logged gradient norm `1.192395` at step 970. CPU reconstruction
verified SFT step/scheduler/tracker 1,000, loader device-batch step 4,000,
optimizer and scheduler LR `1e-5`, and validation current/minimum `0.791659`.
The step-1,010 LR multiplier was `0.99145`, exactly beginning warmdown; every
logged SFT multiplier through the subsequent audit matched the closed-form
schedule within `1.11e-16`. W&B remained exact with zero remote artifacts.

At step 1,250, assistant BPB improved another `0.008421` (1.06%) to
`0.783238` and promoted `best.pt`; the checkpoint files were each
1,331,007,399 bytes, and training resumed at step 1,260 with logged LR
multiplier `0.75395`. Steps 1,010--1,250 had mean/median loss
`2.455598`/`2.448630`, mean throughput 16,326.47 tokens/s, and maximum logged
gradient norm `1.147575` at step 1,190. CPU reconstruction verified SFT
step/scheduler/tracker 1,250, loader device-batch step 5,000, saved optimizer
LR `7.625e-6`, validation current/minimum `0.783238`, and all expected
identities. W&B remained exact with zero remote artifacts.

At step 1,500, assistant BPB improved another `0.005702` (0.73%) to
`0.777536`, its sixth strict improvement, and promoted `best.pt`; the new
checkpoint files were each 1,331,007,271 bytes. Training resumed at step 1,510
with LR multiplier `0.51645`. Steps 1,260--1,500 had mean/median loss
`2.458387`/`2.464086`, mean throughput 16,318.34 tokens/s, and maximum logged
gradient norm `1.220591` at step 1,490. CPU reconstruction verified SFT
step/scheduler/tracker 1,500, loader device-batch step 6,000, saved optimizer
LR `5.25e-6`, validation current/minimum `0.777536`, and all expected
identities. W&B remained exact with zero remote artifacts.

At step 1,750, assistant BPB improved another `0.003513` (0.45%) to
`0.774023`, its seventh strict improvement, and promoted `best.pt`; the new
checkpoint files were each 1,331,007,399 bytes. Training resumed at step 1,760
with LR multiplier `0.27895`. Steps 1,510--1,750 had mean/median loss
`2.401143`/`2.388572`, mean throughput 16,275.54 tokens/s, and maximum logged
gradient norm `1.161564` at step 1,710. CPU reconstruction verified SFT
step/scheduler/tracker 1,750, loader device-batch step 7,000, saved optimizer
LR `2.875e-6`, validation current/minimum `0.774023`, and all expected
identities. W&B remained exact with zero remote artifacts.

At terminal step 2,000, assistant BPB improved another `0.001378` (0.18%) to
`0.772646`, its eighth strict improvement. From the first gate it fell by
`0.066564`, or 7.93%. The terminal window, steps 1,760--2,000, had mean/median
loss `2.428137`/`2.442010`, mean throughput 16,269.04 tokens/s, and maximum
logged gradient norm `1.052212` at step 1,940. Across all 200 logged training
rows, mean throughput was 16,309.10 tokens/s, mean MFU 35.47%, median gradient
norm `0.958791`, p95 `1.132069`, p99 `1.220591`, and maximum `1.413963` at
step 10. All values were finite, and every LR multiplier matched the schedule
within `1.11e-16`.

`best.pt`, `step_002000.pt`, and `last.pt` were each 1,331,007,271 bytes and
all reconstructed as SFT step 2,000 with the same validation, W&B, and base
checkpoint identities. Full `last.pt` reconstruction verified scheduler and
tracker step 2,000, saved LR `5e-7`, 110,906,112 parameters, 76 model tensors,
75 optimizer states, 8,000 loader device batches, `5.0714973831168e16`
training FLOPs, and 4,019.001 seconds (66m 59s) of optimizer time. The loader
had seen 112,273 conversations, packed 111,149, cropped 26,478, skipped 1,025
with zero assistant supervision, and emitted 64,000 rows with no padding rows
and 1,164,176 padding tokens. The selected checkpoint identity is
`sha256:66fcfef337f93b48bd4057d2d49275e7dc4382026aa9ca6702afaf789b6f6874`.

The final local log held 235 complete records: one config, 200 training rows on
the exact 10-step grid, eight validation rows, and 26 unique artifact events.
Ten retained checkpoints total 13,310,073,222 bytes (12.40 GiB). From W&B
tracking-state publication to the completed SFT report, wall time was
4,355.472 seconds (72m 35s). W&B uploaded exactly the two permitted evaluation
reports and no model/checkpoint, dataset, or tokenizer artifact; downloading
them reproduced local bytes exactly. `sft_eval.json` is
`sha256:6f2e8c74b6f0a3e63e65c19c812ef0fa60347cfa597737a68c23b5f8e8077460`
and `sft_samples.md` is
`sha256:3d64e40f0b97eb5caf83243132cb71f6fe10d917416c13d4d46f099ff63b733f`.

The final assistant BPB covers eight complete device batches, 65,536 processed
model tokens, 69 source conversations, 53,924 supervised assistant tokens, and
241,739 assistant bytes. The five fixed samples generated 1,185 tokens in
5.841 seconds, or 202.89 tokens/s in aggregate. SFT taught the checkpoint to
emit assistant-shaped prose and use the assistant-end token on two prompts,
but task quality remained poor: it did not explain gradient descent, produced
invalid string-reversal code, gave irrelevant PyTorch project ideas, failed
`17 * 23`, and did not return JSON. Thus SFT improved format behavior and its
own held-out likelihood without producing dependable completions from this
weak base model.

## Post-SFT regression evaluation

The SFT lifecycle separates chat-native evaluation from base-regression
evaluation. At SFT completion, `train_sft` performs the final assistant-only BPB
and renders the frozen five-prompt chat suite to `metrics/sft_samples.md`. The
supervisor then runs `eval_base` on the selected SFT checkpoint to measure both
base BPB protocols on the same pretraining validation corpus, render the frozen
seven-prompt base-completion suite, and run full CORE.

The corrected post-SFT pass began at approximately 00:11:13 EDT on August 6
and atomically published its reports at 01:28:18 EDT, about 77m 05s later. It
completed on attempt 1 after the earlier configuration-guard rejections, which
had evaluated no examples.

| Post-SFT evaluation item | Result | Change from base |
| --- | ---: | ---: |
| Selected SFT step | 2,000 | -- |
| Checkpoint identity | `sha256:66fcfef337f93b48bd4057d2d49275e7dc4382026aa9ca6702afaf789b6f6874` | -- |
| Nanochat-compatible BPB | 1.130756 | +0.042838 (+3.94%, worse) |
| Full-document BPB | 1.164907 | +0.043314 (+3.86%, worse) |
| Effective byte choices (`2^BPB`) | 2.190 compatible / 2.242 full | +0.064 / +0.066 |
| CORE v1 | 0.057101 | +0.012415 (+27.78%, better) |
| CORE scope | all 22 tasks / 91,037 examples | identical scope |
| CORE elapsed | 3,267.781s (54m 28s) | +5.378s |
| Fixed base-style samples | 7 prompts / 1,792 sampled tokens | +72 tokens |
| Aggregate sample throughput | 203.304 sampled tokens/s | -1.057 tokens/s |

The two BPB protocols reproduce the base run's exact tokenizer and validation
manifest identities. The compatibility pass processed 1,048,576 model tokens
and counted 1,047,054 target tokens / 4,904,886 target bytes. The full pass
processed 56,459,264 model tokens and counted all 54,719,729 source tokens /
252,081,736 source bytes. The higher BPB therefore measures genuine base-text
likelihood regression after the 65.5M-token SFT schedule, not a data or protocol
change.

CORE moved in the opposite direction. Its largest centered-score gain was
BoolQ, from `-0.460647` to `-0.191856` (+0.268791), followed by CommonsenseQA
from `0.088862` to `0.178952` (+0.090090). Smaller gains included CS Algorithms
(+0.025000), ARC Challenge (+0.013652), and Winogrande (+0.011050). The largest
losses were COPA (-0.040000), Operators (-0.019048), OpenBookQA (-0.016000),
PIQA (-0.015234), and ARC Easy (-0.013468). The aggregate improved, but its
absolute `0.057101` remains low and does not contradict the poor generations.

All seven base-style post-SFT samples exhausted the 256-token limit; none
reached the base sampler's BOS stop token or answered its task reliably. They
were generally more repetitive than the base samples and sometimes emitted
literal chat-control text because this plain-completion evaluator does not stop
on `assistant_end`. The France, gold, weekday, planet, and equation prompts all
remained wrong. Together with the five failed chat-native samples, this shows
that assistant formatting improved more than task completion quality.

The three post-SFT report identities are
`sha256:eec65d7b4590c12354635923bff1aaad4c8dd4464f5ee2caf69fccaa369a6fbb`
for `base_eval.json`,
`sha256:35385e28de62e78911c5e8ec5501d8b25ff636f91659c99bcf60a57c5b464269`
for `base_samples.md`, and
`sha256:10f4778dd7de7ff8430dce83a785cd64b3d64191d61a5a7ea3a5a4e4c6ae7984`
for `core_comparison.md`. W&B run `4uitqomj` finished with exactly five
`evaluation` reports total (two chat-native SFT reports and these three), all
downloaded byte-for-byte equal to the local files. It contains no
model/checkpoint, dataset, or tokenizer artifact.

ChatCORE is unavailable in this repository: the README explicitly leaves its
fields absent until that evaluator exists, and Milestone 8 remains future work.
Ordinary CORE is reported as base-regression evidence and is not relabeled as
ChatCORE.

## Closest nanochat references

The closest official January 2026 nanochat miniseries points are:

| Reference | Shape / parameters | Training tokens | Validation BPB | CORE v1 | 8xH100 time |
| --- | --- | ---: | ---: | ---: | ---: |
| d11 parameter match | depth 11, width 704, 112M | 0.89B | 1.0096 | 0.0918 | 6.6 min |
| d12 shape match | depth 12, width 768, 135M | 1.08B | 0.9825 | 0.1059 | 7.8 min |
| This run, base | depth 12, width 768, 110.906M | 1.966B | 1.0879 | 0.0447 | single-3090 base training 54h 55m measured optimizer+gate subtotal |
| This run, post-SFT regression | same | +0.066B SFT | 1.1308 | 0.0571 | 72m 35s SFT plus 77m 05s regression eval |

Source: [nanochat January 7 miniseries results](https://github.com/karpathy/nanochat/discussions/420).

| This run minus reference | Validation BPB | CORE v1 |
| --- | ---: | ---: |
| Base versus d11 parameter match | +0.078318 | -0.047113 |
| Base versus d12 shape match | +0.105418 | -0.061213 |
| Post-SFT versus d11 parameter match | +0.121156 | -0.034699 |
| Post-SFT versus d12 shape match | +0.148256 | -0.048799 |

Positive BPB is worse because lower is better; negative CORE is worse because
higher is better.

Our run schedules 1.966B tokens, or approximately 17.73 tokens per parameter:
2.21x the d11 reference's tokens and 1.82x the d12 reference's tokens. These
are useful anchors, not controlled apples-to-apples baselines. Differences
include tokenizer and data realization, 1,024 versus 2,048 context, AdamW
versus Muon, LR schedule, architecture details, parameter count for d12, and
independent training/evaluation implementations. The nanochat sweep used a
524,288-token global batch on 8xH100, a compute-optimal D:N target near 8, and
40% warmdown; this run uses a 65,536-token batch on one RTX 3090, D:N 17.73,
and 50% warmdown. Both report CORE v1, but the implementation path is still
independent. Final comparisons therefore report absolute deltas while avoiding
causal claims.

The official d11/d12 miniseries reports pretraining BPB and CORE only; it does
not publish an architecture-matched SFT or ChatCORE point. Consequently,
post-SFT base-regression BPB/CORE can be compared to those same base anchors,
but this run's assistant-only BPB cannot: it uses chat-rendered validation data
and masks all non-assistant targets. Chat samples are reported qualitatively,
and ChatCORE is listed as unavailable rather than replaced with ordinary CORE.

## Timing, interruptions, and resumability

The step-500 `best.pt` was independently reconstructed on CPU through the full
training-checkpoint loader. It restored model and optimizer state, scheduler
step 500, the exact document-packing loader continuation, RNG state, tracker
step 500, both validation minima, and W&B run ID `ngbuyhxj`. This proves an
exact local continuation is available if the live process is interrupted.

The first process was stopped during the step-1,000 validation gate. Its local
training telemetry had reached step 1,000, but no step-1,000 validation record
or exact checkpoint had been committed. The original 103-line JSONL was
preserved as `interrupted_step1000_metrics.jsonl` with identity
`sha256:79bf4221da54bb56c776ee19ee6f224e4326ab7e0fad9b7a204166f77af7f20c`.
The active JSONL and compact summary were then rolled back to the verified
step-500 checkpoint boundary without deleting the interrupted evidence.

That recovery exposed a validator inconsistency: pretraining rejected the two
legitimate metrics events at a validation step (one training event and one
validation event), while SFT already accepted monotonically non-decreasing
steps. Commit `8561942` adds regression coverage, permits equal adjacent steps,
and continues to reject true step regressions. Five focused resume/validation
tests and Ruff passed before the resumed process was launched.

The run resumed the original W&B ID with model, optimizer, scheduler, RNG, and
packed-loader state from step 500. The replayed step-510 loss
(`9.41638469696045`) and gradient norm (`9.611682891845703`) matched the
interrupted attempt exactly. A later audit at step 810 also matched loss,
gradient norm, and LR multiplier exactly, providing an additional bit-exact
continuation check. Timing-only fields differed, as expected. Because W&B had
already accepted the first attempt's training rows through step 1,000, it
ignored replayed rows below its remote step 1,001; those original deterministic
rows remain in W&B, while the rebuilt local JSONL is authoritative for the
resumed lifecycle.

A full W&B-history reconciliation through step 14,000 found 27 remote
validation rows matching the 28 local rows exactly, with no mismatches or
extra steps. The sole remote gap is the recovered step-1,000 validation row:
W&B retains the exact step-1,000 training loss from the first attempt, but the
first attempt was stopped before validation and the monotonic remote cursor
later rejected the replayed step-1,000 validation event. Local JSONL and all
three exact step-1,000 checkpoints preserve both BPBs. The live W&B run still
had zero uploaded artifacts.

The replay then completed the previously interrupted step-1,000 gate.
Compatibility BPB improved from `2.842049` to `2.176092`, while full-document
BPB improved from `2.949778` to `2.240685`. Local `best.pt`,
`step_001000.pt`, and `last.pt` were each 1,331,002,343 bytes. A full CPU load
of `last.pt` verified step/scheduler/tracker 1,000, packed-loader position
64,000, both validation results, and W&B run ID `ngbuyhxj` before training
continued at step 1,010. The W&B API still reported zero uploaded artifacts.

At step 1,500, compatibility BPB improved again to `2.110908` and
full-document BPB to `2.169972`. The 1,331,002,343-byte `best.pt` reconstructed
fully on CPU at step/scheduler/tracker 1,500 with packed-loader position 96,000
before training continued at step 1,520. The remote artifact count remained
zero.

At step 2,000, compatibility BPB reached `2.053895` and full-document BPB
`2.112036`. Local `best.pt`, `step_002000.pt`, and `last.pt` were each
1,331,002,343 bytes. Full CPU reconstruction of `last.pt` verified
step/scheduler/tracker 2,000, packed-loader position 128,000, both validation
values, and total training FLOPs `1.01429947662336e17` before step 2,010. W&B
again reported zero uploaded artifacts.

At step 2,500, compatibility BPB improved by `0.061025` to `1.992870`, and
full-document BPB improved by `0.067416` to `2.044620`. The gate again took
approximately 21m 54s. A full CPU load of the 1,331,002,343-byte `best.pt`
verified step/scheduler/tracker/validation 2,500, packed-loader position
160,000, total training FLOPs `1.2678743457792e17`, both BPBs, and W&B run ID
`ngbuyhxj`. Training continued at step 2,510, and the W&B API continued to
report zero uploaded artifacts.

At step 3,000, compatibility BPB improved by `0.035758` to `1.957112`, and
full-document BPB improved by `0.035116` to `2.009505`. The gate took
approximately 22m 01s. Full CPU reconstruction of `best.pt`,
`step_003000.pt`, and `last.pt` verified that every 1,331,002,343-byte file
contained step/scheduler/tracker/validation 3,000, packed-loader position
192,000, total training FLOPs `1.52144921493504e17`, both BPBs, and W&B run ID
`ngbuyhxj`. Training continued at step 3,010 and reached step 3,020 before the
milestone was recorded; W&B remained online with zero uploaded artifacts.

At step 3,500, compatibility BPB improved by `0.084141` to `1.872970`, and
full-document BPB improved by `0.086814` to `1.922690`. The gate took
approximately 22m 02s. A full CPU load of the 1,331,002,343-byte `best.pt`
verified step/scheduler/tracker/validation 3,500, packed-loader position
224,000, total training FLOPs `1.77502408409088e17`, both BPBs, and W&B run ID
`ngbuyhxj`. Training continued at step 3,510 and reached step 3,520; W&B
remained online with zero uploaded artifacts.

At step 4,000, compatibility BPB improved by `0.062841` to `1.810129`, and
full-document BPB improved by `0.066060` to `1.856630`. The gate took
approximately 22m 15s through durable `last.pt` publication. Full CPU
reconstruction of `best.pt`, `step_004000.pt`, and `last.pt` verified that
every 1,331,002,343-byte file contained step/scheduler/tracker/validation
4,000, packed-loader position 256,000, total training FLOPs
`2.02859895324672e17`, both BPBs, and W&B run ID `ngbuyhxj`. Training
continued at step 4,010 and reached step 4,030 before the milestone was
recorded. The W&B API reported the run online with zero uploaded artifacts and
the step-4,000 validation values in its remote summary.

At step 4,500, compatibility BPB improved by `0.048730` to `1.761399`, and
full-document BPB improved by `0.050092` to `1.806538`. The gate took
approximately 21m 56s through durable `best.pt` publication. This 500-step
gate does not write a numbered or `last.pt` checkpoint because periodic saves
occur every 1,000 steps. Full CPU reconstruction of the 1,331,002,343-byte
`best.pt` verified step/scheduler/tracker/validation 4,500, packed-loader
position 288,000, total training FLOPs `2.28217382240256e17`, both BPBs, and
W&B run ID `ngbuyhxj`. Training continued at step 4,510. The W&B API reported
the run online with the step-4,500 values in its remote summary and zero
uploaded artifacts.

At step 5,000, compatibility BPB improved by `0.049185` to `1.712214`, and
full-document BPB improved by `0.050025` to `1.756513`. The gate took
approximately 22m 23s through durable `last.pt` publication. Full CPU
reconstruction of `best.pt`, `step_005000.pt`, and `last.pt` verified that
every 1,331,002,343-byte file contained step/scheduler/tracker/validation
5,000, packed-loader position 320,000, total training FLOPs
`2.5357486915584e17`, both BPBs, and W&B run ID `ngbuyhxj`. Training continued
at step 5,010 with 16.67K tokens/s. The W&B API reported the run online with
zero uploaded artifacts; its summary was temporarily one logging interval
behind the local stream when first queried after the gate, then caught up to
step 5,010 with both step-5,000 validation values intact.

At step 5,500, compatibility BPB improved by `0.042728` to `1.669486`, and
full-document BPB improved by `0.038938` to `1.717575`. The gate took
approximately 21m 56s through durable `best.pt` publication. This 500-step
gate did not write a numbered or `last.pt` checkpoint because periodic saves
occur every 1,000 steps. Full CPU reconstruction of the 1,331,002,343-byte
`best.pt` verified step/scheduler/tracker/validation 5,500, packed-loader
position 352,000, total training FLOPs `2.78932356071424e17`, both BPBs, and
W&B run ID `ngbuyhxj`. Training continued at step 5,510 with 16.63K tokens/s.
The initial W&B API query reported the run online with zero uploaded artifacts
while its summary lagged one logging interval behind the local stream; it then
caught up to step 5,510 with both step-5,500 validation values intact.

At step 6,000, compatibility BPB improved by `0.048653` to `1.620833`, and
full-document BPB improved by `0.050665` to `1.666910`. The gate took
approximately 22m 14s through durable `last.pt` publication. Full CPU
reconstruction of `best.pt`, `step_006000.pt`, and `last.pt` verified that
every 1,331,002,343-byte file contained step/scheduler/tracker/validation
6,000, packed-loader position 384,000, total training FLOPs
`3.04289842987008e17`, both BPBs, and W&B run ID `ngbuyhxj`. Training
continued at step 6,010 with 16.66K tokens/s. The initial W&B API query
reported the run online with zero uploaded artifacts while its summary lagged
one logging interval behind the local stream; it then caught up to step 6,010
with both step-6,000 validation values intact.

At step 6,500, compatibility BPB improved by `0.037335` to `1.583498`, and
full-document BPB improved by `0.041373` to `1.625537`. The gate took
approximately 21m 56s through durable `best.pt` publication. This 500-step
gate did not write a numbered or `last.pt` checkpoint because periodic saves
occur every 1,000 steps. Full CPU reconstruction of the 1,331,002,343-byte
`best.pt` verified step/scheduler/tracker/validation 6,500, packed-loader
position 416,000, total training FLOPs `3.29647329902592e17`, both BPBs, and
W&B run ID `ngbuyhxj`. Training continued at step 6,510 with 16.63K tokens/s.
The W&B API reported the step-6,500 validation summary, run state `running`,
and zero uploaded artifacts.

At step 7,000, compatibility BPB improved by `0.050049` to `1.533449`, and
full-document BPB improved by `0.051621` to `1.573916`. The gate took
approximately 22m 18s through durable `last.pt` publication. Full CPU
reconstruction of `best.pt`, `step_007000.pt`, and `last.pt` verified that
every 1,331,002,343-byte file contained step/scheduler/tracker/validation
7,000, packed-loader position and row position 448,000, total training FLOPs
`3.55004816818176e17`, both BPBs, the expected tokenized-data manifest, and
W&B run ID `ngbuyhxj`. Training continued at step 7,010 with loss `5.021323`
and 16.65K tokens/s. The W&B API reported the step-7,000 validation summary,
run state `running`, and zero uploaded artifacts.

At step 7,500, compatibility BPB improved by `0.049511` to `1.483938`, and
full-document BPB improved by `0.045890` to `1.528027`. The gate took
approximately 21m 57s through durable `best.pt` publication. This 500-step
gate did not write a numbered or `last.pt` checkpoint because periodic saves
occur every 1,000 steps. Full CPU reconstruction of the 1,331,002,343-byte
`best.pt` verified step/scheduler/tracker/validation 7,500, packed-loader
position and row position 480,000, total training FLOPs
`3.8036230373376e17`, both BPBs, the expected tokenized-data manifest, and W&B
run ID `ngbuyhxj`. Training continued at step 7,510 with loss `4.810125` and
16.63K tokens/s. The first post-gate W&B API query reported run state
`running` and zero uploaded artifacts while its summary briefly lagged behind
the local JSONL; the local checkpoint and JSONL remain authoritative. A
follow-up API query caught up to step 7,510 with both step-7,500 BPB values
intact and the remote artifact count still zero.

At step 8,000, compatibility BPB improved by `0.050267` to `1.433670`, and
full-document BPB improved by `0.054606` to `1.473420`. The gate took
approximately 22m 17s through durable `last.pt` publication. Full CPU
reconstruction of `best.pt`, `step_008000.pt`, and `last.pt` verified that
every 1,331,002,343-byte file contained step/scheduler/tracker/validation
8,000, packed-loader position and row position 512,000, total training FLOPs
`4.05719790649344e17`, both BPBs, the expected tokenized-data manifest, and
W&B run ID `ngbuyhxj`. Training returned to full GPU load after checkpoint
publication. The first post-gate W&B API query reported run state `running`
and zero uploaded artifacts while its summary briefly lagged at step 7,990;
the local checkpoint and JSONL remain authoritative. A follow-up API query
caught up to step 8,020 with both step-8,000 BPB values intact and the remote
artifact count still zero.

At step 8,500, compatibility BPB improved by `0.031902` to `1.401768`, and
full-document BPB improved by `0.032892` to `1.440529`. The gate took
approximately 21m 58s through durable `best.pt` publication. This 500-step
gate did not write a numbered or `last.pt` checkpoint because periodic saves
occur every 1,000 steps. Full CPU reconstruction of the 1,331,002,343-byte
`best.pt` verified step/scheduler/tracker/validation 8,500, packed-loader
position and row position 544,000, total training FLOPs
`4.31077277564928e17`, both BPBs, the expected tokenized-data manifest, and
W&B run ID `ngbuyhxj`. Training returned to full GPU load after checkpoint
publication. The first post-gate W&B API query reported run state `running`
and zero uploaded artifacts while its summary briefly lagged at step 8,490;
the local checkpoint and JSONL remain authoritative. A follow-up API query
caught up to step 8,510 with both step-8,500 BPB values intact and the remote
artifact count still zero.

At step 9,000, compatibility BPB improved by `0.025504` to `1.376264`, and
full-document BPB improved by `0.025297` to `1.415232`. The gate took
approximately 22m 14s through durable `last.pt` publication. Full CPU
reconstruction of `best.pt`, `step_009000.pt`, and `last.pt` verified that
every 1,331,002,343-byte file contained step/scheduler/tracker/validation
9,000, packed-loader position and row position 576,000, total training FLOPs
`4.56434764480512e17`, both BPBs, the expected tokenized-data manifest, and
W&B run ID `ngbuyhxj`. Training returned to full GPU load after checkpoint
publication. The first post-gate W&B API query reported run state `running`
and zero uploaded artifacts while its summary briefly lagged at step 8,990;
the local checkpoint and JSONL remain authoritative. A follow-up API query
caught up to step 9,020 with both step-9,000 BPB values intact and the remote
artifact count still zero.

At step 9,500, compatibility BPB improved by `0.024074` to `1.352190`, and
full-document BPB improved by `0.024926` to `1.390306`. The gate took
approximately 21m 57s through durable `best.pt` publication. This 500-step
gate did not write a numbered or `last.pt` checkpoint because periodic saves
occur every 1,000 steps. Full CPU reconstruction of the 1,331,002,343-byte
`best.pt` verified step/scheduler/tracker/validation 9,500, packed-loader
position and row position 608,000, total training FLOPs
`4.81792251396096e17`, both BPBs, the expected tokenized-data manifest, and
W&B run ID `ngbuyhxj`. Training resumed at step 9,510 with loss `4.488977` and
16.62K tokens/s. The first post-gate W&B API query reported run state
`running` and zero uploaded artifacts while its summary briefly lagged at the
step-9,000 validation values; the local JSONL and checkpoint remain
authoritative. A follow-up API query caught up to step 9,510 with both
step-9,500 BPB values intact and the remote artifact count still zero.

At step 10,000, compatibility BPB improved by `0.019731` to `1.332459`, and
full-document BPB improved by `0.016606` to `1.373700`. The gate took
approximately 22m 18s through durable `last.pt` publication. Full CPU
reconstruction of `best.pt`, `step_010000.pt`, and `last.pt` verified that
every 1,331,002,343-byte file contained step/scheduler/tracker/validation
10,000, packed-loader position and row position 640,000, total training FLOPs
`5.0714973831168e17`, both BPBs, the expected tokenized-data manifest, and W&B
run ID `ngbuyhxj`. The first post-gate W&B API query reported run state
`running` and zero uploaded artifacts while its summary briefly lagged at the
step-9,500 validation values; the local JSONL and checkpoints remain
authoritative. A follow-up API query caught up to step 10,010 with both
step-10,000 BPB values intact and the remote artifact count still zero.

At step 10,500, compatibility BPB improved by `0.016512` to `1.315947`, and
full-document BPB improved by `0.016740` to `1.356959`. The gate took
approximately 21m 57s through durable `best.pt` publication. This 500-step
gate did not write a numbered or `last.pt` checkpoint because periodic saves
occur every 1,000 steps. Full CPU reconstruction of the 1,331,002,343-byte
`best.pt` verified step/scheduler/tracker/validation 10,500, packed-loader
position and row position 672,000, total training FLOPs
`5.32507225227264e17`, both BPBs, the expected tokenized-data manifest, the
pretraining stage, and W&B run ID `ngbuyhxj`. The first post-gate W&B API
query reported run state `running` and zero uploaded artifacts while its
summary briefly lagged at the step-10,000 validation values; the local JSONL
and checkpoint remain authoritative. A follow-up API query caught up to step
10,520 with both step-10,500 BPB values intact and the remote artifact count
still zero.

At step 11,000, compatibility BPB improved by `0.015227` to `1.300720`, and
full-document BPB improved by `0.018129` to `1.338830`. The gate took
approximately 22m 17s through durable `last.pt` publication. Full CPU
reconstruction of `best.pt`, `step_011000.pt`, and `last.pt` verified that
every 1,331,002,343-byte file contained step/scheduler/tracker/validation
11,000, packed-loader position and row position 704,000, total training FLOPs
`5.57864712142848e17`, both BPBs, the expected tokenized-data manifest, the
pretraining stage, 76 model tensors, 75 optimizer states, and W&B run ID
`ngbuyhxj`. Training resumed through step 11,020 at about 16.63K tokens/s.
The first post-gate W&B API query reported run state `running` and zero
uploaded artifacts while its summary briefly lagged at step 10,990 with the
step-10,500 validation values; the local JSONL and checkpoints remain
authoritative. A follow-up API query caught up to step 11,020 with both
step-11,000 BPB values intact and the remote artifact count still zero.

At step 11,500, compatibility BPB improved by `0.015692` to `1.285029`, and
full-document BPB improved by `0.016662` to `1.322168`. The gate took
approximately 21m 57s through durable `best.pt` publication. This 500-step
gate did not write a numbered or `last.pt` checkpoint because periodic saves
occur every 1,000 steps. Full CPU reconstruction of the 1,331,002,343-byte
`best.pt` verified step/scheduler/tracker/validation 11,500, packed-loader
position and row position 736,000, total training FLOPs
`5.83222199058432e17`, both BPBs, the expected tokenized-data manifest, the
pretraining stage, 76 model tensors, 75 optimizer states, and W&B run ID
`ngbuyhxj`. Training resumed at step 11,510 with loss `4.259425` and 16.64K
tokens/s. The post-gate W&B API query was already synchronized at step 11,500
with both validation values, run state `running`, and zero uploaded artifacts.

At step 12,000, compatibility BPB improved by `0.012392` to `1.272637`, and
full-document BPB improved by `0.012287` to `1.309881`. The gate took
approximately 22m 21s through durable `last.pt` publication. Full CPU
reconstruction of `best.pt`, `step_012000.pt`, and `last.pt` verified that
every 1,331,002,343-byte file contained step/scheduler/tracker/validation
12,000, packed-loader position and row position 768,000, total training FLOPs
`6.08579685974016e17`, both BPBs, the expected tokenized-data manifest, the
pretraining stage, 76 model tensors, 75 optimizer states, and W&B run ID
`ngbuyhxj`. Training resumed through step 12,020 with loss `4.226473` and
16.62K tokens/s. The post-gate W&B API query was already synchronized at step
12,000 with both validation values, run state `running`, and zero uploaded
artifacts.

At step 12,500, compatibility BPB improved by `0.012851` to `1.259785`, and
full-document BPB improved by `0.013151` to `1.296730`. The gate took
approximately 21m 57s through durable `best.pt` publication. This 500-step
gate did not write a numbered or `last.pt` checkpoint because periodic saves
occur every 1,000 steps. Full CPU reconstruction of the 1,331,002,343-byte
`best.pt` verified step/scheduler/tracker/validation 12,500, packed-loader
position and row position 800,000, total training FLOPs
`6.339371728896e17`, both BPBs, the expected tokenized-data manifest, the
pretraining stage, 76 model tensors, 75 optimizer states, and W&B run ID
`ngbuyhxj`. Training resumed at step 12,510 with loss `3.995852` and 16.62K
tokens/s. The initial post-gate W&B API query reported run state `running` and
zero uploaded artifacts while its summary briefly lagged at step 12,490 with
the step-12,000 validation values; local JSONL and checkpoint remain
authoritative. A follow-up API query caught up to step 12,500 with both new
validation values intact and the remote artifact count still zero.

At step 13,000, compatibility BPB improved by `0.008523` to `1.251262`, and
full-document BPB improved by `0.005852` to `1.290878`. The gate took
approximately 22m 20s through durable `last.pt` publication. Full CPU
reconstruction of `best.pt`, `step_013000.pt`, and `last.pt` verified that
every 1,331,002,343-byte file contained step/scheduler/tracker/validation
13,000, packed-loader position and row position 832,000, total training FLOPs
`6.59294659805184e17`, both BPBs, the expected tokenized-data manifest and
validation identity, the pretraining stage, 76 model tensors, 75 optimizer
states, and W&B run ID `ngbuyhxj`. Training resumed at step 13,010 with loss
`4.309958` and 16.64K tokens/s. The post-gate W&B API query was synchronized
at step 13,010 with both step-13,000 validation values, run state `running`,
and zero uploaded artifacts.

A read-only recovery audit while training was at step 13,260 confirmed that
the supervisor's newest-checkpoint selection resolves to `last.pt` at exact
step 13,000. Its scheduler, tracker, and validation steps all equal 13,000;
the packed-loader position and row position are 832,000; the two BPBs and W&B
run ID are intact. Fifteen retained base checkpoint files occupied about 19
GiB, with 976 GiB free, leaving ample room for the full retention plan.

At step 13,500, compatibility BPB improved by `0.010493` to `1.240769`, and
full-document BPB improved by `0.012959` to `1.277918`. The gate took
approximately 21m 57s through durable `best.pt` publication. This 500-step
gate did not write a numbered or `last.pt` checkpoint because periodic saves
occur every 1,000 steps. Full CPU reconstruction of the 1,331,002,343-byte
`best.pt` verified step/scheduler/tracker/validation 13,500, packed-loader
position and row position 864,000, total training FLOPs
`6.84652146720768e17`, both BPBs, the expected tokenized-data manifest and
validation identity, the pretraining stage, 76 model tensors, 75 optimizer
states, and W&B run ID `ngbuyhxj`. Training resumed at step 13,510 with loss
`4.106626` and 16.62K tokens/s. The post-gate W&B API query was synchronized
at step 13,500 with both validation values, run state `running`, and zero
uploaded artifacts.

At step 14,000, compatibility BPB improved by `0.007795` to `1.232974`, and
full-document BPB improved by `0.008671` to `1.269248`. Validation was observed
to enter by 15:39:42 EDT; `best.pt`, `step_014000.pt`, and `last.pt` were
durable by 16:00:43, 16:00:52, and 16:01:04, respectively, for an observed
entry-to-durable-last interval of approximately 21m 22s. Each file is
1,331,002,343 bytes. Full sequential CPU reconstruction of all three verified
step/scheduler/tracker/validation 14,000, packed-loader position and row
position 896,000, total training FLOPs `7.10009633636352e17`, the expected
tokenized-data manifest, both new BPBs, pretraining stage, 76 model tensors,
75 optimizer states, and W&B run ID `ngbuyhxj`. Training resumed at step
14,010 with loss `4.010866` and 16.65K tokens/s. W&B caught up to step 14,010
with both validation values, run state `running`, and zero uploaded artifacts.

Step 14,500 entered dual-BPB validation by 16:34:52 EDT with training loss
`4.046722`, gradient norm `0.541424`, throughput 16.61K tokens/s, and total
training FLOPs `7.35367120551936e17`. The complete step-14,010--14,500 window
averaged loss `4.074951`, improving on the prior complete-window mean
`4.115318`; all 50 rows were numeric, mean throughput was 16.615K tokens/s,
and the maximum gradient norm was `0.9399486`. Across all 551 logged rows from
step 9,000 through 14,500, gradient norms remained finite with median
`0.708107`, p95 `1.026229`, p99 `1.451016`, and maximum `3.697129`. The only
four values above 2.0 remain the previously documented steps 10,150, 10,850,
11,110, and 12,440; no new spike occurred. Compatibility BPB improved by
`0.010483` to `1.222491`, while full-document BPB improved by `0.010221` to
`1.259026`. The 1,331,002,343-byte `best.pt` was durable at 16:56:05 EDT, an
observed entry-to-durable interval of approximately 21m 13s. This 500-step
gate did not write a numbered or `last.pt` checkpoint because periodic saves
occur every 1,000 steps. Full CPU reconstruction verified step, scheduler,
tracker, and validation step 14,500; packed-loader position and row position
928,000; total training FLOPs `7.35367120551936e17`; the expected tokenized-
data manifest and validation identity; pretraining stage; 110,906,112 model
parameters; 76 model tensors; 75 optimizer states; and W&B run ID `ngbuyhxj`.
Training resumed through step 14,520 at about 16.62K tokens/s. W&B synchronized
to step 14,500 with both validation values, run state `running`, five small run
files, and zero uploaded artifacts.

Step 15,000 entered dual-BPB validation at 17:29:14 EDT with training loss
`4.021072`, gradient norm `0.745356`, throughput 16.634K tokens/s, LR
multiplier `1.0`, and total training FLOPs `7.6072460746752e17`. This is 50%
of the scheduled optimizer steps and 983,040,000 processed model tokens; the
token-normalized epoch field is `0.487297`. The complete step-14,510--15,000
window averaged loss `4.030666`, improving on `4.074951` for the preceding
window. All 50 rows were finite; mean throughput was 16.614K tokens/s and the
maximum gradient norm was `0.749396`. Across the 601 logged gradient rows from
step 9,000 through 15,000, the median was `0.688940`, p95 `1.015764`, p99
`1.451016`, and maximum `3.697129`; the same four historical rows exceeded
2.0, with no new spike. Compatibility BPB improved by `0.008895` to
`1.213596`, and full-document BPB improved by `0.009499` to `1.249528`.
`best.pt`, `step_015000.pt`, and `last.pt` became durable at 17:51:04,
17:51:14, and 17:51:24 EDT, respectively, for an observed entry-to-durable-
last interval of approximately 22m 10s. All three files are 1,331,002,343
bytes. Full sequential CPU reconstruction verified identical resume-critical
state: step/scheduler/tracker/validation 15,000; optimizer and scheduler LR
`3e-4`; packed-loader position and row position 960,000; expected manifest and
validation identities; total training FLOPs `7.6072460746752e17`; pretraining
stage; 110,906,112 model parameters; 76 model tensors; 75 optimizer states;
and W&B run ID `ngbuyhxj`. Training resumed at step 15,010 with loss `3.971377`,
LR multiplier `0.99943`, and 16.647K tokens/s, confirming the scheduled
warmdown transition. W&B synchronized to the halfway BPBs with run state
`running`, five small files, and zero uploaded artifacts.

Step 15,500 entered dual-BPB validation at 18:24:44 EDT with training loss
`3.955152`, gradient norm `0.562073`, throughput 16.614K tokens/s, LR
multiplier `0.968397`, and total training FLOPs `7.86082094383104e17`. The
complete first warmdown window, steps 15,010--15,500, averaged loss `3.996215`,
improving on `4.030666` for the preceding full-rate window. Its 50 rows were
finite, mean throughput was 16.620K tokens/s, and maximum gradient norm was
`1.839263` at step 15,300; clipping handled it and it did not add a fifth
greater-than-2.0 spike. All logged warmdown multipliers matched the closed-form
schedule within `1.11e-16`. Across the 651 gradient rows from step 9,000
through 15,500, median was `0.679088`, p95 `1.013403`, p99 `1.458531`, and
maximum `3.697129`;
the same four historical rows exceeded 2.0. Compatibility BPB improved by
`0.006556` to `1.207040`, and full-document BPB improved by `0.006190` to
`1.243338`. The 1,331,002,343-byte `best.pt` became durable at 18:46:24 EDT,
approximately 21m 40s after observed entry. This best-only gate wrote no
numbered or `last.pt` checkpoint because periodic saves occur every 1,000
steps. Full CPU reconstruction verified step/scheduler/tracker/validation
15,500; optimizer and scheduler LR `0.0002905`; packed-loader position and row
position 992,000; expected manifest and validation identities; total training
FLOPs `7.86082094383104e17`; pretraining stage; 110,906,112 parameters; 76
model tensors; 75 optimizer states; and W&B run ID `ngbuyhxj`. Training resumed
at step 15,510 with loss `3.994005`, LR multiplier `0.967763`, and 16.633K
tokens/s. W&B synchronized to the new gate with run state `running`, five small
files, and zero uploaded artifacts.

Step 16,000 entered dual-BPB validation at 19:19:40 EDT with training loss
`4.416873`, gradient norm `0.792678`, throughput 16.608K tokens/s, LR
multiplier `0.93673`, and total training FLOPs `8.11439581298688e17`. The
point loss is the largest in the step-15,510--16,000 window, but the complete
50-row mean declined from `3.996215` to `3.983534`; all rows were finite, mean
throughput was 16.611K tokens/s, and no gradient exceeded the step-16,000
value. Every logged warmdown multiplier through this boundary matched the
closed-form schedule within `1.11e-16`. Across 701 gradient rows from step
9,000 through 16,000, median was `0.667422`, p95 `1.007691`, p99 `1.451016`,
and maximum `3.697129`; the same four historical rows exceeded 2.0.
Compatibility BPB improved by `0.007209` to `1.199832`, and full-document BPB
improved by `0.006922` to `1.236416`. `best.pt`, `step_016000.pt`, and
`last.pt` became durable at 19:41:27, 19:41:37, and 19:41:48 EDT,
respectively, for an observed entry-to-durable-last interval of approximately
22m 08s. All three files are 1,331,002,343 bytes. Full sequential CPU
reconstruction verified identical resume-critical state:
step/scheduler/tracker/validation 16,000; optimizer and scheduler LR
`0.000281`; packed-loader position and row position 1,024,000; expected
manifest and validation identities; total training FLOPs
`8.11439581298688e17`; pretraining stage; 110,906,112 parameters; 76 model
tensors; 75 optimizer states; and W&B run ID `ngbuyhxj`. Training resumed at
step 16,010 with loss `4.127469`, LR multiplier `0.936097`, and 16.637K
tokens/s. W&B synchronized to the gate with run state `running`, five small
files, and zero uploaded artifacts.

Step 16,500 entered dual-BPB validation at 20:14:54 EDT with training loss
`3.871204`, gradient norm `0.518243`, throughput 16.621K tokens/s, LR
multiplier `0.905063`, and total training FLOPs `8.36797068214272e17`. The
complete step-16,010--16,500 window averaged loss `3.950307`, improving on
`3.983534` for the preceding complete window. All 50 rows were finite; mean
throughput was 16.616K tokens/s, and the maximum gradient norm was `0.757575`
at step 16,110. Every logged warmdown multiplier through this boundary matched
the closed-form schedule within `1.11e-16`. Across 751 gradient rows from step
9,000 through 16,500, median was `0.654431`, p95 `1.002118`, p99 `1.423839`,
and maximum `3.697129`; the same four historical rows exceeded 2.0. The two
BPBs improved by `0.008793` and `0.009391`, respectively, to compatibility
`1.191039` and full-document `1.227025`. `best.pt` became durable at 20:36:48
EDT, approximately 21m 54s after entry; no periodic or terminal checkpoint was
due at this 500-step-only boundary. The file is 1,331,002,343 bytes. Full CPU
reconstruction verified step/scheduler/tracker/validation 16,500; optimizer and
scheduler LR `0.0002715`; packed-loader position and row position 1,056,000;
expected manifest and validation identities; total training FLOPs
`8.36797068214272e17`; pretraining stage; 110,906,112 parameters; 76 model
tensors; 75 optimizer states; and W&B run ID `ngbuyhxj`. Training resumed at
step 16,510 with loss `3.881496`, LR multiplier `0.90443`, and 16.642K
tokens/s. W&B synchronized through that row with run state `running`, five
small files, and zero uploaded artifacts.

Step 17,000 entered dual-BPB validation at 21:09:53 EDT with training loss
`3.974002`, gradient norm `0.479789`, throughput 16.635K tokens/s, LR
multiplier `0.873397`, and total training FLOPs `8.62154555129856e17`. All 50
step-16,510--17,000 rows were finite and averaged 16.610K tokens/s. One
unusually easy batch at step 16,840 had loss `0.415052` and pre-clipping
gradient norm `4.796720`; throughput remained normal, the next logged loss was
`3.976934`, and gradient norm returned below 1.0 by step 16,860. Consequently,
the window arithmetic mean was `3.848478`, its median was `3.934275`, and a
sensitivity mean excluding only that outlier was `3.918548`; all three remain
below the preceding window's `3.950307`, but no training metric was actually
filtered. Across 801 gradient rows from step 9,000 through 17,000, median was
`0.640316`, p95 `1.000101`, p99 `1.451016`, and maximum `4.796720`; this is the
fifth row above 2.0. Every logged warmdown multiplier through this boundary
matched the closed-form schedule within `1.11e-16`. Compatibility BPB improved
by `0.007144` to `1.183895`, and full-document BPB improved by `0.005642` to
`1.221383`; both series have now improved strictly at all 34 gates.
`best.pt`, `step_017000.pt`, and `last.pt` became durable at 21:31:49,
21:32:00, and 21:32:10 EDT, respectively, for an observed entry-to-durable-last
interval of approximately 22m 17s. All three files are 1,331,002,343 bytes.
Full sequential CPU reconstruction verified identical critical state:
step/scheduler/tracker/validation 17,000; optimizer and scheduler LR
`0.000262`; packed-loader position and row position 1,088,000; expected
manifest and validation identities; total training FLOPs
`8.62154555129856e17`; pretraining stage; 110,906,112 parameters; 76 model
tensors; 75 optimizer states; and W&B run ID `ngbuyhxj`. Training resumed at
step 17,010 with loss `3.887802`, LR multiplier `0.872763`, and 16.669K
tokens/s. W&B synchronized through that row with the exact BPBs, run state
`running`, five small files, and zero uploaded artifacts.

Step 17,500 entered dual-BPB validation at 22:05:14 EDT with training loss
`3.929624`, gradient norm `0.518391`, throughput 16.621K tokens/s, LR
multiplier `0.84173`, and total training FLOPs `8.8751204204544e17`. All 50
step-17,010--17,500 rows were finite, with loss range `3.696765`--`4.168075`,
mean `3.889750`, median `3.868427`, and mean throughput 16.609K tokens/s. This
mean improves on the preceding window's outlier-excluded sensitivity value
`3.918548`; it is appropriately higher than that window's outlier-depressed
raw mean. The window's maximum gradient norm was `0.670767` at step 17,150.
Across 851 gradient rows from step 9,000 through 17,500, median was `0.631046`,
p95 `0.992896`, p99 `1.423839`, and maximum `4.796720`; no new row joined the
five historical values above 2.0. Every logged warmdown multiplier through
this boundary matched the closed-form schedule within `1.11e-16`.
Compatibility BPB improved by `0.005850` to `1.178044`, and full-document BPB
improved by `0.008134` to `1.213249`; both series have improved strictly at all
35 gates. The 1,331,002,343-byte `best.pt` became durable at 22:27:11 EDT,
approximately 21m 57s after entry; no periodic or terminal checkpoint was due
at this 500-step-only boundary. Full CPU reconstruction verified
step/scheduler/tracker/validation 17,500; optimizer and scheduler LR
`0.0002525`; packed-loader position and row position 1,120,000; expected
manifest and validation identities; total training FLOPs
`8.8751204204544e17`; pretraining stage; 110,906,112 parameters; 76 model
tensors; 75 optimizer states; and W&B run ID `ngbuyhxj`. Training resumed at
step 17,510 with loss `3.921936`, LR multiplier `0.841097`, and 16.625K
tokens/s. W&B synchronized through that row with the exact BPBs, run state
`running`, five small files, and zero uploaded artifacts.

Step 18,000 entered dual-BPB validation at 23:00:15 EDT with training loss
`4.107933`, gradient norm `0.468042`, throughput 16.606K tokens/s, LR
multiplier `0.810063`, and total training FLOPs `9.12869528961024e17`. All 50
step-17,510--18,000 rows were finite, with loss range `3.731288`--`4.107933`,
mean `3.886593`, median `3.869556`, and mean throughput 16.603K tokens/s. The
mean improved slightly from `3.889750` while the median was effectively flat.
The window's maximum gradient norm was `0.670230` at step 17,790. Across 901
gradient rows from step 9,000 through 18,000, median was `0.621418`, p95
`0.988095`, p99 `1.396663`, and maximum `4.796720`; no new row joined the five
historical values above 2.0. Every logged warmdown multiplier through this
boundary matched the closed-form schedule within `1.11e-16`. Compatibility
BPB improved by `0.006848` to `1.171197`, and full-document BPB improved by
`0.006155` to `1.207094`; both series have improved strictly at all 36 gates.
`best.pt`, `step_018000.pt`, and `last.pt` became durable at 23:22:12,
23:22:22, and 23:22:33 EDT, respectively, for an observed entry-to-durable-last
interval of approximately 22m 18s. All three files are 1,331,002,343 bytes.
Full sequential CPU reconstruction verified identical critical state:
step/scheduler/tracker/validation 18,000; optimizer and scheduler LR
`0.000243`; packed-loader position and row position 1,152,000; expected
manifest and validation identities; total training FLOPs
`9.12869528961024e17`; pretraining stage; 110,906,112 parameters; 76 model
tensors; 75 optimizer states; and W&B run ID `ngbuyhxj`. Training resumed at
step 18,010 with loss `3.855928`, LR multiplier `0.80943`, and 16.642K
tokens/s. W&B synchronized through that row with the exact BPBs, run state
`running`, five small files, and zero uploaded artifacts.

Step 18,500 entered dual-BPB validation at 23:55:37 EDT with training loss
`3.745098`, gradient norm `0.492046`, throughput 16.616K tokens/s, LR
multiplier `0.778397`, and total training FLOPs `9.38227015876608e17`. All 50
step-18,010--18,500 rows were finite, with loss range `3.526724`--`4.420314`,
mean `3.847935`, median `3.834223`, and mean throughput 16.605K tokens/s. Both
mean and median improved from the preceding window. The window's maximum
gradient norm was `0.687141` at step 18,090. Across 951 gradient rows from step
9,000 through 18,500, median was `0.612455`, p95 `0.981726`, p99 `1.358588`,
and maximum `4.796720`; no new row joined the five historical values above
2.0. Every logged warmdown multiplier through this boundary matched the
closed-form schedule within `1.11e-16`. Compatibility BPB improved by
`0.005679` to `1.165518`, and full-document BPB improved by `0.005416` to
`1.201678`; both series have improved strictly at all 37 gates. The
1,331,002,343-byte `best.pt` became durable at 00:17:34 EDT, approximately
21m 57s after entry; no periodic or terminal checkpoint was due at this
500-step-only boundary. Full CPU reconstruction verified
step/scheduler/tracker/validation 18,500; optimizer and scheduler LR
`0.0002335`; packed-loader position and row position 1,184,000; expected
manifest and validation identities; total training FLOPs
`9.38227015876608e17`; pretraining stage; 110,906,112 parameters; 76 model
tensors; 75 optimizer states; and W&B run ID `ngbuyhxj`. Training resumed at
step 18,510 with loss `3.745628`, LR multiplier `0.777763`, and 16.619K
tokens/s. W&B synchronized through that row with the exact BPBs, run state
`running`, five small files, and zero uploaded artifacts.

Step 19,000 entered dual-BPB validation at 00:50:40 EDT with training loss
`3.851731`, gradient norm `0.501283`, throughput 16.596K tokens/s, LR
multiplier `0.74673`, and total training FLOPs `9.63584502792192e17`. All 50
step-18,510--19,000 rows were finite, with loss range `3.611170`--`4.207469`,
mean `3.836103`, median `3.837784`, and mean throughput 16.608K tokens/s. The
mean improved from the preceding window while the median increased slightly.
The window's maximum gradient norm was `0.553615` at step 18,730. Across 1,001
gradient rows from step 9,000 through 19,000, median was `0.600853`, p95
`0.969321`, p99 `1.320512`, and maximum `4.796720`; no new row joined the five
historical values above 2.0. Every logged warmdown multiplier through this
boundary matched the closed-form schedule within `1.11e-16`. Compatibility
BPB improved by `0.007551` to `1.157966`, and full-document BPB improved by
`0.008578` to `1.193100`; both series have improved strictly at all 38 gates.
The 1,331,002,343-byte `best.pt`, `step_019000.pt`, and `last.pt` became durable
at 01:12:37, 01:12:49, and 01:13:02 EDT, respectively, approximately 22m 22s
from entry to durable `last.pt`. Full sequential CPU reconstruction found
identical critical state in all three files: step/scheduler/tracker/validation
19,000; optimizer and scheduler LR `0.000224`; packed-loader position and row
position 1,216,000; expected manifest and validation identities; total training
FLOPs `9.63584502792192e17`; pretraining stage; 110,906,112 parameters; 76
model tensors; 75 optimizer states; and W&B run ID `ngbuyhxj`. Training resumed
at step 19,010 with loss `3.830538`, LR multiplier `0.746097`, and 16.680K
tokens/s. W&B synchronized through that row with the exact BPBs, run state
`running`, five small files, and zero uploaded artifacts.

Step 19,500 entered dual-BPB validation at 01:46:15 EDT with training loss
`3.802067`, gradient norm `0.457546`, throughput 16.597K tokens/s, LR
multiplier `0.715063`, and total training FLOPs `9.88941989707776e17`. All 50
step-19,010--19,500 rows were finite, with loss range `3.142054`--`4.089184`,
mean `3.781057`, median `3.795575`, and mean throughput 16.607K tokens/s. Both
mean and median improved from the preceding window. The window's maximum
gradient norm was `0.957561` at step 19,410. Across 1,051 gradient rows from
step 9,000 through 19,500, median was `0.595563`, p95 `0.961933`, p99
`1.320329`, and maximum `4.796720`; no new row joined the five historical
values above 2.0. Every logged warmdown multiplier through this boundary
matched the closed-form schedule within `1.11e-16`. Compatibility BPB improved
by `0.004074` to `1.153893`, and full-document BPB improved by `0.003571` to
`1.189530`; both series have improved strictly at all 39 gates. The
1,331,002,343-byte `best.pt` became durable at 02:08:10 EDT, approximately
21m 55s after entry; no periodic or terminal checkpoint was due at this
500-step-only boundary. Full CPU reconstruction verified
step/scheduler/tracker/validation 19,500; optimizer and scheduler LR
`0.0002145`; packed-loader position and row position 1,248,000; expected
manifest and validation identities; total training FLOPs
`9.88941989707776e17`; pretraining stage; 110,906,112 parameters; 76 model
tensors; 75 optimizer states; and W&B run ID `ngbuyhxj`. Training resumed at
step 19,510 with loss `3.740071`, LR multiplier `0.71443`, and 16.601K tokens/s.
W&B synchronized through that row with the exact BPBs, run state `running`,
five small files, and zero uploaded artifacts.

Step 20,000 entered dual-BPB validation at 02:41:16 EDT with training loss
`4.201722`, gradient norm `0.585925`, throughput 16.596K tokens/s, LR
multiplier `0.683397`, and total training FLOPs `1.01429947662336e18`. All 50
step-19,510--20,000 rows were finite, with loss range `3.420005`--`4.201722`,
mean `3.772247`, median `3.781346`, and mean throughput 16.606K tokens/s. Both
mean and median improved from the preceding window despite the noisy final
batch. The window's maximum gradient norm was `0.604941` at step 19,550.
Across 1,101 gradient rows from step 9,000 through 20,000, median was
`0.586255`, p95 `0.956745`, p99 `1.320146`, and maximum `4.796720`; no new row
joined the five historical values above 2.0. Every logged warmdown multiplier
through this boundary matched the closed-form schedule within `1.11e-16`.
Compatibility BPB improved by `0.002555` to `1.151338`, and full-document BPB
improved by `0.003120` to `1.186410`; both series have improved strictly at all
40 gates. The 1,331,002,343-byte `best.pt`, `step_020000.pt`, and `last.pt`
became durable at 03:03:14, 03:03:23, and 03:03:34 EDT, respectively,
approximately 22m 18s from entry to durable `last.pt`. Full sequential CPU
reconstruction found identical critical state in all three files:
step/scheduler/tracker/validation 20,000; optimizer and scheduler LR `0.000205`;
packed-loader position and row position 1,280,000; expected manifest and
validation identities; total training FLOPs `1.01429947662336e18`; pretraining
stage; 110,906,112 parameters; 76 model tensors; 75 optimizer states; and W&B
run ID `ngbuyhxj`. Training resumed at step 20,010 with loss `3.781255`, LR
multiplier `0.682763`, and 16.643K tokens/s. W&B synchronized through that row
with the exact BPBs, run state `running`, five small files, and zero uploaded
artifacts.

Step 20,500 entered dual-BPB validation at 03:36:41 EDT with training loss
`3.769477`, gradient norm `0.527373`, throughput 16.608K tokens/s, LR
multiplier `0.651730`, and total training FLOPs `1.039656963538944e18`. All 50
step-20,010--20,500 rows were finite, with loss range
`3.493636`--`3.911407`, mean `3.758186`, median `3.762661`, and mean throughput
16.602K tokens/s; both mean and median improved from the preceding window. The
window's maximum gradient norm was `1.016214` at step 20,160. Across 1,151
gradient rows from step 9,000 through 20,500, median was `0.576415`, p95
`0.953165`, p99 `1.319190`, and maximum `4.796720`; no new row joined the five
historical values above 2.0. Every logged warmdown multiplier through this
boundary matched the closed-form schedule within `1.11e-16`. Compatibility BPB
improved by `0.004349` to `1.146989`, and full-document BPB improved by
`0.004233` to `1.182177`; both series have improved strictly at all 41 gates.
The 1,331,002,343-byte `best.pt` became durable at 03:58:38 EDT, approximately
21m 57s after entry; no periodic or terminal checkpoint was due at this
500-step-only boundary. Full CPU reconstruction verified
step/scheduler/tracker/validation 20,500; optimizer and scheduler LR
`0.0001955`; packed-loader position and row position 1,312,000; expected
manifest and validation identities; total training FLOPs
`1.039656963538944e18`; pretraining stage; 110,906,112 parameters; 76 model
tensors; 75 optimizer states; and W&B run ID `ngbuyhxj`. Training resumed at
step 20,510 with loss `3.674820`, LR multiplier `0.651097`, and 16.633K
tokens/s. A settled local audit through step 20,520 found 2,175 complete
newline-terminated records, the full 10-step training grid, 41 validation rows,
finite and monotonic metrics, and 81 unique checkpoint events. W&B synchronized
through step 20,510 with the exact BPBs, run state `running`, five small files,
and zero uploaded artifacts. Checkpoint retention remained 22 files
(approximately 27.3 GiB), with 967 GB free.

Step 21,000 entered dual-BPB validation at 04:31:47 EDT with training loss
`3.968399`, gradient norm `0.534121`, throughput 16.609K tokens/s, LR
multiplier `0.620063`, and total training FLOPs `1.065014450454528e18`. All 50
step-20,510--21,000 rows were finite, with loss range
`2.845719`--`4.018404`, mean `3.742003`, median `3.744022`, and mean throughput
16.600K tokens/s; both mean and median improved from the preceding window. The
window's maximum gradient norm was `1.052912` at step 20,850. Across 1,201
gradient rows from step 9,000 through 21,000, median was `0.568962`, p95
`0.947818`, p99 `1.318234`, and maximum `4.796720`; no new row joined the five
historical values above 2.0. Every logged warmdown multiplier through this
boundary matched the closed-form schedule within `1.11e-16`. Compatibility BPB
improved by `0.007126` to `1.139862`, and full-document BPB improved by
`0.004801` to `1.177376`; both series have improved strictly at all 42 gates.
The 1,331,002,343-byte `best.pt`, `step_021000.pt`, and `last.pt` became durable
at 04:53:42, 04:53:56, and 04:54:08 EDT, respectively, approximately 22m 21s
from entry through durable `last.pt`. Full sequential CPU reconstruction found
identical critical state in all three files: step/scheduler/tracker/validation
21,000; optimizer and scheduler LR `0.000186`; packed-loader position and row
position 1,344,000; expected manifest and validation identities; total training
FLOPs `1.065014450454528e18`; pretraining stage; 110,906,112 parameters; 76
model tensors; 75 optimizer states; and W&B run ID `ngbuyhxj`. Training resumed
at step 21,010 with loss `3.814822`, LR multiplier `0.619430`, and 16.665K
tokens/s. A settled local audit through step 21,020 found 2,229 complete
newline-terminated records, the full 10-step training grid, 42 validation rows,
finite and monotonic metrics, and 84 unique checkpoint events. W&B synchronized
through step 21,010 with the exact BPBs, run state `running`, five small files,
and zero uploaded artifacts. Checkpoint retention increased to 23 files
(approximately 28.5 GiB), with 966 GB free.

Step 21,500 entered dual-BPB validation at 05:27:16 EDT with training loss
`3.668236`, gradient norm `0.463731`, throughput 16.609K tokens/s, LR
multiplier `0.588397`, and total training FLOPs `1.090371937370112e18`. All 50
step-21,010--21,500 rows were finite, with loss range
`3.396114`--`3.947486`, mean `3.741130`, median `3.746887`, and mean throughput
16.609K tokens/s. The mean improved marginally from the preceding window while
the median rose by `0.002866`. The window's maximum gradient norm was
`0.718156` at step 21,200. Across 1,251 gradient rows from step 9,000 through
21,500, median was `0.561672`, p95 `0.936444`, p99 `1.312226`, and maximum
`4.796720`; no new row joined the five historical values above 2.0. Every
logged warmdown multiplier through this boundary matched the closed-form
schedule within `1.11e-16`. Compatibility BPB improved by `0.003089` to
`1.136773`, and full-document BPB improved by `0.005111` to `1.172265`; both
series have improved strictly at all 43 gates. The 1,331,002,343-byte
`best.pt` became durable at 05:49:13 EDT, approximately 21m 57s after entry; no
periodic or terminal checkpoint was due at this 500-step-only boundary. Full
CPU reconstruction verified step/scheduler/tracker/validation 21,500;
optimizer and scheduler LR `0.0001765`; packed-loader position and row position
1,376,000; expected manifest and validation identities; total training FLOPs
`1.090371937370112e18`; pretraining stage; 110,906,112 parameters; 76 model
tensors; 75 optimizer states; and W&B run ID `ngbuyhxj`. Training resumed at
step 21,510 with loss `3.719697`, LR multiplier `0.587763`, and 16.627K
tokens/s. A settled local audit through step 21,530 found 2,282 complete
newline-terminated records, the full 10-step training grid, 43 validation rows,
finite and monotonic metrics, and 85 unique checkpoint events. W&B synchronized
through step 21,520 with the exact BPBs, run state `running`, five small files,
and zero uploaded artifacts. Checkpoint retention remained 23 files
(approximately 28.5 GiB), with 966 GB free.

Step 22,000 entered dual-BPB validation at 06:22:18 EDT with training loss
`3.682498`, gradient norm `0.449968`, throughput 16.610K tokens/s, LR
multiplier `0.556730`, and total training FLOPs `1.115729424285696e18`. All 50
step-21,510--22,000 rows were finite, with loss range
`3.302727`--`3.891446`, mean `3.720873`, median `3.737093`, and mean throughput
16.604K tokens/s; both mean and median improved from the preceding window. The
window's maximum gradient norm was `0.744432` at step 21,560. Across 1,301
gradient rows from step 9,000 through 22,000, median was `0.553615`, p95
`0.922458`, p99 `1.306217`, and maximum `4.796720`; no new row joined the five
historical values above 2.0. Every logged warmdown multiplier through this
boundary matched the closed-form schedule within `1.11e-16`. Compatibility BPB
improved by `0.005681` to `1.131092`, and full-document BPB improved by
`0.006274` to `1.165991`; both series have improved strictly at all 44 gates.
The 1,331,002,343-byte `best.pt`, `step_022000.pt`, and `last.pt` became durable
at 06:44:15, 06:44:26, and 06:44:36 EDT, respectively, approximately 22m 18s
from entry through durable `last.pt`. Full sequential CPU reconstruction found
identical critical state in all three files: step/scheduler/tracker/validation
22,000; optimizer and scheduler LR `0.000167`; packed-loader position and row
position 1,408,000; expected manifest and validation identities; total training
FLOPs `1.115729424285696e18`; pretraining stage; 110,906,112 parameters; 76
model tensors; 75 optimizer states; and W&B run ID `ngbuyhxj`. Training resumed
at step 22,010 with loss `3.722515`, LR multiplier `0.556097`, and 16.644K
tokens/s. A settled local audit through step 22,030 found 2,336 complete
newline-terminated records, the full 10-step training grid, 44 validation rows,
finite and monotonic metrics, and 88 unique checkpoint events. W&B synchronized
through step 22,020 with the exact BPBs, run state `running`, five small files,
and zero uploaded artifacts. Checkpoint retention increased to 24 files
(approximately 29.8 GiB), with 965 GB free.

Step 22,500 entered dual-BPB validation at 07:17:42 EDT with training loss
`3.696497`, gradient norm `0.478224`, throughput 16.599K tokens/s, LR
multiplier `0.525063`, and total training FLOPs `1.14108691120128e18`. All 50
step-22,010--22,500 rows were finite, with loss range
`3.317015`--`4.022859`, mean `3.741128`, median `3.736093`, and mean throughput
16.608K tokens/s. The mean rose by `0.020254` from the preceding window while
the median improved slightly. The window's maximum gradient norm was `1.475930`
at step 22,230. Across 1,351 gradient rows from step 9,000 through 22,500,
median was `0.546142`, p95 `0.916072`, p99 `1.312226`, and maximum `4.796720`;
no new row joined the five historical values above 2.0. Every logged warmdown
multiplier through this boundary matched the closed-form schedule within
`1.11e-16`. Compatibility BPB improved by `0.001612` to `1.129480`, remaining
strictly improved at all 45 gates. Full-document BPB increased by `0.000180` to
`1.166171`, its first gate-to-gate regression, so its independent minimum
remained `1.165991` at step 22,000. Because compatibility BPB is the pinned
ranking metric, the 1,331,002,343-byte `best.pt` correctly advanced and became
durable at 07:39:38 EDT, approximately 21m 56s after entry; no periodic or
terminal checkpoint was due. Full CPU reconstruction verified
step/scheduler/tracker/validation 22,500; current/minimum compatibility BPB
`1.129480`/`1.129480`; current/minimum full-document BPB
`1.166171`/`1.165991`; optimizer and scheduler LR `0.0001575`; packed-loader
position and row position 1,440,000; expected manifest and validation
identities; total training FLOPs `1.14108691120128e18`; pretraining stage;
110,906,112 parameters; 76 model tensors; 75 optimizer states; and W&B run ID
`ngbuyhxj`. Training resumed at step 22,510 with loss `3.618581`, LR multiplier
`0.524430`, and 16.622K tokens/s. A settled local audit through step 22,530
found 2,388 complete newline-terminated records, the full 10-step training
grid, 45 validation rows, finite and monotonic metrics, and 89 unique checkpoint
events. W&B synchronized through step 22,520 with the exact current and minimum
BPBs, run state `running`, five small files, and zero uploaded artifacts.
Checkpoint retention remained 24 files (approximately 29.8 GiB), with 965 GB
free.

Step 23,000 entered dual-BPB validation at 08:12:49 EDT with training loss
`3.539738`, gradient norm `0.473929`, throughput 16.596K tokens/s, LR
multiplier `0.493397`, and total training FLOPs `1.166444398116864e18`. All 50
step-22,510--23,000 rows were finite, with loss range
`2.293702`--`4.417295`, mean `3.701603`, median `3.704086`, and mean throughput
16.603K tokens/s. The mean and median improved by `0.039524` and `0.032007`,
respectively, from the preceding window. Step 22,710 was an isolated easy batch
with loss `2.293702` and pre-clip gradient norm `2.923918`; step 22,720
immediately returned to loss `3.827365` and gradient norm `0.540726`, and the
following rows remained ordinary. Across 1,401 gradient rows from step 9,000
through 23,000, median was `0.539956`, p95 `0.914680`, p99 `1.318234`, and
maximum `4.796720`; the step-22,710 transient raised the count above 2.0 from
five to six. Every logged warmdown multiplier through this boundary matched
the closed-form schedule within `1.11e-16`. Compatibility BPB improved by
`0.004530` to `1.124950`, remaining strictly improved at all 46 gates.
Full-document BPB improved by `0.006646` from the preceding gate and by
`0.006465` from its prior minimum, setting a new minimum of `1.159525`. The
compatibility-ranked `best.pt` was therefore correctly promoted at 08:34:45
EDT, approximately 21m 56s after entry; `step_023000.pt` followed at 08:34:54
and `last.pt` at 08:35:03, approximately 22m 14s after entry. All three files
were 1,331,002,343 bytes. Sequential full CPU reconstruction proved their
critical states equal: step/scheduler/tracker/validation 23,000; scheduler step
count 23,001; current/minimum compatibility BPB `1.124950`/`1.124950`;
current/minimum full-document BPB `1.159525`/`1.159525`; optimizer and scheduler
LR `0.000148`; packed-loader position and row position 1,472,000; expected
manifest and validation identities; total training FLOPs
`1.166444398116864e18`; pretraining stage; 110,906,112 parameters; 76 model
tensors; 75 optimizer states; and W&B run ID `ngbuyhxj`. Training resumed at
step 23,010 with loss `3.651335`, LR multiplier `0.492763`, and 16.643K
tokens/s. A settled local audit through step 23,020 found 2,441 complete
newline-terminated records, the full 10-step training grid, 46 validation rows,
finite and monotonic metrics, and 92 unique checkpoint events. W&B synchronized
through step 23,020 with the exact current and minimum BPBs, run state
`running`, five small files, and zero uploaded artifacts. Checkpoint retention
increased to 25 files totaling 33,275,058,575 bytes (approximately 31.0 GiB),
with 964 GB free.

Step 23,500 entered dual-BPB validation at 09:08:10 EDT with training loss
`3.568047`, gradient norm `0.464810`, throughput 16.626K tokens/s, LR
multiplier `0.461730`, cumulative training FLOPs `1.191801885032448e18`, and
optimizer timer `92678.526313` seconds. All 50 step-23,010--23,500 rows were
finite and complete on the 10-step grid, with loss range
`3.370876`--`3.948747`, mean `3.698908`, median `3.701723`, and mean throughput
16.607K tokens/s. The mean and median improved by `0.002695` and `0.002363`,
respectively, from the preceding window. This window's maximum pre-clipping
gradient norm was `0.698788` at step 23,190. Across 1,451 gradient rows from
step 9,000 through 23,500, median was `0.534662`, p95 `0.910294`, p99
`1.312226`, and maximum `4.796720` at step 16,840; no new value exceeded 2.0,
so the six isolated outliers remained unchanged. Every logged warmdown
multiplier from step 15,000 through this boundary matched the closed-form
schedule within `1.11e-16`. Compatibility BPB improved by `0.004150` to
`1.120800`, and full-document BPB improved by `0.004742` to `1.154783`; both
were new minima. The compatibility-ranked `best.pt` was therefore correctly
promoted at 09:30:05 EDT, approximately 21m 55s after entry. Training resumed
at 09:30:55 with step-23,510 loss `3.407630`, LR multiplier `0.461097`, and
16.609K tokens/s. Full CPU reconstruction verified step/scheduler/tracker/
validation 23,500; scheduler step count 23,501; current/minimum compatibility
BPB `1.120800`/`1.120800`; current/minimum full-document BPB
`1.154783`/`1.154783`; optimizer and scheduler LR `0.0001385`; packed-loader
position and row position 1,504,000; expected manifest and validation
identities; total training FLOPs `1.191801885032448e18`; pretraining stage;
110,906,112 parameters; 76 model tensors; 75 optimizer states; and W&B run ID
`ngbuyhxj`. A settled local audit through step 23,530 found 2,494 complete
newline-terminated records, all 2,353 training rows on the step-10 grid from
10 through 23,530, 47 validation rows, finite and monotonic metrics, and 93
unique checkpoint events. W&B held the exact step-23,500 validation values,
run state `running`, five small files, and zero uploaded artifacts. Checkpoint
retention remained 25 files totaling 33,275,058,575 bytes (approximately 31.0
GiB), with 964 GB free.

Step 24,000 entered dual-BPB validation at 10:03:17 EDT with training loss
`3.571117`, gradient norm `0.467690`, throughput 16.623K tokens/s, LR
multiplier `0.430063`, cumulative training FLOPs `1.217159371948032e18`, and
optimizer timer `94650.876750` seconds. All 50 step-23,510--24,000 rows were
finite and complete on the 10-step grid, with loss range
`3.380820`--`4.234838`, mean `3.688268`, median `3.689858`, and mean throughput
16.613K tokens/s. The mean and median improved by `0.010640` and `0.011865`,
respectively, from the preceding window. This window's maximum pre-clipping
gradient norm was `0.586497` at step 23,520. Across 1,501 gradient rows from
step 9,000 through 24,000, median was `0.529547`, p95 `0.909297`, p99
`1.306217`, and maximum `4.796720` at step 16,840; the same six isolated rows
exceeded 2.0, with none new. Every logged warmdown multiplier through this
boundary matched the closed-form schedule within `1.11e-16`. Compatibility BPB
improved by `0.003178` to `1.117621`, and full-document BPB improved by
`0.001605` to `1.153179`; both were new minima. The compatibility-ranked
`best.pt` was promoted at 10:25:06 EDT, approximately 21m 49s after entry;
`step_024000.pt` followed at 10:25:18 and `last.pt` at 10:25:31,
approximately 22m 14s after entry. All three files were 1,331,002,343 bytes.
Sequential full CPU reconstruction proved their critical states equal: step,
scheduler, tracker, and validation 24,000; scheduler step count 24,001;
current/minimum compatibility BPB `1.117621`/`1.117621`; current/minimum
full-document BPB `1.153179`/`1.153179`; optimizer and scheduler LR
`0.000129`; packed-loader position and row position 1,536,000; expected
manifest and validation identities; total training FLOPs
`1.217159371948032e18`; pretraining stage; 110,906,112 parameters; 76 model
tensors; 75 optimizer states; and W&B run ID `ngbuyhxj`. Training resumed at
10:26:29 with step-24,010 loss `3.633157`, LR multiplier `0.429430`, and
16.686K tokens/s. A settled local audit through step 24,020 found 2,547
complete newline-terminated records, all 2,402 training rows on the step-10
grid from 10 through 24,020, 48 validation rows, finite and monotonic metrics,
and 96 unique checkpoint events. W&B held the exact step-24,000 values, run
state `running`, five small files, and zero uploaded artifacts. Checkpoint
retention increased to 26 files totaling 34,606,060,918 bytes (approximately
32.2 GiB), with 963 GB free.

Step 24,500 entered dual-BPB validation at 10:58:38 EDT with training loss
`3.851119`, gradient norm `0.493012`, throughput 16.599K tokens/s, LR
multiplier `0.398397`, cumulative training FLOPs `1.242516858863616e18`, and
optimizer timer `96622.914818` seconds. All 50 step-24,010--24,500 rows were
finite and complete on the 10-step grid, with loss range
`3.529440`--`4.079950`, mean `3.687776`, median `3.676737`, and mean throughput
16.614K tokens/s. The mean improved marginally by `0.000492`, while the median
improved by `0.013121` from the preceding window. This window's maximum
pre-clipping gradient norm was `0.699274` at step 24,290. Across 1,551 gradient
rows from step 9,000 through 24,500, median was `0.525085`, p95 `0.904686`, p99
`1.284247`, and maximum `4.796720` at step 16,840; the same six isolated rows
exceeded 2.0, with none new. Every logged warmdown multiplier through this
boundary matched the closed-form schedule within `1.11e-16`. Compatibility BPB
improved by `0.003738` to `1.113883`, and full-document BPB improved by
`0.005346` to `1.147832`; both were new minima. The compatibility-ranked
`best.pt` was therefore promoted at 11:20:36 EDT, approximately 21m 58s after
entry. Full CPU reconstruction verified step, scheduler, tracker, and
validation 24,500; scheduler step count 24,501; current/minimum compatibility
BPB `1.113883`/`1.113883`; current/minimum full-document BPB
`1.147832`/`1.147832`; optimizer and scheduler LR `0.0001195`; packed-loader
position and row position 1,568,000; expected manifest and validation
identities; total training FLOPs `1.242516858863616e18`; pretraining stage;
110,906,112 parameters; 76 model tensors; 75 optimizer states; and W&B run ID
`ngbuyhxj`. Training resumed at 11:21:31 with step-24,510 loss `3.688686`, LR
multiplier `0.397763`, and 16.648K tokens/s. A settled local audit through step
24,520 found 2,599 complete newline-terminated records, all 2,452 training rows
on the step-10 grid from 10 through 24,520, 49 validation rows, finite and
monotonic metrics, and 97 unique checkpoint events. W&B held the exact
step-24,500 values, run state `running`, five small files, and zero uploaded
artifacts. Checkpoint retention remained 26 files totaling 34,606,060,918
bytes (approximately 32.2 GiB), with 963 GB free.

Step 25,000 entered dual-BPB validation at 11:53:41 EDT with training loss
`3.550405`, gradient norm `0.479394`, throughput 16.631K tokens/s, LR
multiplier `0.366730`, cumulative training FLOPs `1.2678743457792e18`, and
optimizer timer `98595.116320` seconds. All 50 step-24,510--25,000 rows were
finite and complete on the 10-step grid, with loss range
`3.470174`--`3.836337`, mean `3.648065`, median `3.640162`, and mean throughput
16.613K tokens/s. The mean and median improved by `0.039710` and `0.036575`,
respectively, from the preceding window. This window's maximum pre-clipping
gradient norm was `1.032850` at step 24,900. Across 1,601 gradient rows from
step 9,000 through 25,000, median was `0.522281`, p95 `0.903566`, p99
`1.262277`, and maximum `4.796720` at step 16,840; the same six isolated rows
exceeded 2.0, with none new. Every logged warmdown multiplier through this
boundary matched the closed-form schedule within `1.11e-16`. Compatibility BPB
improved by `0.002595` to `1.111288`, remaining strictly improved at every
gate. Full-document BPB rose by `0.000789` to `1.148621`, while its minimum
correctly remained `1.147832` from step 24,500. The compatibility-ranked
`best.pt` was promoted at 12:15:37 EDT, approximately 21m 56s after entry;
`step_025000.pt` followed at 12:15:47 and `last.pt` at 12:15:57,
approximately 22m 16s after entry. All three files were 1,331,002,343 bytes.
Sequential full CPU reconstruction proved their critical states equal: step,
scheduler, tracker, and validation 25,000; scheduler step count 25,001;
current/minimum compatibility BPB `1.111288`/`1.111288`; current/minimum
full-document BPB `1.148621`/`1.147832`; optimizer and scheduler LR `0.00011`;
packed-loader position and row position 1,600,000; expected manifest and
validation identities; total training FLOPs `1.2678743457792e18`; pretraining
stage; 110,906,112 parameters; 76 model tensors; 75 optimizer states; and W&B
run ID `ngbuyhxj`. Training resumed at 12:16:53 with step-25,010 loss
`3.612008`, LR multiplier `0.366097`, and 16.656K tokens/s. A settled local
audit through step 25,020 found 2,653 complete newline-terminated records, all
2,502 training rows on the step-10 grid from 10 through 25,020, 50 validation
rows, finite and monotonic metrics, and 100 unique checkpoint events. W&B held
the exact step-25,000 current and minimum values, run state `running`, five
small files, and zero uploaded artifacts. Checkpoint retention increased to 27
files totaling 35,937,063,261 bytes (approximately 33.5 GiB), with 962 GB free.

Step 25,500 entered dual-BPB validation at 12:49:01 EDT with training loss
`3.707666`, gradient norm `0.462795`, throughput 16.643K tokens/s, LR
multiplier `0.335063`, cumulative training FLOPs `1.293231832694784e18`, and
optimizer timer `100566.855227` seconds. All 50 step-25,010--25,500 rows were
finite and complete on the 10-step grid, with loss range
`3.424618`--`3.875909`, mean `3.647485`, median `3.640877`, and mean throughput
16.616K tokens/s. The mean improved marginally by `0.000580`, while median rose
by only `0.000715` from the preceding window. This window's maximum
pre-clipping gradient norm was `1.214433` at step 25,020. Across 1,651 gradient
rows from step 9,000 through 25,500, median was `0.520232`, p95 `0.901932`, p99
`1.246117`, and maximum `4.796720` at step 16,840; the same six isolated rows
exceeded 2.0, with none new. Every logged warmdown multiplier through this
boundary matched the closed-form schedule within `1.11e-16`. Compatibility BPB
improved by `0.003390` to `1.107898`. Full-document BPB improved by `0.006621`
from the preceding gate and by `0.005832` from its prior minimum, setting a new
minimum of `1.142000`. The compatibility-ranked `best.pt` was therefore
promoted at 13:10:58 EDT, approximately 21m 57s after entry. Full CPU
reconstruction verified step, scheduler, tracker, and validation 25,500;
scheduler step count 25,501; current/minimum compatibility BPB
`1.107898`/`1.107898`; current/minimum full-document BPB
`1.142000`/`1.142000`; optimizer and scheduler LR `0.0001005`; packed-loader
position and row position 1,632,000; expected manifest and validation
identities; total training FLOPs `1.293231832694784e18`; pretraining stage;
110,906,112 parameters; 76 model tensors; 75 optimizer states; and W&B run ID
`ngbuyhxj`. Training resumed at 13:11:53 with step-25,510 loss `3.523238`, LR
multiplier `0.334430`, and 16.631K tokens/s. A settled local audit through step
25,520 found 2,705 complete newline-terminated records, all 2,552 training rows
on the step-10 grid from 10 through 25,520, 51 validation rows, finite and
monotonic metrics, and 101 unique checkpoint events. W&B held the exact
step-25,500 values, run state `running`, five small files, and zero uploaded
artifacts. Checkpoint retention remained 27 files totaling 35,937,063,261
bytes (approximately 33.5 GiB), with 962 GB free.

Step 26,000 entered dual-BPB validation at 13:44:05 EDT with training loss
`3.631985`, gradient norm `0.488800`, throughput 16.631K tokens/s, LR
multiplier `0.303397`, cumulative training FLOPs `1.318589319610368e18`, and
optimizer timer `102538.690290` seconds. All 50 step-25,510--26,000 rows were
finite and complete on the 10-step grid, with loss range
`3.389269`--`3.912430`, mean `3.644720`, median `3.633050`, and mean throughput
16.616K tokens/s. Mean and median improved by `0.002765` and `0.007827` from
the preceding window. This window's maximum pre-clipping gradient norm was
`0.785318` at step 25,660. Across 1,701 gradient rows from step 9,000 through
26,000, median was `0.517381`, p95 `0.895507`, p99 `1.229957`, and maximum
`4.796720` at step 16,840; the same six isolated rows exceeded 2.0, with none
new. Every logged warmdown multiplier through this boundary matched the
closed-form schedule within `1.11e-16`. Compatibility BPB improved by
`0.002559` to `1.105340`, remaining strictly improved at all 52 gates.
Full-document BPB improved by `0.002467` to a new minimum of `1.139533`.
The compatibility-ranked `best.pt` was promoted at 14:06:02 EDT;
`step_026000.pt` followed at 14:06:15 and `last.pt` at 14:06:28,
approximately 22m 23s after entry. All three files were 1,331,002,343 bytes.
Sequential full CPU reconstruction proved their critical states equal: step,
scheduler, tracker, and validation 26,000; scheduler step count 26,001;
current/minimum compatibility BPB `1.105340`/`1.105340`; current/minimum
full-document BPB `1.139533`/`1.139533`; optimizer and scheduler LR `0.000091`;
packed-loader position and row position 1,664,000; expected manifest and
validation identities; total training FLOPs `1.318589319610368e18`;
pretraining stage; 110,906,112 parameters; 76 model tensors; 75 optimizer
states; and W&B run ID `ngbuyhxj`. Training resumed at 14:07:28 with
step-26,010 loss `3.604769`, LR multiplier `0.302763`, and 16.678K tokens/s. A
settled local audit through step 26,040 found 2,761 complete
newline-terminated records, all 2,604 training rows on the step-10 grid from
10 through 26,040, 52 validation rows, finite and monotonic metrics, and 104
unique checkpoint events. W&B held the exact step-26,000 values, run state
`running`, five small files, and zero uploaded model, dataset, or tokenizer
artifacts. Checkpoint retention increased to 28 files totaling 37,268,065,604
bytes (approximately 34.7 GiB), with 960 GB free.

Step 26,500 entered dual-BPB validation at 14:39:46 EDT with training loss
`3.635831`, gradient norm `0.491752`, throughput 16.626K tokens/s, LR
multiplier `0.271730`, cumulative training FLOPs `1.343946806525952e18`, and
optimizer timer `104510.056129` seconds. All 50 step-26,010--26,500 rows were
finite and complete on the 10-step grid, with loss range
`3.383379`--`3.929838`, mean `3.629354`, median `3.638493`, and mean throughput
16.619K tokens/s. Mean improved by `0.015366`, while median rose by `0.005443`
from the preceding window. This window's maximum pre-clipping gradient norm
was `0.563128` at step 26,190. Across 1,751 gradient rows from step 9,000
through 26,500, median was `0.514806`, p95 `0.891131`, p99 `1.222195`, and
maximum `4.796720` at step 16,840; the same six isolated rows exceeded 2.0,
with none new. Every logged warmdown multiplier through this boundary matched
the closed-form schedule within `1.11e-16`. Compatibility BPB improved by
`0.002749` to `1.102590`, remaining strictly improved at all 53 gates.
Full-document BPB improved by `0.002769` to a new minimum of `1.136765`.
The compatibility-ranked `best.pt` was promoted at 15:01:34 EDT,
approximately 21m 48s after entry, and remained 1,331,002,343 bytes. Full CPU
reconstruction verified step, scheduler, tracker, and validation 26,500;
scheduler step count 26,501; current/minimum compatibility BPB
`1.102590`/`1.102590`; current/minimum full-document BPB
`1.136765`/`1.136765`; optimizer and scheduler LR `0.0000815`; packed-loader
position and row position 1,696,000; expected manifest and validation
identities; total training FLOPs `1.343946806525952e18`; pretraining stage;
110,906,112 parameters; 76 model tensors; 75 optimizer states; and W&B run ID
`ngbuyhxj`. Training resumed at 15:02:33 with step-26,510 loss `3.601475`, LR
multiplier `0.271097`, and 16.640K tokens/s. A settled local audit through step
26,520 found 2,811 complete newline-terminated records, all 2,652 training rows
on the step-10 grid from 10 through 26,520, 53 validation rows, finite and
monotonic metrics, and 105 unique checkpoint events. W&B held the exact
step-26,500 values, run state `running`, five small files, and zero uploaded
model, dataset, or tokenizer artifacts. Checkpoint retention remained 28 files
totaling 37,268,065,604 bytes (approximately 34.7 GiB), with 960 GB free.

Step 27,000 entered dual-BPB validation at 15:34:43 EDT with training loss
`3.477205`, gradient norm `0.494279`, throughput 16.604K tokens/s, LR
multiplier `0.240063`, cumulative training FLOPs `1.369304293441536e18`, and
optimizer timer `106481.370284` seconds. All 50 step-26,510--27,000 rows were
finite and complete on the 10-step grid, with loss range
`3.432989`--`3.949111`, mean `3.643077`, median `3.645444`, and mean throughput
16.617K tokens/s. Mean and median rose by `0.013723` and `0.006951` from the
preceding window. This window's maximum pre-clipping gradient norm was
`0.606425` at step 26,580. Across 1,801 gradient rows from step 9,000 through
27,000, median was `0.512652`, p95 `0.878518`, p99 `1.214433`, and maximum
`4.796720` at step 16,840; the same six isolated rows exceeded 2.0, with none
new. Every logged warmdown multiplier through this boundary matched the
closed-form schedule within `1.11e-16`. Compatibility BPB improved by
`0.001962` to `1.100628`, remaining strictly improved at all 54 gates.
Full-document BPB improved by `0.002072` to a new minimum of `1.134693`.
The compatibility-ranked `best.pt` was promoted at 15:56:35 EDT;
`step_027000.pt` followed at 15:56:45 and `last.pt` at 15:56:55,
approximately 22m 12s after entry. All three files were 1,331,002,343 bytes.
Sequential full CPU reconstruction proved their critical states equal: step,
scheduler, tracker, and validation 27,000; scheduler step count 27,001;
current/minimum compatibility BPB `1.100628`/`1.100628`; current/minimum
full-document BPB `1.134693`/`1.134693`; optimizer and scheduler LR `0.000072`;
packed-loader position and row position 1,728,000; expected manifest and
validation identities; total training FLOPs `1.369304293441536e18`;
pretraining stage; 110,906,112 parameters; 76 model tensors; 75 optimizer
states; and W&B run ID `ngbuyhxj`. Training resumed at 15:57:45 with
step-27,010 loss `3.572006`, LR multiplier `0.239430`, and 16.672K tokens/s. A
settled local audit through step 27,020 found 2,865 complete
newline-terminated records, all 2,702 training rows on the step-10 grid from
10 through 27,020, 54 validation rows, finite and monotonic metrics, and 108
unique checkpoint events. W&B held the exact step-27,000 values, run state
`running`, five small files, and zero uploaded model, dataset, or tokenizer
artifacts. Checkpoint retention increased to 29 files totaling 38,599,067,947
bytes (approximately 35.9 GiB), with 959 GB free.

Step 27,500 entered dual-BPB validation at 16:30:04 EDT with training loss
`3.686575`, gradient norm `0.476520`, throughput 16.629K tokens/s, LR
multiplier `0.208397`, cumulative training FLOPs `1.39466178035712e18`, and
optimizer timer `108452.474950` seconds. All 50 step-27,010--27,500 rows were
finite and complete on the 10-step grid, with loss range
`3.421737`--`3.860883`, mean `3.644958`, median `3.636748`, and mean throughput
16.621K tokens/s. Mean rose by only `0.001881`, while median improved by
`0.008697` from the preceding window. This window's maximum pre-clipping
gradient norm was `0.737880` at step 27,220. Across 1,851 gradient rows from
step 9,000 through 27,500, median was `0.511491`, p95 `0.877667`, p99
`1.208450`, and maximum `4.796720` at step 16,840; the same six isolated rows
exceeded 2.0, with none new. Every logged warmdown multiplier through this
boundary matched the closed-form schedule within `1.11e-16`. Compatibility BPB
improved by `0.003146` to `1.097482`, remaining strictly improved at all 55
gates. Full-document BPB improved by `0.003347` to a new minimum of `1.131346`.
The compatibility-ranked `best.pt` was promoted at 16:51:55 EDT,
approximately 21m 51s after entry, and remained 1,331,002,343 bytes. Full CPU
reconstruction verified step, scheduler, tracker, and validation 27,500;
scheduler step count 27,501; current/minimum compatibility BPB
`1.097482`/`1.097482`; current/minimum full-document BPB
`1.131346`/`1.131346`; optimizer and scheduler LR `0.0000625`; packed-loader
position and row position 1,760,000; expected manifest and validation
identities; total training FLOPs `1.39466178035712e18`; pretraining stage;
110,906,112 parameters; 76 model tensors; 75 optimizer states; and W&B run ID
`ngbuyhxj`. Training resumed at 16:52:51 with step-27,510 loss `3.557314`, LR
multiplier `0.207763`, and 16.632K tokens/s. A settled local audit through step
27,520 found 2,917 complete newline-terminated records, all 2,752 training rows
on the step-10 grid from 10 through 27,520, 55 validation rows, finite and
monotonic metrics, and 109 unique checkpoint events. W&B held the exact
step-27,500 values, run state `running`, five small files, and zero uploaded
model, dataset, or tokenizer artifacts. Checkpoint retention remained 29 files
totaling 38,599,067,947 bytes (approximately 35.9 GiB), with 959 GB free.

Step 28,000 entered dual-BPB validation at 17:25:03 EDT with training loss
`3.570409`, gradient norm `0.476669`, throughput 16.627K tokens/s, LR
multiplier `0.176730`, cumulative training FLOPs `1.420019267272704e18`, and
optimizer timer `110423.883396` seconds. All 50 step-27,510--28,000 rows were
finite and complete on the 10-step grid, with loss range
`3.454027`--`3.835724`, mean `3.607238`, median `3.606569`, and mean throughput
16.620K tokens/s. Mean and median improved by `0.037720` and `0.030178` from
the preceding window. This window's maximum pre-clipping gradient norm was
`0.636398` at step 27,910. Across 1,901 gradient rows from step 9,000 through
28,000, median was `0.510111`, p95 `0.873570`, p99 `1.202467`, and maximum
`4.796720` at step 16,840; the same six isolated rows exceeded 2.0, with none
new. Every logged warmdown multiplier through this boundary matched the
closed-form schedule within `1.11e-16`. Compatibility BPB improved by
`0.002110` to `1.095372`, remaining strictly improved at all 56 gates.
Full-document BPB improved by `0.002099` to a new minimum of `1.129247`.
The compatibility-ranked `best.pt` was promoted at 17:46:57 EDT;
`step_028000.pt` followed at 17:47:07 and `last.pt` at 17:47:20,
approximately 22m 17s after entry. All three files were 1,331,002,343 bytes.
Sequential full CPU reconstruction proved their critical states equal: step,
scheduler, tracker, and validation 28,000; scheduler step count 28,001;
current/minimum compatibility BPB `1.095372`/`1.095372`; current/minimum
full-document BPB `1.129247`/`1.129247`; optimizer and scheduler LR `0.000053`;
packed-loader position and row position 1,792,000; expected manifest and
validation identities; total training FLOPs `1.420019267272704e18`;
pretraining stage; 110,906,112 parameters; 76 model tensors; 75 optimizer
states; and W&B run ID `ngbuyhxj`. Training resumed at 17:48:20 with
step-28,010 loss `3.608949`, LR multiplier `0.176097`, and 16.706K tokens/s. A
settled local audit through step 28,020 found 2,971 complete
newline-terminated records, all 2,802 training rows on the step-10 grid from
10 through 28,020, 56 validation rows, finite and monotonic metrics, and 112
unique checkpoint events. W&B held the exact step-28,000 values, run state
`running`, five small files, and zero uploaded model, dataset, or tokenizer
artifacts. Checkpoint retention increased to 30 files totaling 39,930,070,290
bytes (approximately 37.2 GiB), with 958 GB free.

Step 28,500 entered dual-BPB validation at 18:20:25 EDT with training loss
`3.551093`, gradient norm `0.572360`, throughput 16.607K tokens/s, LR
multiplier `0.145063`, cumulative training FLOPs `1.445376754188288e18`, and
optimizer timer `112394.614255` seconds. All 50 step-28,010--28,500 rows were
finite and complete on the 10-step grid, with loss range
`3.258898`--`4.150836`, mean `3.614773`, median `3.607547`, and mean throughput
16.625K tokens/s. Mean and median rose by `0.007535` and `0.000977` from the
preceding window. This window's maximum pre-clipping gradient norm was
`1.039049` at step 28,370. Across 1,951 gradient rows from step 9,000 through
28,500, median was `0.508850`, p95 `0.870215`, p99 `1.199682`, and maximum
`4.796720` at step 16,840; the same six isolated rows exceeded 2.0, with none
new. Every logged warmdown multiplier through this boundary matched the
closed-form schedule within `1.11e-16`. Compatibility BPB improved by
`0.001495` to `1.093877`, remaining strictly improved at all 57 gates.
Full-document BPB improved by `0.001500` to a new minimum of `1.127747`.
The compatibility-ranked `best.pt` was promoted at 18:42:22 EDT,
approximately 21m 57s after entry, and remained 1,331,002,343 bytes. Full CPU
reconstruction verified step, scheduler, tracker, and validation 28,500;
scheduler step count 28,501; current/minimum compatibility BPB
`1.093877`/`1.093877`; current/minimum full-document BPB
`1.127747`/`1.127747`; optimizer and scheduler LR `0.0000435`; packed-loader
position and row position 1,824,000; expected manifest and validation
identities; total training FLOPs `1.445376754188288e18`; pretraining stage;
110,906,112 parameters; 76 model tensors; 75 optimizer states; and W&B run ID
`ngbuyhxj`. Training resumed at 18:43:27 with step-28,510 loss `3.665422`, LR
multiplier `0.144430`, and 16.657K tokens/s. A settled local audit through step
28,520 found 3,023 complete newline-terminated records, all 2,852 training rows
on the step-10 grid from 10 through 28,520, 57 validation rows, finite and
monotonic metrics, and 113 unique checkpoint events. W&B held the exact
step-28,500 values, run state `running`, five small files, and zero uploaded
model, dataset, or tokenizer artifacts. Checkpoint retention remained 30 files
totaling 39,930,070,290 bytes (approximately 37.2 GiB), with 958 GB free.

Step 29,000 entered dual-BPB validation at 19:15:38 EDT with training loss
`3.656514`, gradient norm `0.468507`, throughput 16.609K tokens/s, LR
multiplier `0.113397`, cumulative training FLOPs `1.470734241103872e18`, and
optimizer timer `114365.499320` seconds. All 50 step-28,510--29,000 rows were
finite and complete on the 10-step grid, with loss range
`3.437267`--`4.132912`, mean `3.616492`, median `3.613277`, and mean throughput
16.624K tokens/s. Mean and median rose by `0.001719` and `0.005730` from the
preceding window. This window's maximum pre-clipping gradient norm was
`0.950499` at step 28,560. Across 2,001 gradient rows from step 9,000 through
29,000, median was `0.507192`, p95 `0.865113`, p99 `1.196898`, and maximum
`4.796720` at step 16,840; the same six isolated rows exceeded 2.0, with none
new. Every logged warmdown multiplier through this boundary matched the
closed-form schedule within `1.11e-16`. Validation, conditional best
promotion, periodic `step_029000.pt`, and `last.pt` publication all completed.
Compatibility BPB improved by `0.002722` to `1.091155`, remaining strictly
improved at all 58 gates. Full-document BPB improved by `0.002775` to a new
minimum of `1.124971`. The compatibility-ranked `best.pt` was durable at
19:37:26 EDT, `step_029000.pt` at 19:37:37, and `last.pt` at 19:37:50,
approximately 22m 12s after entry; all three files were 1,331,002,343 bytes.
Sequential full CPU reconstruction proved their critical states equal: step,
scheduler, tracker, and validation 29,000; scheduler step count 29,001;
current/minimum compatibility BPB `1.091155`/`1.091155`; current/minimum
full-document BPB `1.124971`/`1.124971`; optimizer and scheduler LR
`0.000034`; packed-loader position and row position 1,856,000; expected
manifest and validation identities; total training FLOPs
`1.470734241103872e18`; optimizer timer `114365.499320` seconds; pretraining
stage; 110,906,112 parameters; 76 model tensors; 75 optimizer states; and W&B
run ID `ngbuyhxj`. Training resumed at 19:38:55 with step-29,010 loss
`4.007753`, LR multiplier `0.112763`, and 16.673K tokens/s. A settled local
audit through step 29,020 found 3,077 complete newline-terminated records, all
2,902 training rows on the step-10 grid from 10 through 29,020, 58 validation
rows, finite and monotonic metrics, and 116 unique checkpoint events. W&B held
the exact step-29,000 values, run state `running`, five small files, and zero
uploaded model, dataset, or tokenizer artifacts. Checkpoint retention increased
to 31 files totaling 41,261,072,633 bytes (approximately 38.4 GiB), with 957 GB
free.

Step 29,500 entered dual-BPB validation at 20:10:58 EDT with training loss
`3.824175`, gradient norm `0.499844`, throughput 16.617K tokens/s, LR
multiplier `0.081730`, cumulative training FLOPs `1.496091728019456e18`, and
optimizer timer `116336.401732` seconds. All 50 step-29,010--29,500 rows were
finite and complete on the 10-step grid, with loss range
`3.340239`--`4.007753`, mean `3.620917`, median `3.637766`, and mean throughput
16.624K tokens/s. Mean and median rose by `0.004425` and `0.024489` from the
preceding window. This window's maximum pre-clipping gradient norm was
`0.637464` at step 29,010. Across 2,051 gradient rows from step 9,000 through
29,500, median was `0.506444`, p95 `0.862737`, p99 `1.177342`, and maximum
`4.796720` at step 16,840; the same six isolated rows exceeded 2.0, with none
new. Every logged warmdown multiplier through this boundary matched the
closed-form schedule within `1.11e-16`. Validation and conditional `best.pt`
promotion both completed; no periodic or `last.pt` checkpoint was due at this
boundary. Compatibility BPB improved by `0.001492` to `1.089663`, remaining
strictly improved at all 59 gates. Full-document BPB improved by `0.001505` to
a new minimum of `1.123466`. The compatibility-ranked `best.pt` was durable at
20:32:54 EDT, approximately 21m 56s after entry, and remained 1,331,002,343
bytes. Full CPU reconstruction verified step, scheduler, tracker, and
validation 29,500; scheduler step count 29,501; current/minimum compatibility
BPB `1.089663`/`1.089663`; current/minimum full-document BPB
`1.123466`/`1.123466`; optimizer and scheduler LR `0.0000245`; packed-loader
position and row position 1,888,000; expected manifest and validation
identities; total training FLOPs `1.496091728019456e18`; optimizer timer
`116336.401732` seconds; pretraining stage; 110,906,112 parameters; 76 model
tensors; 75 optimizer states; and W&B run ID `ngbuyhxj`. Training resumed at
20:33:48 with step-29,510 loss `3.817093`, LR multiplier `0.081097`, and
16.636K tokens/s. A settled local audit through step 29,520 found 3,129
complete newline-terminated records, all 2,952 training rows on the step-10
grid from 10 through 29,520, 59 validation rows, finite and monotonic metrics,
and 117 unique checkpoint events. W&B held the exact step-29,500 values, run
state `running`, five small files, and zero uploaded model, dataset, or
tokenizer artifacts. Checkpoint retention remained 31 files totaling
41,261,072,633 bytes (approximately 38.4 GiB), with 957 GB free.

Step 30,000 entered terminal dual-BPB validation at 21:06:04 EDT with training
loss `3.608021`, gradient norm `0.451849`, throughput 16.610K tokens/s, LR
multiplier `0.050063`, cumulative training FLOPs `1.52144921493504e18`, and
optimizer timer `118307.259754` seconds. All 50 step-29,510--30,000 rows were
finite and complete on the 10-step grid, with loss range
`3.271101`--`3.817093`, mean `3.623239`, median `3.617522`, and mean throughput
16.623K tokens/s. Mean rose by only `0.002323` while median improved by
`0.020244` from the preceding window. This window's maximum pre-clipping
gradient norm was `0.738976` at step 29,950. Across 2,101 gradient rows from
step 9,000 through 30,000, median was `0.505869`, p95 `0.859413`, p99
`1.157787`, and maximum `4.796720` at step 16,840; the same six isolated rows
exceeded 2.0, with none new. Every logged warmdown multiplier through the
terminal update matched the closed-form schedule within `1.11e-16`.
Validation, conditional `best.pt` promotion, `step_030000.pt`, `last.pt`, and
clean training-process exit all completed. Compatibility BPB improved by
`0.001745` to `1.087918`, remaining strictly improved at all 60 gates.
Full-document BPB improved by `0.001874` to a new minimum of `1.121593`. The
compatibility-ranked `best.pt` was durable at 21:27:57 EDT,
`step_030000.pt` at 21:28:10, and the final `last.pt` at 21:28:36,
approximately 22m 32s after entry; all three files were 1,331,002,343 bytes.
Sequential full CPU reconstruction proved their critical states equal: step,
scheduler, tracker, and validation 30,000; scheduler step count 30,001;
current/minimum compatibility BPB `1.087918`/`1.087918`; current/minimum
full-document BPB `1.121593`/`1.121593`; optimizer and scheduler LR
`0.000015`; packed-loader position and row position 1,920,000; expected
manifest and validation identities; total training FLOPs
`1.52144921493504e18`; optimizer timer `118307.259754` seconds; pretraining
stage; 110,906,112 parameters; 76 model tensors; 75 optimizer states; and W&B
run ID `ngbuyhxj`. The original process exited cleanly at 21:28:54. A final
local training-log audit found 3,181 complete newline-terminated records, all
3,000 training rows on the step-10 grid, 60 validation rows, finite and
monotonic metrics, and 120 unique checkpoint events. W&B held the exact final
values and zero uploaded artifacts when standalone evaluation began.
Checkpoint retention finished at 32 files totaling 42,592,074,976 bytes
(approximately 39.7 GiB), with 955 GB free. The guarded supervisor then
launched full base evaluation against the selected step-30,000 `best.pt`.

Standalone base evaluation began at 21:29:07 EDT and atomically published all
three reports at 22:46:10 EDT. Its approximately 77m 03s wall time included
54m 22s for full CORE, both BPB protocols, and seven fixed samples. A separate
audit reopened the final JSON and Markdown files, checked their identities and
complete scopes, downloaded all three remote `evaluation` reports, and matched
them byte-for-byte. Their SHA-256 identities are
`c3322bada8bf3fc23158cd6fd67f8b44d6a54d4f5cb55ef2fafae0ab10c3ad7e` for
`base_eval.json`,
`ccafd7a6417a2064023fed6f8c7aaefdb24432b7ef1cdb273dd3ae7a8e25cd20` for
`base_samples.md`, and
`b0c9831a4e006d8f3e37cb29340632b1226baa785af308351174d9ab62800b93` for
`core_comparison.md`. The base W&B run then finished normally.

The automatic handoff launched weighted SFT at 22:46:12 EDT. A user-requested
pause stopped that first attempt after its local step-100 record and before the
first step-250 checkpoint. Its W&B run `61lxq6dl` was explicitly finished as
failed with zero artifacts, and all four local telemetry/config files remain
preserved in `runs/sft-111m-base30k-3090`. Because no exact SFT continuation
existed yet, a retry had to replay from the unchanged selected base checkpoint.
A subsequent detached-shell launch (`ahxld1qg`) did not survive its parent
shell and produced no optimizer row; it was likewise marked failed with zero
artifacts and preserved under its distinct `r2` run path. The persistent `r3`
attempt (`4uitqomj`) is the final run. Its early losses replayed the first
attempt exactly where compared, confirming deterministic seed and loader order.
Across every logged step from 10 through 100, training loss, pre-clipping
gradient norm, LR multiplier, and cumulative FLOPs matched bit-for-bit. Timing
fields differed, as expected; cumulative optimizer time differed by only
1.406 seconds at worst across those ten rows.

The `r3` attempt completed all 2,000 optimizer steps and eight assistant-BPB
gates. Optimizer time was 4,019.001 seconds (66m 59s), while wall time from
tracking-state publication through the final SFT reports was 4,355.472 seconds
(72m 35s). The terminal checkpoint passed a full CPU reconstruction and has
identity
`sha256:66fcfef337f93b48bd4057d2d49275e7dc4382026aa9ca6702afaf789b6f6874`.
Its W&B run contained exactly the two permitted evaluation reports and no
model/checkpoint, dataset, or tokenizer upload when post-SFT regression
evaluation began.

An external guarded supervisor watches the resumed base process. A premature
exit selects the newest local exact checkpoint, preserves any post-checkpoint
telemetry under a distinct interruption filename, rolls the active JSONL and
summary back to the checkpoint boundary, and invokes pretraining with
`--wandb-resume same`. It allows at most three failures per unchanged
checkpoint. Base evaluation cannot launch until both a step-30,000 training
record and step-30,000 `last.pt` exist, and SFT cannot launch until base
evaluation exits successfully. Full base and post-SFT evaluations may each
retry three times. Once SFT has written an exact checkpoint, a premature exit
similarly resumes the newest SFT checkpoint and its W&B run, with the same
bounded retry rule. An SFT interruption before its first checkpoint fails
safely without deleting partial state.

The original syntax-checked full-pipeline supervisor has SHA-256
`ea5927d3ee497d082c5489b262c9df4a0182df7706f19e9c37954285783a5ecf`.
The SFT-recovery supervisor used after the checkpoint-free interruptions has
SHA-256
`9762b8a5d5f2e084fc05913d073864490b0f899eda4e5d92092b15a540c2016d`.
The first three post-SFT launch attempts stopped at the resolved-configuration
guard before evaluating any examples: the persisted SFT run config contains
the CLI-injected `sft.base_checkpoint`, while the original evaluation command
overrode only `run.name`. The corrected, syntax-checked post-SFT supervisor has
SHA-256
`f37d60557de3c84093ad91d09a78bb241ca4cdec8da7caa84abb3e118bf2af9b` and
replays both persisted overrides. No checkpoint, metric report, or dataset was
modified by the rejected attempts.
Its effective stage commands are:

```bash
uv run --extra tracking python -m scripts.eval_base \
  --config configs/base_111m_3090.yaml \
  --checkpoint runs/base-111m-3090-2b/checkpoints/best.pt \
  --eval core,bpb,sample --core-bundle data/eval/eval_bundle.zip

uv run --extra tracking python -m scripts.train_sft \
  --config configs/sft_111m_3090.yaml \
  --override run.name=sft-111m-base30k-3090-r3 \
  --base-checkpoint runs/base-111m-3090-2b/checkpoints/best.pt

uv run --extra tracking python -m scripts.eval_base \
  --config configs/sft_111m_3090.yaml \
  --override run.name=sft-111m-base30k-3090-r3 \
  --override sft.base_checkpoint=runs/base-111m-3090-2b/checkpoints/best.pt \
  --checkpoint runs/sft-111m-base30k-3090-r3/checkpoints/best.pt \
  --eval core,bpb,sample --core-bundle data/eval/eval_bundle.zip
```

The first and third commands may retry without changing their checkpoint or
protocol. Any training retry instead uses the latest exact local checkpoint
plus `--wandb-resume same`; it never restarts an interrupted stage from random
weights.

Measured lifecycle timing is:

| Stage or interval | Duration |
| --- | ---: |
| Data preparation | 8,937.638s (2h 28m 58s) |
| Base optimizer timer | 118,307.260s (32h 51m 47s) |
| Base validation/checkpoint intervals | approximately 22h 03m 35s |
| Base optimizer + gate subtotal | approximately 54h 55m 22s |
| Standalone base evaluation | approximately 77m 03s |
| Final SFT run through its reports | 4,355.472s (72m 35s) |
| Post-SFT regression evaluation | approximately 77m 05s |
| Inclusive data-prep start to post-SFT report publication | approximately 62h 25m 09s |

The inclusive interval runs from the data-preparation W&B creation timestamp
at 11:03:09 EDT on August 3 through final publication at 01:28:18 EDT on
August 6. It includes the user-requested base pause, exact-resume work, both
checkpoint-free SFT attempts, process-launch gaps, and all evaluations. The
optimizer-plus-gate subtotal excludes those out-of-process intervals and is
the better measure of base compute cost. The compatibility-ranked base and SFT
checkpoints were also their exact terminal checkpoints (steps 30,000 and 2,000),
so no earlier-step selection caveat applies to the final comparisons.

## Qualitative samples

Both stages use frozen public prompt suites rather than hand-selected outputs:

| Sampling item | Base | SFT |
| --- | --- | --- |
| Prompt count | 7 | 5 |
| Prompt-set identity | `sha256:bdc98dda4e1c0dd435c30836d8f8c2987384c48a30d8c7a9a3bc86e210bf2fa7` | `sha256:563529654c2e745fb1c659cf326f2466f5b54de4df253c3a25c3150b6302a383` |
| Temperature / top-k | 0.8 / 50 | 0.8 / 50 |
| Maximum new tokens | 256 | 256 |
| Seeds | 1337 plus prompt index | 1337 plus prompt index |
| Stop policy | tokenizer BOS | assistant-end, then BOS safety |
| Renderer | plain completion | `scratch_llm_chat_renderer_v1` |

The base suite covers factual completion, simple temporal/arithmetic reasoning,
enumeration, and open-ended continuation. All seven final base outputs were
grammatical at a local level, but none gave a dependable direct answer: the
model confused Paris with New York/state prose, failed to emit `Au`, missed
Sunday, did not enumerate the planets, and repeated invalid algebra. Five
samples exhausted all 256 tokens and two reached the BOS stop token. The SFT
suite covers explanation, Python generation, ideation, worked arithmetic, and
constrained JSON. Its five outputs adopted an assistant-like voice, but none
completed the requested task correctly; two reached the assistant-end token
and three exhausted the 256-token limit. Nothing is cherry-picked, and the
complete canonical Markdown reports remain in their respective run
directories.

## Conclusions

The requested single-RTX-3090 pipeline completed from fresh base initialization
through weighted SFT and both evaluation stages. The 110,906,112-parameter base
model consumed 1.966B scheduled model tokens without repeating a packed row,
though that was 94.60% of its first shuffled packed plan rather than a complete
source epoch. Its final nanochat-compatible BPB was `1.087918` and CORE was
`0.044687`. Those trail the closest official nanochat d11 point by `0.078318`
BPB and `0.047113` CORE, with major tokenizer, data, context, optimizer,
architecture, and hardware differences preventing causal attribution.

SFT produced a clean 7.93% improvement in its own assistant-only held-out BPB,
from `0.839210` at the first gate to `0.772646` at step 2,000. It also improved
ordinary CORE to `0.057101`, led by BoolQ and CommonsenseQA. That benefit came
with a 3.94% regression in compatible base-text BPB to `1.130756`. More
importantly, all five chat-native prompts and all seven post-SFT base-style
prompts still failed their requested tasks. This checkpoint is useful as a
fully reproducible educational base-to-SFT run, but it is not yet a dependable
assistant or a source of proper completions.

Operationally, batch 8 was safe in FP32, mature base throughput was about 16.6K
tokens/s, SFT averaged 16.3K tokens/s, and local exact recovery worked after an
interrupted base validation. The base optimizer-plus-validation subtotal was
54h 55m; the successful SFT and its regression evaluation added about 2h 30m.
W&B retained scalar telemetry and five small final-run evaluation reports while
all model/checkpoint, dataset, and tokenizer uploads remained blocked. ChatCORE
was not available and was not replaced with ordinary CORE.
