# Residual/input-scalar bounded comparison

This is the bounded, same-seed diagnostic for
`gpt-training-sandbox-as7.10`. It compares the compatibility-default disabled
mode with two experimental initializers for one learned residual-stream scalar
and one learned normalized-input scalar per layer. The resolved configs differ
only in `run.name`, `model.use_residual_scalars`, and
`model.residual_scalar_init`.

Both enabled variants apply this recurrence immediately before every block:

```text
x0 = parameter_free_rmsnorm(initial_token_representation)
x = residual_scalar[layer] * x + input_scalar[layer] * x0
```

`x0` is computed once and remains immutable. Neutral initialization uses
residual `[1, 1]` and input `[0, 0]`, making the enabled recurrence an exact
functional no-op before learning. The pinned depth schedule uses residual
`[1.15, 1.05]` and input `[0.20, 0.05]`, following the
[pinned nanochat implementation](https://github.com/karpathy/nanochat/blob/92d63d4e8bb4df75c3b71618f31ddde2378b2bcd/nanochat/gpt.py).

All runs used the tracked `configs/smoke.yaml` byte model, seed 1337,
float32/manual full-context attention, two ordinary MHA heads, LayerNorm,
bias-free projections, learned absolute positions, context 128, device batch
2, four accumulation microsteps, and 50 optimizer steps (51,200 processed
model tokens). Throughput aggregates exclude steps 1–5. All runs use the
explicit RTX 3090 FP32 35.58-TFLOP/s MFU basis. Peak memory is observed CUDA
allocated memory. The post-run BPB check predicts all 261 bytes in
`data/fixtures/chat/validation.jsonl` (SHA-256
`b41bd8ad00945ea6c0223a2f1964835de0d6b7105fe76dcfa80639e78823ac3f`)
in contiguous, non-overlapping model windows; it is a bounded diagnostic, not
a CORE or full-corpus evaluation.

| Training metric | Disabled | Neutral | Pinned depth |
| --- | ---: | ---: | ---: |
| Unique parameters | 444,160 | 444,164 | 444,164 |
| Validation BPB | 12.307663 | 12.361151 | 13.035627 |
| BPB delta vs disabled | — | +0.053487 (+0.43%) | +0.727964 (+5.91%) |
| Warm tokens/sec, median | 45,856.47 | 40,005.74 | 40,016.34 |
| Median throughput delta | — | -5,850.73 (-12.76%) | -5,840.12 (-12.74%) |
| Warm tokens/sec, mean | 45,737.19 | 39,839.66 | 40,240.65 |
| Peak allocated memory | 28.6328 MiB | 29.0127 MiB | 29.0127 MiB |
| Final training loss | 2.809594 | 2.539336 | 2.603706 |

The learned vectors after step 50 were:

| Initializer | Residual: initial → learned | Input: initial → learned |
| --- | --- | --- |
| Neutral | `[1, 1]` → `[0.965919, 0.984849]` | `[0, 0]` → `[-0.033345, -0.070374]` |
| Pinned depth | `[1.15, 1.05]` → `[1.102085, 1.017043]` | `[0.20, 0.05]` → `[0.153108, -0.032277]` |

The inference benchmark used a 16-token prompt, 64 generated tokens, two
warmups, and ten timed iterations per checkpoint. Scalar recurrence adds only
four float32 parameter elements at depth two and does not change the KV-cache
allocation or its logical traffic.

| Cached inference metric | Disabled | Neutral | Pinned depth |
| --- | ---: | ---: | ---: |
| Model parameter bytes | 1,776,640 B | 1,776,656 B | 1,776,656 B |
| Physical cache allocation | 262,144 B | 262,144 B | 262,144 B |
| Cache reads/request | 6,193,152 B | 6,193,152 B | 6,193,152 B |
| Cache writes/request | 129,024 B | 129,024 B | 129,024 B |
| Decode latency, p50 | 1.1674 ms/token | 1.2367 ms/token | 1.2368 ms/token |
| Cached throughput, p50 | 856.62 tok/s | 808.59 tok/s | 808.56 tok/s |
| Peak allocated memory | 10.2041 MiB | 10.2129 MiB | 10.2129 MiB |

Both enabled runs reached lower training loss than the disabled control, and
the learned vectors moved independently from their initial values. The neutral
run nevertheless increased validation BPB by 0.43%, while the pinned schedule
increased it by 5.91%. Both reduced measured median training throughput by
about 12.7% and cached throughput by about 5.6%. This tiny eager-mode budget is
too small to establish a broader quality or performance trend, but it provides
no evidence for enabling the feature by default.

The implementation additionally passes hand-computed recurrence and immutable
`x0` checks; disabled and neutral exact-logit checks; learned-position/RoPE,
MHA/GQA, manual/SDPA, full/prefill/cached decode, activation-checkpointing,
optimizer, checkpoint, gradient, parameter-accounting, resource, telemetry,
OOM, and tiny-overfit tests. Exact identities and controls are in
`summary.json`; standard `scripts.compare_runs` artifacts are under
`offline-run-comparison/`. The comparator records all three parameterizations
and emits no BPB ranking because these bounded runs intentionally have no full
base-evaluation report.

The training runs were launched sequentially with this common command, using
`off`, `neutral`, or `pinned` for the run suffix and the corresponding model
overrides:

```bash
uv run python -m scripts.pretrain --config configs/smoke.yaml \
  --override run.name=m10-scalars-<variant> \
  --override run.device=cuda \
  --override train.max_steps=50 \
  --override train.sample_every=50 \
  --override train.save_every=50 \
  --override train.eval_every=50 \
  --override train.mfu_peak_flops_per_second=35580000000000 \
  --override train.mfu_peak_flops_basis=rtx3090_fp32_35.58_tflops \
  --no-wandb
```

The enabled runs additionally used
`model.use_residual_scalars=true`; the pinned run used
`model.residual_scalar_init=nanochat_depth`. Inference used
`scripts.benchmark_inference` with each matching checkpoint,
`generation.max_new_tokens=64`, two warmups, ten timed iterations, and the RTX
3090 936.2-GB/s memory-bandwidth basis.
