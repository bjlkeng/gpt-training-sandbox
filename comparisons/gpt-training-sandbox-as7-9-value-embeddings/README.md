# Gated value-embedding bounded comparison

This is the bounded, same-seed diagnostic for
`gpt-training-sandbox-as7.9`. It compares the compatibility-default disabled
mode with the experimental gated value embedding. At depth two, final-parity
placement selects only layer 1. The resolved configs differ only in
`run.name` and `model.use_value_embeddings`.

Both runs used the tracked `configs/smoke.yaml` byte model, seed 1337,
float32/manual full-context attention, two ordinary MHA heads, LayerNorm,
bias-free projections, learned absolute positions, context 128, device batch
2, four accumulation microsteps, and 50 optimizer steps (51,200 processed
model tokens). Throughput aggregates exclude steps 1–5; the median is the
more robust statistic because CUDA work is asynchronous and several enabled
steps had unusually short host timings. Both use the explicit RTX 3090 FP32
35.58-TFLOP/s MFU basis. Peak memory is observed CUDA allocated memory. The
post-run BPB check predicts all 261 bytes in
`data/fixtures/chat/validation.jsonl` (SHA-256
`b41bd8ad00945ea6c0223a2f1964835de0d6b7105fe76dcfa80639e78823ac3f`)
in contiguous, non-overlapping model windows; it is a bounded diagnostic, not
a CORE or full-corpus evaluation.

| Training metric | Disabled | Enabled, layer 1 | Enabled delta |
| --- | ---: | ---: | ---: |
| Unique parameters | 444,160 | 478,104 | +33,944 (+7.64%) |
| Value-table parameters | 0 | 33,920 | +33,920 |
| Gate parameters | 0 | 24 | +24 |
| Validation BPB | 12.307663 | 13.814755 | +1.507091 (+12.25%) |
| Warm tokens/sec, median | 45,191.67 | 40,042.68 | -5,148.98 (-11.39%) |
| Warm tokens/sec, mean | 45,101.36 | 42,919.30 | -2,182.06 (-4.84%) |
| Peak allocated memory | 28.6328 MiB | 29.2812 MiB | +0.6484 MiB |
| Final training loss | 2.809594 | 2.815056 | +0.005462 |

The paired inference benchmark used a 16-token prompt, 64 generated tokens,
two warmups, and ten timed iterations per checkpoint. Value embeddings add
parameter reads but do not alter the compact physical KV cache or its logical
read/write traffic.

| Cached inference metric | Disabled | Enabled, layer 1 | Enabled delta |
| --- | ---: | ---: | ---: |
| Model parameter bytes | 1,776,640 B | 1,912,416 B | +135,776 B (+7.64%) |
| Physical cache allocation | 262,144 B | 262,144 B | 0 |
| Cache reads/request | 6,193,152 B | 6,193,152 B | 0 |
| Cache writes/request | 129,024 B | 129,024 B | 0 |
| Decode latency, p50 | 1.1868 ms/token | 1.2768 ms/token | +0.0900 (+7.58%) |
| Cached throughput, p50 | 842.61 tok/s | 783.24 tok/s | -59.37 tok/s (-7.05%) |
| Peak allocated memory | 10.2041 MiB | 10.3340 MiB | +0.1299 MiB |

Both runs completed with finite telemetry and exact declared placement. In
this tiny budget, the added capacity did not improve the tracked outcomes: it
increased BPB, reduced median training and decode throughput, and increased
observed memory. The fixture and 50-step budget are intentionally too small to
establish a broader quality trend, but they provide no evidence for enabling
the feature by default.

The implementation additionally passes hand-computed gate arithmetic; proof
that Q/K remain untouched; manual/SDPA forward and gradient parity; MHA/GQA,
odd/even depth, dtype/device, checkpoint, optimizer, and tiny-overfit tests;
and sliding-window cache prefill/decode parity. Exact identities and controls
are in `summary.json`; standard `scripts.compare_runs` artifacts are under
`offline-run-comparison/`. The comparator preserves the parameterization
difference and emits no BPB ranking because these bounded runs intentionally
have no full base-evaluation report.

The training runs were launched sequentially with:

```bash
uv run python -m scripts.pretrain --config configs/smoke.yaml \
  --override run.name=m10-value-off \
  --override run.device=cuda \
  --override train.max_steps=50 \
  --override train.sample_every=50 \
  --override train.save_every=50 \
  --override train.eval_every=50 \
  --override train.mfu_peak_flops_per_second=35580000000000 \
  --override train.mfu_peak_flops_basis=rtx3090_fp32_35.58_tflops \
  --no-wandb

uv run python -m scripts.pretrain --config configs/smoke.yaml \
  --override run.name=m10-value-on \
  --override run.device=cuda \
  --override model.use_value_embeddings=true \
  --override train.max_steps=50 \
  --override train.sample_every=50 \
  --override train.save_every=50 \
  --override train.eval_every=50 \
  --override train.mfu_peak_flops_per_second=35580000000000 \
  --override train.mfu_peak_flops_basis=rtx3090_fp32_35.58_tflops \
  --no-wandb
```

Inference used `scripts.benchmark_inference` with each matching checkpoint,
`generation.max_new_tokens=64`, two warmups, ten timed iterations, the same
FP32 peak basis, and the RTX 3090 936.2-GB/s memory-bandwidth basis.
