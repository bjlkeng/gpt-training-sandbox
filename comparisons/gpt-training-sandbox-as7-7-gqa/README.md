# Grouped-query attention bounded comparison

This is the bounded, same-seed diagnostic for
`gpt-training-sandbox-as7.7`. It compares ordinary four-head MHA with
experimental four-query/two-KV-head GQA. The resolved training configs differ
only in `run.name`, `model.n_kv_head`, and `model.use_gqa`; both use the same
query-head geometry, optimization budget, data order, and seed.

Both runs used the tracked `configs/smoke.yaml` byte model with `n_head=4`,
seed 1337, float32/manual attention, LayerNorm, bias-free projections, learned
absolute positions, context 128, device batch 2, four accumulation
microsteps, and 50 optimizer steps (51,200 processed model tokens).
Throughput aggregates exclude steps 1–5. Both use the explicit RTX 3090 FP32
35.58-TFLOP/s MFU basis. Peak memory is observed CUDA allocated memory. The
post-run BPB check predicts all 261 bytes in
`data/fixtures/chat/validation.jsonl` (SHA-256
`b41bd8ad00945ea6c0223a2f1964835de0d6b7105fe76dcfa80639e78823ac3f`)
in contiguous, non-overlapping model windows; it is a bounded diagnostic, not
a CORE or full-corpus evaluation.

| Training metric | MHA (4 KV) | GQA (2 KV) | GQA delta |
| --- | ---: | ---: | ---: |
| Unique parameters | 444,160 | 411,392 | -32,768 (-7.38%) |
| Validation BPB | 12.432264 | 18.120358 | +5.688094 (+45.75%) |
| Warm tokens/sec, median | 45,289.27 | 42,910.32 | -2,378.95 (-5.25%) |
| Warm tokens/sec, mean | 45,209.35 | 46,772.29 | +1,562.94 (+3.46%) |
| Peak allocated memory | 29.1328 MiB | 28.6328 MiB | -0.5000 MiB |
| Final training loss | 2.774105 | 2.792745 | +0.018640 |

The paired inference benchmark used a 16-token prompt, 64 generated tokens,
two warmups, and ten timed iterations per checkpoint. It compares naive and
cached shared generation; the table below reports cached p50 decode latency.

| Inference metric | MHA (4 KV) | GQA (2 KV) | GQA delta |
| --- | ---: | ---: | ---: |
| Cache bytes/token | 2,048 | 1,024 | -1,024 (-50.00%) |
| Full cache allocation (128 tokens) | 262,144 B | 131,072 B | -131,072 B |
| Cached decode latency, p50 | 1.1530 ms/token | 1.1898 ms/token | +0.0368 (+3.19%) |
| Cached throughput, p50 | 867.32 tok/s | 840.48 tok/s | -26.83 tok/s |
| Cached peak allocated memory | 10.2041 MiB | 9.9541 MiB | -0.2500 MiB |

Both training runs completed with finite telemetry, and the reduced-head model
halved its KV-cache storage exactly. On this tiny manual-attention workload,
the reduced projection did not improve median training or decode latency, and
its bounded validation BPB was materially worse. The mean/median throughput
disagreement also shows that this short run is timing-noisy. These results do
not establish a quality or performance trend.

The implementation additionally passes hand-built forward/gradient
references for MHA, GQA, and MQA; manual, SDPA, and fake-Flash backend tests;
compact-cache prefill/decode parity; lossless legacy-MHA checkpoint migration;
architecture-mismatch errors; exact resource/FLOPs accounting; and tiny
overfit tests. Exact identities and controls are in `summary.json`; standard
`scripts.compare_runs` artifacts are under `offline-run-comparison/`. The
comparator keeps the parameterization difference visible and emits no BPB
ranking because these bounded runs intentionally have no full base-evaluation
report.

The training runs were launched sequentially with:

```bash
uv run python -m scripts.pretrain --config configs/smoke.yaml \
  --override run.name=m10-gqa-mha \
  --override run.device=cuda \
  --override model.n_head=4 \
  --override model.n_kv_head=4 \
  --override model.use_gqa=false \
  --override train.max_steps=50 \
  --override train.sample_every=50 \
  --override train.save_every=50 \
  --override train.eval_every=50 \
  --override train.mfu_peak_flops_per_second=35580000000000 \
  --override train.mfu_peak_flops_basis=rtx3090_fp32_35.58_tflops \
  --no-wandb

uv run python -m scripts.pretrain --config configs/smoke.yaml \
  --override run.name=m10-gqa-2kv \
  --override run.device=cuda \
  --override model.n_head=4 \
  --override model.n_kv_head=2 \
  --override model.use_gqa=true \
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
