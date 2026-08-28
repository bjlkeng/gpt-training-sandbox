# Query/key normalization bounded comparison

This is the bounded, same-seed diagnostic for
`gpt-training-sandbox-as7.6`. It compares the compatibility-default attention
path with experimental parameter-free per-head QK normalization. The resolved
configs differ only in `run.name` and `model.use_qk_norm`; normalization adds
no parameters or persistent state.

Both runs used the tracked `configs/smoke.yaml` byte model, seed 1337,
float32/manual attention, LayerNorm, bias-free projections, learned absolute
positions, context 128, device batch 2, four accumulation microsteps, and 50
optimizer steps (51,200 processed model tokens). Throughput aggregates exclude
steps 1–5. Both use the explicit RTX 3090 FP32 35.58-TFLOP/s MFU basis. Peak
memory is observed CUDA allocated memory. The post-run BPB check predicts all
261 bytes in `data/fixtures/chat/validation.jsonl` (SHA-256
`b41bd8ad00945ea6c0223a2f1964835de0d6b7105fe76dcfa80639e78823ac3f`)
in contiguous, non-overlapping model windows; it is a bounded diagnostic, not
a CORE or full-corpus evaluation.

| Metric | Disabled | Enabled | Enabled delta |
| --- | ---: | ---: | ---: |
| Unique parameters | 444,160 | 444,160 | 0 |
| Validation BPB | 12.307663 | 12.283795 | -0.023868 (-0.19%) |
| Warm tokens/sec, median | 45,595.76 | 42,566.21 | -3,029.54 (-6.64%) |
| Warm tokens/sec, mean | 45,450.12 | 42,703.37 | -2,746.74 (-6.04%) |
| Peak allocated memory | 28.6328 MiB | 29.3906 MiB | +0.7578 MiB |
| Final training loss | 2.809594 | 2.648833 | -0.160761 |

All recorded training telemetry was finite and both runs completed. The
enabled path also passes hand-computed ordering/scale, per-head normalization,
manual/SDPA/Flash parity, RoPE, cache-prefill/decode, extreme-magnitude,
checkpoint-state, and tiny-overfit tests. The small BPB improvement accompanies
a measurable short-run throughput and peak-allocation cost. This bounded
fixture does not establish a quality or performance trend.

Exact identities and controls are in `summary.json`; standard
`scripts.compare_runs` artifacts are under `offline-run-comparison/`. That
comparator keeps the configuration difference visible but emits no BPB ranking
because these bounded runs intentionally have no full base-evaluation report.
Learned sharpening constants and a logit softcap were not enabled or tested.

The runs were launched sequentially with:

```bash
uv run python -m scripts.pretrain --config configs/smoke.yaml \
  --override run.name=m10-qk-norm-off \
  --override run.device=cuda \
  --override train.max_steps=50 \
  --override train.sample_every=50 \
  --override train.save_every=50 \
  --override train.eval_every=50 \
  --override train.mfu_peak_flops_per_second=35580000000000 \
  --override train.mfu_peak_flops_basis=rtx3090_fp32_35.58_tflops \
  --no-wandb

uv run python -m scripts.pretrain --config configs/smoke.yaml \
  --override run.name=m10-qk-norm-on \
  --override run.device=cuda \
  --override model.use_qk_norm=true \
  --override train.max_steps=50 \
  --override train.sample_every=50 \
  --override train.save_every=50 \
  --override train.eval_every=50 \
  --override train.mfu_peak_flops_per_second=35580000000000 \
  --override train.mfu_peak_flops_basis=rtx3090_fp32_35.58_tflops \
  --no-wandb
```
