# Tied versus untied embedding bounded comparison

This is the bounded, same-seed diagnostic for
`gpt-training-sandbox-as7.5`. It compares the compatibility-default shared
token-embedding/LM-head parameter with the experimental independent LM head.
The resolved configs differ only in `run.name` and `model.tie_weights`.

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

| Metric | Tied | Untied | Untied delta |
| --- | ---: | ---: | ---: |
| Unique parameters | 444,160 | 478,080 | +33,920 (+7.64%) |
| Validation BPB | 12.307663 | 9.649763 | -2.657901 (-21.60%) |
| Warm tokens/sec, median | 45,715.87 | 45,311.11 | -404.76 (-0.89%) |
| Warm tokens/sec, mean | 45,605.79 | 45,231.09 | -374.70 (-0.82%) |
| Peak allocated memory | 28.6328 MiB | 29.1504 MiB | +0.5176 MiB |
| Final training loss | 2.809594 | 1.732469 | -1.077125 |

The parameter delta is exactly `265 * 128 = 33,920`, and the estimator's
parameter, gradient, and two-float32-AdamW-moment deltas reconcile with the
constructed models. Both modes complete forward/loss, independent-gradient,
optimizer-visibility, checkpoint, and tiny-overfit tests. The untied run
improved this tiny byte-fixture BPB while using slightly more memory and
slightly less throughput, but the short run and 261-byte validation fixture do
not establish a quality or performance trend for 32K-token training.

Exact identities and controls are in `summary.json`. The standard
`scripts.compare_runs` artifacts are under `offline-run-comparison/`; the
version-2 comparison displays the tied/untied parameterization mismatch and
correctly emits no BPB ranking because the runs have different unique
parameter counts.

The runs were launched sequentially with:

```bash
uv run python -m scripts.pretrain --config configs/smoke.yaml \
  --override run.name=m10-weight-tied \
  --override run.device=cuda \
  --override train.max_steps=50 \
  --override train.sample_every=50 \
  --override train.save_every=50 \
  --override train.eval_every=50 \
  --override train.mfu_peak_flops_per_second=35580000000000 \
  --override train.mfu_peak_flops_basis=rtx3090_fp32_35.58_tflops \
  --no-wandb

uv run python -m scripts.pretrain --config configs/smoke.yaml \
  --override run.name=m10-weight-untied \
  --override run.device=cuda \
  --override model.tie_weights=false \
  --override train.max_steps=50 \
  --override train.sample_every=50 \
  --override train.save_every=50 \
  --override train.eval_every=50 \
  --override train.mfu_peak_flops_per_second=35580000000000 \
  --override train.mfu_peak_flops_basis=rtx3090_fp32_35.58_tflops \
  --no-wandb
```
