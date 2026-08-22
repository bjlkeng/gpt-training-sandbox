# ReLU-squared bounded comparison

This is the bounded, same-seed diagnostic for
`gpt-training-sandbox-as7.4`. It compares the compatibility-default GELU with
the experimental `relu_squared` MLP activation. The resolved configs differ
only in `run.name` and `model.activation`; both models have exactly the same
parameter count, state keys, projection geometry, initialization, dropout, and
residual ordering.

Both runs used the tracked `configs/smoke.yaml` byte model, seed 1337,
float32/manual attention, LayerNorm, bias-free projections, learned absolute
positions, context 128, device batch 2, four accumulation microsteps, and 50
optimizer steps (51,200 processed model tokens). Throughput aggregates exclude
steps 1–5. Both use the explicit RTX 3090 FP32 35.58-TFLOP/s MFU basis. Peak
memory is observed CUDA allocated memory. The post-run BPB check predicts all
261 bytes in `data/fixtures/chat/validation.jsonl` (SHA-256
`b41bd8ad00945ea6c0223a2f1964835de0d6b7105fe76dcfa80639e78823ac3f`);
it is a bounded diagnostic, not a CORE or full-corpus evaluation.

| Metric | GELU | ReLU-squared | ReLU-squared delta |
| --- | ---: | ---: | ---: |
| Unique parameters | 444,160 | 444,160 | 0 |
| Validation BPB | 12.307664 | 12.602589 | +0.294925 (+2.40%) |
| Warm tokens/sec, median | 45,814.73 | 44,016.98 | -1,797.75 (-3.92%) |
| Warm tokens/sec, mean | 46,279.30 | 43,890.84 | -2,388.46 (-5.16%) |
| Peak allocated memory | 28.6328 MiB | 29.3589 MiB | +0.7261 MiB |
| Final training loss | 2.809594 | 2.432719 | -0.376875 |

All recorded telemetry was finite and both runs completed. Both activations
pass hand-computed derivative/forward, arbitrary-shape, exact state,
checkpoint round-trip, dropout, finite-gradient, and tiny-overfit tests. The
experimental activation reached a lower final training loss but slightly worse
bounded validation BPB, throughput, and peak allocation. This mixed short-run
result does not establish a quality or performance trend. Exact identities and
controls are in `summary.json`; standard `scripts.compare_runs` artifacts are
under `offline-run-comparison/` and correctly block BPB ranking without a full
base evaluation.

The runs were launched sequentially with:

```bash
uv run python -m scripts.pretrain --config configs/smoke.yaml \
  --override run.name=m10-activation-gelu \
  --override run.device=cuda \
  --override train.max_steps=50 \
  --override train.sample_every=50 \
  --override train.save_every=50 \
  --override train.eval_every=50 \
  --override train.mfu_peak_flops_per_second=35580000000000 \
  --override train.mfu_peak_flops_basis=rtx3090_fp32_35.58_tflops \
  --no-wandb

uv run python -m scripts.pretrain --config configs/smoke.yaml \
  --override run.name=m10-activation-relu2 \
  --override run.device=cuda \
  --override model.activation=relu_squared \
  --override train.max_steps=50 \
  --override train.sample_every=50 \
  --override train.save_every=50 \
  --override train.eval_every=50 \
  --override train.mfu_peak_flops_per_second=35580000000000 \
  --override train.mfu_peak_flops_basis=rtx3090_fp32_35.58_tflops \
  --no-wandb
```
