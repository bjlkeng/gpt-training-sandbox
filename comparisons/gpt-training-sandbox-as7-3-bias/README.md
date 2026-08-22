# Bias-policy bounded comparison

This is the architecture audit and bounded same-seed diagnostic for
`gpt-training-sandbox-as7.3`. The simple-GPT baseline is already bias-free:
`configs/smoke.yaml` sets `model.bias: false`. The comparison changes only
`run.name` and `model.bias`; it does not introduce a custom Linear class or
change the LM head.

For this two-layer, width-128, MLP-ratio-4 LayerNorm model, enabling the flag
adds exactly 13 tensors and 2,944 elements:

| Tensor family | Count | Elements |
| --- | ---: | ---: |
| Attention QKV projection bias | 2 | 768 |
| Attention output projection bias | 2 | 256 |
| MLP input projection bias | 2 | 1,024 |
| MLP output projection bias | 2 | 256 |
| Block LayerNorm bias | 4 | 512 |
| Final LayerNorm bias | 1 | 128 |
| LM-head bias | 0 | 0 |
| **Total** | **13** | **2,944** |

The last two nonzero rows matter: this is not a Linear-only comparison because
the legacy switch also controls LayerNorm bias. With RMSNorm, the normalization
rows disappear and only the 2,304 projection-bias elements change.

Both runs used the tracked `configs/smoke.yaml` byte model, seed 1337,
float32/manual attention, learned absolute positions, context 128, device batch
2, four accumulation microsteps, and 50 optimizer steps (51,200 processed model
tokens). Throughput aggregates exclude steps 1–5. Both use the explicit RTX
3090 FP32 35.58-TFLOP/s MFU basis. Peak memory is observed CUDA allocated
memory. The post-run BPB check predicts all 261 bytes in
`data/fixtures/chat/validation.jsonl` (SHA-256
`b41bd8ad00945ea6c0223a2f1964835de0d6b7105fe76dcfa80639e78823ac3f`);
it is a bounded diagnostic, not a CORE or full-corpus evaluation.

| Metric | Bias-free | Biased | Biased delta |
| --- | ---: | ---: | ---: |
| Unique parameters | 444,160 | 447,104 | +2,944 |
| Validation BPB | 12.307664 | 16.277262 | +3.969599 (+32.25%) |
| Warm tokens/sec, median | 45,814.55 | 42,288.33 | -3,526.22 (-7.70%) |
| Warm tokens/sec, mean | 45,781.82 | 42,188.56 | -3,593.26 (-7.85%) |
| Peak allocated memory | 28.6328 MiB | 28.6777 MiB | +0.0449 MiB |
| Final training loss | 2.809594 | 2.986202 | +0.176608 |

All recorded telemetry was finite and both runs completed. Both policies pass
shape, causal, finite-gradient, checkpoint round-trip, architecture-mismatch,
and tiny-overfit tests. The bounded result favors the already bias-free default
on this fixture, but the run is too short and the validation set too small to
establish a quality or performance trend. Exact identities, inventory, and
controls are in `summary.json`; standard `scripts.compare_runs` artifacts are
under `offline-run-comparison/` and correctly block BPB ranking without a full
base evaluation.

The runs were launched sequentially with:

```bash
uv run python -m scripts.pretrain --config configs/smoke.yaml \
  --override run.name=m10-bias-free \
  --override run.device=cuda \
  --override train.max_steps=50 \
  --override train.sample_every=50 \
  --override train.save_every=50 \
  --override train.eval_every=50 \
  --override train.mfu_peak_flops_per_second=35580000000000 \
  --override train.mfu_peak_flops_basis=rtx3090_fp32_35.58_tflops \
  --no-wandb

uv run python -m scripts.pretrain --config configs/smoke.yaml \
  --override run.name=m10-biased \
  --override run.device=cuda \
  --override model.bias=true \
  --override train.max_steps=50 \
  --override train.sample_every=50 \
  --override train.save_every=50 \
  --override train.eval_every=50 \
  --override train.mfu_peak_flops_per_second=35580000000000 \
  --override train.mfu_peak_flops_basis=rtx3090_fp32_35.58_tflops \
  --no-wandb
```
