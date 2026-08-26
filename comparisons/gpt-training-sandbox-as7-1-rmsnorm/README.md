# RMSNorm bounded comparison

This is the bounded, same-seed diagnostic for
`gpt-training-sandbox-as7.1`. It compares the compatibility-default
parameterized LayerNorm with parameter-free RMSNorm on one RTX 3090. The
resolved configs differ only in `run.name`, `model.norm`, and
`model.use_rmsnorm`. RMSNorm uses PyTorch's native `F.rms_norm` operation with
`weight=None` and the project's explicit `1e-5` epsilon.

Both runs used the tracked `configs/smoke.yaml` byte model, seed 1337,
float32/manual attention, context 128, device batch 2, four accumulation
microsteps, and 50 optimizer steps (51,200 processed model tokens). Throughput
aggregates exclude steps 1–5. Both configs use the same explicit RTX 3090 FP32
35.58-TFLOP/s MFU basis so the existing offline run comparator can load the
telemetry. Peak memory is observed CUDA allocated memory.
The fixed post-run BPB check predicts every next byte in the tracked
`data/fixtures/chat/validation.jsonl` fixture (SHA-256
`b41bd8ad00945ea6c0223a2f1964835de0d6b7105fe76dcfa80639e78823ac3f`);
it is a 261-byte bounded diagnostic, not a CORE or full-corpus evaluation.

| Metric | LayerNorm | RMSNorm | RMSNorm delta |
| --- | ---: | ---: | ---: |
| Unique parameters | 444,160 | 443,520 | -640 |
| Validation BPB | 12.307664 | 12.438693 | +0.131029 (+1.06%) |
| Warm tokens/sec, median | 45,731.02 | 45,530.11 | -200.91 (-0.44%) |
| Warm tokens/sec, mean | 45,721.02 | 45,477.53 | -243.49 (-0.53%) |
| Peak allocated memory | 28.6328 MiB | 28.6182 MiB | -0.0146 MiB |
| Final training loss | 2.809594 | 2.768610 | -0.040984 |

All recorded losses, gradients, throughput values, and memory values were
finite; both runs completed all requested steps. RMSNorm also passed the
deterministic tiny-overfit test, while the unchanged LayerNorm path passed the
full regression suite. Moving from the original handwritten operation chain to
the native operation reduced the bounded median-throughput gap from 18.27% to
0.44%. The remaining timing difference and small BPB increase do not establish
a quality or performance trend. The exact machine-readable values, identities,
protocol, and controls are in `summary.json`.
The repository-standard `scripts.compare_runs` output is preserved under
`offline-run-comparison/`; it correctly leaves BPB rankings blocked because a
261-byte diagnostic is not a complete base evaluation.

The training runs were launched sequentially with these commands:

```bash
uv run python -m scripts.pretrain --config configs/smoke.yaml \
  --override run.name=m10-rmsnorm-layernorm-native-r3 \
  --override run.device=cuda \
  --override train.max_steps=50 \
  --override train.sample_every=50 \
  --override train.save_every=50 \
  --override train.eval_every=50 \
  --override train.mfu_peak_flops_per_second=35580000000000 \
  --override train.mfu_peak_flops_basis=rtx3090_fp32_35.58_tflops \
  --no-wandb

uv run python -m scripts.pretrain --config configs/smoke.yaml \
  --override run.name=m10-rmsnorm-native-r3 \
  --override run.device=cuda \
  --override model.norm=rmsnorm \
  --override model.use_rmsnorm=true \
  --override train.max_steps=50 \
  --override train.sample_every=50 \
  --override train.save_every=50 \
  --override train.eval_every=50 \
  --override train.mfu_peak_flops_per_second=35580000000000 \
  --override train.mfu_peak_flops_basis=rtx3090_fp32_35.58_tflops \
  --no-wandb
```
