# RoPE bounded comparison

This is the bounded, same-seed diagnostic for
`gpt-training-sandbox-as7.2`. It compares the compatibility-default learned
absolute position embedding with rotary position embeddings on one RTX 3090.
The resolved configs differ only in `run.name` and `model.use_rope`;
`model.rope_theta` is explicitly `10000.0` in both config identities.

Both runs used the tracked `configs/smoke.yaml` byte model, seed 1337,
float32/manual attention, context 128, device batch 2, four accumulation
microsteps, and 50 optimizer steps (51,200 processed model tokens). Throughput
aggregates exclude steps 1–5. Both configs use the same explicit RTX 3090 FP32
35.58-TFLOP/s MFU basis so the repository offline comparator can load the
telemetry. Peak memory is observed CUDA allocated memory. The fixed post-run
BPB check predicts all 261 bytes in the tracked
`data/fixtures/chat/validation.jsonl` fixture (SHA-256
`b41bd8ad00945ea6c0223a2f1964835de0d6b7105fe76dcfa80639e78823ac3f`);
it is a bounded diagnostic, not a CORE or full-corpus evaluation.

| Metric | Learned positions | RoPE | RoPE delta |
| --- | ---: | ---: | ---: |
| Unique parameters | 444,160 | 427,776 | -16,384 |
| Validation BPB | 12.307664 | 20.131045 | +7.823382 (+63.57%) |
| Warm tokens/sec, median | 46,165.86 | 33,511.00 | -12,654.86 (-27.41%) |
| Warm tokens/sec, mean | 46,061.72 | 33,761.82 | -12,299.90 (-26.70%) |
| Peak allocated memory | 28.6328 MiB | 28.6318 MiB | -0.0010 MiB |
| Final training loss | 2.809594 | 2.126777 | -0.682817 |

All recorded losses, gradients, throughput values, and memory values were
finite; both runs completed all requested steps. RoPE also passed hand-worked
rotation, norm/relative-position, manual/SDPA, full/prefill/decode parity,
checkpoint round-trip, gradient, causal, and tiny-overfit coverage. The short
run shows a large bounded BPB and throughput regression despite its lower final
training loss. It does not establish a quality or performance trend. The exact
values, theta/protocol identities, and controls are in `summary.json`.
The repository-standard `scripts.compare_runs` output is preserved under
`offline-run-comparison/`; it correctly leaves BPB rankings blocked because a
261-byte diagnostic is not a complete base evaluation.

The training runs were launched sequentially with these commands:

```bash
uv run python -m scripts.pretrain --config configs/smoke.yaml \
  --override run.name=m10-rope-learned \
  --override run.device=cuda \
  --override train.max_steps=50 \
  --override train.sample_every=50 \
  --override train.save_every=50 \
  --override train.eval_every=50 \
  --override train.mfu_peak_flops_per_second=35580000000000 \
  --override train.mfu_peak_flops_basis=rtx3090_fp32_35.58_tflops \
  --no-wandb

uv run python -m scripts.pretrain --config configs/smoke.yaml \
  --override run.name=m10-rope \
  --override run.device=cuda \
  --override model.use_rope=true \
  --override model.rope_theta=10000.0 \
  --override train.max_steps=50 \
  --override train.sample_every=50 \
  --override train.save_every=50 \
  --override train.eval_every=50 \
  --override train.mfu_peak_flops_per_second=35580000000000 \
  --override train.mfu_peak_flops_basis=rtx3090_fp32_35.58_tflops \
  --no-wandb
```
