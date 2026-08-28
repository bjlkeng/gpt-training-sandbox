# Sliding-window attention bounded comparison

This is the bounded, same-seed diagnostic for
`gpt-training-sandbox-as7.8`. It compares the compatibility-default `L`
full-context pattern with the experimental `S` pattern at short-window size
16. At depth two, tiling plus the forced-full final block resolves these to
`L,L` and `S,L`. The resolved configs differ only in `run.name`,
`model.sliding_window_pattern`, and `model.sliding_window_size`.

Both runs used the tracked `configs/smoke.yaml` byte model, seed 1337,
float32/manual attention, two ordinary MHA heads, LayerNorm, bias-free
projections, learned absolute positions, context 128, device batch 2, four
accumulation microsteps, and 50 optimizer steps (51,200 processed model
tokens). Throughput aggregates exclude steps 1–5. Both use the explicit RTX
3090 FP32 35.58-TFLOP/s MFU basis. Peak memory is observed CUDA allocated
memory. The post-run BPB check predicts all 261 bytes in
`data/fixtures/chat/validation.jsonl` (SHA-256
`b41bd8ad00945ea6c0223a2f1964835de0d6b7105fe76dcfa80639e78823ac3f`)
in contiguous, non-overlapping model windows; it is a bounded diagnostic, not
a CORE or full-corpus evaluation.

| Training metric | Full `L,L` | Sliding `S,L` | Sliding delta |
| --- | ---: | ---: | ---: |
| Unique parameters | 444,160 | 444,160 | 0 |
| Effective layer key spans | 128, 128 | 17, 128 | declared reduction |
| Validation BPB | 12.307663 | 12.126275 | -0.181388 (-1.47%) |
| Warm tokens/sec, median | 45,274.46 | 40,581.47 | -4,692.99 (-10.37%) |
| Warm tokens/sec, mean | 45,186.51 | 40,532.20 | -4,654.31 (-10.30%) |
| Peak allocated memory | 28.6328 MiB | 32.3872 MiB | +3.7544 MiB |
| Final training loss | 2.809594 | 2.839205 | +0.029611 |

The paired inference benchmark used a 16-token prompt, 64 generated tokens,
two warmups, and ten timed iterations per checkpoint. Both caches retain the
same physical 128-token capacity and 262,144-byte allocation; only logical
reads differ.

| Cached inference metric | Full `L,L` | Sliding `S,L` | Sliding delta |
| --- | ---: | ---: | ---: |
| Physical cache allocation | 262,144 B | 262,144 B | 0 |
| Cache writes/request | 129,024 B | 129,024 B | 0 |
| Logical cache reads/request | 6,193,152 B | 4,193,280 B | -1,999,872 B (-32.29%) |
| Decode latency, p50 | 1.1594 ms/token | 1.2060 ms/token | +0.0466 (+4.02%) |
| Cached throughput, p50 | 862.55 tok/s | 829.22 tok/s | -33.33 tok/s |
| Peak allocated memory | 10.2041 MiB | 10.2944 MiB | +0.0903 MiB |

Both runs completed with finite telemetry, and the declared pattern reduced
logical cached K/V reads by 32.29% while leaving physical capacity and writes
unchanged. This tiny manual-attention implementation did not turn that traffic
reduction into a latency, throughput, or observed-memory improvement. Its
bounded BPB was modestly lower, but the fixture and 50-step budget do not
establish a quality trend.

The implementation additionally passes hand-calculated visibility masks;
manual/SDPA forward and gradient references at window boundaries; explicit
Flash window-capability and fallback tests; alternating full/short GQA cache
prefill/decode parity; exact cache traffic, FLOP, and resource identities;
checkpoint state round-trip; and tiny overfit. Exact identities and controls
are in `summary.json`; standard `scripts.compare_runs` artifacts are under
`offline-run-comparison/`. The comparator keeps the pattern identity visible
and emits no BPB ranking because these bounded runs intentionally have no full
base-evaluation report.

The training runs were launched sequentially with:

```bash
uv run python -m scripts.pretrain --config configs/smoke.yaml \
  --override run.name=m10-window-full \
  --override run.device=cuda \
  --override train.max_steps=50 \
  --override train.sample_every=50 \
  --override train.save_every=50 \
  --override train.eval_every=50 \
  --override train.mfu_peak_flops_per_second=35580000000000 \
  --override train.mfu_peak_flops_basis=rtx3090_fp32_35.58_tflops \
  --no-wandb

uv run python -m scripts.pretrain --config configs/smoke.yaml \
  --override run.name=m10-window-s16 \
  --override run.device=cuda \
  --override model.sliding_window_pattern=S \
  --override model.sliding_window_size=16 \
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
