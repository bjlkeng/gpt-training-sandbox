# Prepared FlashAttention 2 benchmark

- Date: 2026-08-09
- Bead: `gpt-training-sandbox-244`
- Source commit before the change: `e963922bfa9338f1724a6e15179975b6b017d78f`
- GPU: NVIDIA GeForce RTX 3090, compute capability 8.6
- Python/PyTorch/CUDA: 3.12.13 / 2.13.0+cu130 / CUDA 13.0
- FlashAttention: 2.8.3.post1

## Change

The production runtimes already resolved the requested attention backend before
building a model, but they used the result only for progress output. Each Flash
decoder block therefore repeated runtime capability and provider resolution
inside `forward`. TorchDynamo traced the `functools.lru_cache`-wrapped provider
loader, producing five observed recompilations and fragmenting the compiled
path.

The model now accepts the validated preflight resolution before
`torch.compile`. Prepared Flash blocks directly invoke the selected provider;
prepared fallback blocks directly invoke SDPA or manual attention. Standalone
eager modules retain lazy resolution, and kernel-launch failure still follows
the configured strict-or-fallback policy.

## Protocol

Both finalists used the same 235,963,392-parameter model, BF16 autocast,
device batch 16, sequence length 1024, 65,536 tokens per optimizer step,
Inductor `default` compilation, no activation checkpointing, and the same
tokenizer/data/model identities. Each run completed 5 excluded warmup steps
and 30 timed production optimizer steps (1,966,080 timed tokens). The strict
Flash policy made fallback an error. Three FA2 runs and three fresh SDPA control
runs were measured.

## Results

| Backend | Repeated tokens/sec | Median | Range | Peak allocated MiB | Peak reserved MiB | Recompilations |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Prepared native FA2 | 37,668.28; 37,650.27; 37,572.23 | **37,650.27** | 0.255% | 14,479.16 | 15,028 | 1 |
| Fresh SDPA control | 37,596.71; 37,536.04; 37,525.37 | 37,536.04 | 0.190% | 14,479.17 | 15,028 | 1 |
| Native FA2 before the fix | 35,942.22; 35,969.90; 36,046.00 | 35,969.90 | 0.289% | 17,997.28 | 20,494 | 5 |

Prepared FA2 is 0.304% (114.23 tokens/sec) faster than the fresh SDPA median
and 0.307% faster than the previous best-config median of 37,535.08 tokens/sec.
It is 4.672% faster than the pre-fix FA2 median and reduces peak allocated
memory by 3,518.13 MiB. All three prepared-FA2 repeats completed with native
provider `fa2`, no fallback, finite telemetry, and one observed recompilation.

## Decision

The acceptance gate passes: native FA2 is measurably faster than the best
existing BF16 + SDPA + default-compile configuration on this machine. With
FlashAttention installed, the fastest measured 236M configuration is now BF16
+ native FA2 + default compile + device batch 16, without activation
checkpointing. The end-to-end gain over SDPA is small because attention is only
one part of a full optimizer step, even though the isolated FA2 kernel has a
larger advantage.

Local source reports:

- `runs/m9-236m-bf16-fa2-prepared-v2-compile-default-b16-r{1,2,3}/metrics/throughput_benchmark.json`
- `runs/m9-236m-bf16-sdpa-prepared-control-b16-r{1,2,3}/metrics/throughput_benchmark.json`
