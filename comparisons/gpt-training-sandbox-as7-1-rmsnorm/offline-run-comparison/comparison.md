# Training run comparison

Compared runs: `2`

## Identity differences

| Field | Values by run |
| --- | --- |
| config_identity | {"m10-rmsnorm-layernorm-r2":"sha256:c753c6d4d4e531e2e0836dcd0de80c61fe4d8d22944700f7e16d5ae514417c26","m10-rmsnorm-r2":"sha256:c9fc427f3f8f7257ffb5e335c5d33d31a370d00245f9f9a3516f1ca2308a9f50"} |

Unavailable across all runs: `checkpoint_identity`, `tokenizer_identity`, `validation_manifest_identity`, `code_identity`.

## Numeric deltas

| Metric | Baseline | Deltas by run |
| --- | --- | --- |
| best_compatibility_bpb | m10-rmsnorm-layernorm-r2 | {"m10-rmsnorm-layernorm-r2":null,"m10-rmsnorm-r2":null} |
| best_full_document_bpb | m10-rmsnorm-layernorm-r2 | {"m10-rmsnorm-layernorm-r2":null,"m10-rmsnorm-r2":null} |
| best_loss | m10-rmsnorm-layernorm-r2 | {"m10-rmsnorm-layernorm-r2":0.0,"m10-rmsnorm-r2":-0.014405012130737305} |
| latest_loss | m10-rmsnorm-layernorm-r2 | {"m10-rmsnorm-layernorm-r2":0.0,"m10-rmsnorm-r2":-0.04097723960876465} |
| tokens_per_second | m10-rmsnorm-layernorm-r2 | {"m10-rmsnorm-layernorm-r2":0.0,"m10-rmsnorm-r2":-9098.866724066087} |
| mfu | m10-rmsnorm-layernorm-r2 | {"m10-rmsnorm-layernorm-r2":0.0,"m10-rmsnorm-r2":-0.0007559455087148546} |
| peak_memory_mib | m10-rmsnorm-layernorm-r2 | {"m10-rmsnorm-layernorm-r2":0.0,"m10-rmsnorm-r2":-0.0146484375} |

## Training summary

| Run | Status | Step | Configured parameters | Unique parameters | Processed tokens |
| --- | --- | ---: | ---: | ---: | ---: |
| m10-rmsnorm-layernorm-r2 | completed | 50 | 478080 | 444160 | 51200 |
| m10-rmsnorm-r2 | completed | 50 | 477440 | 443520 | 51200 |

## Training quality

| Run | Latest loss | Best loss | Latest compatibility BPB | Best compatibility BPB | Latest full-document BPB | Best full-document BPB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| m10-rmsnorm-layernorm-r2 | 2.80959 | 2.57122 | — | — | — | — |
| m10-rmsnorm-r2 | 2.76862 | 2.55682 | — | — | — | — |

## Performance

| Run | Tokens/sec | MFU | Peak MiB | Total FLOPs | Training seconds |
| --- | ---: | ---: | ---: | ---: | ---: |
| m10-rmsnorm-layernorm-r2 | 46198.2 | 0.0038382 | 28.6328 | 1.51349e+11 | 1.40379 |
| m10-rmsnorm-r2 | 37099.3 | 0.00308226 | 28.6182 | 1.51349e+11 | 1.67455 |

## Ranking blockers

- m10-rmsnorm-layernorm-r2: base_eval.json is missing; nanochat_compat_v1 result is missing; full_documents_v1 result is missing
- m10-rmsnorm-r2: base_eval.json is missing; nanochat_compat_v1 result is missing; full_documents_v1 result is missing

## Compatibility BPB

| Run | BPB | Source-byte retention | Rank |
| --- | ---: | ---: | ---: |
| m10-rmsnorm-layernorm-r2 | — | — | — |
| m10-rmsnorm-r2 | — | — | — |

## Full-document BPB

| Run | BPB | Source-byte retention | Rank |
| --- | ---: | ---: | ---: |
| m10-rmsnorm-layernorm-r2 | — | — | — |
| m10-rmsnorm-r2 | — | — | — |
