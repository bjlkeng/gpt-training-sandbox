# Training run comparison

Compared runs: `2`

## Identity differences

| Field | Values by run |
| --- | --- |
| config_identity | {"m10-bias-free":"sha256:fccb7e47322b0a9e6cd420a552cf5f5fed0097acba93d1a50291000558121b83","m10-biased":"sha256:d89b1e66c5c5268195f0303888f31213c4507d4823cfa525662196974a61d48b"} |

Unavailable across all runs: `checkpoint_identity`, `tokenizer_identity`, `validation_manifest_identity`, `code_identity`.

## Numeric deltas

| Metric | Baseline | Deltas by run |
| --- | --- | --- |
| best_compatibility_bpb | m10-bias-free | {"m10-bias-free":null,"m10-biased":null} |
| best_full_document_bpb | m10-bias-free | {"m10-bias-free":null,"m10-biased":null} |
| best_loss | m10-bias-free | {"m10-bias-free":0.0,"m10-biased":0.11685371398925781} |
| latest_loss | m10-bias-free | {"m10-bias-free":0.0,"m10-biased":0.17660832405090332} |
| tokens_per_second | m10-bias-free | {"m10-bias-free":0.0,"m10-biased":-3939.450217321646} |
| mfu | m10-bias-free | {"m10-bias-free":0.0,"m10-biased":-0.00032729457292888543} |
| peak_memory_mib | m10-bias-free | {"m10-bias-free":0.0,"m10-biased":0.044921875} |

## Training summary

| Run | Status | Step | Configured parameters | Unique parameters | Processed tokens |
| --- | --- | ---: | ---: | ---: | ---: |
| m10-bias-free | completed | 50 | 478080 | 444160 | 51200 |
| m10-biased | completed | 50 | 481024 | 447104 | 51200 |

## Training quality

| Run | Latest loss | Best loss | Latest compatibility BPB | Best compatibility BPB | Latest full-document BPB | Best full-document BPB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| m10-bias-free | 2.80959 | 2.57122 | — | — | — | — |
| m10-biased | 2.9862 | 2.68807 | — | — | — | — |

## Performance

| Run | Tokens/sec | MFU | Peak MiB | Total FLOPs | Training seconds |
| --- | ---: | ---: | ---: | ---: | ---: |
| m10-bias-free | 46227.8 | 0.00384066 | 28.6328 | 1.51349e+11 | 1.40072 |
| m10-biased | 42288.3 | 0.00351337 | 28.6777 | 1.51349e+11 | 1.49954 |

## Ranking blockers

- m10-bias-free: base_eval.json is missing; nanochat_compat_v1 result is missing; full_documents_v1 result is missing
- m10-biased: base_eval.json is missing; nanochat_compat_v1 result is missing; full_documents_v1 result is missing

## Compatibility BPB

| Run | BPB | Source-byte retention | Rank |
| --- | ---: | ---: | ---: |
| m10-bias-free | — | — | — |
| m10-biased | — | — | — |

## Full-document BPB

| Run | BPB | Source-byte retention | Rank |
| --- | ---: | ---: | ---: |
| m10-bias-free | — | — | — |
| m10-biased | — | — | — |
