# Training run comparison

Compared runs: `2`

## Identity differences

| Field | Values by run |
| --- | --- |
| config_identity | {"m10-gqa-2kv":"sha256:abf6491c4c0427ecf392168802339f68402055b19d72cc0016a4e3cd39046e95","m10-gqa-mha":"sha256:160ab5233fa46720ce85e308d8029ef9b837748aa3681a0c34ea9fb419d32bc5"} |
| parameterization | {"m10-gqa-2kv":{"n_head":4,"n_kv_head":2,"tie_weights":true,"unique_parameters":411392,"use_gqa":true},"m10-gqa-mha":{"n_head":4,"n_kv_head":4,"tie_weights":true,"unique_parameters":444160,"use_gqa":false}} |

Unavailable across all runs: `checkpoint_identity`, `tokenizer_identity`, `validation_manifest_identity`, `code_identity`.

## Numeric deltas

| Metric | Baseline | Deltas by run |
| --- | --- | --- |
| best_compatibility_bpb | m10-gqa-2kv | {"m10-gqa-2kv":null,"m10-gqa-mha":null} |
| best_full_document_bpb | m10-gqa-2kv | {"m10-gqa-2kv":null,"m10-gqa-mha":null} |
| best_loss | m10-gqa-2kv | {"m10-gqa-2kv":0.0,"m10-gqa-mha":-0.07361602783203125} |
| latest_loss | m10-gqa-2kv | {"m10-gqa-2kv":0.0,"m10-gqa-mha":-0.018639802932739258} |
| tokens_per_second | m10-gqa-2kv | {"m10-gqa-2kv":0.0,"m10-gqa-mha":-26687.42783026264} |
| mfu | m10-gqa-2kv | {"m10-gqa-2kv":0.0,"m10-gqa-mha":-0.001821375441772868} |
| peak_memory_mib | m10-gqa-2kv | {"m10-gqa-2kv":0.0,"m10-gqa-mha":0.5} |

## Training summary

| Run | Status | Step | Configured parameters | Unique parameters | Processed tokens |
| --- | --- | ---: | ---: | ---: | ---: |
| m10-gqa-2kv | completed | 50 | 445312 | 411392 | 51200 |
| m10-gqa-mha | completed | 50 | 478080 | 444160 | 51200 |

## Training quality

| Run | Latest loss | Best loss | Latest compatibility BPB | Best compatibility BPB | Latest full-document BPB | Best full-document BPB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| m10-gqa-2kv | 2.79274 | 2.60605 | — | — | — | — |
| m10-gqa-mha | 2.7741 | 2.53243 | — | — | — | — |

## Performance

| Run | Tokens/sec | MFU | Peak MiB | Total FLOPs | Training seconds |
| --- | ---: | ---: | ---: | ---: | ---: |
| m10-gqa-2kv | 71636.7 | 0.00555582 | 28.6328 | 1.41283e+11 | 1.40707 |
| m10-gqa-mha | 44949.3 | 0.00373444 | 29.1328 | 1.51349e+11 | 1.40977 |

## Ranking blockers

- m10-gqa-2kv: base_eval.json is missing; nanochat_compat_v1 result is missing; full_documents_v1 result is missing
- m10-gqa-mha: base_eval.json is missing; nanochat_compat_v1 result is missing; full_documents_v1 result is missing

## Compatibility BPB

| Run | BPB | Source-byte retention | Rank |
| --- | ---: | ---: | ---: |
| m10-gqa-2kv | — | — | — |
| m10-gqa-mha | — | — | — |

## Full-document BPB

| Run | BPB | Source-byte retention | Rank |
| --- | ---: | ---: | ---: |
| m10-gqa-2kv | — | — | — |
| m10-gqa-mha | — | — | — |
