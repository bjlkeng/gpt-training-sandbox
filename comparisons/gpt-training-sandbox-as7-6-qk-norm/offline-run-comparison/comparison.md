# Training run comparison

Compared runs: `2`

## Identity differences

| Field | Values by run |
| --- | --- |
| config_identity | {"m10-qk-norm-off":"sha256:6e11ff21a02a4a70e9d1704a79f71e0eae2af72ed8330eb157bdf21d556dc688","m10-qk-norm-on":"sha256:98cc65a23f664fe224799c50756862ff964643dba7aa0bd41f97f63a592521b8"} |

Unavailable across all runs: `checkpoint_identity`, `tokenizer_identity`, `validation_manifest_identity`, `code_identity`.

## Numeric deltas

| Metric | Baseline | Deltas by run |
| --- | --- | --- |
| best_compatibility_bpb | m10-qk-norm-off | {"m10-qk-norm-off":null,"m10-qk-norm-on":null} |
| best_full_document_bpb | m10-qk-norm-off | {"m10-qk-norm-off":null,"m10-qk-norm-on":null} |
| best_loss | m10-qk-norm-off | {"m10-qk-norm-off":0.0,"m10-qk-norm-on":-0.1332252025604248} |
| latest_loss | m10-qk-norm-off | {"m10-qk-norm-off":0.0,"m10-qk-norm-on":-0.16076111793518066} |
| tokens_per_second | m10-qk-norm-off | {"m10-qk-norm-off":0.0,"m10-qk-norm-on":-2465.807218482405} |
| mfu | m10-qk-norm-off | {"m10-qk-norm-off":0.0,"m10-qk-norm-on":-0.00020486242393662028} |
| peak_memory_mib | m10-qk-norm-off | {"m10-qk-norm-off":0.0,"m10-qk-norm-on":0.7578125} |

## Training summary

| Run | Status | Step | Configured parameters | Unique parameters | Processed tokens |
| --- | --- | ---: | ---: | ---: | ---: |
| m10-qk-norm-off | completed | 50 | 478080 | 444160 | 51200 |
| m10-qk-norm-on | completed | 50 | 478080 | 444160 | 51200 |

## Training quality

| Run | Latest loss | Best loss | Latest compatibility BPB | Best compatibility BPB | Latest full-document BPB | Best full-document BPB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| m10-qk-norm-off | 2.80959 | 2.57122 | — | — | — | — |
| m10-qk-norm-on | 2.64883 | 2.43799 | — | — | — | — |

## Performance

| Run | Tokens/sec | MFU | Peak MiB | Total FLOPs | Training seconds |
| --- | ---: | ---: | ---: | ---: | ---: |
| m10-qk-norm-off | 45001.8 | 0.00373881 | 28.6328 | 1.51349e+11 | 1.40511 |
| m10-qk-norm-on | 42536 | 0.00353395 | 29.3906 | 1.51349e+11 | 1.47628 |

## Ranking blockers

- m10-qk-norm-off: base_eval.json is missing; nanochat_compat_v1 result is missing; full_documents_v1 result is missing
- m10-qk-norm-on: base_eval.json is missing; nanochat_compat_v1 result is missing; full_documents_v1 result is missing

## Compatibility BPB

| Run | BPB | Source-byte retention | Rank |
| --- | ---: | ---: | ---: |
| m10-qk-norm-off | — | — | — |
| m10-qk-norm-on | — | — | — |

## Full-document BPB

| Run | BPB | Source-byte retention | Rank |
| --- | ---: | ---: | ---: |
| m10-qk-norm-off | — | — | — |
| m10-qk-norm-on | — | — | — |
