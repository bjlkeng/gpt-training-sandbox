# Training run comparison

Compared runs: `2`

## Identity differences

| Field | Values by run |
| --- | --- |
| config_identity | {"m10-activation-gelu":"sha256:b3193b3834474cfa8cf79d5948f43003ba016e45c8b8b165751365f1d77c8785","m10-activation-relu2":"sha256:12def0e25f3bff13699158dcc6a2471664abcdd92d093b3913c344272e6b9034"} |

Unavailable across all runs: `checkpoint_identity`, `tokenizer_identity`, `validation_manifest_identity`, `code_identity`.

## Numeric deltas

| Metric | Baseline | Deltas by run |
| --- | --- | --- |
| best_compatibility_bpb | m10-activation-gelu | {"m10-activation-gelu":null,"m10-activation-relu2":null} |
| best_full_document_bpb | m10-activation-gelu | {"m10-activation-gelu":null,"m10-activation-relu2":null} |
| best_loss | m10-activation-gelu | {"m10-activation-gelu":0.0,"m10-activation-relu2":-0.25092530250549316} |
| latest_loss | m10-activation-gelu | {"m10-activation-gelu":0.0,"m10-activation-relu2":-0.3768749237060547} |
| tokens_per_second | m10-activation-gelu | {"m10-activation-gelu":0.0,"m10-activation-relu2":-4462.91253889435} |
| mfu | m10-activation-gelu | {"m10-activation-gelu":0.0,"m10-activation-relu2":-0.00037078449348434354} |
| peak_memory_mib | m10-activation-gelu | {"m10-activation-gelu":0.0,"m10-activation-relu2":0.72607421875} |

## Training summary

| Run | Status | Step | Configured parameters | Unique parameters | Processed tokens |
| --- | --- | ---: | ---: | ---: | ---: |
| m10-activation-gelu | completed | 50 | 478080 | 444160 | 51200 |
| m10-activation-relu2 | completed | 50 | 478080 | 444160 | 51200 |

## Training quality

| Run | Latest loss | Best loss | Latest compatibility BPB | Best compatibility BPB | Latest full-document BPB | Best full-document BPB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| m10-activation-gelu | 2.80959 | 2.57122 | — | — | — | — |
| m10-activation-relu2 | 2.43272 | 2.32029 | — | — | — | — |

## Performance

| Run | Tokens/sec | MFU | Peak MiB | Total FLOPs | Training seconds |
| --- | ---: | ---: | ---: | ---: | ---: |
| m10-activation-gelu | 45844.6 | 0.00380883 | 28.6328 | 1.51349e+11 | 1.39578 |
| m10-activation-relu2 | 41381.7 | 0.00343804 | 29.3589 | 1.51349e+11 | 1.76776 |

## Ranking blockers

- m10-activation-gelu: base_eval.json is missing; nanochat_compat_v1 result is missing; full_documents_v1 result is missing
- m10-activation-relu2: base_eval.json is missing; nanochat_compat_v1 result is missing; full_documents_v1 result is missing

## Compatibility BPB

| Run | BPB | Source-byte retention | Rank |
| --- | ---: | ---: | ---: |
| m10-activation-gelu | — | — | — |
| m10-activation-relu2 | — | — | — |

## Full-document BPB

| Run | BPB | Source-byte retention | Rank |
| --- | ---: | ---: | ---: |
| m10-activation-gelu | — | — | — |
| m10-activation-relu2 | — | — | — |
