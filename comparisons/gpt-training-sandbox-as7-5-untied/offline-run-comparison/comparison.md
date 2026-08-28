# Training run comparison

Compared runs: `2`

## Identity differences

| Field | Values by run |
| --- | --- |
| config_identity | {"m10-weight-tied":"sha256:f34a5dd541ec833ea1641abcf9bc48686725638d2678c2534f91527346463250","m10-weight-untied":"sha256:116c385872f9d7bcd719646a9fb3e917b9d12877a8e289c873f97a2b772fe24c"} |
| parameterization | {"m10-weight-tied":{"tie_weights":true,"unique_parameters":444160},"m10-weight-untied":{"tie_weights":false,"unique_parameters":478080}} |

Unavailable across all runs: `checkpoint_identity`, `tokenizer_identity`, `validation_manifest_identity`, `code_identity`.

## Numeric deltas

| Metric | Baseline | Deltas by run |
| --- | --- | --- |
| best_compatibility_bpb | m10-weight-tied | {"m10-weight-tied":null,"m10-weight-untied":null} |
| best_full_document_bpb | m10-weight-tied | {"m10-weight-tied":null,"m10-weight-untied":null} |
| best_loss | m10-weight-tied | {"m10-weight-tied":0.0,"m10-weight-untied":-0.9810471534729004} |
| latest_loss | m10-weight-tied | {"m10-weight-tied":0.0,"m10-weight-untied":-1.0771245956420898} |
| tokens_per_second | m10-weight-tied | {"m10-weight-tied":0.0,"m10-weight-untied":-986.854512158141} |
| mfu | m10-weight-tied | {"m10-weight-tied":0.0,"m10-weight-untied":-8.198913764148e-05} |
| peak_memory_mib | m10-weight-tied | {"m10-weight-tied":0.0,"m10-weight-untied":0.517578125} |

## Training summary

| Run | Status | Step | Configured parameters | Unique parameters | Processed tokens |
| --- | --- | ---: | ---: | ---: | ---: |
| m10-weight-tied | completed | 50 | 478080 | 444160 | 51200 |
| m10-weight-untied | completed | 50 | 478080 | 478080 | 51200 |

## Training quality

| Run | Latest loss | Best loss | Latest compatibility BPB | Best compatibility BPB | Latest full-document BPB | Best full-document BPB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| m10-weight-tied | 2.80959 | 2.57122 | — | — | — | — |
| m10-weight-untied | 1.73247 | 1.59017 | — | — | — | — |

## Performance

| Run | Tokens/sec | MFU | Peak MiB | Total FLOPs | Training seconds |
| --- | ---: | ---: | ---: | ---: | ---: |
| m10-weight-tied | 44923.3 | 0.00373229 | 28.6328 | 1.51349e+11 | 1.40099 |
| m10-weight-untied | 43936.5 | 0.0036503 | 29.1504 | 1.51349e+11 | 1.40931 |

## Ranking blockers

- m10-weight-tied: base_eval.json is missing; nanochat_compat_v1 result is missing; full_documents_v1 result is missing
- m10-weight-untied: base_eval.json is missing; nanochat_compat_v1 result is missing; full_documents_v1 result is missing

## Compatibility BPB

| Run | BPB | Source-byte retention | Rank |
| --- | ---: | ---: | ---: |
| m10-weight-tied | — | — | — |
| m10-weight-untied | — | — | — |

## Full-document BPB

| Run | BPB | Source-byte retention | Rank |
| --- | ---: | ---: | ---: |
| m10-weight-tied | — | — | — |
| m10-weight-untied | — | — | — |
