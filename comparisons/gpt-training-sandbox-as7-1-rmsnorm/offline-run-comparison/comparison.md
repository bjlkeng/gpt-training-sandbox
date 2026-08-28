# Training run comparison

Compared runs: `2`

## Identity differences

| Field | Values by run |
| --- | --- |
| config_identity | {"m10-rmsnorm-layernorm-native-r3":"sha256:af5f0b02c7634fb1f5f9fa288b768ee2f772ab8fa79cf2afd6562353d0fbdb1c","m10-rmsnorm-native-r3":"sha256:2c63a9c4d1b512594b129d421b5c2614cd52294ad9d90d7859956679f595e988"} |

Unavailable across all runs: `checkpoint_identity`, `tokenizer_identity`, `validation_manifest_identity`, `code_identity`.

## Numeric deltas

| Metric | Baseline | Deltas by run |
| --- | --- | --- |
| best_compatibility_bpb | m10-rmsnorm-layernorm-native-r3 | {"m10-rmsnorm-layernorm-native-r3":null,"m10-rmsnorm-native-r3":null} |
| best_full_document_bpb | m10-rmsnorm-layernorm-native-r3 | {"m10-rmsnorm-layernorm-native-r3":null,"m10-rmsnorm-native-r3":null} |
| best_loss | m10-rmsnorm-layernorm-native-r3 | {"m10-rmsnorm-layernorm-native-r3":0.0,"m10-rmsnorm-native-r3":-0.014404773712158203} |
| latest_loss | m10-rmsnorm-layernorm-native-r3 | {"m10-rmsnorm-layernorm-native-r3":0.0,"m10-rmsnorm-native-r3":-0.040984392166137695} |
| tokens_per_second | m10-rmsnorm-layernorm-native-r3 | {"m10-rmsnorm-layernorm-native-r3":0.0,"m10-rmsnorm-native-r3":-2.8630223633590504} |
| mfu | m10-rmsnorm-layernorm-native-r3 | {"m10-rmsnorm-layernorm-native-r3":0.0,"m10-rmsnorm-native-r3":-2.3786356725084834e-07} |
| peak_memory_mib | m10-rmsnorm-layernorm-native-r3 | {"m10-rmsnorm-layernorm-native-r3":0.0,"m10-rmsnorm-native-r3":-0.0146484375} |

## Training summary

| Run | Status | Step | Configured parameters | Unique parameters | Processed tokens |
| --- | --- | ---: | ---: | ---: | ---: |
| m10-rmsnorm-layernorm-native-r3 | completed | 50 | 478080 | 444160 | 51200 |
| m10-rmsnorm-native-r3 | completed | 50 | 477440 | 443520 | 51200 |

## Training quality

| Run | Latest loss | Best loss | Latest compatibility BPB | Best compatibility BPB | Latest full-document BPB | Best full-document BPB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| m10-rmsnorm-layernorm-native-r3 | 2.80959 | 2.57122 | — | — | — | — |
| m10-rmsnorm-native-r3 | 2.76861 | 2.55682 | — | — | — | — |

## Performance

| Run | Tokens/sec | MFU | Peak MiB | Total FLOPs | Training seconds |
| --- | ---: | ---: | ---: | ---: | ---: |
| m10-rmsnorm-layernorm-native-r3 | 45405.3 | 0.00377233 | 28.6328 | 1.51349e+11 | 1.39489 |
| m10-rmsnorm-native-r3 | 45402.4 | 0.00377209 | 28.6182 | 1.51349e+11 | 1.39454 |

## Ranking blockers

- m10-rmsnorm-layernorm-native-r3: base_eval.json is missing; nanochat_compat_v1 result is missing; full_documents_v1 result is missing
- m10-rmsnorm-native-r3: base_eval.json is missing; nanochat_compat_v1 result is missing; full_documents_v1 result is missing

## Compatibility BPB

| Run | BPB | Source-byte retention | Rank |
| --- | ---: | ---: | ---: |
| m10-rmsnorm-layernorm-native-r3 | — | — | — |
| m10-rmsnorm-native-r3 | — | — | — |

## Full-document BPB

| Run | BPB | Source-byte retention | Rank |
| --- | ---: | ---: | ---: |
| m10-rmsnorm-layernorm-native-r3 | — | — | — |
| m10-rmsnorm-native-r3 | — | — | — |
