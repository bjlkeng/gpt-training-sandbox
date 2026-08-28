# Training run comparison

Compared runs: `2`

## Identity differences

| Field | Values by run |
| --- | --- |
| config_identity | {"m10-window-full":"sha256:f0cb9c64728577ed5e518b61b79b4c478fd2ee02e44a0b25d6cc9601831b5f2d","m10-window-s16":"sha256:d86ad409a3948d35b3771c9d284c0aef21ac62488fab1c058396d37ec13586f2"} |
| parameterization | {"m10-window-full":{"n_head":2,"n_kv_head":2,"sliding_window":{"final_layer_forced_full":true,"pattern":"L","resolved_layer_types":["L","L"],"resolved_left_windows":[null,null],"short_window_size":64},"tie_weights":true,"unique_parameters":444160,"use_gqa":false},"m10-window-s16":{"n_head":2,"n_kv_head":2,"sliding_window":{"final_layer_forced_full":true,"pattern":"S","resolved_layer_types":["S","L"],"resolved_left_windows":[16,null],"short_window_size":16},"tie_weights":true,"unique_parameters":444160,"use_gqa":false}} |

Unavailable across all runs: `checkpoint_identity`, `tokenizer_identity`, `validation_manifest_identity`, `code_identity`.

## Numeric deltas

| Metric | Baseline | Deltas by run |
| --- | --- | --- |
| best_compatibility_bpb | m10-window-full | {"m10-window-full":null,"m10-window-s16":null} |
| best_full_document_bpb | m10-window-full | {"m10-window-full":null,"m10-window-s16":null} |
| best_loss | m10-window-full | {"m10-window-full":0.0,"m10-window-s16":0.0505523681640625} |
| latest_loss | m10-window-full | {"m10-window-full":0.0,"m10-window-s16":0.029610872268676758} |
| tokens_per_second | m10-window-full | {"m10-window-full":0.0,"m10-window-s16":-4860.62405564115} |
| mfu | m10-window-full | {"m10-window-full":0.0,"m10-window-s16":-0.0005975980731428641} |
| peak_memory_mib | m10-window-full | {"m10-window-full":0.0,"m10-window-s16":3.75439453125} |

## Training summary

| Run | Status | Step | Configured parameters | Unique parameters | Processed tokens |
| --- | --- | ---: | ---: | ---: | ---: |
| m10-window-full | completed | 50 | 478080 | 444160 | 51200 |
| m10-window-s16 | completed | 50 | 478080 | 444160 | 51200 |

## Training quality

| Run | Latest loss | Best loss | Latest compatibility BPB | Best compatibility BPB | Latest full-document BPB | Best full-document BPB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| m10-window-full | 2.80959 | 2.57122 | — | — | — | — |
| m10-window-s16 | 2.8392 | 2.62177 | — | — | — | — |

## Performance

| Run | Tokens/sec | MFU | Peak MiB | Total FLOPs | Training seconds |
| --- | ---: | ---: | ---: | ---: | ---: |
| m10-window-full | 45297.8 | 0.0037634 | 28.6328 | 1.51349e+11 | 1.40883 |
| m10-window-s16 | 40437.2 | 0.0031658 | 32.3872 | 1.42619e+11 | 1.84313 |

## Ranking blockers

- m10-window-full: base_eval.json is missing; nanochat_compat_v1 result is missing; full_documents_v1 result is missing
- m10-window-s16: base_eval.json is missing; nanochat_compat_v1 result is missing; full_documents_v1 result is missing

## Compatibility BPB

| Run | BPB | Source-byte retention | Rank |
| --- | ---: | ---: | ---: |
| m10-window-full | — | — | — |
| m10-window-s16 | — | — | — |

## Full-document BPB

| Run | BPB | Source-byte retention | Rank |
| --- | ---: | ---: | ---: |
| m10-window-full | — | — | — |
| m10-window-s16 | — | — | — |
