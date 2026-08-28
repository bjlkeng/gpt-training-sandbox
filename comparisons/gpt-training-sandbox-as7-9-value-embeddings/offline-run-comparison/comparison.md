# Training run comparison

Compared runs: `2`

## Identity differences

| Field | Values by run |
| --- | --- |
| config_identity | {"m10-value-off":"sha256:706645c44923bdec1bfea39b1315790c7df85e75e8b61b3a55a60f7878b30585","m10-value-on":"sha256:f64c5d31674dc87fbfb5a32cc30a35bc4c44ef372be9ea855fa3e001a405eadf"} |
| parameterization | {"m10-value-off":{"n_head":2,"n_kv_head":2,"sliding_window":{"final_layer_forced_full":true,"pattern":"L","resolved_layer_types":["L","L"],"resolved_left_windows":[null,null],"short_window_size":64},"tie_weights":true,"unique_parameters":444160,"use_gqa":false,"value_embeddings":{"enabled":false,"gate_channels":12,"gate_scale":3.0,"kv_width":128,"layer_indices":[],"placement":"alternating_by_final_layer_parity"}},"m10-value-on":{"n_head":2,"n_kv_head":2,"sliding_window":{"final_layer_forced_full":true,"pattern":"L","resolved_layer_types":["L","L"],"resolved_left_windows":[null,null],"short_window_size":64},"tie_weights":true,"unique_parameters":478104,"use_gqa":false,"value_embeddings":{"enabled":true,"gate_channels":12,"gate_scale":3.0,"kv_width":128,"layer_indices":[1],"placement":"alternating_by_final_layer_parity"}}} |

Unavailable across all runs: `checkpoint_identity`, `tokenizer_identity`, `validation_manifest_identity`, `code_identity`.

## Numeric deltas

| Metric | Baseline | Deltas by run |
| --- | --- | --- |
| best_compatibility_bpb | m10-value-off | {"m10-value-off":null,"m10-value-on":null} |
| best_full_document_bpb | m10-value-off | {"m10-value-off":null,"m10-value-on":null} |
| best_loss | m10-value-off | {"m10-value-off":0.0,"m10-value-on":0.04672431945800781} |
| latest_loss | m10-value-off | {"m10-value-off":0.0,"m10-value-on":0.005461931228637695} |
| tokens_per_second | m10-value-off | {"m10-value-off":0.0,"m10-value-on":24136.02733938529} |
| mfu | m10-value-off | {"m10-value-off":0.0,"m10-value-on":0.0020055315956701153} |
| peak_memory_mib | m10-value-off | {"m10-value-off":0.0,"m10-value-on":0.6484375} |

## Training summary

| Run | Status | Step | Configured parameters | Unique parameters | Processed tokens |
| --- | --- | ---: | ---: | ---: | ---: |
| m10-value-off | completed | 50 | 478080 | 444160 | 51200 |
| m10-value-on | completed | 50 | 512024 | 478104 | 51200 |

## Training quality

| Run | Latest loss | Best loss | Latest compatibility BPB | Best compatibility BPB | Latest full-document BPB | Best full-document BPB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| m10-value-off | 2.80959 | 2.57122 | — | — | — | — |
| m10-value-on | 2.81506 | 2.61794 | — | — | — | — |

## Performance

| Run | Tokens/sec | MFU | Peak MiB | Total FLOPs | Training seconds |
| --- | ---: | ---: | ---: | ---: | ---: |
| m10-value-off | 44926.5 | 0.00373255 | 28.6328 | 1.51349e+11 | 1.41182 |
| m10-value-on | 69062.5 | 0.00573808 | 29.2812 | 1.51356e+11 | 1.57496 |

## Ranking blockers

- m10-value-off: base_eval.json is missing; nanochat_compat_v1 result is missing; full_documents_v1 result is missing
- m10-value-on: base_eval.json is missing; nanochat_compat_v1 result is missing; full_documents_v1 result is missing

## Compatibility BPB

| Run | BPB | Source-byte retention | Rank |
| --- | ---: | ---: | ---: |
| m10-value-off | — | — | — |
| m10-value-on | — | — | — |

## Full-document BPB

| Run | BPB | Source-byte retention | Rank |
| --- | ---: | ---: | ---: |
| m10-value-off | — | — | — |
| m10-value-on | — | — | — |
