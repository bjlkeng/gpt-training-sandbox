# Training run comparison

Compared runs: `3`

## Identity differences

| Field | Values by run |
| --- | --- |
| config_identity | {"m10-scalars-neutral":"sha256:4cf8c7028bcc6f719ab30fbfb321e1a1bd4cfa2cb71d884c3acb75e11b020ef5","m10-scalars-off":"sha256:4b8a22c19517d8a51ba9af0c6f9c732722419f897a1cdc78151207e06a694fbc","m10-scalars-pinned":"sha256:4e1d0193fc06b83e50395a43220ba3a5714e882b595c2a41b6e7726e5eb59735"} |
| parameterization | {"m10-scalars-neutral":{"n_head":2,"n_kv_head":2,"residual_scalars":{"enabled":true,"initializer":"neutral","input_initial_values":[0.0,0.0],"input_source":"parameter_free_rmsnorm_initial_token_representation","placement":"before_each_transformer_block","residual_initial_values":[1.0,1.0]},"sliding_window":{"final_layer_forced_full":true,"pattern":"L","resolved_layer_types":["L","L"],"resolved_left_windows":[null,null],"short_window_size":64},"tie_weights":true,"unique_parameters":444164,"use_gqa":false,"value_embeddings":{"enabled":false,"gate_channels":12,"gate_scale":3.0,"kv_width":128,"layer_indices":[],"placement":"alternating_by_final_layer_parity"}},"m10-scalars-off":{"n_head":2,"n_kv_head":2,"residual_scalars":{"enabled":false,"initializer":"neutral","input_initial_values":[],"input_source":"parameter_free_rmsnorm_initial_token_representation","placement":"before_each_transformer_block","residual_initial_values":[]},"sliding_window":{"final_layer_forced_full":true,"pattern":"L","resolved_layer_types":["L","L"],"resolved_left_windows":[null,null],"short_window_size":64},"tie_weights":true,"unique_parameters":444160,"use_gqa":false,"value_embeddings":{"enabled":false,"gate_channels":12,"gate_scale":3.0,"kv_width":128,"layer_indices":[],"placement":"alternating_by_final_layer_parity"}},"m10-scalars-pinned":{"n_head":2,"n_kv_head":2,"residual_scalars":{"enabled":true,"initializer":"nanochat_depth","input_initial_values":[0.2,0.05000000000000002],"input_source":"parameter_free_rmsnorm_initial_token_representation","placement":"before_each_transformer_block","residual_initial_values":[1.15,1.0499999999999998]},"sliding_window":{"final_layer_forced_full":true,"pattern":"L","resolved_layer_types":["L","L"],"resolved_left_windows":[null,null],"short_window_size":64},"tie_weights":true,"unique_parameters":444164,"use_gqa":false,"value_embeddings":{"enabled":false,"gate_channels":12,"gate_scale":3.0,"kv_width":128,"layer_indices":[],"placement":"alternating_by_final_layer_parity"}}} |

Unavailable across all runs: `checkpoint_identity`, `tokenizer_identity`, `validation_manifest_identity`, `code_identity`.

## Numeric deltas

| Metric | Baseline | Deltas by run |
| --- | --- | --- |
| best_compatibility_bpb | m10-scalars-neutral | {"m10-scalars-neutral":null,"m10-scalars-off":null,"m10-scalars-pinned":null} |
| best_full_document_bpb | m10-scalars-neutral | {"m10-scalars-neutral":null,"m10-scalars-off":null,"m10-scalars-pinned":null} |
| best_loss | m10-scalars-neutral | {"m10-scalars-neutral":0.0,"m10-scalars-off":0.15939974784851074,"m10-scalars-pinned":0.02498912811279297} |
| latest_loss | m10-scalars-neutral | {"m10-scalars-neutral":0.0,"m10-scalars-off":0.27025771141052246,"m10-scalars-pinned":0.06436944007873535} |
| tokens_per_second | m10-scalars-neutral | {"m10-scalars-neutral":0.0,"m10-scalars-off":5894.810106656078,"m10-scalars-pinned":-300.0053284569876} |
| mfu | m10-scalars-neutral | {"m10-scalars-neutral":0.0,"m10-scalars-off":0.0004897483785609551,"m10-scalars-pinned":-2.4924827180701926e-05} |
| peak_memory_mib | m10-scalars-neutral | {"m10-scalars-neutral":0.0,"m10-scalars-off":-0.3798828125,"m10-scalars-pinned":0.0} |

## Training summary

| Run | Status | Step | Configured parameters | Unique parameters | Processed tokens |
| --- | --- | ---: | ---: | ---: | ---: |
| m10-scalars-neutral | completed | 50 | 478084 | 444164 | 51200 |
| m10-scalars-off | completed | 50 | 478080 | 444160 | 51200 |
| m10-scalars-pinned | completed | 50 | 478084 | 444164 | 51200 |

## Training quality

| Run | Latest loss | Best loss | Latest compatibility BPB | Best compatibility BPB | Latest full-document BPB | Best full-document BPB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| m10-scalars-neutral | 2.53934 | 2.41182 | — | — | — | — |
| m10-scalars-off | 2.80959 | 2.57122 | — | — | — | — |
| m10-scalars-pinned | 2.60371 | 2.43681 | — | — | — | — |

## Performance

| Run | Tokens/sec | MFU | Peak MiB | Total FLOPs | Training seconds |
| --- | ---: | ---: | ---: | ---: | ---: |
| m10-scalars-neutral | 40178.7 | 0.0033381 | 29.0127 | 1.51349e+11 | 1.55612 |
| m10-scalars-off | 46073.6 | 0.00382785 | 28.6328 | 1.51349e+11 | 1.39875 |
| m10-scalars-pinned | 39878.7 | 0.00331318 | 29.0127 | 1.51349e+11 | 1.54509 |

## Ranking blockers

- m10-scalars-neutral: base_eval.json is missing; nanochat_compat_v1 result is missing; full_documents_v1 result is missing
- m10-scalars-off: base_eval.json is missing; nanochat_compat_v1 result is missing; full_documents_v1 result is missing
- m10-scalars-pinned: base_eval.json is missing; nanochat_compat_v1 result is missing; full_documents_v1 result is missing

## Compatibility BPB

| Run | BPB | Source-byte retention | Rank |
| --- | ---: | ---: | ---: |
| m10-scalars-neutral | — | — | — |
| m10-scalars-off | — | — | — |
| m10-scalars-pinned | — | — | — |

## Full-document BPB

| Run | BPB | Source-byte retention | Rank |
| --- | ---: | ---: | ---: |
| m10-scalars-neutral | — | — | — |
| m10-scalars-off | — | — | — |
| m10-scalars-pinned | — | — | — |
