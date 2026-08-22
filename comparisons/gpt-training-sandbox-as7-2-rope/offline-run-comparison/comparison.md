# Training run comparison

Compared runs: `2`

## Identity differences

| Field | Values by run |
| --- | --- |
| config_identity | {"m10-rope":"sha256:4a128df142a80d8e273de83f4ca4787b2de6d27747d990e349eaf44e4455564b","m10-rope-learned":"sha256:88f8a55feac494202d6b608169269d049ad1c56f2b42ee44ba30dfea6eb80fac"} |

Unavailable across all runs: `checkpoint_identity`, `tokenizer_identity`, `validation_manifest_identity`, `code_identity`.

## Numeric deltas

| Metric | Baseline | Deltas by run |
| --- | --- | --- |
| best_compatibility_bpb | m10-rope | {"m10-rope":null,"m10-rope-learned":null} |
| best_full_document_bpb | m10-rope | {"m10-rope":null,"m10-rope-learned":null} |
| best_loss | m10-rope | {"m10-rope":0.0,"m10-rope-learned":0.5211174488067627} |
| latest_loss | m10-rope | {"m10-rope":0.0,"m10-rope-learned":0.6828172206878662} |
| tokens_per_second | m10-rope | {"m10-rope":0.0,"m10-rope-learned":12371.749819521247} |
| mfu | m10-rope | {"m10-rope":0.0,"m10-rope-learned":0.0010278608308740595} |
| peak_memory_mib | m10-rope | {"m10-rope":0.0,"m10-rope-learned":0.0009765625} |

## Training summary

| Run | Status | Step | Configured parameters | Unique parameters | Processed tokens |
| --- | --- | ---: | ---: | ---: | ---: |
| m10-rope | completed | 50 | 461696 | 427776 | 51200 |
| m10-rope-learned | completed | 50 | 478080 | 444160 | 51200 |

## Training quality

| Run | Latest loss | Best loss | Latest compatibility BPB | Best compatibility BPB | Latest full-document BPB | Best full-document BPB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| m10-rope | 2.12678 | 2.0501 | — | — | — | — |
| m10-rope-learned | 2.80959 | 2.57122 | — | — | — | — |

## Performance

| Run | Tokens/sec | MFU | Peak MiB | Total FLOPs | Training seconds |
| --- | ---: | ---: | ---: | ---: | ---: |
| m10-rope | 33551.4 | 0.00278749 | 28.6318 | 1.51349e+11 | 1.91425 |
| m10-rope-learned | 45923.2 | 0.00381536 | 28.6328 | 1.51349e+11 | 1.40162 |

## Ranking blockers

- m10-rope: base_eval.json is missing; nanochat_compat_v1 result is missing; full_documents_v1 result is missing
- m10-rope-learned: base_eval.json is missing; nanochat_compat_v1 result is missing; full_documents_v1 result is missing

## Compatibility BPB

| Run | BPB | Source-byte retention | Rank |
| --- | ---: | ---: | ---: |
| m10-rope | — | — | — |
| m10-rope-learned | — | — | — |

## Full-document BPB

| Run | BPB | Source-byte retention | Rank |
| --- | ---: | ---: | ---: |
| m10-rope | — | — | — |
| m10-rope-learned | — | — | — |
