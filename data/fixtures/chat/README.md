# Tiny chat fixtures

`train.jsonl` and `validation.jsonl` are deliberately small, tracked SFT
fixtures for deterministic CPU tests. Each UTF-8 line is one canonical JSON
object with `schema_version: 1` and a non-empty `messages` array. A conversation
may begin with one `system` string, then alternates `user` and `assistant`
messages beginning with a user. User and system content is always a string.
Assistant content is either a string or a non-empty ordered list of `text`,
`python`, and `python_output` parts.

The `scratch_llm_chat_renderer_v1` token order is:

```text
bos
user_start, user text, user_end
assistant_start, assistant content, assistant_end
...
```

A leading system string is copied into the first user string as
`system + "\n\n" + user`; caller data is never changed. The loss mask supervises
assistant text, `assistant_end`, and assistant-authored Python calls including
their delimiters. BOS, user spans, `assistant_start`, Python outputs, and output
delimiters are masked. The batch boundary shifts once with `mask[1:]` and uses
only `-1` as the ignored label.

Completion prompts use the same rendering and must end in a user message. They
append exactly one `assistant_start` and do not invent an `assistant_end`.

The delimiter semantics are pinned for parity to karpathy/nanochat commit
`92d63d4e8bb4df75c3b71618f31ddde2378b2bcd`; validation and immutable domain
objects are local safety additions.
