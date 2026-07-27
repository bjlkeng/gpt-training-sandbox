# Regex chunk parity fixtures

`regex_chunks.json` stores network-independent expected chunks for the locked
nanochat split pattern. The cases cover empty text, ASCII, case-insensitive
contractions, one- and two-digit Unicode number groups, ambiguous whitespace,
tabs, CRLF/newlines, punctuation adjoining newlines, code, math, multiple
Unicode scripts, Korean, and emoji.

The fixture was verified against `nanochat/tokenizer.py` at commit
`41865401f73ff1c5321ae53297bceb2b78d4c8b4`. Tests do not import nanochat,
RustBPE, or tiktoken and do not use the network.

## Refresh procedure

1. Choose and record an upstream nanochat commit. Retrieve
   `nanochat/tokenizer.py` at that exact commit.
2. Compare its `SPLIT_PATTERN` literal byte-for-byte with both
   `scratch_llm.regex_chunking.SPLIT_PATTERN` and the fixture's `pattern`.
   If it differs, update the roadmap decision before changing local semantics;
   do not silently regenerate expected chunks.
3. Compile the confirmed literal with the project's third-party `regex`
   dependency. Evaluate every existing `text`, inspect each ambiguous boundary
   manually, and add cases for any newly relevant behavior.
4. Update only the reviewed `chunks` and `upstream.commit` fields, then run
   `uv run --extra dev pytest -q tests/test_regex_chunking.py`. The test also
   verifies exact reconstruction and the pinned provenance contract.
