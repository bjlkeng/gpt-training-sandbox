# Synthetic parquet fixture provenance

The three parquet shards in this directory contain synthetic text authored for this repository.
They are not copied or adapted from any external corpus and are distributed
under the repository's MIT license.

- `shard_00000.parquet` and `shard_00001.parquet` are the available training
  prefix.
- `shard_06542.parquet` deliberately uses ClimbMix's fixed final validation
  index so partial local datasets exercise the same split convention as a full
  download.

Each shard has one UTF-8 `text` column and multiple row groups. Together they
cover ASCII, Unicode, and empty-string documents. The corpus can be reproduced
with the pinned PyArrow dependency by running:

```bash
uv run python data/fixtures/parquet/generate_fixture.py
```

`generate_fixture.py` contains the complete source rows and deterministic writer
settings used for the checked-in files.
