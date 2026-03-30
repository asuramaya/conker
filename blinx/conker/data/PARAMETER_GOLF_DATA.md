# Parameter Golf Data

`conker/` now treats the official Parameter Golf dataset as the primary truth source.

Expected local layout:

- `conker/data/datasets/fineweb10B_sp1024/`
- `conker/data/tokenizers/fineweb_1024_bpe.model`

Official upstream workflow:

```bash
python3 data/cached_challenge_fineweb.py --variant sp1024
```

That command comes from the official `openai/parameter-golf` repo and downloads:

- the fixed `fineweb_val_*` split in full
- a prefix of the same frozen shuffled train export
- the matching tokenizer files

Recommended placement for this repo:

1. clone `openai/parameter-golf` somewhere local
2. run the downloader there
3. copy or symlink the resulting `data/datasets/fineweb10B_sp1024` and `data/tokenizers` into `conker/data/`

Helper:

```bash
zsh conker/scripts/link_parameter_golf_data.zsh /path/to/parameter-golf/data
```

For local queues here:

- set `CONKER_GOLF_DATA_ROOT=/absolute/path/to/fineweb10B_sp1024`
- or place the dataset at `conker/data/datasets/fineweb10B_sp1024`

Notes:

- `text8` is now smoke-only for `conker`
- `conker` experiments should prefer official golf shards over archive corpora
- local bridge scripts currently report token loss / bits-per-token; official leaderboard scoring still requires the challenge's final evaluation path
