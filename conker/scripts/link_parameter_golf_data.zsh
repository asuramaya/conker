#!/bin/zsh
set -euo pipefail

ROOT=/Users/asuramaya/Code/carving_machine_v3
CONKER="$ROOT/conker"
SRC_ROOT="${1:-}"

if [[ -z "$SRC_ROOT" ]]; then
  echo "usage: zsh conker/scripts/link_parameter_golf_data.zsh /path/to/parameter-golf/data"
  exit 1
fi

SRC_ROOT="${SRC_ROOT:A}"
DATASET_SRC="$SRC_ROOT/datasets/fineweb10B_sp1024"
TOKENIZER_SRC="$SRC_ROOT/tokenizers"
DATASET_DST="$CONKER/data/datasets/fineweb10B_sp1024"
TOKENIZER_DST="$CONKER/data/tokenizers"

if [[ ! -d "$DATASET_SRC" ]]; then
  echo "missing dataset source: $DATASET_SRC"
  exit 1
fi

if [[ ! -d "$TOKENIZER_SRC" ]]; then
  echo "missing tokenizer source: $TOKENIZER_SRC"
  exit 1
fi

mkdir -p "$CONKER/data/datasets"
rm -f "$DATASET_DST"
rm -f "$TOKENIZER_DST"
ln -s "$DATASET_SRC" "$DATASET_DST"
ln -s "$TOKENIZER_SRC" "$TOKENIZER_DST"

echo "linked:"
echo "  $DATASET_DST -> $DATASET_SRC"
echo "  $TOKENIZER_DST -> $TOKENIZER_SRC"
