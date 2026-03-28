from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import mlx.core as mx
import numpy as np

from .config import RegimeSwitchConfig, Text8Config


REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass
class CharDataset:
    train_tokens: np.ndarray
    test_tokens: np.ndarray
    alphabet: str | tuple[str, ...]
    source_path: str
    source_names: tuple[str, ...] = ()
    train_switch_points: tuple[int, ...] = ()
    test_switch_points: tuple[int, ...] = ()
    tokenizer: str = "char"
    train_char_count: int | None = None
    test_char_count: int | None = None

    def __post_init__(self) -> None:
        vocab = tuple(self.alphabet) if not isinstance(self.alphabet, str) else tuple(self.alphabet)
        self.vocab = vocab
        self.char_to_idx = {token: idx for idx, token in enumerate(vocab)}
        self.idx_to_char = {idx: token for token, idx in self.char_to_idx.items()}
        self.train_char_count = self.train_char_count if self.train_char_count is not None else len(self.train_tokens)
        self.test_char_count = self.test_char_count if self.test_char_count is not None else len(self.test_tokens)

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    @property
    def train_tokens_per_char(self) -> float:
        return len(self.train_tokens) / max(self.train_char_count, 1)

    @property
    def test_tokens_per_char(self) -> float:
        return len(self.test_tokens) / max(self.test_char_count, 1)

    def _split_arrays(self, split: str) -> tuple[np.ndarray, tuple[int, ...]]:
        if split == "train":
            return self.train_tokens, self.train_switch_points
        if split == "test":
            return self.test_tokens, self.test_switch_points
        raise ValueError(f"Unknown split: {split}")

    def batch(self, split: str, batch_size: int, seq_len: int) -> tuple[mx.array, mx.array]:
        data, _ = self._split_arrays(split)
        max_start = len(data) - seq_len - 1
        if max_start <= 0:
            raise ValueError(
                f"Dataset split '{split}' is too short for seq_len={seq_len}. "
                f"Split length: {len(data)}."
            )
        starts = np.random.randint(0, max_start + 1, size=batch_size)
        x = np.stack([data[i : i + seq_len] for i in starts])
        y = np.stack([data[i + 1 : i + seq_len + 1] for i in starts])
        return mx.array(x), mx.array(y)

    def rollout_batch(
        self,
        split: str,
        batch_size: int,
        prompt_len: int,
        rollout_len: int,
        near_boundaries: bool = False,
        boundary_band: int = 128,
    ) -> tuple[mx.array, mx.array]:
        data, switch_points = self._split_arrays(split)
        max_start = len(data) - prompt_len - rollout_len
        if max_start < 0:
            raise ValueError(
                f"Dataset split '{split}' is too short for prompt_len={prompt_len} "
                f"and rollout_len={rollout_len}. Split length: {len(data)}."
            )

        starts = self._sample_rollout_starts(
            max_start=max_start,
            batch_size=batch_size,
            prompt_len=prompt_len,
            switch_points=switch_points,
            near_boundaries=near_boundaries,
            boundary_band=boundary_band,
        )
        prompts = np.stack([data[i : i + prompt_len] for i in starts])
        targets = np.stack([data[i + prompt_len : i + prompt_len + rollout_len] for i in starts])
        return mx.array(prompts), mx.array(targets)

    @staticmethod
    def _sample_rollout_starts(
        max_start: int,
        batch_size: int,
        prompt_len: int,
        switch_points: tuple[int, ...],
        near_boundaries: bool,
        boundary_band: int,
    ) -> np.ndarray:
        if not near_boundaries or not switch_points:
            return np.random.randint(0, max_start + 1, size=batch_size)

        candidates = []
        for boundary in switch_points:
            start_lo = max(0, boundary - prompt_len - boundary_band)
            start_hi = min(max_start, boundary - prompt_len)
            if start_hi < start_lo:
                continue
            candidates.append(np.arange(start_lo, start_hi + 1, dtype=np.int32))

        if not candidates:
            return np.random.randint(0, max_start + 1, size=batch_size)

        pool = np.unique(np.concatenate(candidates))
        replace = len(pool) < batch_size
        return np.random.choice(pool, size=batch_size, replace=replace)


Text8Data = CharDataset


class BPETokenizer:
    def __init__(self, vocab: tuple[str, ...], merges: tuple[tuple[str, str], ...]):
        self.vocab = vocab
        self.merges = merges
        self.token_to_idx = {token: idx for idx, token in enumerate(vocab)}
        self.merge_ranks = {pair: idx for idx, pair in enumerate(merges)}
        self._word_cache: dict[str, tuple[int, ...]] = {}

    def encode_word(self, word: str) -> tuple[int, ...]:
        cached = self._word_cache.get(word)
        if cached is not None:
            return cached
        symbols = tuple(word)
        while len(symbols) > 1:
            best_rank = None
            best_idx = -1
            for idx in range(len(symbols) - 1):
                rank = self.merge_ranks.get((symbols[idx], symbols[idx + 1]))
                if rank is None:
                    continue
                if best_rank is None or rank < best_rank:
                    best_rank = rank
                    best_idx = idx
            if best_rank is None:
                break
            merged = symbols[best_idx] + symbols[best_idx + 1]
            symbols = symbols[:best_idx] + (merged,) + symbols[best_idx + 2 :]
        encoded = tuple(self.token_to_idx[symbol] for symbol in symbols)
        self._word_cache[word] = encoded
        return encoded


def _resolve_bpe_cache_path(cache_path: str | None, vocab_size: int) -> Path:
    if cache_path:
        return Path(cache_path).expanduser()
    return REPO_ROOT / "data" / f"bpe_{vocab_size}.json"


def _space_prefixed_words(text: str) -> list[str]:
    words: list[str] = []
    idx = 0
    limit = len(text)
    while idx < limit:
        start = idx
        if text[idx] == " ":
            idx += 1
            while idx < limit and text[idx] == " ":
                idx += 1
            if idx >= limit:
                break
        while idx < limit and text[idx] != " ":
            idx += 1
        word = text[start:idx]
        if word:
            words.append(word)
    return words


def _merge_pair(symbols: tuple[str, ...], pair: tuple[str, str]) -> tuple[str, ...]:
    merged: list[str] = []
    idx = 0
    left, right = pair
    while idx < len(symbols):
        if idx < len(symbols) - 1 and symbols[idx] == left and symbols[idx + 1] == right:
            merged.append(left + right)
            idx += 2
            continue
        merged.append(symbols[idx])
        idx += 1
    return tuple(merged)


def _train_bpe_tokenizer(text: str, alphabet: str, vocab_size: int) -> BPETokenizer:
    if vocab_size <= len(alphabet):
        raise ValueError(f"BPE vocab_size={vocab_size} must exceed base alphabet size {len(alphabet)}.")

    word_counts: dict[str, int] = {}
    for word in _space_prefixed_words(text):
        word_counts[word] = word_counts.get(word, 0) + 1

    vocab_counts = {tuple(word): count for word, count in word_counts.items()}
    merges: list[tuple[str, str]] = []
    learned_tokens: list[str] = list(alphabet)

    while len(learned_tokens) < vocab_size:
        pair_counts: dict[tuple[str, str], int] = {}
        for symbols, count in vocab_counts.items():
            for idx in range(len(symbols) - 1):
                pair = (symbols[idx], symbols[idx + 1])
                pair_counts[pair] = pair_counts.get(pair, 0) + count
        if not pair_counts:
            break
        best_pair = max(pair_counts.items(), key=lambda item: (item[1], item[0]))[0]
        merged_symbol = best_pair[0] + best_pair[1]
        if merged_symbol in learned_tokens:
            break
        merges.append(best_pair)
        learned_tokens.append(merged_symbol)
        vocab_counts = {_merge_pair(symbols, best_pair): count for symbols, count in vocab_counts.items()}

    return BPETokenizer(tuple(learned_tokens), tuple(merges))


def _save_bpe_tokenizer(path: Path, tokenizer: BPETokenizer) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": 1,
        "vocab": list(tokenizer.vocab),
        "merges": [list(pair) for pair in tokenizer.merges],
    }
    path.write_text(json.dumps(payload, indent=2))


def _load_bpe_tokenizer(path: Path) -> BPETokenizer:
    payload = json.loads(path.read_text())
    vocab = tuple(payload["vocab"])
    merges = tuple(tuple(pair) for pair in payload["merges"])
    return BPETokenizer(vocab, merges)


def _shared_bpe_tokenizer(alphabet: str, vocab_size: int, cache_path: str | None, limit_chars: int = 10_000_000) -> BPETokenizer:
    path = _resolve_bpe_cache_path(cache_path, vocab_size)
    if path.exists():
        return _load_bpe_tokenizer(path)
    text8_path = resolve_text8_path(Text8Config(limit_chars=limit_chars))
    text = text8_path.read_text()[:limit_chars]
    tokenizer = _train_bpe_tokenizer(text, alphabet, vocab_size)
    _save_bpe_tokenizer(path, tokenizer)
    return tokenizer


def _encode_bpe_text(text: str, tokenizer: BPETokenizer) -> np.ndarray:
    token_ids: list[int] = []
    for word in _space_prefixed_words(text):
        token_ids.extend(tokenizer.encode_word(word))
    return np.array(token_ids, dtype=np.int32)


def resolve_text8_path(config: Text8Config) -> Path:
    candidates: list[Path] = []
    if config.path:
        candidates.append(Path(config.path).expanduser())

    env_path = None
    try:
        import os

        env_path = os.environ.get("CARVING_TEXT8")
    except Exception:
        env_path = None
    if env_path:
        candidates.append(Path(env_path).expanduser())

    candidates.extend(
        [
            REPO_ROOT / "data" / "text8",
            REPO_ROOT / "text8",
        ]
    )

    for candidate in candidates:
        if candidate.exists():
            return candidate

    hint_lines = [
        "Could not find the text8 dataset.",
        "Looked in:",
        *[f"  - {candidate}" for candidate in candidates],
        "Pass `--data /path/to/text8` or set `CARVING_TEXT8=/path/to/text8`.",
    ]
    raise FileNotFoundError("\n".join(hint_lines))


def load_text8(config: Text8Config) -> CharDataset:
    path = resolve_text8_path(config)
    text = path.read_text()[: config.limit_chars]
    split = int(len(text) * config.split)
    train_text = text[:split]
    test_text = text[split:]
    if config.tokenizer == "char":
        train_tokens = encode_text(train_text, config.alphabet)
        test_tokens = encode_text(test_text, config.alphabet)
        vocab: str | tuple[str, ...] = config.alphabet
    elif config.tokenizer == "bpe_1024":
        tokenizer = _shared_bpe_tokenizer(
            config.alphabet,
            config.bpe_vocab_size,
            config.bpe_cache_path,
            limit_chars=config.limit_chars,
        )
        train_tokens = _encode_bpe_text(train_text, tokenizer)
        test_tokens = _encode_bpe_text(test_text, tokenizer)
        vocab = tokenizer.vocab
    else:
        raise ValueError(f"Unknown tokenizer mode: {config.tokenizer}")
    return CharDataset(
        train_tokens=train_tokens,
        test_tokens=test_tokens,
        alphabet=vocab,
        source_path=str(path),
        tokenizer=config.tokenizer,
        train_char_count=len(train_text),
        test_char_count=len(test_text),
    )


def resolve_gutenberg_root(config: RegimeSwitchConfig) -> Path:
    candidates = []
    if config.root:
        candidates.append(Path(config.root).expanduser())
    candidates.extend(
        [
            REPO_ROOT / "data" / "gutenberg",
            REPO_ROOT / "gutenberg",
        ]
    )
    for candidate in candidates:
        if candidate.exists() and candidate.is_dir():
            return candidate
    hint_lines = [
        "Could not find the Gutenberg corpus directory.",
        "Looked in:",
        *[f"  - {candidate}" for candidate in candidates],
    ]
    raise FileNotFoundError("\n".join(hint_lines))


def strip_gutenberg_boilerplate(text: str) -> str:
    lines = text.splitlines()
    start = 0
    end = len(lines)
    for idx, line in enumerate(lines):
        upper = line.upper()
        if "*** START OF" in upper or "***START OF" in upper:
            start = idx + 1
            break
    for idx in range(len(lines) - 1, -1, -1):
        upper = lines[idx].upper()
        if "*** END OF" in upper or "***END OF" in upper:
            end = idx
            break
    return "\n".join(lines[start:end])


def normalize_text(text: str, alphabet: str) -> str:
    allowed = set(alphabet)
    chars: list[str] = []
    prev_space = True
    for char in text.lower():
        mapped = char if char in allowed else " "
        if mapped == " ":
            if prev_space:
                continue
            prev_space = True
        else:
            prev_space = False
        chars.append(mapped)
    return "".join(chars).strip()


def encode_text(text: str, alphabet: str) -> np.ndarray:
    char_to_idx = {char: idx for idx, char in enumerate(alphabet)}
    return np.array([char_to_idx.get(char, 0) for char in text], dtype=np.int32)


def extract_gutenberg_title(path: Path) -> str:
    for line in path.read_text(errors="ignore").splitlines()[:200]:
        stripped = line.strip()
        if stripped.startswith("Title:"):
            return stripped.split("Title:", 1)[1].strip()
    return path.stem


def load_regime_switch_corpus(config: RegimeSwitchConfig) -> CharDataset:
    root = resolve_gutenberg_root(config)
    rng = np.random.default_rng(config.seed)
    tokenizer = None
    if config.tokenizer == "bpe_1024":
        tokenizer = _shared_bpe_tokenizer(config.alphabet, config.bpe_vocab_size, config.bpe_cache_path)

    books: list[str] = []
    source_names: list[str] = []
    for file_name in config.book_files:
        path = root / file_name
        if not path.exists():
            raise FileNotFoundError(f"Missing Gutenberg source: {path}")
        text = strip_gutenberg_boilerplate(path.read_text(errors="ignore"))
        text = normalize_text(text, config.alphabet)
        if config.skip_chars > 0:
            text = text[config.skip_chars :]
        if len(text) < config.block_chars * 2:
            raise ValueError(f"Gutenberg source too short after normalization: {path}")
        text = text[: config.chars_per_book]
        books.append(text)
        source_names.append(extract_gutenberg_title(path))

    cursors = [0 for _ in books]
    blocks: list[np.ndarray] = []
    switch_points: list[int] = []
    total_chars = 0
    total_tokens = 0
    made_progress = True

    while total_chars < config.total_chars and made_progress:
        made_progress = False
        for source_idx, source_text in enumerate(books):
            cursor = cursors[source_idx]
            if cursor >= len(source_text):
                continue
            span = config.block_chars
            if config.block_jitter > 0:
                span += int(rng.integers(-config.block_jitter, config.block_jitter + 1))
            span = max(span, config.block_chars // 2)
            end = min(cursor + span, len(source_text), cursor + (config.total_chars - total_chars))
            block_text = source_text[cursor:end]
            if len(block_text) == 0:
                continue
            if config.tokenizer == "char":
                block = encode_text(block_text, config.alphabet)
                vocab: str | tuple[str, ...] = config.alphabet
            elif config.tokenizer == "bpe_1024":
                if tokenizer is None:
                    raise ValueError("BPE tokenizer missing for regime-switch corpus.")
                block = _encode_bpe_text(block_text, tokenizer)
                vocab = tokenizer.vocab
            else:
                raise ValueError(f"Unknown tokenizer mode: {config.tokenizer}")
            if total_tokens > 0:
                switch_points.append(total_tokens)
            blocks.append(block)
            total_chars += len(block_text)
            total_tokens += len(block)
            cursors[source_idx] = end
            made_progress = True
            if total_chars >= config.total_chars:
                break

    if not blocks:
        raise ValueError("Failed to build any regime-switch blocks.")

    data = np.concatenate(blocks)
    split = int(len(data) * config.split)
    train_switch_points = tuple(point for point in switch_points if 0 < point < split)
    test_switch_points = tuple(point - split for point in switch_points if split < point < len(data))
    source_path = "regime_switch:" + ",".join(config.book_files)
    return CharDataset(
        train_tokens=data[:split],
        test_tokens=data[split:],
        alphabet=vocab,
        source_path=source_path,
        source_names=tuple(source_names),
        train_switch_points=train_switch_points,
        test_switch_points=test_switch_points,
        tokenizer=config.tokenizer,
        train_char_count=int(total_chars * config.split),
        test_char_count=total_chars - int(total_chars * config.split),
    )
