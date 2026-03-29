from __future__ import annotations

import copy
import json
from pathlib import Path
import sys
from typing import Any

import mlx.core as mx
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from conker.src.conker10 import ConkerTenConfig, ConkerTenModel, build_packed_tables, scale_config
from conker.src.conker3 import ConkerThreeConfig
from conker.src.golf_data import build_parameter_golf_dataset


class ConkerTenAdapter:
    def __init__(self, summary_path: str, checkpoint_path: str):
        self.summary_path = str(summary_path)
        self.checkpoint_path = str(checkpoint_path)
        summary = json.loads(Path(summary_path).read_text(encoding="utf-8"))
        model_cfg = summary["model"]
        train_cfg = summary["config"]["train"]
        data_root = summary.get("adapter_data_root") or "/tmp/conker_blinx_proxy_data"

        dataset = build_parameter_golf_dataset(data_root, vocab_size=1024)
        tables = build_packed_tables(
            dataset,
            token_budget=int(model_cfg["packed_tokens"]),
            trigram_buckets=int(model_cfg["trigram_buckets"]),
        )
        base_cfg = ConkerThreeConfig(
            max_seq_len=int(train_cfg["seq_len"]),
            linear_modes=256,
            local_window=4,
        )
        config = scale_config(
            ConkerTenConfig(
                base_config=base_cfg,
                freeze_base=bool(model_cfg["freeze_base"]),
                blend_mode=str(model_cfg["blend_mode"]),
                structure_proxy_entropy=bool(model_cfg.get("structure_proxy_entropy", False)),
                structure_proxy_peak=bool(model_cfg.get("structure_proxy_peak", False)),
                structure_proxy_candidate4=bool(model_cfg.get("structure_proxy_candidate4", False)),
                structure_proxy_agreement=bool(model_cfg.get("structure_proxy_agreement", False)),
                alpha_bigram=float(model_cfg["alpha_bigram"]),
                alpha_trigram=float(model_cfg["alpha_trigram"]),
                controller_hidden=int(model_cfg["controller_hidden"]),
                controller_temperature=float(model_cfg["controller_temperature"]),
            ),
            float(model_cfg["scale"]),
        )
        self.model = ConkerTenModel(vocab_size=dataset.vocab_size, tables=tables, config=config)
        state = np.load(checkpoint_path)
        self.model.load_weights([(name, mx.array(state[name])) for name in state.files], strict=False)
        self.vocab_size = dataset.vocab_size
        self._unigram = np.asarray(tables.unigram_probs, dtype=np.float64)

    def fork(self) -> "ConkerTenAdapter":
        return copy.deepcopy(self)

    def describe(self) -> dict[str, Any]:
        return {
            "adapter": "ConkerTenAdapter",
            "summary_path": self.summary_path,
            "checkpoint_path": self.checkpoint_path,
            "vocab_size": int(self.vocab_size),
            "notes": "Causal next-token replay adapter for saved Conker-10 bridge checkpoints.",
        }

    def score_chunk(self, tokens: np.ndarray, sample_positions: np.ndarray | None = None) -> dict[str, Any]:
        seq = np.asarray(tokens, dtype=np.int64).reshape(-1)
        if seq.size == 0:
            return {}
        probs = np.zeros((seq.size, self.vocab_size), dtype=np.float64)
        probs[0] = self._unigram
        if seq.size > 1:
            x = mx.array(seq[:-1][None, :], dtype=mx.int32)
            log_probs = self.model(x)
            mx.eval(log_probs)
            probs[1:] = np.exp(np.asarray(log_probs[0], dtype=np.float64))
        probs /= np.maximum(probs.sum(axis=-1, keepdims=True), 1e-12)
        if sample_positions is None:
            return {}
        idx = np.asarray(sample_positions, dtype=np.int64)
        return {"sample_predictions": probs[idx]}

    def adapt_chunk(self, tokens: np.ndarray) -> None:
        _ = tokens
        return None


def build_adapter(config: dict[str, Any]) -> ConkerTenAdapter:
    return ConkerTenAdapter(
        summary_path=str(config["summary_path"]),
        checkpoint_path=str(config["checkpoint_path"]),
    )
