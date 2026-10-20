"""Chia tập train thành 2 nửa A/B (phân tầng region x center) để dự đoán chéo ŷ cho residual flow.

Mỗi fold là 1 file split: train = nửa đó, val = val gốc (chọn checkpoint), test = nửa còn lại
(model fold dự đoán nửa nó chưa thấy -> ŷ cho tập train của residual flow).

    python scripts/make_folds.py --split configs/splits/split_seed42.json
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.make_split import stratified_split  # noqa: E402
from src.data.preprocess import parse_id  # noqa: E402


def make_folds(split: dict, seed: int) -> dict[str, dict]:
    halves = stratified_split(split["train"], {"train": 0.5, "val": 0.0, "test": 0.5}, seed)
    a, b = halves["train"], halves["test"]
    common = {"seed": seed, "parent": "train split chia đôi", "stratify_by": ["region", "center"],
              "val": split["val"], "special": split.get("special", {})}
    return {"A": {**common, "train": a, "test": b}, "B": {**common, "train": b, "test": a}}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--split", default="configs/splits/split_seed42.json")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    split_path = Path(args.split)
    folds = make_folds(json.loads(split_path.read_text()), args.seed)
    for name, f in folds.items():
        out = split_path.with_name(f"{split_path.stem}_fold{name}.json")
        out.write_text(json.dumps(f, indent=1))
        counts = Counter(f"{parse_id(i)['region']}-{parse_id(i)['center']}" for i in f["train"])
        print(f"fold {name}: train {len(f['train'])} {dict(sorted(counts.items()))}, "
              f"dự đoán chéo {len(f['test'])}, val {len(f['val'])} -> {out}")


if __name__ == "__main__":
    main()
