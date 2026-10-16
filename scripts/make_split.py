"""Chia train/val/test theo bệnh nhân, phân tầng theo region x center, gắn cờ ca đặc biệt.

Ca đặc biệt lấy từ cột ghi chú (Note/note) trong overview/*.xlsx của SynthRAD.

    python -X utf8 scripts/make_split.py --raw-root data/raw/Task1_train
"""

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.preprocess import REGIONS, parse_id  # noqa: E402


def read_notes(raw_root: Path) -> dict[str, str]:
    import pandas as pd

    notes = defaultdict(list)
    for x in sorted(raw_root.glob("*/overview/*.xlsx")):
        for sheet, df in pd.read_excel(x, sheet_name=None).items():
            for col in (c for c in df.columns if str(c).lower() == "note"):
                for pid, note in zip(df["ID"], df[col]):
                    if isinstance(note, str) and note.strip():
                        notes[pid].append(f"{sheet}: {note.strip()}")
    return {pid: "; ".join(v) for pid, v in sorted(notes.items())}


def stratified_split(ids: list[str], ratios: dict[str, float], seed: int) -> dict[str, list[str]]:
    """Chia từng tầng (region, center) riêng; val/test làm tròn, phần còn lại vào train."""
    strata = defaultdict(list)
    for pid in sorted(ids):
        info = parse_id(pid)
        strata[(info["region"], info["center"])].append(pid)

    rng = random.Random(seed)
    split = {"train": [], "val": [], "test": []}
    for key in sorted(strata):
        group = strata[key][:]
        rng.shuffle(group)
        n_test = round(len(group) * ratios["test"])
        n_val = round(len(group) * ratios["val"])
        split["test"] += group[:n_test]
        split["val"] += group[n_test:n_test + n_val]
        split["train"] += group[n_test + n_val:]
    return {k: sorted(v) for k, v in split.items()}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--raw-root", default="data/raw/Task1_train")
    p.add_argument("--out", default="configs/splits/split_seed42.json")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--ratios", type=float, nargs=3, default=[0.7, 0.1, 0.2], metavar=("TRAIN", "VAL", "TEST"))
    args = p.parse_args()

    raw = Path(args.raw_root)
    ids = [d.name for r in REGIONS.values() for d in (raw / r).iterdir() if d.is_dir() and d.name != "overview"]
    ratios = dict(zip(("train", "val", "test"), args.ratios))
    split = stratified_split(ids, ratios, args.seed)
    special = read_notes(raw)

    out = {
        "seed": args.seed,
        "ratios": ratios,
        "stratify_by": ["region", "center"],
        **split,
        "special": special,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=1))

    for name in ("train", "val", "test"):
        counts = defaultdict(int)
        for pid in split[name]:
            info = parse_id(pid)
            counts[f"{info['region']}-{info['center']}"] += 1
        n_special = sum(pid in special for pid in split[name])
        print(f"{name:5}: {len(split[name]):3} BN {dict(sorted(counts.items()))}, đặc biệt: {n_special}")
    print(f"Đã lưu {args.out}")


if __name__ == "__main__":
    main()
