"""Sinh ŷ (sCT regression) cho residual flow và thống kê phần dư.

ŷ lưu ở <processed_dir>/<region>/<id>/reg.npy (int16 HU, cùng khung bbox với ct.npy, ngoài mask -1024);
nguồn của từng file ghi vào <processed_dir>/reg_manifest.json.

    # tập train: dự đoán chéo (mỗi model fold dự đoán nửa nó chưa thấy = split 'test' của fold)
    python -X utf8 scripts/predict_reg.py --run outputs/runs/regression_foldA_<t> --subset test
    python -X utf8 scripts/predict_reg.py --run outputs/runs/regression_foldB_<t> --subset test
    # val/test: model regression đầy đủ
    python -X utf8 scripts/predict_reg.py --run outputs/runs/regression_20261016-230441 --subset val test
    # thống kê phần dư (chọn scale) + kiểm tra MAE dự đoán chéo khớp với val/test
    python -X utf8 scripts/predict_reg.py --stats
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from tqdm.auto import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.dataset import build_dataset, load_split, normalize_ct  # noqa: E402
from src.data.preprocess import CT_CLIP, parse_id  # noqa: E402
from src.eval.evaluate import load_checkpoint  # noqa: E402
from src.inference.predict import predict_volume  # noqa: E402
from src.utils.config import load_config  # noqa: E402


def predict(run: Path, ckpt: str, subsets: list[str], overwrite: bool) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, method, cfg, _ = load_checkpoint(run / f"{ckpt}.pt", device)
    if cfg["method"]["name"] != "regression":
        raise SystemExit(f"{run} không phải run regression")
    split = load_split(cfg["data"]["split"]["file"])
    root = Path(cfg["data"]["processed_dir"])
    manifest_path = root / "reg_manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}

    ids = [pid for s in subsets for pid in split[s]]
    ds = build_dataset(cfg, ids, "val")
    maes = []
    for pid in tqdm(ids, desc=f"ŷ từ {run.name}"):
        out = root / parse_id(pid)["region"] / pid / "reg.npy"
        if out.exists() and not overwrite:
            if manifest.get(pid, {}).get("run") != run.name:
                raise SystemExit(f"{out} đã có từ run khác ({manifest.get(pid)}); dùng --overwrite nếu chắc chắn")
            continue
        hu = predict_volume(method, model, ds, pid, 1, cfg["inference"], device)
        np.save(out, np.round(hu).astype(np.int16))
        ct, mask = ds.load_volume(pid)
        maes.append(float(np.abs(hu - ct)[mask > 0].mean()))
        manifest[pid] = {"run": run.name, "ckpt": ckpt, "split_file": cfg["data"]["split"]["file"],
                         "subset": next(s for s in subsets if pid in split[s]), "mae": maes[-1],
                         "time": time.strftime("%Y-%m-%d %H:%M:%S")}
        manifest_path.write_text(json.dumps(manifest, indent=1))
    if maes:
        print(f"{len(maes)} BN, MAE ŷ = {np.mean(maes):.2f} HU")


def stats(cfg: dict, max_voxels: int, seed: int) -> dict:
    """Phân phối phần dư r = norm(CT) - norm(ŷ) trong mask, theo từng tập của split chính."""
    split = load_split(cfg["data"]["split"]["file"])
    root = Path(cfg["data"]["processed_dir"])
    rng = np.random.default_rng(seed)
    out = {}
    for name in ("train", "val", "test"):
        samples, maes, by_region = [], [], {}
        for pid in tqdm(split[name], desc=f"stats {name}"):
            d = root / parse_id(pid)["region"] / pid
            if not (d / "reg.npy").exists():
                raise SystemExit(f"thiếu {d / 'reg.npy'}")
            m = np.load(d / "mask.npy") > 0
            ct, reg = np.load(d / "ct.npy")[m], np.load(d / "reg.npy")[m]
            maes.append(float(np.abs(np.clip(reg, *CT_CLIP) - ct).mean()))
            by_region.setdefault(parse_id(pid)["region"], []).append(maes[-1])
            r = normalize_ct(ct) - normalize_ct(reg)
            samples.append(r[rng.choice(r.size, min(r.size, max_voxels), replace=False)])
        r = np.concatenate(samples)
        mad = float(np.median(np.abs(r - np.median(r))) * 1.4826)
        out[name] = {
            "n": len(maes), "mae_hu": float(np.mean(maes)),
            "mae_hu_by_region": {k: float(np.mean(v)) for k, v in by_region.items()},
            "mean": float(r.mean()), "std": float(r.std()), "mad_std": mad,
            "p01": float(np.percentile(r, 1)), "p99": float(np.percentile(r, 99)),
            "frac_abs_gt_3std": float((np.abs(r) > 3 * r.std()).mean()),
        }
        s = out[name]
        print(f"{name:5}: {s['n']} BN, MAE ŷ {s['mae_hu']:.2f} HU {({k: round(v, 1) for k, v in s['mae_hu_by_region'].items()})} | "
              f"r mean {s['mean']:+.4f} std {s['std']:.4f} MAD-std {s['mad_std']:.4f} "
              f"p1/p99 {s['p01']:+.3f}/{s['p99']:+.3f} |r|>3std {100 * s['frac_abs_gt_3std']:.2f}%")
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", help="thư mục run regression")
    p.add_argument("--ckpt", default="best")
    p.add_argument("--subset", nargs="+", default=["test"], choices=["train", "val", "test"])
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--stats", action="store_true", help="thống kê phần dư thay vì dự đoán")
    p.add_argument("--config", default="configs/base.yaml")
    p.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="ghi đè config (chế độ --stats)")
    p.add_argument("--max-voxels", type=int, default=200_000, help="số voxel lấy mẫu mỗi BN khi thống kê")
    args = p.parse_args()

    if args.stats:
        cfg = load_config(args.config, None, args.set)
        res = stats(cfg, args.max_voxels, cfg["seed"])
        path = Path(cfg["output_dir"]) / "residual_stats.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(res, indent=1))
        print(f"Đã lưu {path}. scale đề xuất (std train): {res['train']['std']:.4f}")
    else:
        if not args.run:
            raise SystemExit("cần --run (hoặc --stats)")
        predict(Path(args.run), args.ckpt, args.subset, args.overwrite)


if __name__ == "__main__":
    main()
