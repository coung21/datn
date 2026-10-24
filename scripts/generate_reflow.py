"""Sinh cặp reflow bằng teacher (flow matching 1-RF) trên tập train.

Mỗi lát z: noise ε = slice_noise(pid, z, seed) trên lát đã pad -> x̂1 = ODE_teacher(ε | MR), Euler NFE bước.
Lưu x̂1 (HU, int16, khung bbox như ct.npy) tại <processed_dir>/<region>/<id>/reflow_nfe<K>.npy.
ε không lưu: dataset sinh lại từ (pid, z, seed). Nguồn ghi ở <processed_dir>/reflow_manifest.json.

    python -X utf8 scripts/generate_reflow.py --teacher outputs/runs/flow_matching_20261018-094349 --nfe 2 --seed 1
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

from src.data.dataset import build_dataset, load_split  # noqa: E402
from src.data.preprocess import parse_id  # noqa: E402
from src.eval.evaluate import load_checkpoint  # noqa: E402
from src.inference.predict import predict_volume  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--teacher", required=True, help="thư mục run flow_matching")
    p.add_argument("--ckpt", default="best")
    p.add_argument("--nfe", type=int, default=2)
    p.add_argument("--seed", type=int, default=1, help="seed noise (0 là bộ noise dùng khi đánh giá)")
    p.add_argument("--subset", nargs="+", default=["train"], choices=["train", "val", "test"])
    args = p.parse_args()

    teacher = Path(args.teacher)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, method, cfg, _ = load_checkpoint(teacher / f"{args.ckpt}.pt", device)
    if cfg["method"]["name"] != "flow_matching":
        raise SystemExit(f"{teacher} không phải run flow_matching")
    if cfg["data"]["patch_size"] != cfg["inference"]["roi_size"]:
        raise SystemExit("cần patch_size == roi_size để noise lúc train khớp noise lúc sinh")

    split = load_split(cfg["data"]["split"]["file"])
    root = Path(cfg["data"]["processed_dir"])
    name = f"reflow_nfe{args.nfe}"
    info = {"teacher": teacher.name, "ckpt": args.ckpt, "nfe": args.nfe, "seed": args.seed}
    manifest_path = root / "reflow_manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    if manifest.get(name, {}).get("info", info) != info:
        raise SystemExit(f"{name} đã được sinh với cấu hình khác: {manifest[name]['info']}")
    done = manifest.setdefault(name, {"info": info, "patients": {}})["patients"]

    ids = [pid for s in args.subset for pid in split[s]]
    ds = build_dataset(cfg, ids, "val")
    for pid in tqdm(ids, desc=f"{name} (seed {args.seed})"):
        out = root / parse_id(pid)["region"] / pid / f"{name}.npy"
        if pid in done and out.exists():
            continue  # chạy lại sau khi bị ngắt: bỏ qua BN đã xong
        hu = predict_volume(method, model, ds, pid, args.nfe, cfg["inference"], device, noise_seed=args.seed)
        np.save(out, np.round(hu).astype(np.int16))
        ct, mask = ds.load_volume(pid)
        done[pid] = {"mae_vs_ct": float(np.abs(hu - ct)[mask > 0].mean()), "time": time.strftime("%Y-%m-%d %H:%M:%S")}
        manifest_path.write_text(json.dumps(manifest, indent=1))
    maes = [v["mae_vs_ct"] for k, v in done.items() if k in ids]
    print(f"{name}: {len(maes)}/{len(ids)} BN, MAE x̂1 so với CT thật = {np.mean(maes):.2f} HU")


if __name__ == "__main__":
    main()
