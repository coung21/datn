"""Kiểm tra bản đồ cấu trúc của SG-gate trên tập val (lát nguyên kích thước).

In ra theo vùng (não / chậu):
  1) Dice khí / mô mềm / xương của bản đồ dự đoán (argmax) ở từng tầng gate và đầu P0,
     so với nhãn ngưỡng HU hạ phân giải nearest. Dice tính gộp mọi voxel trong mask.
  2) alpha trung bình của gate trong từng lớp mô (theo nhãn), để xem gate có mở/đóng khác nhau theo mô không.
  3) MAE (HU) trên các lát này khi bỏ thành phần cấu trúc khỏi gate (W_s = 0), để xem gate có dùng P không.

    python -m scripts.eval_structure --run outputs/runs/unet2d_gn_sg_<thời gian> [--every 2]
"""

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from src.data.dataset import build_dataset, load_split
from src.data.preprocess import CT_CLIP, parse_id
from src.eval.evaluate import load_checkpoint
from src.methods.regression import structure_labels

CLASSES = ["khí", "mô mềm", "xương"]


def pad16(t: torch.Tensor, value: float, k: int) -> torch.Tensor:
    h, w = t.shape[-2:]
    return F.pad(t, (0, (-w) % k, 0, (-h) % k), value=value)


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--ckpt", default="best")
    p.add_argument("--every", type=int, default=2, help="lấy 1 lát mỗi N lát có mask")
    args = p.parse_args()

    run = Path(args.run)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, method, cfg, _ = load_checkpoint(run / f"{args.ckpt}.pt", device)
    assert model.structure_guided, "checkpoint không phải SG-gate"
    ids = load_split(cfg["data"]["split"]["file"])["val"]
    ds = build_dataset(cfg, ids, mode="val")
    k = 2**model.depth
    lo, hi = CT_CLIP
    thr = method.struct_thresholds

    # inter[region][name][c], union: cho Dice; alpha_sum/alpha_n[region][level][c]
    inter = defaultdict(lambda: defaultdict(lambda: np.zeros(3)))
    total = defaultdict(lambda: defaultdict(lambda: np.zeros(3)))
    a_sum = defaultdict(lambda: defaultdict(lambda: np.zeros(3)))
    a_n = defaultdict(lambda: defaultdict(lambda: np.zeros(3)))
    err = defaultdict(lambda: {"full": 0.0, "no_struct": 0.0, "n": 0.0})
    w_s_saved = [g.w_s[0].weight.clone() for g in model.gates]

    for pid in ids:
        region = parse_id(pid)["region"]
        area = np.asarray(ds.metas[pid]["slice_mask_area"])
        for z in np.flatnonzero(area > 0)[:: args.every]:
            mr, ct, mask = ds.load_slice(pid, int(z))
            h, w = mr.shape[-2:]
            mr = pad16(torch.from_numpy(mr)[None].to(device), -1.0, k)
            ct = pad16(torch.from_numpy(ct)[None].to(device), -1.0, k)
            m = pad16(torch.from_numpy(np.asarray(mask, dtype=np.float32))[None].to(device), 0.0, k)
            lab = structure_labels(ct, thr)

            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                out, structs = model(mr, return_struct=True)
            names = [f"gate l={lv}" for lv, _ in structs[:-1]] + ["đầu P0"]
            for name, (_, lg) in zip(names, structs):
                size = lg.shape[-2:]
                lb = F.interpolate(lab.float(), size=size, mode="nearest").long()[:, 0]
                mm = F.interpolate(m, size=size, mode="nearest")[:, 0] > 0
                pred = lg.float().argmax(1)
                for c in range(3):
                    pc, tc = (pred == c) & mm, (lb == c) & mm
                    inter[region][name][c] += (pc & tc).sum().item()
                    total[region][name][c] += pc.sum().item() + tc.sum().item()
            for (lv, _), gate in zip(structs[:-1], model.gates):
                al = gate.last_alpha.float()
                size = al.shape[-2:]
                lb = F.interpolate(lab.float(), size=size, mode="nearest").long()[:, 0]
                mm = F.interpolate(m, size=size, mode="nearest")[:, 0] > 0
                for c in range(3):
                    sel = (lb == c) & mm
                    a_sum[region][f"gate l={lv}"][c] += al[:, 0][sel].sum().item()
                    a_n[region][f"gate l={lv}"][c] += sel.sum().item()

            hu = lambda x: (x.float().clamp(-1, 1)[..., :h, :w] + 1) / 2 * (hi - lo) + lo
            mk = m[..., :h, :w] > 0
            gt = hu(ct)[mk]
            err[region]["full"] += (hu(out)[mk] - gt).abs().sum().item()
            for g in model.gates:
                g.w_s[0].weight.zero_()
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                out0 = model(mr)
            for g, wsv in zip(model.gates, w_s_saved):
                g.w_s[0].weight.copy_(wsv)
            err[region]["no_struct"] += (hu(out0)[mk] - gt).abs().sum().item()
            err[region]["n"] += mk.sum().item()

    rows = []
    for region in inter:
        for name in inter[region]:
            d = 2 * inter[region][name] / np.maximum(total[region][name], 1)
            rows.append({"vùng": region, "bản đồ": name, **{f"Dice {c}": round(v, 3) for c, v in zip(CLASSES, d)}})
    print("\n== Dice bản đồ cấu trúc (argmax) so với nhãn ngưỡng HU")
    print(pd.DataFrame(rows).to_string(index=False))

    rows = []
    for region in a_sum:
        for name in a_sum[region]:
            a = a_sum[region][name] / np.maximum(a_n[region][name], 1)
            rows.append({"vùng": region, "gate": name, **{f"alpha {c}": round(v, 3) for c, v in zip(CLASSES, a)}})
    print("\n== alpha trung bình của gate theo lớp mô")
    print(pd.DataFrame(rows).to_string(index=False))

    print("\n== MAE (HU) trên các lát này: đủ gate vs bỏ thành phần cấu trúc (W_s = 0)")
    print(pd.DataFrame([{"vùng": r, "đủ": round(e["full"] / e["n"], 2), "W_s=0": round(e["no_struct"] / e["n"], 2)}
                        for r, e in err.items()]).to_string(index=False))


if __name__ == "__main__":
    main()
