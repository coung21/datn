"""Hình so sánh định tính sCT giữa các mô hình (ảnh và bản đồ sai số), 2 bước.

Bước 1, trên server (cần GPU và data/processed): chọn bệnh nhân, dự đoán lại, lưu npz gọn

    python -m scripts.make_qualitative export \
        --model "UNet 2.5D" outputs/runs/unet2d_baseline_<...> \
        --model "GN + AG" outputs/runs/unet2d_gn_ag_<...> \
        --model "GN + AG, 9 slices" outputs/runs/unet2d_gn_ag_ctx9_<...> \
        --model "Multi-view" outputs/runs/unet2d_gn_ag_mv_<...> \
        --ref "GN + AG" --new "Multi-view" [--patients ID ...] --out outputs/hinh_dinh_tinh

Bước 2, ở local (chỉ cần các npz): vẽ hình axial và coronal

    python -m scripts.make_qualitative plot --dir outputs/hinh_dinh_tinh

Chọn bệnh nhân (nếu không truyền --patients) theo MAE test từng bệnh nhân (eval/test_best/per_patient.csv):
  - mỗi vùng 1 ca có MAE của mô hình --ref ở trung vị (ca đại diện);
  - ca chậu mà --new cải thiện nhiều nhất so với --ref;
  - ca mà --new kém hơn --ref nhiều nhất.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

CT_WINDOW = (-1000, 1500)  # HU
ERR_LIMIT = 300            # HU, thang bản đồ sai số
REGION_NAMES = {"brain": "brain", "pelvis": "pelvis"}


def select_patients(ref: pd.DataFrame, new: pd.DataFrame) -> dict[str, str]:
    """{id: lý do chọn}. ref, new: per_patient (index id, cột region, mae)."""
    new = new.loc[ref.index]
    picked: dict[str, str] = {}
    for region in ("brain", "pelvis"):
        m = ref.loc[ref["region"] == region, "mae"].sort_values()
        picked[m.index[(len(m) - 1) // 2]] = f"median {region}"
    diff = new["mae"] - ref["mae"]
    left = diff.drop(index=list(picked))
    picked[left[ref.loc[left.index, "region"] == "pelvis"].idxmin()] = "largest gain (pelvis)"
    picked[left.drop(index=list(picked)[-1:]).idxmax()] = "largest loss"
    return picked


def middle_index(mask: np.ndarray, axis: int) -> int:
    """Chỉ số giữa của phần có mask theo trục axis."""
    other = tuple(a for a in range(mask.ndim) if a != axis)
    idx = np.flatnonzero(mask.any(axis=other))
    return int(idx[len(idx) // 2])


def take_view(vol: np.ndarray, view: str, index: int) -> np.ndarray:
    """Lát 2D từ volume [z, y, x]: axial -> [y, x], coronal -> [z, x] (z tăng lên trên)."""
    return vol[index] if view == "axial" else vol[::-1, index]


def masked_mae(sct: np.ndarray, ct: np.ndarray, mask: np.ndarray) -> float:
    m = mask > 0
    return float(np.abs(sct[m] - ct[m]).mean())


def export(args):
    import torch

    from src.data.dataset import build_dataset, normalize_mr
    from src.eval.evaluate import load_checkpoint
    from src.inference.predict import predict_volume

    runs = {label: Path(run) for label, run in args.model}
    per_patient = {l: pd.read_csv(r / "eval/test_best/per_patient.csv").set_index("id") for l, r in runs.items()}
    if args.patients:
        picked = {pid: "manual" for pid in args.patients}
    else:
        picked = select_patients(per_patient[args.ref], per_patient[args.new])
    for pid, why in picked.items():
        print(f"{pid}: {why}; " + ", ".join(f"{l} {d.loc[pid, 'mae']:.1f}" for l, d in per_patient.items()))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    data = {pid: {} for pid in picked}
    for i, (label, run) in enumerate(runs.items()):
        model, method, cfg, _ = load_checkpoint(run / f"{args.ckpt}.pt", device)
        ds = build_dataset(cfg, list(picked), mode="val")
        for pid in picked:
            d = data[pid]
            d[f"sct_{i}"] = np.round(predict_volume(method, model, ds, pid, 1, cfg["inference"], device)).astype(np.int16)
            if "ct" not in d:
                ct, mask = ds.load_volume(pid)
                meta = ds.metas[pid]
                p = meta["mr_percentiles"]
                mr = normalize_mr(np.asarray(ds._arrays(pid)["mr"]), p[ds.mr_keys[0]], p[ds.mr_keys[1]])
                d.update(ct=np.round(ct).astype(np.int16), mask=mask.astype(np.uint8), mr=mr.astype(np.float16),
                         spacing=np.asarray(meta["spacing"], np.float32))
        del model
        print(f"{label}: xong", flush=True)
    for pid, d in data.items():
        np.savez_compressed(out / f"{pid}.npz", **d)
    info = {"labels": list(runs), "patients": picked,
            "mae": {pid: {l: float(per_patient[l].loc[pid, "mae"]) for l in runs} for pid in picked}}
    (out / "info.json").write_text(json.dumps(info, indent=1, ensure_ascii=False))
    print(f"Đã ghi {out}")


def draw(view: str, info: dict, data: dict, out_stem: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels, pids = info["labels"], list(info["patients"])
    n_col = 2 + len(labels)
    panels, heights = [], []
    for pid in pids:
        d = data[pid]
        sx, sy, sz = (float(s) for s in d["spacing"])
        index = middle_index(d["mask"], 0 if view == "axial" else 1)
        aspect = sy / sx if view == "axial" else sz / sx
        cut = lambda v: take_view(np.asarray(v), view, index)  # noqa: E731
        mask = cut(d["mask"]) > 0
        ys, xs = np.flatnonzero(mask.any(1)), np.flatnonzero(mask.any(0))
        box = (slice(ys[0], ys[-1] + 1), slice(xs[0], xs[-1] + 1))
        panels.append({"pid": pid, "aspect": aspect, "mask": mask[box], "mr": cut(d["mr"]).astype(np.float32)[box],
                       "ct": cut(d["ct"]).astype(np.float32)[box],
                       "sct": [cut(d[f"sct_{i}"]).astype(np.float32)[box] for i in range(len(labels))]})
        h, w = panels[-1]["mask"].shape
        heights.append(h * aspect / w)

    col_w = 2.0
    ratios = [h for h in heights for _ in range(2)]
    fig, axes = plt.subplots(2 * len(pids), n_col, figsize=(col_w * n_col + 0.7, col_w * sum(ratios) + 0.5),
                             gridspec_kw={"height_ratios": ratios, "wspace": 0.03, "hspace": 0.05}, squeeze=False)
    fig.patch.set_facecolor("black")
    im_ct = im_err = None
    for r, pn in enumerate(panels):
        top, bot = axes[2 * r], axes[2 * r + 1]
        kw = {"aspect": pn["aspect"], "interpolation": "nearest"}
        top[0].imshow(pn["mr"], cmap="gray", vmin=-1, vmax=1, **kw)
        im_ct = top[1].imshow(pn["ct"], cmap="gray", vmin=CT_WINDOW[0], vmax=CT_WINDOW[1], **kw)
        for c, (label, sct) in enumerate(zip(labels, pn["sct"])):
            top[2 + c].imshow(sct, cmap="gray", vmin=CT_WINDOW[0], vmax=CT_WINDOW[1], **kw)
            err = np.where(pn["mask"], sct - pn["ct"], np.nan)
            im_err = bot[2 + c].imshow(err, cmap="RdBu_r", vmin=-ERR_LIMIT, vmax=ERR_LIMIT, **kw)
            bot[2 + c].text(0.5, 0.02, f"MAE {info['mae'][pn['pid']][label]:.1f} HU", color="black", fontsize=7,
                            ha="center", va="bottom", transform=bot[2 + c].transAxes,
                            bbox={"facecolor": "white", "alpha": 0.8, "pad": 1.5, "edgecolor": "none"})
        bot[0].text(0.03, 0.95, f"{pn['pid']}\n{info['patients'][pn['pid']]}", color="white", fontsize=8,
                    ha="left", va="top", transform=bot[0].transAxes)
        if r == 0:
            for ax, title in zip(top, ["MRI", "Ground truth CT", *labels]):
                ax.set_title(title, color="white", fontsize=9)
    for ax in axes.ravel():
        ax.set_xticks([]), ax.set_yticks([])
        ax.set_facecolor("black")
        for s in ax.spines.values():
            s.set_visible(False)
    for im, label, y in ((im_ct, "HU", 0.55), (im_err, "sCT − CT (HU)", 0.12)):
        cax = fig.add_axes([0.925, y, 0.012, 0.3])
        cb = fig.colorbar(im, cax=cax)
        cb.set_label(label, color="white", fontsize=8)
        cb.ax.tick_params(colors="white", labelsize=7)
    fig.subplots_adjust(left=0.01, right=0.915, top=0.96, bottom=0.01)
    for ext in ("png", "pdf"):
        fig.savefig(out_stem.with_suffix(f".{ext}"), dpi=200, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Đã ghi {out_stem}.png / .pdf")


def plot(args):
    d = Path(args.dir)
    info = json.loads((d / "info.json").read_text())
    data = {pid: np.load(d / f"{pid}.npz") for pid in info["patients"]}
    for view in ("axial", "coronal"):
        draw(view, info, data, d / f"qualitative_{view}")


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("export")
    e.add_argument("--model", nargs=2, action="append", required=True, metavar=("LABEL", "RUN"))
    e.add_argument("--ref", default=None, help="nhãn mô hình tham chiếu khi chọn bệnh nhân (mặc định: mô hình thứ 2)")
    e.add_argument("--new", default=None, help="nhãn mô hình đề xuất (mặc định: mô hình cuối)")
    e.add_argument("--patients", nargs="+", default=None)
    e.add_argument("--ckpt", default="best")
    e.add_argument("--out", default="outputs/hinh_dinh_tinh")
    e.set_defaults(func=export)
    q = sub.add_parser("plot")
    q.add_argument("--dir", default="outputs/hinh_dinh_tinh")
    q.set_defaults(func=plot)
    args = p.parse_args()
    if args.cmd == "export":
        labels = [m[0] for m in args.model]
        args.ref = args.ref or labels[min(1, len(labels) - 1)]
        args.new = args.new or labels[-1]
    args.func(args)


if __name__ == "__main__":
    main()
