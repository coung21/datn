"""Hình so sánh định tính sCT giữa các mô hình (ảnh và bản đồ sai số), 2 bước.

Bước 1, trên server (cần GPU và data/processed): chọn bệnh nhân, dự đoán lại, lưu npz gọn

    python -m scripts.make_qualitative export \
        --model "UNet 2.5D" outputs/runs/unet2d_baseline_<...> \
        --model "GN + AG" outputs/runs/unet2d_gn_ag_<...> \
        --model "GN + AG, 9 slices" outputs/runs/unet2d_gn_ag_ctx9_<...> \
        --model "Multi-view" outputs/runs/unet2d_gn_ag_mv_<...> \
        --ref "GN + AG" --new "Multi-view" [--patients ID ...] --out outputs/hinh_dinh_tinh

Bước 2, ở local (chỉ cần các npz): vẽ hình axial và coronal, mỗi hướng 1 hình toàn lát và 1 hình phóng to
(ZOOM_MM mm quanh chỗ --new giảm sai số nhiều nhất so với --ref trên lát đó). Có thêm hàng hiệu sai số
|e_mô hình| - |e_ref| (xanh: tốt hơn --ref).

    python -m scripts.make_qualitative plot --dir outputs/hinh_dinh_tinh

Chọn bệnh nhân (nếu không truyền --patients) theo MAE test từng bệnh nhân (eval/test_best/per_patient.csv),
bỏ các ca đặc biệt (cột special):
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
    """{id: lý do chọn}. ref, new: per_patient (index id, cột region, mae, có thể có cột special).
    Bỏ các ca đặc biệt (special = True, vd có kim loại cấy ghép) vì không đại diện cho đa số."""
    if "special" in ref.columns:
        ref = ref[~ref["special"].astype(bool)]
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
    info = {"labels": list(runs), "patients": picked, "ref": args.ref, "new": args.new,
            "mae": {pid: {l: float(per_patient[l].loc[pid, "mae"]) for l in runs} for pid in picked}}
    (out / "info.json").write_text(json.dumps(info, indent=1, ensure_ascii=False))
    print(f"Đã ghi {out}")


DIFF_LIMIT = 100          # HU, thang bản đồ hiệu sai số |e_mô hình| - |e_ref|
ZOOM_MM = 80              # cạnh vùng phóng to (mm)


def gain_index(d, axis: int, ref: int, new: int, trim: float = 0.15, smooth: int = 7) -> int:
    """Lát theo trục axis mà --new giảm MAE nhiều nhất so với --ref (trung bình trượt `smooth` lát),
    chỉ xét lát có diện tích mask >= 50% lớn nhất và bỏ `trim` mỗi đầu (lát biên thiếu ngữ cảnh)."""
    from scipy.ndimage import uniform_filter1d

    m = d["mask"] > 0
    ct = d["ct"].astype(np.float32)
    gain = np.where(m, np.abs(d[f"sct_{ref}"] - ct) - np.abs(d[f"sct_{new}"] - ct), 0.0)
    other = tuple(a for a in range(3) if a != axis)
    area = m.sum(other)
    g = uniform_filter1d(gain.sum(other) / np.maximum(area, 1), smooth)
    idx = np.flatnonzero(area > 0)
    k = int(trim * len(idx))
    ok = area >= 0.5 * area.max()
    ok[:idx[0] + k] = False
    ok[idx[-1] - k + 1:] = False
    return int(np.argmax(np.where(ok, g, -np.inf)))


def slice_panel(view: str, pid: str, d, n_models: int, index: int | None = None) -> dict:
    """Lát `index` (mặc định lát giữa theo mask) của 1 bệnh nhân, cắt sát khung mask: MR, CT, các sCT và tỉ lệ khung."""
    sx, sy, sz = (float(s) for s in d["spacing"])
    axis = 0 if view == "axial" else 1
    index = middle_index(d["mask"], axis) if index is None else index
    cut = lambda v: take_view(np.asarray(v), view, index)  # noqa: E731
    mask = cut(d["mask"]) > 0
    ys, xs = np.flatnonzero(mask.any(1)), np.flatnonzero(mask.any(0))
    box = (slice(ys[0], ys[-1] + 1), slice(xs[0], xs[-1] + 1))
    return {"pid": pid, "aspect": sy / sx if view == "axial" else sz / sx, "px_mm": (sy if view == "axial" else sz, sx),
            "mask": mask[box], "mr": cut(d["mr"]).astype(np.float32)[box], "ct": cut(d["ct"]).astype(np.float32)[box],
            "sct": [cut(d[f"sct_{i}"]).astype(np.float32)[box] for i in range(n_models)]}


def zoom_box(pn: dict, ref: int, new: int, size_mm: float = ZOOM_MM) -> tuple[slice, slice]:
    """Khung size_mm x size_mm quanh chỗ --new giảm sai số nhiều nhất so với --ref (trung bình trượt trong mask)."""
    from scipy.ndimage import uniform_filter

    gain = np.where(pn["mask"], np.abs(pn["sct"][ref] - pn["ct"]) - np.abs(pn["sct"][new] - pn["ct"]), 0.0)
    h, w = gain.shape
    size = [min(int(round(size_mm / mm)), n) for mm, n in zip(pn["px_mm"], (h, w))]
    smooth = uniform_filter(gain, size=[max(s // 2, 1) for s in size], mode="constant")
    cy, cx = np.unravel_index(np.argmax(smooth), smooth.shape)
    y0 = int(np.clip(cy - size[0] // 2, 0, h - size[0]))
    x0 = int(np.clip(cx - size[1] // 2, 0, w - size[1]))
    return slice(y0, y0 + size[0]), slice(x0, x0 + size[1])


def crop_panel(pn: dict, box) -> dict:
    out = dict(pn)
    for k in ("mask", "mr", "ct"):
        out[k] = pn[k][box]
    out["sct"] = [s[box] for s in pn["sct"]]
    return out


def draw(view: str, info: dict, data: dict, out_stem: Path, zoom: bool = False, slice_by: str = "middle") -> None:
    """Mỗi bệnh nhân: hàng ảnh (MRI, CT, sCT), hàng bản đồ sai số và, nếu info có "ref", hàng hiệu sai số
    |e_mô hình| - |e_ref| (xanh: tốt hơn mô hình ref). zoom=True: chỉ vẽ vùng ZOOM_MM quanh chỗ info["new"]
    cải thiện nhiều nhất so với ref trên lát đó."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels, pids = info["labels"], list(info["patients"])
    ref = labels.index(info["ref"]) if info.get("ref") in labels else None
    new = labels.index(info["new"]) if info.get("new") in labels else len(labels) - 1
    if zoom and ref is None:
        raise ValueError("zoom cần info['ref']")
    per = 3 if ref is not None else 2
    n_col = 2 + len(labels)
    panels, heights = [], []
    for pid in pids:
        index = None
        if slice_by == "gain" and ref is not None:
            index = gain_index(data[pid], 0 if view == "axial" else 1, ref, new)
            print(f"{pid} {view}: lát {index}")
        pn = slice_panel(view, pid, data[pid], len(labels), index)
        if zoom:
            pn = crop_panel(pn, zoom_box(pn, ref, new))
        panels.append(pn)
        h, w = pn["mask"].shape
        heights.append(h * pn["aspect"] / w)

    col_w = 2.0
    ratios = [h for h in heights for _ in range(per)]
    fig, axes = plt.subplots(per * len(pids), n_col, figsize=(col_w * n_col + 0.7, col_w * sum(ratios) + 0.5),
                             gridspec_kw={"height_ratios": ratios, "wspace": 0.03, "hspace": 0.05}, squeeze=False)
    fig.patch.set_facecolor("black")
    im_ct = im_err = im_diff = None
    for r, pn in enumerate(panels):
        top, bot = axes[per * r], axes[per * r + 1]
        kw = {"aspect": pn["aspect"], "interpolation": "nearest"}
        top[0].imshow(pn["mr"], cmap="gray", vmin=-1, vmax=1, **kw)
        im_ct = top[1].imshow(pn["ct"], cmap="gray", vmin=CT_WINDOW[0], vmax=CT_WINDOW[1], **kw)
        err_ref = None if ref is None else np.abs(pn["sct"][ref] - pn["ct"])
        for c, (label, sct) in enumerate(zip(labels, pn["sct"])):
            top[2 + c].imshow(sct, cmap="gray", vmin=CT_WINDOW[0], vmax=CT_WINDOW[1], **kw)
            err = np.where(pn["mask"], sct - pn["ct"], np.nan)
            im_err = bot[2 + c].imshow(err, cmap="RdBu_r", vmin=-ERR_LIMIT, vmax=ERR_LIMIT, **kw)
            if zoom:
                text = f"MAE {np.nanmean(np.abs(err)):.1f} HU (zoom)"
            else:
                text = f"MAE {info['mae'][pn['pid']][label]:.1f} HU"
            bot[2 + c].text(0.5, 0.02, text, color="black", fontsize=7, ha="center", va="bottom",
                            transform=bot[2 + c].transAxes,
                            bbox={"facecolor": "white", "alpha": 0.8, "pad": 1.5, "edgecolor": "none"})
            if ref is not None:
                ax = axes[per * r + 2][2 + c]
                if c == ref:
                    ax.text(0.5, 0.5, "ref", color="white", fontsize=9, ha="center", va="center", transform=ax.transAxes)
                else:
                    diff = np.where(pn["mask"], np.abs(sct - pn["ct"]) - err_ref, np.nan)
                    im_diff = ax.imshow(diff, cmap="RdBu_r", vmin=-DIFF_LIMIT, vmax=DIFF_LIMIT, **kw)
        bot[0].text(0.03, 0.95, f"{pn['pid']}\n{info['patients'][pn['pid']]}", color="white", fontsize=8,
                    ha="left", va="top", transform=bot[0].transAxes)
        if ref is not None:
            axes[per * r + 2][0].text(0.03, 0.95, f"|e| − |e {info['ref']}|", color="white", fontsize=8,
                                      ha="left", va="top", transform=axes[per * r + 2][0].transAxes)
        if r == 0:
            for ax, title in zip(top, ["MRI", "Ground truth CT", *labels]):
                ax.set_title(title, color="white", fontsize=9)
    for ax in axes.ravel():
        ax.set_xticks([]), ax.set_yticks([])
        ax.set_facecolor("black")
        for s in ax.spines.values():
            s.set_visible(False)
    bars = [(im_ct, "HU"), (im_err, "sCT − CT (HU)")] + ([(im_diff, "|e| − |e ref| (HU)")] if im_diff is not None else [])
    step = 0.9 / len(bars)
    for k, (im, label) in enumerate(bars):
        cax = fig.add_axes([0.925, 0.96 - step * (k + 1) + 0.04, 0.012, step - 0.08])
        cb = fig.colorbar(im, cax=cax)
        cb.set_label(label, color="white", fontsize=8)
        cb.ax.tick_params(colors="white", labelsize=7)
    fig.subplots_adjust(left=0.01, right=0.915, top=0.96, bottom=0.01)
    for ext in ("png", "pdf"):
        fig.savefig(out_stem.with_suffix(f".{ext}"), dpi=200, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Đã ghi {out_stem}.png / .pdf")


def draw_boxed(view: str, info: dict, data: dict, out_stem: Path, slice_by: str = "middle") -> None:
    """Mỗi bệnh nhân 4 hàng: lát đầy đủ có khung đỏ quanh vùng info["new"] cải thiện nhiều nhất so với ref,
    rồi phóng to vùng đó: ảnh, sai số sCT − CT, hiệu sai số |e| − |e ref|."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    labels, pids = info["labels"], list(info["patients"])
    ref, new = labels.index(info["ref"]), labels.index(info["new"])
    n_col = 2 + len(labels)
    rows, ratios = [], []
    for pid in pids:
        index = gain_index(data[pid], 0 if view == "axial" else 1, ref, new) if slice_by == "gain" else None
        full = slice_panel(view, pid, data[pid], len(labels), index)
        box = zoom_box(full, ref, new)
        zoom = crop_panel(full, box)
        h, w = full["mask"].shape
        zh, zw = zoom["mask"].shape
        rows.append((full, box, zoom))
        ratios += [h * full["aspect"] / w] + [zh * zoom["aspect"] / zw] * 3

    col_w = 2.0
    fig, axes = plt.subplots(len(ratios), n_col, figsize=(col_w * n_col + 0.7, col_w * sum(ratios) + 0.5),
                             gridspec_kw={"height_ratios": ratios, "wspace": 0.03, "hspace": 0.05}, squeeze=False)
    fig.patch.set_facecolor("black")
    im_ct = im_err = im_diff = None
    for r, (full, box, zoom) in enumerate(rows):
        a_full, a_img, a_err, a_diff = axes[4 * r: 4 * r + 4]
        (y, x) = box
        for row, pn in ((a_full, full), (a_img, zoom)):
            kw = {"aspect": pn["aspect"], "interpolation": "nearest"}
            row[0].imshow(pn["mr"], cmap="gray", vmin=-1, vmax=1, **kw)
            im_ct = row[1].imshow(pn["ct"], cmap="gray", vmin=CT_WINDOW[0], vmax=CT_WINDOW[1], **kw)
            for c, sct in enumerate(pn["sct"]):
                row[2 + c].imshow(sct, cmap="gray", vmin=CT_WINDOW[0], vmax=CT_WINDOW[1], **kw)
        for ax in a_full:
            ax.add_patch(Rectangle((x.start - 0.5, y.start - 0.5), x.stop - x.start, y.stop - y.start,
                                   fill=False, edgecolor="red", linewidth=1.2))
        kw = {"aspect": zoom["aspect"], "interpolation": "nearest"}
        err_ref = np.abs(zoom["sct"][ref] - zoom["ct"])
        for c, (label, sct) in enumerate(zip(labels, zoom["sct"])):
            err = np.where(zoom["mask"], sct - zoom["ct"], np.nan)
            im_err = a_err[2 + c].imshow(err, cmap="RdBu_r", vmin=-ERR_LIMIT, vmax=ERR_LIMIT, **kw)
            a_full[2 + c].text(0.5, 0.02, f"MAE {info['mae'][full['pid']][label]:.1f} HU", color="black", fontsize=7,
                               ha="center", va="bottom", transform=a_full[2 + c].transAxes,
                               bbox={"facecolor": "white", "alpha": 0.8, "pad": 1.5, "edgecolor": "none"})
            a_err[2 + c].text(0.5, 0.02, f"{np.nanmean(np.abs(err)):.1f} HU", color="black", fontsize=7,
                              ha="center", va="bottom", transform=a_err[2 + c].transAxes,
                              bbox={"facecolor": "white", "alpha": 0.8, "pad": 1.5, "edgecolor": "none"})
            if c == ref:
                a_diff[2 + c].text(0.5, 0.5, "ref", color="white", fontsize=9, ha="center", va="center",
                                   transform=a_diff[2 + c].transAxes)
            else:
                diff = np.where(zoom["mask"], np.abs(sct - zoom["ct"]) - err_ref, np.nan)
                im_diff = a_diff[2 + c].imshow(diff, cmap="RdBu_r", vmin=-DIFF_LIMIT, vmax=DIFF_LIMIT, **kw)
        side = f"{ZOOM_MM:.0f} mm"
        for ax, text in ((a_full[0], f"{full['pid']}\n{info['patients'][full['pid']]}"),
                         (a_img[0], f"zoom {side}"), (a_err[0], "sCT − CT"), (a_diff[0], f"|e| − |e {info['ref']}|")):
            ax.text(0.03, 0.95, text, color="white", fontsize=8, ha="left", va="top", transform=ax.transAxes,
                    bbox={"facecolor": "black", "alpha": 0.6, "pad": 1.5, "edgecolor": "none"})
        if r == 0:
            for ax, title in zip(a_full, ["MRI", "Ground truth CT", *labels]):
                ax.set_title(title, color="white", fontsize=9)
    for ax in axes.ravel():
        ax.set_xticks([]), ax.set_yticks([])
        ax.set_facecolor("black")
        for sp in ax.spines.values():
            sp.set_visible(False)
    bars = [(im_ct, "HU"), (im_err, "sCT − CT (HU)"), (im_diff, "|e| − |e ref| (HU)")]
    step = 0.9 / len(bars)
    for k, (im, label) in enumerate(bars):
        cax = fig.add_axes([0.925, 0.96 - step * (k + 1) + 0.04, 0.012, step - 0.08])
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
        draw(view, info, data, d / f"qualitative_{view}", slice_by=args.slice)
        if info.get("ref") in info["labels"]:
            draw(view, info, data, d / f"qualitative_{view}_zoom", zoom=True, slice_by=args.slice)
            draw_boxed(view, info, data, d / f"qualitative_{view}_boxed", slice_by=args.slice)


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
    q.add_argument("--slice", choices=["middle", "gain"], default="middle",
                   help="middle: lát giữa theo mask; gain: lát --new cải thiện nhiều nhất so với --ref (bỏ 15%% mỗi đầu)")
    q.set_defaults(func=plot)
    args = p.parse_args()
    if args.cmd == "export":
        labels = [m[0] for m in args.model]
        args.ref = args.ref or labels[min(1, len(labels) - 1)]
        args.new = args.new or labels[-1]
    args.func(args)


if __name__ == "__main__":
    main()
