"""Thống kê kết quả test qua nhiều seed huấn luyện và chênh lệch MAE giữa 2 mô hình.

    python -m scripts.seed_stats \
        --model "GN + AG" "outputs/runs/unet2d_gn_ag_2026*" "outputs/runs/unet2d_gn_ag_s[0-9]_*" \
        --model "Multi-view" "outputs/runs/unet2d_gn_ag_mv_2026*" "outputs/runs/unet2d_gn_ag_mv_s[0-9]_*" \
        [--out outputs/seed_stats]

Mỗi --model: nhãn rồi các glob thư mục run (mỗi run 1 seed, cần eval/test_best/per_patient.csv).
In ra:
  1) MAE / PSNR / SSIM của từng run (tất cả, não, chậu);
  2) mean ± std giữa các seed cho từng mô hình;
  3) chênh lệch MAE (mô hình 2 − mô hình 1), tất cả / não / chậu:
     - CI 95% theo seed: Welch t trên MAE trung bình mỗi seed;
     - CI 95% bootstrap phân cấp: lấy mẫu lại seed của mỗi mô hình và bệnh nhân (chung cho 2 mô hình);
     - Wilcoxon trên MAE từng bệnh nhân đã trung bình qua các seed.
"""

import argparse
import glob
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

REGIONS = ["all", "brain", "pelvis"]
METRICS = ["mae", "psnr", "ssim"]


def load_model(patterns: list[str]) -> dict[str, pd.DataFrame]:
    """{tên run: per_patient (index id)}, chỉ các run đã có kết quả test."""
    runs = sorted({p for pat in patterns for p in glob.glob(pat)})
    out = {}
    for r in runs:
        f = Path(r) / "eval/test_best/per_patient.csv"
        if f.exists():
            out[Path(r).name] = pd.read_csv(f).set_index("id")
    return out


def region_mask(df: pd.DataFrame, region: str) -> np.ndarray:
    return np.ones(len(df), bool) if region == "all" else (df["region"] == region).to_numpy()


def welch_ci(a: np.ndarray, b: np.ndarray, level: float = 0.95) -> tuple[float, float, float, float]:
    """Chênh lệch trung bình b − a, CI Welch, p hai phía."""
    d = b.mean() - a.mean()
    va, vb = a.var(ddof=1) / len(a), b.var(ddof=1) / len(b)
    se = np.sqrt(va + vb)
    df = (va + vb) ** 2 / (va**2 / (len(a) - 1) + vb**2 / (len(b) - 1))
    t = stats.t.ppf(0.5 + level / 2, df)
    p = 2 * stats.t.sf(abs(d) / se, df)
    return d, d - t * se, d + t * se, p


def hier_bootstrap(A: np.ndarray, B: np.ndarray, n_boot: int = 10000, level: float = 0.95, seed: int = 0):
    """A [seed_a, bệnh nhân], B [seed_b, bệnh nhân] (cùng thứ tự bệnh nhân). CI của mean(B) − mean(A)."""
    rng = np.random.default_rng(seed)
    n_p = A.shape[1]
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        pts = rng.integers(0, n_p, n_p)
        sa = rng.integers(0, A.shape[0], A.shape[0])
        sb = rng.integers(0, B.shape[0], B.shape[0])
        diffs[i] = B[np.ix_(sb, pts)].mean() - A[np.ix_(sa, pts)].mean()
    lo, hi = np.percentile(diffs, [50 * (1 - level), 50 * (1 + level)])
    return B.mean() - A.mean(), lo, hi


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", nargs="+", action="append", required=True, metavar=("LABEL", "GLOB"))
    p.add_argument("--n-boot", type=int, default=10000)
    p.add_argument("--out", default=None, help="thư mục ghi các bảng CSV")
    args = p.parse_args()

    models = {m[0]: load_model(m[1:]) for m in args.model}
    for label, runs in models.items():
        assert runs, f"không có run nào cho {label}"
    ids = sorted(next(iter(next(iter(models.values())).values())).index)
    for label, runs in models.items():
        for name, df in runs.items():
            assert sorted(df.index) == ids, f"{name}: tập bệnh nhân khác"

    rows = []
    for label, runs in models.items():
        for name, df in runs.items():
            df = df.loc[ids]
            row = {"mô hình": label, "run": name}
            for r in REGIONS:
                m = region_mask(df, r)
                for k in METRICS:
                    row[f"{k}_{r}"] = df[k].to_numpy()[m].mean()
            rows.append(row)
    per_run = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    print("== Từng run (seed)")
    print(per_run.round(3).to_string(index=False))

    print("\n== Mean ± std giữa các seed")
    cols = [c for c in per_run.columns if c not in ("mô hình", "run")]
    agg = per_run.groupby("mô hình", sort=False)[cols].agg(["mean", "std", "count"])
    summary = pd.DataFrame({c: [f"{agg.loc[l, (c, 'mean')]:.3f} ± {agg.loc[l, (c, 'std')]:.3f}" for l in agg.index]
                            for c in cols}, index=agg.index)
    summary.insert(0, "n seed", agg[(cols[0], "count")].astype(int))
    print(summary.T.to_string())

    labels = list(models)
    diff_rows = []
    if len(labels) >= 2:
        a_lab, b_lab = labels[0], labels[1]
        A = np.stack([df.loc[ids, "mae"].to_numpy() for df in models[a_lab].values()])
        B = np.stack([df.loc[ids, "mae"].to_numpy() for df in models[b_lab].values()])
        ref = next(iter(models[a_lab].values())).loc[ids]
        print(f"\n== Chênh lệch MAE: {b_lab} − {a_lab} (âm = {b_lab} tốt hơn); "
              f"{A.shape[0]} vs {B.shape[0]} seed, {len(ids)} bệnh nhân")
        for r in REGIONS:
            m = region_mask(ref, r)
            d, lo, hi, pw = welch_ci(A[:, m].mean(1), B[:, m].mean(1))
            _, blo, bhi = hier_bootstrap(A[:, m], B[:, m], args.n_boot)
            pa, pb = A[:, m].mean(0), B[:, m].mean(0)
            pwx = stats.wilcoxon(pb, pa).pvalue
            diff_rows.append({"vùng": r, "ΔMAE": d, "CI seed (Welch) thấp": lo, "CI seed cao": hi, "p Welch": pw,
                              "CI bootstrap thấp": blo, "CI bootstrap cao": bhi,
                              "BN tốt hơn (MAE TB qua seed)": f"{int((pb < pa).sum())}/{m.sum()}",
                              "p Wilcoxon": pwx})
        print(pd.DataFrame(diff_rows).round(4).to_string(index=False))

    if args.out:
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        per_run.to_csv(out / "per_run.csv", index=False)
        summary.to_csv(out / "summary.csv")
        if diff_rows:
            pd.DataFrame(diff_rows).to_csv(out / "diff.csv", index=False)
        print(f"\nĐã ghi {out}")


if __name__ == "__main__":
    main()
