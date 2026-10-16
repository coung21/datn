import json
import math
import sys

import numpy as np
import pandas as pd
import pytest
import torch
from test_dataset import make_patient

from src.eval import evaluate
from src.eval.metrics import all_metrics
from src.methods import Regression, build_method, masked_l1
from src.models import UNetModel
from src.train import cosine_with_warmup
from src.train import main as train_main
from src.utils.config import apply_overrides, deep_merge, load_config
from src.utils.logger import WandbLogger, example_panel

TINY_MODEL = dict(image_size=32, in_channels=3, model_channels=32, channel_mult=(1, 2), num_res_blocks=1,
                  attention_resolutions=(16,), num_head_channels=32)


def test_masked_l1():
    pred, target = torch.zeros(1, 1, 2, 2), torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]])
    mask = torch.tensor([[[[1.0, 1.0], [0.0, 0.0]]]])
    assert masked_l1(pred, target, mask).item() == pytest.approx(1.5)
    assert masked_l1(pred, target, torch.zeros_like(mask)).item() == 0


def test_regression_loss_and_predict():
    torch.manual_seed(0)
    model = UNetModel(**TINY_MODEL)
    method = build_method("regression")
    assert isinstance(method, Regression) and method.nfe_values([1, 4, 10]) == [1]
    batch = {"cond": torch.randn(2, 3, 32, 32), "ct": torch.rand(2, 1, 32, 32) * 2 - 1,
             "mask": torch.ones(2, 1, 32, 32)}
    loss, logs = method.loss(model, batch)
    loss.backward()
    assert loss.ndim == 0 and logs["l1"] == pytest.approx(loss.item())
    assert model.input_blocks[0][0].weight.grad is not None
    out = method.predict(model.eval(), batch["cond"])
    assert out.shape == (2, 1, 32, 32) and out.min() >= -1 and out.max() <= 1


def test_metrics():
    rng = np.random.default_rng(0)
    ct = rng.uniform(-1024, 3000, (8, 32, 32)).astype(np.float32)
    mask = np.zeros_like(ct, dtype=np.uint8)
    mask[:, 8:24, 8:24] = 1
    same = all_metrics(ct, ct, mask)
    assert same["mae"] == 0 and math.isinf(same["psnr"]) and same["ssim"] == pytest.approx(1.0)
    sct = ct.copy()
    sct[mask > 0] = np.clip(ct[mask > 0] + 100, -1024, 3000)
    sct[mask == 0] = 5000  # ngoài mask không được tính
    r = all_metrics(sct, ct, mask)
    exp_mae = np.abs(np.clip(ct + 100, -1024, 3000) - ct)[mask > 0].mean()
    assert r["mae"] == pytest.approx(exp_mae, rel=1e-5)
    mse = ((np.clip(ct + 100, -1024, 3000) - ct)[mask > 0] ** 2).mean()
    assert r["psnr"] == pytest.approx(10 * np.log10(4024**2 / mse), rel=1e-5)
    assert 0 < r["ssim"] < 1


def test_cosine_with_warmup():
    fn = cosine_with_warmup(total_steps=100, warmup_steps=10, min_ratio=0.01)
    assert fn(0) == pytest.approx(0.1) and fn(9) == pytest.approx(1.0)
    assert fn(10) == pytest.approx(1.0)
    assert fn(55) == pytest.approx(0.01 + 0.99 * 0.5)
    assert fn(100) == pytest.approx(0.01) and fn(500) == pytest.approx(0.01)


def test_config(tmp_path):
    assert deep_merge({"a": {"b": 1, "c": 2}}, {"a": {"c": 3}}) == {"a": {"b": 1, "c": 3}}
    cfg = apply_overrides({"a": {"b": 1}}, ["a.b=2.5", "x.y=[1, 2]", "flag=false"])
    assert cfg == {"a": {"b": 2.5}, "x": {"y": [1, 2]}, "flag": False}
    cfg = load_config("configs/base.yaml", "configs/exp/regression.yaml")
    assert cfg["model"]["in_channels"] == 3 and cfg["method"]["name"] == "regression"
    assert cfg["model"]["model_channels"] == 64  # giữ từ base


@pytest.fixture
def tiny_run(tmp_path, monkeypatch):
    """Dữ liệu giả + ghi đè config để train vài step trên CPU."""
    ids = {"train": ["1BA001", "1BA002", "1PA001", "1PA002"], "val": ["1BB001", "1PC001"], "test": ["1BC001"]}
    for i, pid in enumerate(sum(ids.values(), [])):
        make_patient(tmp_path / "processed", pid, spacing_z=1.0 if pid[1] == "B" else 2.5, seed=i)
    (tmp_path / "split.json").write_text(json.dumps({**ids, "special": {"1PC001": "MR: test"}}))
    overrides = [
        f"output_dir={tmp_path.as_posix()}", f"data.processed_dir={(tmp_path / 'processed').as_posix()}",
        f"data.split.file={(tmp_path / 'split.json').as_posix()}", "data.patch_size=32", "data.num_workers=0",
        "data.pin_memory=false", *(f"model.{k}={list(v) if isinstance(v, tuple) else v}" for k, v in TINY_MODEL.items()),
        "train.batch_size=2", "train.grad_accum=2", "train.total_images=24", "train.log_every=2",
        "train.val_every=3", "train.save_every=3", "train.lr_schedule.warmup_steps=2", "train.precision=fp32",
        "train.cudnn_benchmark=false", "inference.precision=fp32", "inference.roi_size=32", "inference.sw_batch_size=4",
        "logging.wandb.enabled=false",
    ]

    def run(*extra, resume=False):
        argv = ["train", "--exp", "configs/exp/regression.yaml", "--set", *overrides, *extra]
        monkeypatch.setattr(sys, "argv", argv + (["--resume"] if resume else []))
        train_main()

    return tmp_path / "runs" / "regression", run


def test_train_end_to_end(tiny_run, monkeypatch, capsys):
    run_dir, run = tiny_run
    run()  # 24 ảnh / (2 x 2) = 6 optimizer step
    log = pd.read_csv(run_dir / "train_log.csv")
    assert log["step"].tolist() == [2, 4, 6] and log["loss"].notna().all()
    assert pd.read_csv(run_dir / "val_log.csv")["step"].tolist() == [3, 6]
    assert {p.name for p in (run_dir / "checkpoints").iterdir()} == {"step_000003.pt", "step_000006.pt"}
    vf = pd.read_csv(run_dir / "val_full.csv")
    assert set(vf["step"]) == {3, 6} and len(vf) == 4 and (vf["mae"] > 0).all()
    assert (run_dir / "best.pt").exists() and (run_dir / "last.pt").exists()

    with pytest.raises(SystemExit):  # không ghi đè run cũ
        run()
    run("train.total_images=32", resume=True)  # train tiếp 6 -> 8
    assert pd.read_csv(run_dir / "train_log.csv")["step"].tolist() == [2, 4, 6, 8]

    monkeypatch.setattr(sys, "argv", ["evaluate", "--run", str(run_dir), "--ckpt", "best", "--split", "test"])
    evaluate.main()
    out = run_dir / "eval" / "test_best"
    per = pd.read_csv(out / "per_patient.csv")
    assert per["id"].tolist() == ["1BC001"] and per["nfe"].tolist() == [1]
    summary = pd.read_csv(out / "summary.csv")
    assert set(summary["subset"]) == {"all", "no_special"}


def test_val_fast_examples(tiny_run):
    run_dir, run = tiny_run
    run()
    model, method, cfg, _ = evaluate.load_checkpoint(run_dir / "best.pt", torch.device("cpu"))
    val_ds = evaluate.build_dataset(cfg, ["1BB001", "1PC001"], "val")
    res, examples = evaluate.val_fast(method, model, val_ds, [1, 4], cfg["inference"], torch.device("cpu"))
    assert list(res) == ["mae_nfe1"]
    assert [e["id"] for e in examples] == ["1BB001", "1PC001"]  # 1 lát mỗi region
    e = examples[0]
    assert e["mr"].shape == e["sct"].shape == e["ct"].shape == e["mask"].shape
    panel = example_panel(e["mr"], e["sct"], e["ct"], e["mask"])
    assert panel.dtype == np.uint8 and panel.shape == (e["mr"].shape[0], 4 * e["mr"].shape[1])


def test_wandb_disabled_is_noop(tmp_path):
    wb = WandbLogger({"logging": {"wandb": {"enabled": False}}}, "x", tmp_path, resume=False)
    wb.log({"a": 1}, step=1)
    wb.log_examples("k", [{"id": "1BA001"}], step=1)
    wb.summary({"a": 1})
    wb.finish()
    assert wb.run is None and not (tmp_path / "wandb_id.txt").exists()


def test_summarize_special():
    df = pd.DataFrame({"id": ["a", "b", "c"], "region": ["brain", "brain", "pelvis"], "special": [False, True, False],
                       "nfe": [1, 1, 1], "mae": [10.0, 30.0, 50.0], "psnr": [30.0] * 3, "ssim": [0.9] * 3})
    s = evaluate.summarize(df).set_index(["subset", "region"])
    assert s.loc[("all", "brain"), "mae_mean"] == 20 and s.loc[("no_special", "brain"), "mae_mean"] == 10
    assert s.loc[("all", "all"), "n"] == 3 and s.loc[("no_special", "all"), "n"] == 2
