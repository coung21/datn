import math
import re
import sys

import numpy as np
import pandas as pd
import pytest
import torch
from conftest import make_tiny_run

from src.eval import evaluate
from src.eval.metrics import all_metrics
from src.methods import Regression, build_method, masked_l1
from src.models import UNetModel
from src.train import cosine_with_warmup
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
    model.eval()
    out = method.sample(lambda inp, t: model(inp, t), batch["cond"], batch["mask"], torch.randn(2, 1, 32, 32), nfe=1)
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
    return make_tiny_run(tmp_path, monkeypatch, "configs/exp/regression.yaml", in_channels=3)


def test_train_end_to_end(tiny_run, monkeypatch, capsys):
    runs, run = tiny_run
    run_dir = run()  # 24 ảnh / (2 x 2) = 6 optimizer step
    assert re.fullmatch(r"regression_\d{8}-\d{6}", run_dir.name)
    log = pd.read_csv(run_dir / "train_log.csv")
    assert log["step"].tolist() == [2, 4, 6] and log["loss"].notna().all()
    assert pd.read_csv(run_dir / "val_log.csv")["step"].tolist() == [3, 6]
    assert {p.name for p in (run_dir / "checkpoints").iterdir()} == {"step_000003.pt", "step_000006.pt"}
    vf = pd.read_csv(run_dir / "val_full.csv")
    assert set(vf["step"]) == {3, 6} and len(vf) == 4 and (vf["mae"] > 0).all()
    assert (run_dir / "best.pt").exists() and (run_dir / "last.pt").exists()

    run("train.total_images=32", resume=run_dir)  # train tiếp 6 -> 8, config đọc từ run
    assert pd.read_csv(run_dir / "train_log.csv")["step"].tolist() == [2, 4, 6, 8]
    with pytest.raises(SystemExit):  # thư mục không phải run
        run(resume=runs / "khong_ton_tai")

    monkeypatch.setattr(sys, "argv", ["evaluate", "--run", str(run_dir), "--ckpt", "best", "--split", "test"])
    evaluate.main()
    out = run_dir / "eval" / "test_best"
    per = pd.read_csv(out / "per_patient.csv")
    assert per["id"].tolist() == ["1BC001"] and per["nfe"].tolist() == [1]
    summary = pd.read_csv(out / "summary.csv")
    assert set(summary["subset"]) == {"all", "no_special"}


def test_val_fast_examples(tiny_run):
    _, run = tiny_run
    run_dir = run()
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
    wb = WandbLogger({"logging": {"wandb": {"enabled": False}}}, tmp_path, resume=False)
    wb.log({"a": 1}, step=1)
    wb.log_examples("k", [{"id": "1BA001"}], step=1)
    wb.summary({"a": 1})
    wb.finish()
    assert wb.run is None and not (tmp_path / "wandb_id.txt").exists()


@pytest.fixture
def fake_wandb(monkeypatch):
    """Module wandb giả, chỉ có init / Image / Table: không phụ thuộc API nội bộ đổi theo phiên bản."""
    calls = {"inits": [], "log": [], "summary": {}}

    class FakeRun:
        def __init__(self, run_id):
            self.id = run_id
            self.summary = calls["summary"]

        def log(self, data, step=None):
            calls["log"].append((data, step))

        def finish(self):
            calls["finish"] = True

    def fake_init(**kw):
        calls["init"] = kw
        calls["inits"].append(kw)
        return FakeRun(kw["id"])

    fake = type(sys)("wandb")
    fake.init, fake.Image, fake.Table = fake_init, lambda arr, caption=None: arr, lambda dataframe: dataframe
    monkeypatch.setitem(sys.modules, "wandb", fake)
    return calls


def test_wandb_enabled_with_fake_module(tmp_path, fake_wandb):
    calls = fake_wandb
    run_dir = tmp_path / "regression_20261016-230229"
    run_dir.mkdir()
    cfg = {"run_name": "regression", "logging": {"wandb": {"enabled": True, "project": "mri2ct", "entity": "e"}}}
    wb = WandbLogger(cfg, run_dir, resume=False)
    assert calls["init"]["name"] == run_dir.name and calls["init"]["group"] == "regression"
    run_id = (run_dir / "wandb_id.txt").read_text()
    assert re.fullmatch(r"[0-9a-f]{8}", run_id)
    wb.log({"train/loss": 1.0}, step=3)
    wb.finish()
    assert calls["log"] == [({"train/loss": 1.0}, 3)] and calls["finish"]
    WandbLogger(cfg, run_dir, resume=True)
    assert calls["init"]["id"] == run_id  # resume giữ nguyên id


def test_wandb_train_then_evaluate(tiny_run, fake_wandb, monkeypatch):
    calls = fake_wandb
    _, run = tiny_run
    run_dir = run("logging.wandb.enabled=true", "logging.wandb.entity=e")
    train_id = calls["inits"][0]["id"]

    # ảnh mẫu: mỗi region 1 key riêng
    example_keys = {k for data, _ in calls["log"] for k in data if k.startswith("val/examples")}
    assert example_keys == {"val/examples/brain", "val/examples/pelvis"}

    monkeypatch.setattr(sys, "argv", ["evaluate", "--run", str(run_dir), "--ckpt", "best", "--split", "test"])
    evaluate.main()
    assert calls["inits"][-1]["id"] == train_id  # ghi tiếp vào đúng run train
    logged = {k for data, _ in calls["log"] for k in data}
    assert {"eval/test_best/per_patient", "eval/test_best/summary"} <= logged
    assert calls["summary"]["test_best/nfe1/all/mae"] > 0
    assert "test_best_no_special/nfe1/brain/mae" in calls["summary"]


def test_summary_metrics_keys():
    summary = pd.DataFrame([
        {"subset": "all", "nfe": 4, "region": "pelvis", "n": 2, "mae_mean": 50.0, "mae_std": 1.0,
         "psnr_mean": 29.0, "psnr_std": 1.0, "ssim_mean": 0.8, "ssim_std": 0.1},
        {"subset": "no_special", "nfe": 4, "region": "all", "n": 1, "mae_mean": 40.0, "mae_std": 0.0,
         "psnr_mean": 30.0, "psnr_std": 0.0, "ssim_mean": 0.9, "ssim_std": 0.0},
    ])
    m = evaluate.summary_metrics(summary, "test_best")
    assert m == {"test_best/nfe4/pelvis/mae": 50.0, "test_best/nfe4/pelvis/psnr": 29.0,
                 "test_best/nfe4/pelvis/ssim": 0.8, "test_best_no_special/nfe4/all/mae": 40.0,
                 "test_best_no_special/nfe4/all/psnr": 30.0, "test_best_no_special/nfe4/all/ssim": 0.9}


def test_summarize_special():
    df = pd.DataFrame({"id": ["a", "b", "c"], "region": ["brain", "brain", "pelvis"], "special": [False, True, False],
                       "nfe": [1, 1, 1], "mae": [10.0, 30.0, 50.0], "psnr": [30.0] * 3, "ssim": [0.9] * 3})
    s = evaluate.summarize(df).set_index(["subset", "region"])
    assert s.loc[("all", "brain"), "mae_mean"] == 20 and s.loc[("no_special", "brain"), "mae_mean"] == 10
    assert s.loc[("all", "all"), "n"] == 3 and s.loc[("no_special", "all"), "n"] == 2
