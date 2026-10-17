import sys

import pandas as pd
import pytest
import torch
from conftest import TINY_MODEL, make_tiny_run
from monai.inferers import sliding_window_inference

from src.eval import evaluate
from src.inference.predict import padded_shape, predict_slices, slice_noise
from src.methods import FlowMatching, Regression, build_method
from src.models import UNetModel

INF = dict(roi_size=32, sw_batch_size=4, overlap=0.25, blend_mode="gaussian", pad_value=-1.0, precision="fp32")


def tiny_model(in_channels):
    torch.manual_seed(0)
    return UNetModel(**TINY_MODEL, in_channels=in_channels).eval()


def make_batch(n=2, size=32):
    torch.manual_seed(1)
    mask = torch.zeros(n, 1, size, size)
    mask[..., 8:24, 6:26] = 1
    return {"cond": torch.randn(n, 3, size, size), "ct": torch.rand(n, 1, size, size) * 2 - 1, "mask": mask}


@pytest.mark.parametrize("h, w", [(20, 28), (40, 50), (32, 45)])
def test_regression_same_as_monai_padding(h, w):
    """Pad trước rồi sliding window phải cho kết quả y hệt để MONAI tự pad (baseline không đổi)."""
    model = tiny_model(3)
    cond, mask = torch.randn(2, 3, h, w), torch.ones(2, 1, h, w)
    noise = torch.zeros(2, 1, *padded_shape(h, w, 32))
    new = predict_slices(Regression(), model, cond, mask, noise, 1, INF, torch.device("cpu"))
    with torch.no_grad():
        old = sliding_window_inference(cond, (32, 32), 4, lambda x: model(x, torch.zeros(x.shape[0])),
                                       overlap=0.25, mode="gaussian", cval=-1.0).clamp(-1, 1)
    assert new.shape == (2, 1, h, w)
    torch.testing.assert_close(new, old, atol=1e-5, rtol=1e-5)


def test_fm_loss_masked_and_background():
    method = build_method("flow_matching", fix_background=True)
    assert isinstance(method, FlowMatching) and method.extra_in_channels == 1
    seen = {}

    class Spy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.w = torch.nn.Parameter(torch.ones(()))

        def forward(self, x, t):
            seen["x"], seen["t"] = x, t
            return x[:, :1] * self.w

    batch = make_batch()
    torch.manual_seed(0)
    loss, logs = method.loss(Spy(), batch)
    loss.backward()
    assert loss.ndim == 0 and logs["mse"] == pytest.approx(loss.item())
    assert seen["x"].shape == (2, 4, 32, 32) and seen["t"].shape == (2,)
    assert (seen["x"][:, :1][batch["mask"] == 0] == -1).all()  # nền cố định = -1
    assert (seen["x"][:, 1:] == batch["cond"]).all()

    # CT ngoài mask không ảnh hưởng loss
    other = {**batch, "ct": torch.where(batch["mask"] > 0, batch["ct"], torch.full_like(batch["ct"], 0.7))}
    torch.manual_seed(0)
    loss2, _ = method.loss(Spy(), other)
    assert loss2.item() == pytest.approx(loss.item())


def test_fm_loss_trains_real_unet():
    model = tiny_model(4).train()
    loss, _ = FlowMatching().loss(model, make_batch())
    loss.backward()
    assert torch.isfinite(loss) and model.input_blocks[0][0].weight.grad is not None


@pytest.mark.parametrize("nfe", [1, 2, 4, 10])
def test_fm_euler_reaches_target_with_oracle_velocity(nfe):
    """Vận tốc đúng (hằng số x1 - x0 trên đường thẳng) -> Euler tới đúng x1 với mọi NFE."""
    batch = make_batch()
    target, mask = batch["ct"], batch["mask"]
    noise = torch.randn_like(target)
    x0 = torch.where(mask > 0, noise, -1.0)
    calls = []

    def net(inp, t):
        calls.append(float(t))
        assert inp.shape[1] == 4
        return target - x0

    out = FlowMatching().sample(net, batch["cond"], mask, noise, nfe)
    assert len(calls) == nfe and calls[0] == 0.0
    torch.testing.assert_close(out[mask > 0], target[mask > 0], atol=1e-5, rtol=0)
    assert (out[mask == 0] == -1).all()


def test_fm_sample_deterministic_given_noise():
    model = tiny_model(4)
    batch = make_batch()
    noise = torch.randn(2, 1, 32, 32)
    net = lambda inp, t: model(inp, t)  # noqa: E731
    a = FlowMatching().sample(net, batch["cond"], batch["mask"], noise, 4)
    b = FlowMatching().sample(net, batch["cond"], batch["mask"], noise, 4)
    c = FlowMatching().sample(net, batch["cond"], batch["mask"], torch.randn(2, 1, 32, 32), 4)
    assert torch.equal(a, b) and not torch.equal(a, c)


def test_slice_noise_seeded():
    assert torch.equal(slice_noise("1BA001", 5, (8, 9)), slice_noise("1BA001", 5, (8, 9)))
    assert not torch.equal(slice_noise("1BA001", 5, (8, 9)), slice_noise("1BA001", 6, (8, 9)))
    assert not torch.equal(slice_noise("1BA001", 5, (8, 9)), slice_noise("1BA002", 5, (8, 9)))
    assert padded_shape(20, 50, 32) == (32, 50)


def test_fm_end_to_end(tmp_path, monkeypatch):
    runs, run = make_tiny_run(tmp_path, monkeypatch, "configs/exp/flow_matching.yaml", in_channels=4)
    run_dir = run()
    assert run_dir.name.startswith("flow_matching_")
    log = pd.read_csv(run_dir / "train_log.csv")
    assert log["loss"].notna().all()
    val = pd.read_csv(run_dir / "val_log.csv")
    assert {"mae_nfe1", "mae_nfe4"} <= set(val.columns) and (val["mae_nfe4"] > 0).all()
    vf = pd.read_csv(run_dir / "val_full.csv")
    assert set(vf["nfe"]) == {4}

    argv = ["evaluate", "--run", str(run_dir), "--ckpt", "best", "--split", "test", "--nfe", "1", "2", "--no-wandb"]
    monkeypatch.setattr(sys, "argv", argv)
    evaluate.main()
    per = pd.read_csv(run_dir / "eval" / "test_best" / "per_patient.csv")
    assert sorted(per["nfe"]) == [1, 2]

    # chạy lại evaluate: cùng seed noise -> cùng kết quả
    evaluate.main()
    again = pd.read_csv(run_dir / "eval" / "test_best" / "per_patient.csv")
    pd.testing.assert_frame_equal(per.drop(columns="seconds"), again.drop(columns="seconds"))
