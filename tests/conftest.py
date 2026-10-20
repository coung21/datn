import json
import sys

import numpy as np

from test_dataset import make_patient

from src.train import main as train_main

# UNet nhỏ để train vài step trên CPU (in_channels đặt theo method)
TINY_MODEL = dict(image_size=32, model_channels=32, channel_mult=(1, 2), num_res_blocks=1,
                  attention_resolutions=(16,), num_head_channels=32)


def write_fake_reg(patient_dir, seed: int, err_hu: float = 60.0) -> None:
    """reg.npy giả: CT + nhiễu ~err_hu HU trong mask, -1024 ngoài mask (như output của predict_reg)."""
    ct, mask = np.load(patient_dir / "ct.npy"), np.load(patient_dir / "mask.npy") > 0
    noisy = ct + np.random.default_rng(seed).normal(0, err_hu, ct.shape)
    np.save(patient_dir / "reg.npy", np.where(mask, np.clip(noisy, -1024, 3000), -1024).round().astype(np.int16))


def make_tiny_run(tmp_path, monkeypatch, exp: str, in_channels: int, with_reg: bool = False):
    """Dữ liệu giả + ghi đè config. Trả về (thư mục runs, run(*extra, resume=None) -> thư mục run)."""
    ids = {"train": ["1BA001", "1BA002", "1PA001", "1PA002"], "val": ["1BB001", "1PC001"], "test": ["1BC001"]}
    for i, pid in enumerate(sum(ids.values(), [])):
        make_patient(tmp_path / "processed", pid, spacing_z=1.0 if pid[1] == "B" else 2.5, seed=i)
        if with_reg:
            write_fake_reg(tmp_path / "processed" / ("brain" if pid[1] == "B" else "pelvis") / pid, seed=100 + i)
    (tmp_path / "split.json").write_text(json.dumps({**ids, "special": {"1PC001": "MR: test"}}))
    model = {**TINY_MODEL, "in_channels": in_channels}
    overrides = [
        f"output_dir={tmp_path.as_posix()}", f"data.processed_dir={(tmp_path / 'processed').as_posix()}",
        f"data.split.file={(tmp_path / 'split.json').as_posix()}", "data.patch_size=32", "data.num_workers=0",
        "data.pin_memory=false", *(f"model.{k}={list(v) if isinstance(v, tuple) else v}" for k, v in model.items()),
        "train.batch_size=2", "train.grad_accum=2", "train.total_images=24", "train.log_every=2",
        "train.val_every=3", "train.save_every=3", "train.lr_schedule.warmup_steps=2", "train.precision=fp32",
        "train.cudnn_benchmark=false", "inference.precision=fp32", "inference.roi_size=32", "inference.sw_batch_size=4",
        "logging.wandb.enabled=false",
    ]
    runs = tmp_path / "runs"

    def run(*extra, resume=None):
        """Train mới (trả về thư mục run vừa tạo) hoặc train tiếp run `resume`."""
        before = set(runs.glob("*")) if runs.exists() else set()
        if resume:
            argv = ["train", "--resume", str(resume), "--set", *extra]
        else:
            argv = ["train", "--exp", exp, "--set", *overrides, *extra]
        monkeypatch.setattr(sys, "argv", argv)
        train_main()
        if resume:
            return resume
        (new,) = set(runs.glob("*")) - before
        return new

    return runs, run
