"""Log lên Weights & Biases (tắt được qua config logging.wandb.enabled)."""

from pathlib import Path

import numpy as np

CT_WINDOW = (-1024.0, 1500.0)  # cửa sổ hiển thị HU
ERR_MAX = 500.0                # |sCT - CT| >= 500 HU hiển thị trắng


def _to_uint8(x: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return (np.clip((x - lo) / (hi - lo), 0, 1) * 255).astype(np.uint8)


def example_panel(mr: np.ndarray, sct: np.ndarray, ct: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Ghép ngang 1 lát: MR (chuẩn hóa [-1,1]) | sCT | CT (HU) | |sCT - CT| trong mask."""
    err = np.where(mask > 0, np.abs(sct - ct), 0)
    return np.concatenate(
        [_to_uint8(mr, -1, 1), _to_uint8(sct, *CT_WINDOW), _to_uint8(ct, *CT_WINDOW), _to_uint8(err, 0, ERR_MAX)],
        axis=1,
    )


class WandbLogger:
    def __init__(self, cfg: dict, run_name: str, run_dir: Path, resume: bool):
        wc = cfg.get("logging", {}).get("wandb", {})
        self.run = None
        if not wc.get("enabled", False) or wc.get("mode") == "disabled":
            return
        import wandb

        self._wandb = wandb
        id_file = run_dir / "wandb_id.txt"
        run_id = id_file.read_text().strip() if resume and id_file.exists() else wandb.util.generate_id()
        id_file.write_text(run_id)
        self.run = wandb.init(
            project=wc["project"], entity=wc.get("entity"), name=run_name, id=run_id, resume="allow",
            config=cfg, dir=str(run_dir), mode=wc.get("mode", "online"), tags=wc.get("tags") or None,
        )

    def log(self, data: dict, step: int) -> None:
        if self.run:
            self.run.log(data, step=step)

    def log_examples(self, key: str, examples: list[dict], step: int) -> None:
        if self.run and examples:
            images = [
                self._wandb.Image(example_panel(e["mr"], e["sct"], e["ct"], e["mask"]),
                                  caption=f"{e['id']} z={e['z']} | MR | sCT | CT | |err|<={ERR_MAX:.0f}HU")
                for e in examples
            ]
            self.run.log({key: images}, step=step)

    def log_table(self, key: str, df) -> None:
        if self.run:
            self.run.log({key: self._wandb.Table(dataframe=df)})

    def summary(self, data: dict) -> None:
        if self.run:
            self.run.summary.update(data)

    def finish(self) -> None:
        if self.run:
            self.run.finish()
