"""I/O conventions: prompts (JSONL), behavior (parquet), activation caches (fp16 memmaps),
run directories with resolved config and metadata. See PLAN §7.3."""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import platform
import shutil
import subprocess
from dataclasses import asdict, dataclass, field
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = REPO_ROOT / "results"
CACHE_DIR = REPO_ROOT / "data" / "cache"


# ---------------------------------------------------------------------------
# Run configs
# ---------------------------------------------------------------------------


@dataclass
class RunConfig:
    """One experiment run. Loaded from `configs/runs/*.yaml`; unknown keys go in `extra`."""

    exp: str
    model: str
    prompt_set: str
    desc: str = "run"
    positions: list[str] = field(default_factory=lambda: ["query_item", "decision"])
    layers: list[int] | str = "all"
    batch_size: int = 16
    dtype: str = "float32"
    device: str = "auto"
    seed: int = 0
    output_dir: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_yaml(cls, path: str | Path) -> RunConfig:
        raw = yaml.safe_load(Path(path).read_text())
        known = {f for f in cls.__dataclass_fields__ if f != "extra"}
        # a saved config (to_dict) already has an `extra` mapping: merge it, don't nest it
        extra = dict(raw.pop("extra", None) or {})
        extra |= {k: v for k, v in raw.items() if k not in known}
        return cls(**{k: v for k, v in raw.items() if k in known}, extra=extra)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def model_short_name(model_id: str) -> str:
    """'Qwen/Qwen3.5-0.8B' -> 'qwen3.5-0.8b' (used in run IDs and cache paths)."""
    return model_id.split("/")[-1].lower()


def make_run_id(exp: str, model: str, desc: str, now: dt.datetime | None = None) -> str:
    """`<YYYYMMDD-HHMM>_<exp>_<model>_<desc>`."""
    now = now or dt.datetime.now()
    return f"{now:%Y%m%d-%H%M}_{exp}_{model_short_name(model)}_{desc}"


# ---------------------------------------------------------------------------
# Run directories and metadata
# ---------------------------------------------------------------------------


def _git(*args: str) -> str | None:
    try:
        out = subprocess.run(
            ["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, timeout=10
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    return out.stdout.strip() if out.returncode == 0 else None


def git_state() -> dict[str, Any]:
    commit = _git("rev-parse", "HEAD")
    status = _git("status", "--porcelain")
    return {"commit": commit, "dirty": bool(status) if status is not None else None}


def package_versions(names: tuple[str, ...] = (
    "torch", "transformers", "numpy", "scipy", "pandas", "scikit-learn", "jlens",
)) -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for name in names:
        try:
            versions[name] = importlib_metadata.version(name)
        except importlib_metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def device_info() -> dict[str, Any]:
    info: dict[str, Any] = {"platform": platform.platform(), "machine": platform.machine()}
    try:
        import torch

        info["cuda"] = torch.cuda.is_available()
        if info["cuda"]:
            info["gpu"] = torch.cuda.get_device_name(0)
        info["mps"] = torch.backends.mps.is_available()
    except ImportError:
        pass
    return info


class Run:
    """A results directory: `results/<run_id>/` with `config.yaml` and `metadata.json`.

    Use as a context manager so the end time and status are recorded even on failure.
    """

    def __init__(self, config: RunConfig, root: Path = RESULTS_DIR, run_id: str | None = None):
        self.config = config
        self.run_id = run_id or make_run_id(config.exp, config.model, config.desc)
        self.dir = Path(config.output_dir) if config.output_dir else root / self.run_id
        self.metadata: dict[str, Any] = {}

    def __enter__(self) -> Run:
        if not self.config.output_dir:
            base, k = self.dir, 2
            while self.dir.exists():  # two runs started in the same minute
                self.dir = base.with_name(f"{base.name}-{k}")
                k += 1
            self.run_id = self.dir.name
        self.dir.mkdir(parents=True, exist_ok=False)
        config_text = yaml.safe_dump(self.config.to_dict(), sort_keys=False)
        (self.dir / "config.yaml").write_text(config_text)
        self.metadata = {
            "run_id": self.run_id,
            "git": git_state(),
            "packages": package_versions(),
            "device": device_info(),
            "host": platform.node(),
            "start": dt.datetime.now().isoformat(timespec="seconds"),
        }
        self._write_metadata()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.metadata["end"] = dt.datetime.now().isoformat(timespec="seconds")
        self.metadata["status"] = "failed" if exc_type else "ok"
        if exc_type:
            self.metadata["error"] = repr(exc)
        self._write_metadata()

    def _write_metadata(self) -> None:
        (self.dir / "metadata.json").write_text(json.dumps(self.metadata, indent=2, default=str))

    def path(self, name: str) -> Path:
        return self.dir / name


# ---------------------------------------------------------------------------
# Prompts and behavior
# ---------------------------------------------------------------------------


def write_jsonl(records: list[dict[str, Any]], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w") as f:
        for rec in records:
            f.write(json.dumps(rec, sort_keys=True) + "\n")
    os.replace(tmp, path)


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open() as f:
        return [json.loads(line) for line in f if line.strip()]


def prompt_set_hash(records: list[dict[str, Any]]) -> str:
    """Content hash of a prompt set (order-sensitive), 12 hex chars. Used in cache paths
    so a changed prompt set never reuses a stale activation cache."""
    h = hashlib.sha256()
    for rec in records:
        h.update(json.dumps(rec, sort_keys=True).encode())
        h.update(b"\n")
    return h.hexdigest()[:12]


def write_parquet(df: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_parquet(tmp, index=False)
    os.replace(tmp, path)


def read_parquet(path: str | Path) -> pd.DataFrame:
    return pd.read_parquet(path)


# ---------------------------------------------------------------------------
# Activation caches
# ---------------------------------------------------------------------------


def activation_dir(model: str, set_hash: str, root: Path = CACHE_DIR) -> Path:
    return root / model_short_name(model) / set_hash


class ActivationWriter:
    """Writes `<dir>/<position>.npy` fp16 memmaps of shape [n_prompts, n_layers+1, d_model]
    plus `<dir>/index.parquet` (row -> prompt_id).

    Everything is written into `<dir>.partial/` and renamed on `close()`, so an interrupted
    run never leaves a cache that looks complete.
    """

    def __init__(
        self,
        out_dir: str | Path,
        prompt_ids: list[str],
        positions: list[str],
        n_states: int,
        d_model: int,
    ):
        self.out_dir = Path(out_dir)
        if self.out_dir.exists():
            raise FileExistsError(f"activation cache already exists: {self.out_dir}")
        self.tmp_dir = self.out_dir.with_name(self.out_dir.name + ".partial")
        if self.tmp_dir.exists():
            shutil.rmtree(self.tmp_dir)
        self.tmp_dir.mkdir(parents=True)
        self.prompt_ids = list(prompt_ids)
        self.shape = (len(prompt_ids), n_states, d_model)
        self.arrays = {
            pos: np.lib.format.open_memmap(
                self.tmp_dir / f"{pos}.npy", mode="w+", dtype=np.float16, shape=self.shape
            )
            for pos in positions
        }
        self.written = np.zeros(len(prompt_ids), dtype=bool)

    def write(self, rows: np.ndarray | list[int], position: str, values: np.ndarray) -> None:
        """values: [len(rows), n_states, d_model]."""
        self.arrays[position][rows] = values.astype(np.float16)
        self.written[rows] = True

    def close(self) -> Path:
        missing = int((~self.written).sum())
        if missing:
            raise RuntimeError(f"{missing} rows were never written; refusing to finalize")
        for arr in self.arrays.values():
            arr.flush()
        self.arrays.clear()
        write_parquet(
            pd.DataFrame({"row": np.arange(len(self.prompt_ids)), "prompt_id": self.prompt_ids}),
            self.tmp_dir / "index.parquet",
        )
        os.replace(self.tmp_dir, self.out_dir)
        return self.out_dir


def load_activations(cache_dir: str | Path, position: str) -> tuple[np.ndarray, pd.DataFrame]:
    """Returns (read-only memmap [n_prompts, n_states, d_model], index dataframe)."""
    cache_dir = Path(cache_dir)
    acts = np.load(cache_dir / f"{position}.npy", mmap_mode="r")
    index = pd.read_parquet(cache_dir / "index.parquet")
    return acts, index
