"""Configuration loading and the edit-config grid."""
from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

KIT_ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class EditConfig:
    t_start: float
    t_end: float
    t_delta: float = 0.0
    cleanup: bool = True

    @property
    def id(self) -> str:
        return f"ts{self.t_start:.2f}_te{self.t_end:.2f}_d{self.t_delta:.2f}_c{int(self.cleanup)}"

    def as_edit_config(self, base: Dict[str, Any]) -> Dict[str, Any]:
        cfg = dict(base)
        cfg.update(
            t_start=float(self.t_start),
            t_end=float(self.t_end),
            t_delta=float(self.t_delta),
            cleanup=bool(self.cleanup),
        )
        return cfg

    @staticmethod
    def from_id(config_id: str) -> "EditConfig":
        ts = te = d = c = None
        for token in config_id.split("_"):
            if token.startswith("ts"):
                ts = float(token[2:])
            elif token.startswith("te"):
                te = float(token[2:])
            elif token.startswith("d"):
                d = float(token[1:])
            elif token.startswith("c"):
                c = bool(int(token[1:]))
        if None in (ts, te, d, c):
            raise ValueError(f"Bad config id: {config_id}")
        return EditConfig(ts, te, d, c)


@dataclass
class Settings:
    raw: Dict[str, Any]
    root: Path = KIT_ROOT
    named: Dict[str, str] = field(default_factory=dict)  # name -> config id

    def path(self, key: str) -> Path:
        value = Path(self.raw["paths"][key]).expanduser()
        return value if value.is_absolute() else (self.root / value)

    @property
    def work(self) -> Path:
        p = self.path("work_dir")
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def results(self) -> Path:
        p = self.work / "results"
        p.mkdir(parents=True, exist_ok=True)
        return p

    def base_edit_config(self) -> Dict[str, Any]:
        g = self.raw["grid"]
        return {
            "noise_samples": int(g.get("noise_samples", 1)),
            "n_steps": int(g.get("n_steps", 1)),
            "step_scale": float(g.get("step_scale", 1.0)),
        }

    def grid(self) -> List[EditConfig]:
        g = self.raw["grid"]
        configs: List[EditConfig] = []
        seen = set()
        for ts, te, d, c in product(g["t_start"], g["t_end"], g["t_delta"], g["cleanup"]):
            ec = EditConfig(float(ts), float(te), float(d), bool(c))
            if ec.id not in seen:
                seen.add(ec.id)
                configs.append(ec)
        for extra in g.get("extra", []) or []:
            ec = EditConfig(
                float(extra["t_start"]),
                float(extra["t_end"]),
                float(extra.get("t_delta", 0.0)),
                bool(extra.get("cleanup", True)),
            )
            self.named[extra["name"]] = ec.id
            if ec.id not in seen:
                seen.add(ec.id)
                configs.append(ec)
        return configs

    def seeds(self, quick: bool = False) -> List[int]:
        seeds = [int(s) for s in self.raw["grid"]["seeds"]]
        return seeds[:1] if quick else seeds

    def reference_id(self) -> str:
        if not self.named:
            self.grid()
        ref = self.raw.get("reference_config", "paper_default")
        return self.named.get(ref, ref)


def load_settings(path: Optional[str] = None, overrides: Optional[Dict[str, Any]] = None) -> Settings:
    cfg_path = Path(path) if path else KIT_ROOT / "configs" / "default.yaml"
    with cfg_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if overrides:
        for dotted, value in overrides.items():
            node = raw
            keys = dotted.split(".")
            for key in keys[:-1]:
                node = node.setdefault(key, {})
            node[keys[-1]] = value
    settings = Settings(raw=raw)
    settings.grid()  # populate named configs
    return settings
