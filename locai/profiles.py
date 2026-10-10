"""Business profiles (YAML) and place-group matching."""
import hashlib
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import yaml

from .config import PROFILES_DIR


@dataclass
class Profile:
    id: str
    label: str
    walk_minutes: float
    tau_min: float
    segment: dict[str, float]
    competitors: list[str]
    competition_curve: dict
    synergy: dict[str, float]
    weights: dict[str, float]
    metrics: dict[str, dict[str, float]]
    caps: list[dict] = field(default_factory=list)
    fingerprint: str = ""


def load_groups() -> dict[str, set[str]]:
    with open(PROFILES_DIR / "_groups.yaml") as f:
        return {k: set(v) for k, v in yaml.safe_load(f).items()}


def list_profiles() -> list[str]:
    return sorted(p.stem for p in PROFILES_DIR.glob("*.yaml") if not p.stem.startswith("_"))


def load_profile(pid: str) -> Profile:
    path = PROFILES_DIR / f"{pid}.yaml"
    if not path.exists():
        raise ValueError(f"Unknown profile '{pid}'. Available: {', '.join(list_profiles())}")
    raw_text = path.read_text()
    groups_text = (PROFILES_DIR / "_groups.yaml").read_text()
    raw = yaml.safe_load(raw_text)
    return Profile(
        id=raw["id"],
        label=raw["label"],
        walk_minutes=raw["catchment"]["walk_minutes"],
        tau_min=raw["catchment"]["tau_min"],
        segment=raw["segment"],
        competitors=raw["competitors"],
        competition_curve=raw["competition_curve"],
        synergy=raw["synergy"],
        weights=raw["weights"],
        metrics=raw["metrics"],
        caps=raw.get("caps", []),
        fingerprint=hashlib.sha1((raw_text + groups_text).encode()).hexdigest()[:10],
    )


def group_masks(places: pd.DataFrame, groups: dict[str, set[str]]) -> dict[str, np.ndarray]:
    """Boolean mask per group over open places."""
    is_open = (places["status"] != "permanently_closed").to_numpy()
    tokens = [set(h.split("|")) if h else set() for h in places["hierarchy"]]
    return {
        g: np.fromiter((bool(t & members) for t in tokens), dtype=bool, count=len(tokens)) & is_open
        for g, members in groups.items()
    }
