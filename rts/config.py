from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

# tomllib is stdlib in Py >=3.11; fallback to tomli on earlier Pythons
try:
    import tomllib as toml
except Exception:  # pragma: no cover - fallback for older Pythons
    import tomli as toml  # type: ignore

@dataclass
class HydrideCfg:
    exp_csv: Path
    dft_csv: Path
    out: Path = Path("out_hydride")
    extra: str = ""

@dataclass
class SquidCfg:
    exp_csv: Path
    out: Path = Path("out_squid")
    extra: str = ""

@dataclass
class RTSConfig:
    hydride: Optional[HydrideCfg] = None
    squid: Optional[SquidCfg] = None

def load_config(path: Path) -> RTSConfig:
    with open(path, "rb") as f:
        data = toml.load(f)
    hyd = None
    if "hydride" in data:
        h = data["hydride"]
        hyd = HydrideCfg(
            exp_csv=Path(h["exp_csv"]),
            dft_csv=Path(h["dft_csv"]),
            out=Path(h.get("out", "out_hydride")),
            extra=h.get("extra", ""),
        )
    sq = None
    if "squid" in data:
        s = data["squid"]
        sq = SquidCfg(
            exp_csv=Path(s["exp_csv"]),
            out=Path(s.get("out", "out_squid")),
            extra=s.get("extra", ""),
        )
    return RTSConfig(hydride=hyd, squid=sq)
