from __future__ import annotations
import json
from pathlib import Path
from typing import Iterable, Tuple, List

def discover_outputs(out_dir: Path) -> Tuple[List[Path], List[Path]]:
    pngs = sorted(out_dir.glob("*.png"))
    jsons = sorted(out_dir.glob("*.json"))
    return pngs, jsons

def summarize_json(json_paths: Iterable[Path]) -> str:
    lines = []
    for jp in json_paths:
        try:
            data = json.loads(jp.read_text())
            keys = [k for k in data if isinstance(data.get(k), (int, float, str))]
            keys = keys[:10]
            parts = [f"{k}={data[k]}" for k in keys]
            lines.append(f"- **{jp.name}**: " + ", ".join(parts))
        except Exception:
            lines.append(f"- **{jp.name}**: (could not parse)")
    return "\n".join(lines) if lines else "_No JSON reports found._"

def build_markdown(out_dir: Path, title: str = "RTS Report") -> str:
    pngs, jsons = discover_outputs(out_dir)
    md = [f"# {title}", ""]
    md.append("## Figures")
    if pngs:
        for p in pngs:
            md.append(f"![{p.name}]({p.name})")
    else:
        md.append("_No images found._")
    md.append("")
    md.append("## JSON summaries")
    md.append(summarize_json(jsons))
    return "\n".join(md)
