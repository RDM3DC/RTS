"""
Thin CLI wrapper that calls the existing scripts so labs can run:

  rts hydride --exp-csv sample_exp_H3S.csv --dft-csv sample_dft_hydride_curves.csv --out outdir/

and

  rts squid --exp-csv TEMPLATE_squid_film_inputs.csv --out outdir/
"""
from __future__ import annotations
import subprocess, sys
from pathlib import Path
import typer
from rich import print
from rich.panel import Panel
from .config import load_config
from .reporting import build_markdown

app = typer.Typer(help="RTS: Rapid Test Suite (hydrides & Bell-chip overlays)")

ROOT = Path(__file__).resolve().parents[1]

def _py() -> str:
    return sys.executable

@app.command()
def hydride(
    exp_csv: Path = typer.Option(..., "--exp-csv", help="Experimental hydride CSV (e.g., sample_exp_H3S.csv)"),
    dft_csv: Path = typer.Option(..., "--dft-csv", help="DFT curves CSV (e.g., sample_dft_hydride_curves.csv)"),
    out: Path = typer.Option(Path("out_hydride"), "--out", help="Output directory"),
    extra: str = typer.Option("", "--extra", help="Extra args passed to arp_fit_overlay.py"),
):
    """Run the hydride ARP overlay."""
    out.mkdir(parents=True, exist_ok=True)
    script = ROOT / "arp_fit_overlay.py"
    if not script.exists():
        raise typer.BadParameter(f"Cannot find {script}")
    cmd = [
        _py(), str(script),
        "--exp", str(exp_csv),
        "--dft", str(dft_csv),
        "--out", str(out),
    ]
    if extra:
        cmd += extra.split()
    print(Panel.fit("Running hydride overlay…", title="RTS"))
    res = subprocess.run(cmd, check=False, capture_output=True, text=True)
    print(res.stdout)
    if res.returncode != 0:
        print(Panel.fit(res.stderr, title="Error", style="red"))
        raise typer.Exit(res.returncode)
    print(Panel.fit(f"Done. Outputs in: {out}", style="green"))

@app.command()
def squid(
    exp_csv: Path = typer.Option(..., "--exp-csv", help="SQUID/film CSV (e.g., TEMPLATE_squid_film_inputs.csv)"),
    out: Path = typer.Option(Path("out_squid"), "--out", help="Output directory"),
    extra: str = typer.Option("", "--extra", help="Extra args passed to arp_fit_new.py"),
):
    """Run the SQUID/film ARP overlay (Bell-chip analogue)."""
    out.mkdir(parents=True, exist_ok=True)
    script = ROOT / "arp_fit_new.py"
    if not script.exists():
        raise typer.BadParameter(f"Cannot find {script}")
    cmd = [
        _py(), str(script),
        "--exp", str(exp_csv),
        "--out", str(out),
    ]
    if extra:
        cmd += extra.split()
    print(Panel.fit("Running SQUID/film overlay…", title="RTS"))
    res = subprocess.run(cmd, check=False, capture_output=True, text=True)
    print(res.stdout)
    if res.returncode != 0:
        print(Panel.fit(res.stderr, title="Error", style="red"))
        raise typer.Exit(res.returncode)
    print(Panel.fit(f"Done. Outputs in: {out}", style="green"))

@app.command("from-config")
def from_config(
    cfg: Path = typer.Option(Path("rts.toml"), "--cfg", help="Path to rts.toml"),
    run: str = typer.Option("all", "--run", help="Which block to run: hydride|squid|all"),
):
    """Run tasks as specified in rts.toml."""
    conf = load_config(cfg)
    if run in ("hydride", "all") and conf.hydride:
        hyd = conf.hydride
        hyd.out.mkdir(parents=True, exist_ok=True)
        cmd = [
            _py(), str(ROOT / "arp_fit_overlay.py"),
            "--exp", str(hyd.exp_csv),
            "--dft", str(hyd.dft_csv),
            "--out", str(hyd.out),
        ] + (hyd.extra.split() if hyd.extra else [])
        print(Panel.fit(f"Hydride from {cfg} …", title="RTS"))
        res = subprocess.run(cmd, capture_output=True, text=True)
        print(res.stdout or "")
        if res.returncode != 0:
            print(Panel.fit(res.stderr, title="Hydride error", style="red"))
    if run in ("squid", "all") and conf.squid:
        sq = conf.squid
        sq.out.mkdir(parents=True, exist_ok=True)
        cmd = [
            _py(), str(ROOT / "arp_fit_new.py"),
            "--exp", str(sq.exp_csv),
            "--out", str(sq.out),
        ] + (sq.extra.split() if sq.extra else [])
        print(Panel.fit(f"SQUID/film from {cfg} …", title="RTS"))
        res = subprocess.run(cmd, capture_output=True, text=True)
        print(res.stdout or "")
        if res.returncode != 0:
            print(Panel.fit(res.stderr, title="SQUID error", style="red"))

@app.command()
def report(
    in_: Path = typer.Option(..., "--in", help="Directory with PNG/JSON outputs"),
    out: Path = typer.Option(Path("report.md"), "--out", help="Markdown path to write"),
    title: str = typer.Option("RTS Report", "--title"),
):
    """Bundle outputs (PNG/JSON) into a Markdown report in the same folder."""
    md = build_markdown(in_, title=title)
    out.write_text(md, encoding="utf-8")
    print(Panel.fit(f"Wrote report: {out}", style="green"))
