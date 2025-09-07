"""
Command-line interface for RTS: Rapid Test Suite (hydrides & Bell-chip overlays).
Wraps the existing scripts so labs can run one-liners like:

  rts hydride --exp-csv sample_exp_H3S.csv --dft-csv sample_dft_hydride_curves.csv --out outdir/

and

  rts squid --exp-csv TEMPLATE_squid_film_inputs.csv --out outdir/
"""
from __future__ import annotations
import argparse, subprocess, sys
from pathlib import Path
from .config import load_config
from .reporting import build_markdown

ROOT = Path(__file__).resolve().parents[1]


def _py() -> str:
    return sys.executable


def run_hydride(exp_csv: Path, dft_csv: Path, out: Path, extra: str = "") -> None:
    out.mkdir(parents=True, exist_ok=True)
    script = ROOT / "arp_fit_overlay.py"
    cmd = [_py(), str(script), "--mode", "hydride", "--exp_csv", str(exp_csv), "--dft_csv", str(dft_csv), "--outdir", str(out)]
    if extra:
        cmd += extra.split()
    subprocess.run(cmd, check=True)


def run_squid(exp_csv: Path, out: Path, extra: str = "") -> None:
    out.mkdir(parents=True, exist_ok=True)
    script = ROOT / "arp_fit_new.py"
    cmd = [_py(), str(script), "--exp", str(exp_csv), "--out", str(out)]
    if extra:
        cmd += extra.split()
    subprocess.run(cmd, check=True)


def cmd_hydride(args: argparse.Namespace) -> None:
    run_hydride(args.exp_csv, args.dft_csv, args.out, args.extra)


def cmd_squid(args: argparse.Namespace) -> None:
    run_squid(args.exp_csv, args.out, args.extra)


def cmd_from_config(args: argparse.Namespace) -> None:
    conf = load_config(args.cfg)
    if args.run in ("hydride", "all") and conf.hydride:
        h = conf.hydride
        run_hydride(h.exp_csv, h.dft_csv, h.out, h.extra)
    if args.run in ("squid", "all") and conf.squid:
        s = conf.squid
        run_squid(s.exp_csv, s.out, s.extra)


def cmd_report(args: argparse.Namespace) -> None:
    md = build_markdown(args.in_dir, title=args.title)
    args.out.write_text(md, encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="rts", description="RTS: Rapid Test Suite")
    sub = parser.add_subparsers(dest="command")

    p = sub.add_parser("hydride", help="Run the hydride ARP overlay")
    p.add_argument("--exp-csv", required=True, type=Path)
    p.add_argument("--dft-csv", required=True, type=Path)
    p.add_argument("--out", type=Path, default=Path("out_hydride"))
    p.add_argument("--extra", default="")
    p.set_defaults(func=cmd_hydride)

    p = sub.add_parser("squid", help="Run the SQUID/film ARP overlay")
    p.add_argument("--exp-csv", required=True, type=Path)
    p.add_argument("--out", type=Path, default=Path("out_squid"))
    p.add_argument("--extra", default="")
    p.set_defaults(func=cmd_squid)

    p = sub.add_parser("from-config", help="Run tasks as specified in rts.toml")
    p.add_argument("--cfg", type=Path, default=Path("rts.toml"))
    p.add_argument("--run", choices=["hydride", "squid", "all"], default="all")
    p.set_defaults(func=cmd_from_config)

    p = sub.add_parser("report", help="Bundle PNG/JSON outputs into a Markdown report")
    p.add_argument("--in", dest="in_dir", required=True, type=Path)
    p.add_argument("--out", type=Path, default=Path("report.md"))
    p.add_argument("--title", default="RTS Report")
    p.set_defaults(func=cmd_report)

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if hasattr(args, "func"):
        args.func(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
