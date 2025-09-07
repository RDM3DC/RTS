#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, json, os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

K_PER_EV = 11604.5  # Kelvin per eV

# -----------------------------
# Core formulas / utilities
# -----------------------------
def Tc_allen_dynes(lambda_eff, omega_logK, mu_star):
    lam = np.asarray(lambda_eff, dtype=float)
    omega = np.asarray(omega_logK, dtype=float)
    if omega.ndim == 0:
        omega = np.full_like(lam, float(omega))
    elif omega.shape != lam.shape:
        omega = np.full_like(lam, float(np.mean(omega)))
    denom = lam - mu_star*(1 + 0.62*lam)
    Tc = np.zeros_like(lam)
    mask = denom > 1e-12
    Tc[mask] = (omega[mask]/1.2) * np.exp(-1.04*(1 + lam[mask]) / denom[mask])
    return Tc

def interp_to(x_src, y_src, x_tgt, fill=0.0):
    x_src = np.asarray(x_src, dtype=float)
    y_src = np.asarray(y_src, dtype=float)
    x_tgt = np.asarray(x_tgt, dtype=float)
    if x_src.size == 0:
        return np.full_like(x_tgt, fill)
    return np.interp(x_tgt, x_src, y_src, left=fill, right=fill)

# -----------------------------
# Hydride pipeline (LaH10 etc.)
# -----------------------------
def lambda_geom_hydride(P, L0, beta, mu0, eta, D=1.0):
    """
    Simple geometry/bath proxy:
      L(P) = L0 / (1 + beta*P)
      mu(P) = mu0 + eta*P
      xi = sqrt(D / mu),  W = exp(-L/xi)/L,  lambda_geom ∝ mu * W
    """
    L = L0 / (1 + beta*np.asarray(P))
    mu = mu0 + eta*np.asarray(P)
    xi = np.sqrt(D / np.maximum(mu, 1e-12))
    W = np.exp(-L/np.maximum(xi,1e-12)) / np.maximum(L, 1e-12)
    return mu * W

def fit_hydride(exp_df, dft_df, outdir, params):
    """
    Fits experimental Tc(P) with:
      λ_base = s_lambda * λ_geom(P)
      Δ_eff ∝ (g/ω)^2 * k_scale
      λ_tot = λ_base * (1 + k_scale * (g/ω)^2)
      Tc = Allen–Dynes(λ_tot, ω_log, μ*)
    """
    os.makedirs(outdir, exist_ok=True)
    P_exp = exp_df["Pressure_GPa"].to_numpy(float)
    Tc_exp = exp_df["Tc_K"].to_numpy(float)

    if dft_df is None or dft_df.empty:
        P_dft = P_exp
        g_eV = np.full_like(P_exp, params.get("g_const_eV", 0.3), dtype=float)
        omega_logK = np.full_like(P_exp, params.get("omega_const_K", 1200.0), dtype=float)
        mu_star = np.full_like(P_exp, params.get("mu_star", 0.10), dtype=float)
    else:
        # Accept either Kelvin or meV columns
        P_dft = dft_df.get("Pressure_GPa", dft_df.get("P_GPa")).to_numpy(float)
        g_eV = dft_df.get("g_eV", pd.Series(np.full_like(P_dft, params.get("g_const_eV", 0.3), dtype=float))).to_numpy(float)
        if "omega_log_K" in dft_df.columns:
            omega_logK = dft_df["omega_log_K"].to_numpy(float)
        elif "omega_log_meV" in dft_df.columns:
            omega_logK = (dft_df["omega_log_meV"].to_numpy(float) * K_PER_EV / 1000.0)
        else:
            omega_logK = np.full_like(P_dft, params.get("omega_const_K", 1200.0), dtype=float)
        mu_star = dft_df.get("mu_star", pd.Series(np.full_like(P_dft, params.get("mu_star", 0.10), dtype=float))).to_numpy(float)

    # Interp DFT → exp P grid
    g_eV = interp_to(P_dft, g_eV, P_exp, fill=g_eV[0] if g_eV.size>0 else 0.3)
    omega_logK = interp_to(P_dft, omega_logK, P_exp, fill=omega_logK[0] if omega_logK.size>0 else 1200.0)
    mu_star = interp_to(P_dft, mu_star, P_exp, fill=params.get("mu_star", 0.10))

    lam_geom = lambda_geom_hydride(P_exp,
                                   params.get("L0",1.5),
                                   params.get("beta",0.01),
                                   params.get("mu0",1.0),
                                   params.get("eta",0.02),
                                   params.get("D",1.0))

    # Coarse grid over scaling knobs
    s_grid = np.linspace(0.5, 3.0, 51)
    k_grid = np.linspace(0.0, 3.0, 61)
    best = {"rmse": 1e9}
    omega_eV = omega_logK / K_PER_EV
    r2 = (g_eV / np.maximum(omega_eV, 1e-9))**2  # (g/ω)^2

    for s in s_grid:
        lam_base = s * lam_geom
        Tc_base = Tc_allen_dynes(lam_base, omega_logK, mu_star)
        for k in k_grid:
            lam_eff = lam_base * (1 + k * r2)
            Tc_model = Tc_allen_dynes(lam_eff, omega_logK, mu_star)
            rmse = float(np.sqrt(np.mean((Tc_model - Tc_exp)**2)))
            if rmse < best["rmse"]:
                best = {
                    "rmse": rmse, "s_lambda": float(s), "k_scale": float(k),
                    "P": P_exp, "Tc_exp": Tc_exp, "Tc_base": Tc_base, "Tc_model": Tc_model
                }

    # Plot overlay
    plt.figure(figsize=(8,5))
    plt.plot(best["P"], best["Tc_exp"], "o", label="Experiment")
    plt.plot(best["P"], best["Tc_base"], label=f"ARP baseline (sλ={best['s_lambda']:.2f})")
    plt.plot(best["P"], best["Tc_model"], label=f"ARP + Δ (k={best['k_scale']:.2f})")
    plt.xlabel("Pressure (GPa)"); plt.ylabel("Tc (K)")
    plt.title("Hydride: Tc(P) — Experiment vs ARP fit")
    plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(outdir, "hydride_fit_overlay.png"), dpi=180)
    plt.close()

    with open(os.path.join(outdir, "hydride_fit_report.json"), "w") as f:
        json.dump({k:v for k,v in best.items() if k not in ["P","Tc_exp","Tc_base","Tc_model"]}, f, indent=2)

# -----------------------------
# Cuprate quick proxy (Presland)
# -----------------------------
def presland_Tc(p, Tcmax=93.0, popt=0.16):
    return Tcmax * np.maximum(0.0, 1.0 - 82.6*(p - popt)**2)

def fit_cuprate(exp_df, outdir, params):
    os.makedirs(outdir, exist_ok=True)
    p = exp_df["Doping_p"].to_numpy(float)
    Tc_exp = exp_df["Tc_K"].to_numpy(float)

    S_grid = np.linspace(60, 120, 61)
    dp_grid = np.linspace(-0.02, 0.02, 41)
    best = {"rmse": 1e9}

    for S in S_grid:
        for dp in dp_grid:
            Tc_model = presland_Tc(p, Tcmax=S, popt=0.16+dp)
            rmse = float(np.sqrt(np.mean((Tc_model - Tc_exp)**2)))
            if rmse < best["rmse"]:
                best = {"rmse": rmse, "Tcmax": float(S), "dp": float(dp), "Tc_model": Tc_model}

    plt.figure(figsize=(8,5))
    plt.plot(p, Tc_exp, "o", label="Experiment")
    plt.plot(p, best["Tc_model"], label=f"ARP ridge proxy (Tcmax={best['Tcmax']:.1f} K, Δp={best['dp']:+.3f})")
    plt.xlabel("Hole doping p"); plt.ylabel("Tc (K)")
    plt.title("Cuprate: Tc(p) — Experiment vs proxy fit")
    plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(outdir, "cuprate_fit_overlay.png"), dpi=180)
    plt.close()

    with open(os.path.join(outdir, "cuprate_fit_report.json"), "w") as f:
        json.dump({k:v for k,v in best.items() if k!="Tc_model"}, f, indent=2)

# -----------------------------
# (Optional) Organics adapter
# -----------------------------
def fit_organics(exp_df, dft_df, outdir, params):
    """
    For convenience, reuse hydride pipeline (Pressure_GPa & omega_log* columns).
    """
    # Rename columns to match hydride expectations if needed
    exp = exp_df.copy()
    if "P_GPa" in exp.columns and "Pressure_GPa" not in exp.columns:
        exp = exp.rename(columns={"P_GPa":"Pressure_GPa"})
    dft = None
    if dft_df is not None:
        dft = dft_df.copy()
        if "P_GPa" in dft.columns and "Pressure_GPa" not in dft.columns:
            dft = dft.rename(columns={"P_GPa":"Pressure_GPa"})
    fit_hydride(exp, dft, outdir, params)

# -----------------------------
# NEW: Maps exporter (Tc vs P, Δ) + sensitivity & slope
# -----------------------------
def export_tc_maps(args):
    """
    Generates:
      - Tc_map_capped.png / Tc_map_uncapped.png
      - dTc_dDelta_vsP.png  (Δ=0.5)
      - dTc_dP_at_Delta0p5.png (capped vs uncapped)
      - sensitivity_stats.json
    Parameters (CLI):
      lambda0, lambda_cap, omega0_K, omega_slope_K_per_GPa, pmax_GPa, nP, nDelta, s, mu_star
    """
    os.makedirs(args.outdir, exist_ok=True)

    mu_star = float(args.mu_star)
    lambda_cap = float(args.lambda_cap)
    s = float(getattr(args, "s", 1.0))
    lambda0 = float(getattr(args, "lambda0", 2.0))

    P = np.linspace(0.0, float(args.pmax_GPa), int(args.nP))
    Delta = np.linspace(0.0, 1.0, int(args.nDelta))
    # ω_log(P) in Kelvin; e.g., 100 meV baseline with slope in meV/GPa
    omega_logK = float(args.omega0_K) + float(args.omega_slope_K_per_GPa)*P

    def Tc_line(lam_line, omegaK):
        return Tc_allen_dynes(lam_line, omegaK, mu_star)

    Tc_unc = np.zeros((P.size, Delta.size))
    Tc_cap = np.zeros_like(Tc_unc)

    for i, p in enumerate(P):
        lam_unc = s*lambda0*(1.0 + Delta)          # treat Δ_eff ∈ [0,1] directly
        lam_cap = np.minimum(lambda_cap, lam_unc)   # capped λ_tot
        Tc_unc[i,:] = Tc_line(lam_unc, omega_logK[i])
        Tc_cap[i,:] = Tc_line(lam_cap, omega_logK[i])

    # Heatmaps
    for tag, M in [("capped", Tc_cap), ("uncapped", Tc_unc)]:
        plt.figure(figsize=(8,6))
        levels = np.linspace(220, 420, 30)
        cp = plt.contourf(Delta, P, M, levels=levels)
        plt.colorbar(cp, label="Tc (K)")
        try:
            CS = plt.contour(Delta, P, M, levels=[300], colors="cyan", linewidths=1.5)
            if CS.collections:
                plt.clabel(CS, inline=1, fmt=lambda v: "300 K")
        except Exception:
            pass
        plt.xlabel("Δ_eff (0–1)"); plt.ylabel("Pressure (GPa)")
        plt.title(f"Tc(P, Δ_eff) — {tag}, s={s}, λ0={lambda0:.2f}, ω(P) baseline in K")
        plt.tight_layout()
        plt.savefig(os.path.join(args.outdir, f"Tc_map_{tag}.png"), dpi=300)
        plt.close()

    # Sensitivity ∂Tc/∂Δ @ Δ=0.5 (capped)
    j = np.argmin(np.abs(Delta - 0.5))
    dTc_dDelta_cap = np.gradient(Tc_cap, Delta, axis=1)[:, j]
    plt.figure(figsize=(7,4))
    plt.plot(P, dTc_dDelta_cap, label="∂Tc/∂Δ @ Δ=0.5")
    plt.axhspan(70,120, color="gray", alpha=0.2, label="target 70–120 K/Δ")
    plt.xlabel("Pressure (GPa)"); plt.ylabel("K per Δ unit")
    plt.title("Sensitivity vs P (capped)")
    plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, "dTc_dDelta_vsP.png"), dpi=300)
    plt.close()

    # Pressure slope dTc/dP at Δ=0.5 (capped & uncapped) — falsifier #3
    dTc_dP_cap = np.gradient(Tc_cap[:, j], P)
    dTc_dP_unc = np.gradient(Tc_unc[:, j], P)
    plt.figure(figsize=(7,4))
    plt.plot(P, dTc_dP_cap, label="capped λ_tot")
    plt.plot(P, dTc_dP_unc, label="uncapped")
    plt.axhline(0, color="k", linewidth=1)
    plt.xlabel("Pressure (GPa)"); plt.ylabel("dTc/dP @ Δ=0.5 (K/GPa)")
    plt.title("Pressure slope at fixed Δ (falsifier #3)")
    plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, "dTc_dP_at_Delta0p5.png"), dpi=300)
    plt.close()

    # Save quick stats (matches what you posted)
    stats = {
        "s": s,
        "lambda0": lambda0,
        "omega0_K": float(args.omega0_K),
        "omega_slope_K_per_GPa": float(args.omega_slope_K_per_GPa),
        "mu_star": mu_star,
        "mean": float(np.mean(dTc_dDelta_cap)),
        "median": float(np.median(dTc_dDelta_cap)),
        "min": float(np.min(dTc_dDelta_cap)),
        "max": float(np.max(dTc_dDelta_cap)),
        "fraction_in_70_120_band": float(np.mean((dTc_dDelta_cap>=70)&(dTc_dDelta_cap<=120)))
    }
    with open(os.path.join(args.outdir, "sensitivity_stats.json"), "w") as f:
        json.dump(stats, f, indent=2)

# -----------------------------
# CLI
# -----------------------------
def main():
    ap = argparse.ArgumentParser(description="ARP overlays & maps")
    ap.add_argument("--mode", choices=["hydride","cuprate","organics","maps"], required=True)
    ap.add_argument("--exp_csv", default=None, help="Experimental CSV")
    ap.add_argument("--dft_csv", default=None, help="DFT CSV (optional)")
    ap.add_argument("--outdir", required=True)

    # Hydride/organics geometry params
    ap.add_argument("--L0", type=float, default=1.5)
    ap.add_argument("--beta", type=float, default=0.01)
    ap.add_argument("--mu0", type=float, default=1.0)
    ap.add_argument("--eta", type=float, default=0.02)
    ap.add_argument("--D", type=float, default=1.0)

    # DFT defaults (when missing)
    ap.add_argument("--g_const_eV", type=float, default=0.3)
    ap.add_argument("--omega_const_K", type=float, default=1200.0)
    ap.add_argument("--mu_star", type=float, default=0.10)

    # MAPS-only knobs
    ap.add_argument("--lambda0", type=float, default=2.0, help="Baseline λ0 in maps")
    ap.add_argument("--lambda_cap", type=float, default=5.0, help="Cap for λ_tot in capped map")
    ap.add_argument("--omega0_K", type=float, default=1160.0, help="ω_log at P=0 [K]")
    ap.add_argument("--omega_slope_K_per_GPa", type=float, default=12.0, help="ω_log slope [K/GPa]")
    ap.add_argument("--pmax_GPa", type=float, default=10.0, help="Max pressure for map")
    ap.add_argument("--nP", type=int, default=101, help="Grid: # of P points")
    ap.add_argument("--nDelta", type=int, default=201, help="Grid: # of Δ points")
    ap.add_argument("--s", type=float, default=1.0, help="Scale for λ in maps mode")

    args = ap.parse_args()

    # Load CSVs if present (non-maps modes)
    exp_df = pd.read_csv(args.exp_csv) if (args.exp_csv and os.path.exists(args.exp_csv)) else None
    dft_df = pd.read_csv(args.dft_csv) if (args.dft_csv and os.path.exists(args.dft_csv)) else None

    if args.mode == "hydride":
        if exp_df is None:
            raise SystemExit("ERROR: --exp_csv is required for --mode hydride")
        fit_hydride(exp_df, dft_df, args.outdir, vars(args))
    elif args.mode == "organics":
        if exp_df is None:
            raise SystemExit("ERROR: --exp_csv is required for --mode organics")
        fit_organics(exp_df, dft_df, args.outdir, vars(args))
    elif args.mode == "cuprate":
        if exp_df is None:
            raise SystemExit("ERROR: --exp_csv is required for --mode cuprate")
        fit_cuprate(exp_df, args.outdir, vars(args))
    else:  # maps
        export_tc_maps(args)

if __name__ == "__main__":
    main()
