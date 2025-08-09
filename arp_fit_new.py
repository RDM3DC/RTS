#!/usr/bin/env python3
# Generates Tc(P, Δ_eff) maps + sensitivity and dTc/dP curves
# Latest settings per thread: ω(P)=100+2.5·P meV/GPa, s=1.2, λ_cap=6, μ*=0.12

import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# ---- Tunables (edit here if needed) ----
MU_STAR = 0.12          # Coulomb pseudopotential
LAMBDA0 = 2.0           # baseline λ0
S_SCALE = 1.2           # s
LAMBDA_CAP = 6.0        # cap for λ_tot in "capped" map
OMEGA0_MEV = 100.0      # ω(0) in meV
OMEGA_SLOPE_MEV_PER_GPA = 2.5  # ω slope in meV/GPa
P_MAX = 10.0            # GPa
N_P = 101               # pressure grid points
N_DELTA = 201           # Δ grid points
OUTDIR = "outputs/maps_latest"

# ---- Helpers ----
def tc_allen_dynes(lam, omegaK, mu_star=MU_STAR):
    lam = np.asarray(lam, dtype=float)
    # omegaK already in Kelvin
    denom = lam - mu_star * (1.0 + 0.62 * lam)
    tc = np.zeros_like(lam, dtype=float)
    mask = denom > 1e-12
    tc[mask] = (omegaK/1.2) * np.exp(-1.04*(1.0 + lam[mask]) / denom[mask])
    return tc

def build_grids():
    P = np.linspace(0.0, P_MAX, N_P)                 # GPa
    Delta = np.linspace(0.0, 1.0, N_DELTA)           # dimensionless
    # ω(P) = (100 + slope*P) meV → Kelvin (1 meV ≈ 11.6045 K)
    omega_logK = (OMEGA0_MEV + OMEGA_SLOPE_MEV_PER_GPA*P) * 11.6045
    return P, Delta, omega_logK

def make_maps(P, Delta, omega_logK, lambda0=LAMBDA0, s=S_SCALE, cap=LAMBDA_CAP):
    Tc_unc = np.zeros((P.size, Delta.size))
    Tc_cap = np.zeros((P.size, Delta.size))
    for i, p in enumerate(P):
        lam_unc = s*lambda0*(1.0 + Delta)       # Δ_eff scanned directly 0..1
        lam_cap = np.minimum(cap, lam_unc)      # capped version
        Tc_unc[i,:] = tc_allen_dynes(lam_unc, omega_logK[i])
        Tc_cap[i,:] = tc_allen_dynes(lam_cap, omega_logK[i])
    return Tc_unc, Tc_cap

def main():
    os.makedirs(OUTDIR, exist_ok=True)
    P, Delta, omega_logK = build_grids()
    Tc_unc, Tc_cap = make_maps(P, Delta, omega_logK)

    # Heatmaps (capped/uncapped)
    for tag, M in [("capped", Tc_cap), ("uncapped", Tc_unc)]:
        plt.figure(figsize=(8,6))
        levels = np.linspace(220, 440, 30)
        cp = plt.contourf(Delta, P, M, levels=levels)
        plt.colorbar(cp, label="Tc (K)")
        try:
            CS = plt.contour(Delta, P, M, levels=[300], colors="cyan", linewidths=1.5)
            if CS.collections:
                plt.clabel(CS, inline=1, fmt=lambda v: "300 K")
        except Exception:
            pass
        plt.xlabel("Δ_eff (0–1)")
        plt.ylabel("Pressure (GPa)")
        plt.title(f"Tc(P, Δ_eff) — {tag}, s={S_SCALE}, λ0={LAMBDA0:.2f}, ω(P)=(100+{OMEGA_SLOPE_MEV_PER_GPA}·P) meV")
        plt.tight_layout()
        plt.savefig(os.path.join(OUTDIR, f"Tc_map_{tag}.png"), dpi=300)
        plt.close()

    # Sensitivity ∂Tc/∂Δ at Δ=0.5 (capped)
    j = np.argmin(np.abs(Delta - 0.5))
    dTc_dDelta_cap = np.gradient(Tc_cap, Delta, axis=1)[:, j]
    plt.figure(figsize=(7,4))
    plt.plot(P, dTc_dDelta_cap, label="∂Tc/∂Δ @ Δ=0.5 (capped)")
    plt.axhspan(70, 120, color="gray", alpha=0.2, label="target 70–120 K/Δ")
    plt.xlabel("Pressure (GPa)"); plt.ylabel("K per Δ unit")
    plt.title(f"Sensitivity vs P (capped), s={S_SCALE}, cap={LAMBDA_CAP}")
    plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, "dTc_dDelta_vsP.png"), dpi=300)
    plt.close()

    # dTc/dP @ Δ=0.5 (capped & uncapped) — falsifier #3
    dTc_dP_cap = np.gradient(Tc_cap[:, j], P)
    dTc_dP_unc = np.gradient(Tc_unc[:, j], P)
    plt.figure(figsize=(7,4))
    plt.plot(P, dTc_dP_cap, label="capped λ_tot")
    plt.plot(P, dTc_dP_unc, label="uncapped")
    plt.axhline(0, color="k", linewidth=1)
    plt.xlabel("Pressure (GPa)"); plt.ylabel("dTc/dP @ Δ=0.5 (K/GPa)")
    plt.title(f"Pressure slope at fixed Δ — s={S_SCALE}, cap={LAMBDA_CAP}")
    plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, "dTc_dP_at_Delta0p5.png"), dpi=300)
    plt.close()

    # Small table @ Δ=0.5 for P = 0,2,4,6,8
    P_tab = np.array([0,2,4,6,8], dtype=float)
    iP = [np.argmin(np.abs(P - v)) for v in P_tab]
    rows = []
    for idx, p in zip(iP, P_tab):
        rows.append((p, float(Tc_unc[idx,j]), float(Tc_cap[idx,j]), float(dTc_dP_cap[idx])))
    df = pd.DataFrame(rows, columns=["P_GPa","Tc_uncapped_K","Tc_capped_K","dTc_dP_cap_K_per_GPa"])
    fig = plt.figure(figsize=(6.5,2.0))
    plt.axis('off')
    tbl = plt.table(cellText=df.values, colLabels=df.columns, loc='center', cellLoc='center')
    tbl.scale(1,1.2)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, "table_Tc_and_slope_at_Delta0p5.png"), dpi=300)
    plt.close()

    # Quick stats to quote on X
    stats = {
        "mu_star": MU_STAR,
        "s": S_SCALE,
        "lambda0": LAMBDA0,
        "lambda_cap": LAMBDA_CAP,
        "omega0_meV": OMEGA0_MEV,
        "omega_slope_meV_per_GPa": OMEGA_SLOPE_MEV_PER_GPA,
        "dTc_dDelta_mean": float(np.mean(dTc_dDelta_cap)),
        "dTc_dDelta_min": float(np.min(dTc_dDelta_cap)),
        "dTc_dDelta_max": float(np.max(dTc_dDelta_cap)),
        "dTc_dP_cap_mean": float(np.mean(dTc_dP_cap)),
    }
    with open(os.path.join(OUTDIR, "sensitivity_stats.json"), "w") as f:
        import json; json.dump(stats, f, indent=2)
    print("Done. Wrote PNGs and sensitivity_stats.json in:", OUTDIR)

if __name__ == "__main__":
    main()