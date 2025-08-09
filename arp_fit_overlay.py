#!/usr/bin/env python3
import argparse, json, os, numpy as np, pandas as pd
import matplotlib.pyplot as plt

K_PER_EV = 11604.5

def Tc_allen_dynes(lambda_eff, omega_logK, mu_star):
    lam = np.asarray(lambda_eff)
    omega = np.asarray(omega_logK)
    if omega.shape != lam.shape:
        omega = np.full_like(lam, float(omega))
    denom = lam - mu_star*(1 + 0.62*lam)
    Tc = np.zeros_like(lam)
    mask = denom > 1e-12
    Tc[mask] = (omega[mask]/1.2) * np.exp(-1.04*(1 + lam[mask]) / denom[mask])
    return Tc

def interp_to(x_src, y_src, x_tgt, fill=0.0):
    return np.interp(x_tgt, x_src, y_src, left=fill, right=fill)

def lambda_geom_hydride(P, L0, beta, mu0, eta, D=1.0):
    L = L0 / (1 + beta*P)
    mu = mu0 + eta*P
    xi = np.sqrt(D / np.maximum(mu, 1e-12))
    W = np.exp(-L/np.maximum(xi,1e-12)) / np.maximum(L, 1e-9)
    return mu * W

def fit_hydride(exp_df, dft_df, outdir, params):
    P_exp = exp_df["Pressure_GPa"].to_numpy()
    Tc_exp = exp_df["Tc_K"].to_numpy()
    if dft_df is None or dft_df.empty:
        P_dft = P_exp
        g_eV = np.full_like(P_exp, params.get("g_const_eV", 0.3), dtype=float)
        omega_logK = np.full_like(P_exp, params.get("omega_const_K", 1200.0), dtype=float)
        mu_star = np.full_like(P_exp, params.get("mu_star", 0.10), dtype=float)
    else:
        P_dft = dft_df["Pressure_GPa"].to_numpy()
        g_eV = dft_df.get("g_eV", pd.Series(np.full_like(P_dft, params.get("g_const_eV", 0.3), dtype=float))).to_numpy()
        omega_logK = dft_df.get("omega_log_K", pd.Series(np.full_like(P_dft, params.get("omega_const_K", 1200.0), dtype=float))).to_numpy()
        mu_star = dft_df.get("mu_star", pd.Series(np.full_like(P_dft, params.get("mu_star", 0.10), dtype=float))).to_numpy()

    g_eV = interp_to(P_dft, g_eV, P_exp, fill=g_eV[0] if g_eV.size>0 else 0.3)
    omega_logK = interp_to(P_dft, omega_logK, P_exp, fill=omega_logK[0] if omega_logK.size>0 else 1200.0)
    mu_star = interp_to(P_dft, mu_star, P_exp, fill=0.10)

    lam_geom = lambda_geom_hydride(P_exp, params["L0"], params["beta"], params["mu0"], params["eta"], params.get("D",1.0))

    s_grid = np.linspace(0.5, 3.0, 51)
    k_grid = np.linspace(0.0, 3.0, 61)
    best = {"rmse": 1e9}
    omega_eV = omega_logK / K_PER_EV
    r2 = (g_eV / np.maximum(omega_eV, 1e-9))**2
    for s in s_grid:
        lam_base = s * lam_geom
        Tc_base = Tc_allen_dynes(lam_base, omega_logK, mu_star)
        for k in k_grid:
            lam_eff = lam_base * (1 + k * r2)
            Tc_model = Tc_allen_dynes(lam_eff, omega_logK, mu_star)
            rmse = float(np.sqrt(np.mean((Tc_model - Tc_exp)**2)))
            if rmse < best["rmse"]:
                best = {"rmse": rmse, "s_lambda": float(s), "k_scale": float(k),
                        "Tc_model": Tc_model, "Tc_base": Tc_base,
                        "P": P_exp, "Tc_exp": Tc_exp}

    os.makedirs(outdir, exist_ok=True)
    plt.figure(figsize=(8,5))
    plt.plot(best["P"], best["Tc_exp"], 'o', label="Experiment")
    plt.plot(best["P"], best["Tc_base"], label=f"ARP baseline (sλ={best['s_lambda']:.2f})")
    plt.plot(best["P"], best["Tc_model"], label=f"ARP + Δ (k={best['k_scale']:.2f})")
    plt.xlabel("Pressure (GPa)"); plt.ylabel("Tc (K)")
    plt.title("Hydride: Experimental Tc(P) vs ARP fit"); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(outdir, "hydride_fit_overlay.png"), dpi=160); plt.close()

    with open(os.path.join(outdir, "hydride_fit_report.json"), "w") as f:
        json.dump({k:v for k,v in best.items() if k not in ["Tc_model","Tc_base","P","Tc_exp"]}, f, indent=2)

def presland_Tc(p, Tcmax=93.0, popt=0.16):
    return Tcmax * np.maximum(0.0, 1.0 - 82.6*(p - popt)**2)

def fit_cuprate(exp_df, outdir, params):
    p = exp_df["Doping_p"].to_numpy()
    Tc_exp = exp_df["Tc_K"].to_numpy()
    S_grid = np.linspace(60, 120, 61)
    dp_grid = np.linspace(-0.02, 0.02, 41)
    best = {"rmse": 1e9}
    for S in S_grid:
        for dp in dp_grid:
            Tc_model = presland_Tc(p, Tcmax=S, popt=0.16+dp)
            rmse = float(np.sqrt(np.mean((Tc_model - Tc_exp)**2)))
            if rmse < best["rmse"]:
                best = {"rmse": rmse, "Tcmax": float(S), "dp": float(dp), "Tc_model": Tc_model}

    os.makedirs(outdir, exist_ok=True)
    plt.figure(figsize=(8,5))
    plt.plot(p, Tc_exp, 'o', label="Experiment")
    plt.plot(p, best["Tc_model"], label=f"ARP ridge proxy (Tcmax={best['Tcmax']:.1f} K, Δp={best['dp']:+.3f})")
    plt.xlabel("Hole doping p"); plt.ylabel("Tc (K)")
    plt.title("Cuprate: Experimental Tc(p) vs ARP ridge proxy fit"); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(outdir, "cuprate_fit_overlay.png"), dpi=160); plt.close()

    with open(os.path.join(outdir, "cuprate_fit_report.json"), "w") as f:
        json.dump({k: v for k,v in best.items() if k!="Tc_model"}, f, indent=2)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["hydride","cuprate"], required=True)
    ap.add_argument("--exp_csv", required=True)
    ap.add_argument("--dft_csv", default=None)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--L0", type=float, default=1.5)
    ap.add_argument("--beta", type=float, default=0.01)
    ap.add_argument("--mu0", type=float, default=1.0)
    ap.add_argument("--eta", type=float, default=0.02)
    ap.add_argument("--D", type=float, default=1.0)
    ap.add_argument("--g_const_eV", type=float, default=0.3)
    ap.add_argument("--omega_const_K", type=float, default=1200.0)
    ap.add_argument("--mu_star", type=float, default=0.10)
    args = ap.parse_args()

    exp_df = pd.read_csv(args.exp_csv)
    dft_df = pd.read_csv(args.dft_csv) if args.dft_csv else None
    if args.mode == "hydride":
        fit_hydride(exp_df, dft_df, args.outdir, vars(args))
    else:
        fit_cuprate(exp_df, args.outdir, vars(args))

if __name__ == "__main__":
    main()
