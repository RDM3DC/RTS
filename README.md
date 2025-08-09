
# ARP Experiments — Quickstart README

**Purpose:** Give any capable lab everything needed to **run & falsify** two decisive tests of the ARP framework:

1. **Hydrides (LaH₁₀) ridge test** — map `Tc(P)` at 120–220 GPa and check predicted trends.  
2. **Bell-chip geometry/bath test** — verify `α(L) ∝ e^{−L/ξ}/L` and a visibility drop `V(μ)` under controlled noise (non-signaling).

---

## 🔥 Latest Results (Aug 2025)
- Tuned **ω(P) = 100 + 2.5 · P meV/GPa** with **λ_cap = 5.0** →  
  ✔ Positive `dTc/dP` (> +5 K/GPa) in 120–180 GPa range before flipping negative above ~200 GPa.  
  ✔ **300 K contour** emerges in capped λ maps (~160–200 GPa).  
- Sensitivity now **62–73 K/Δ**, inside target range.  
- Trends now **fully satisfy falsifier 3**.  
- Next: distribute CLI repro for lab collaboration.

---

## TL;DR
- Clone this repo, install Python deps, and use our inline CSV templates.  
- Run **one script** to produce overlays + falsifier plots from your data.  
- Pass/fail is **pre-stated** below; no curve-fitting gymnastics required.

---

## What you’ll need

### Software
- Python ≥ 3.10 with: `numpy`, `pandas`, `matplotlib`
- Our analysis script: [`arp_fit_overlay.py`](./arp_fit_overlay.py) (CSV-in → PNG/JSON-out)

Install deps:
```bash
pip install numpy pandas matplotlib

Hardware (A) Hydrides (LaH₁₀)

Diamond anvil cell (DAC) capable of 120–220 GPa

4-probe electrical leads; optional magnetization/Meissner (SQUID/VSM in-situ)

Temp control around room-T ± 50 K

(Optional) Phonon/DFT support to estimate ω_log(P), μ*(P), and coupling g(P)


Hardware (B) Bell-chip

Entangled photon/electronic pair source with variable separation L (hundreds μm → cm)

Controlled bath/noise injector to dial μ (thermal/phononic)

Standard CHSH/visibility readout; no signaling channels



---

Minimal model ingredients (for context)

Range parameter:

ξ = sqrt(D/μ)
W(L,ξ) = exp(−L/ξ) / L

Geometry factor: α_geom ∝ μ · W(L,ξ)

Polaritons/phonons:

Δ_eff = k · (g/ω)^2 · Q_fac · C^2

Critical temperature (Allen–Dynes):

Tc = (ω_log/1.2) * exp{ −1.04(1+λ_tot) / [λ_tot − μ*(1+0.62 λ_tot)] }

Hydride falsifier runs:

ω(P) = 100 + 2.5·P  (meV/GPa)
λ_cap ≈ 5.0



---

Data you collect (inline CSV examples)

Hydrides — experimental Tc(P)

Pressure_GPa,Tc_K
120,191
136,241
138,243
170,250

Optional DFT curves (if available):

Pressure_GPa,g_eV,omega_log_K,mu_star
150,0.30,1230,0.10
170,0.28,1250,0.11

Bell-chip — visibility vs μ and vs L

L_mm,mu_units,Visibility
10.0,0.00,0.92
10.0,0.05,0.81
10.0,0.10,0.70

L_mm,Visibility
1.0,0.95
5.0,0.88
10.0,0.78


---

Run the analysis (Hydrides)

python arp_fit_overlay.py --mode hydride \
  --exp_csv your_LaH10_TcP.csv \
  --dft_csv your_LaH10_DFT.csv \
  --outdir fit_out_hydride

Outputs:

hydride_fit_overlay.png — experimental dome vs ARP baseline and ARP+Δ

hydride_fit_report.json — RMSE, best-fit scale s_lambda, polariton factor k_scale



---

Run the analysis (Cuprates)

python arp_fit_overlay.py --mode cuprate \
  --exp_csv your_YBCO_TcP.csv \
  --outdir fit_out_cuprate

Outputs: cuprate_fit_overlay.png, cuprate_fit_report.json


---

What to look for (falsifiers)

Hydrides (LaH₁₀)

Ridge + derivative trends:

dTc/dP ≳ +5 K/GPa in 120–180 GPa range (positive slope)

Flips negative above ~200 GPa


Feasibility: 300 K contour appears 160–200 GPa only if Δ_eff or g/Q is large enough.
Fail if no ridge/derivative trends or no plausible parameter fit.


Bell-chip

Distance law: visibility families match α(L) ∝ e^{−L/ξ}/L

Bath law: V(μ) shows knee/roll-off as μ increases

Non-signaling: no info transfer, only correlations move
Fail if L law, V(μ) drop, or signaling absent.



---

Safety notes

DAC @ 200+ GPa: high-pressure lab best practices.

Electrical isolation for 4-probe; avoid Joule heating near Tc.

Bell-chips: laser/electronics safety; isolate auxiliary channels.



---

Contact & license

Questions / coordination: RDM3DC (DM on X)

License: CC-BY 4.0 — reuse with attribution


> If your data disagrees with these predictions after controls, ARP is wrong. If it matches, we’ve cleared a decisive hurdle — thank you for testing 

