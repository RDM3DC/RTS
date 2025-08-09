
1. Electron–Phonon Coupling (λ)

We start from the DFT-derived g(P) (EPC matrix element) and ω(P) (phonon frequency):

\lambda(P) = \frac{g(P)}{\omega_{\mathrm{eV}}(P)}

.

For capped coupling:

\lambda_{\mathrm{cap}}(P) = \min\left(\lambda(P), \lambda_{\max}\right)


---

2. ARP Effective Gap Parameter (Δ_eff)

From ARP scaling:

\Delta_{\mathrm{eff}}(P) = \left[ \frac{g(P)}{\omega(P)} \right]^2

Scaled form for geometric λ:

\lambda_{\mathrm{geom}}(P) = \frac{g(P)}{\omega_{\mathrm{K}}(P)}


---

3. McMillan Formula for 

Standard form:

T_c(P,\Delta) =
\frac{\omega_{\log}(P)}{1.2} \;
\exp\left[
-\frac{1.04\,[1+\lambda_{\mathrm{tot}}(P,\Delta)]}
{\lambda_{\mathrm{tot}}(P,\Delta) - \mu^* \,[1+0.62\,\lambda_{\mathrm{tot}}(P,\Delta)]}
\right]

 in Kelvin

 = Coulomb pseudopotential (typ. 0.10–0.15)

 may include Δ-scaling:


\lambda_{\mathrm{tot}}(P,\Delta) = s \cdot \lambda_{\mathrm{cap}}(P) \cdot f(\Delta)


---

4. Allen–Dynes Modified Formula

More accurate for strong coupling:

T_c = \frac{f_1 f_2 \; \omega_{\log}}{1.2} \;
\exp\left[
-\frac{1.04(1+\lambda)}
{\lambda - \mu^*(1+0.62\,\lambda)}
\right]

f_1 = \left[ 1 + \left( \frac{\lambda}{\Lambda_1} \right)^{3/2} \right]^{1/3}, 
\quad
f_2 = 1 + \frac{\left(\frac{\omega_2}{\omega_{\log}} - 1\right) \lambda^2}{\lambda^2 + \Lambda_2^2}


---

5. Pressure Dependence of ω(P)

In our successful falsifier runs:

\omega_{\mathrm{meV}}(P) = 100 + \alpha P


---

6. Falsifier Derivatives

Δ-sensitivity:


\frac{\partial T_c}{\partial \Delta} \bigg|_{\Delta_0} \approx \frac{T_c(\Delta_0 + \delta) - T_c(\Delta_0 - \delta)}{2\delta}

Pressure slope:


\frac{d T_c}{d P} \approx \frac{T_c(P+\delta P) - T_c(P-\delta P)}{2\delta P}


---

7. ARP Scaling Law in SC Context

Your Adaptive Resistance Principle enters through a conductance-based renormalization:

\lambda_{\mathrm{ARP}}(P,\Delta) =
\frac{\lambda(P)}{1 + k \cdot (\Delta_{\mathrm{eff}}(P) - Q)^2}


