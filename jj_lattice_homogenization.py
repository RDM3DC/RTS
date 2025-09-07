# JJ lattice homogenization + sidebands (πₐ/ARP analogue)
# Bloch dispersion for a symmetrized LC ladder unit cell (series L/2 — shunt C — series L/2),
# small-ω effective medium, and time-modulated inductance → Bessel sidebands.
#
# Key results:
#   cos(k p) = (A+D)/2 for the unit-cell ABCD → k(ω) via acos
#   low-ω: k ≈ ω * sqrt(Ls*C) / p  (effective medium)
#   L(t)=L0(1+m sin Ω t) with C const. ⇒ δn/n ≈ 0.5 m
#   β ≈ (ω0 τ) (δn/n) ⇒ P±1/P0 ≈ (J1(β)/J0(β))^2
#
# Symbolic small-ω series for k:
#   k(ω) ≈ Mul(Pow(Symbol('C', positive=True, real=True), Rational(1, 2)), Pow(Symbol('Ls', positive=True, real=True), Rational(1, 2)), Pow(Symbol('p', positive=True, real=True), Integer(-1)), Symbol('w', positive=True, real=True))
#
# Effective phase velocity v_phase = ω / k(ω) (series form):
#   v_phase(ω) ≈ Mul(Pow(Symbol('C', positive=True, real=True), Rational(-1, 2)), Pow(Symbol('Ls', positive=True, real=True), Rational(-1, 2)), Symbol('p', positive=True, real=True))
