"""Thermal-dose coverage metrics for the transient tumor model (same discretization and parameters as transient.py).
Writes dose_coverage.json."""
import numpy as np, json, sys, os
from scipy.sparse import diags
from scipy.sparse.linalg import splu, spsolve
Rt = 10e-3; kt = 0.5; rc = 4.0e6; RHOC = 3.6e6; WB = 0.5e-3; ROUT, NR = 60e-3, 600
r = (np.arange(NR) + 0.5) * ROUT / NR; dr = ROUT / NR; rf = np.arange(NR + 1) * dr
vol = (rf[1:]**3 - rf[:-1]**3) / 3; inside = r <= Rt; iB = np.where(inside)[0][-1]
i5 = np.argmin(abs(r - (Rt + 5e-3)))
def operator(wb):
    G = kt * rf[1:-1]**2 / dr
    main = np.zeros(NR); main[:-1] += G; main[1:] += G
    main[-1] += kt * rf[-1]**2 / (dr / 2); main += rc * wb * vol
    return diags([-G, main, -G], [-1, 0, 1], format='csc')
def steady(Q, wb=WB): return spsolve(operator(wb), Q * vol)
def transient(Q, wb=WB, t_heat=3600.0, dt=2.0):
    Kop = operator(wb); Cm = diags(RHOC * vol / dt)
    lu = splu((Cm + 0.5 * Kop).tocsc()); Bm = Cm - 0.5 * Kop
    th = np.zeros(NR); out = [th.copy()]; b = Q * vol
    for k in range(int(t_heat / dt)):
        th = lu.solve(Bm @ th + b); out.append(th.copy())
    return dt, np.array(out)
def cem43(T, dt):
    return np.sum(np.where(T >= 43.0, 0.5, 0.25)**(43.0 - T), axis=0) * dt / 60.0
def vfrac(mask): return float(np.sum(vol[inside & mask]) / np.sum(vol[inside]))
def pct_dose(cem, p):
    """CEM43 reached by at least p of the tumor volume (e.g., p = 0.9 -> CEM43_T90)."""
    ci = cem[inside]; vi = vol[inside]; o = np.argsort(ci)[::-1]; cum = np.cumsum(vi[o]) / vi.sum()
    return float(ci[o][np.searchsorted(cum, p)])
Qu = np.where(inside, 1.0, 0.0); fd = steady(Qu)
out = {}
for target in (42.0, 43.0):
    Q = Qu * (target - 37.0) / fd[iB]
    dt, th = transient(Q); T = 37 + th
    row = {}
    for tm in (30, 60):
        n = int(tm * 60 / dt); c = cem43(T[:n + 1], dt)
        row['%d' % tm] = dict(edge=round(float(c[iB]), 1), center=round(float(c[0]), 0),
                              cem_T90=round(pct_dose(c, 0.9), 1), cem_T50=round(pct_dose(c, 0.5), 0),
                              f10=round(100 * vfrac(c >= 10)), f30=round(100 * vfrac(c >= 30)),
                              f60=round(100 * vfrac(c >= 60)), f240=round(100 * vfrac(c >= 240)),
                              out5_T=round(float(T[n, i5]), 1), out5_cem=round(float(c[i5]), 2))
    row['center_ss'] = round(float(37 + steady(Q)[0]), 1)
    out['edge%.0f' % target] = row
json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dose_coverage.json'), 'w'), indent=1)
print(json.dumps(out, indent=1))
