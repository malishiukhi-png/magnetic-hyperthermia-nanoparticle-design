"""Nonlinear Neel dynamics of immobilized particles: thermally activated two-state (Stoner-Wohlfarth) model.
Compares with linear response theory (model.py) -> nonlinear.json. Run: python nonlinear.py  (about 1 min)"""
import numpy as np, json, sys
sys.path.insert(0, '.')
from model import *

# ---------------------------------------------------------------------------
# Two-state (thermally activated Stoner-Wohlfarth) rate model, immobilized,
# random easy-axis orientation.

def _sw_tables(thetas, hgrid, nphi=4096):
    phi = np.linspace(-np.pi, np.pi, nphi, endpoint=False)
    nt, nh = len(thetas), len(hgrid)
    eA = np.empty((nt, nh)); eB = np.empty((nt, nh)); eS = np.empty((nt, nh))
    cA = np.empty((nt, nh)); cB = np.empty((nt, nh))
    for i, th in enumerate(thetas):
        E = np.sin(phi)[None, :]**2 - 2 * hgrid[:, None] * np.cos(phi[None, :] - th)
        left = np.roll(E, 1, axis=1); right = np.roll(E, -1, axis=1)
        ismin = (E < left) & (E <= right)
        ismax = (E > left) & (E >= right)
        for j in range(nh):
            mins = np.where(ismin[j])[0]; maxs = np.where(ismax[j])[0]
            if len(mins) != 2 or len(maxs) != 2:
                raise RuntimeError(f"expected 2 minima at theta={th}, h={hgrid[j]}")
            a = mins[np.cos(phi[mins]) > 0][0]; b = mins[np.cos(phi[mins]) < 0][0]
            eA[i, j] = E[j, a]; eB[i, j] = E[j, b]
            eS[i, j] = E[j, maxs].min()
            cA[i, j] = np.cos(phi[a] - th); cB[i, j] = np.cos(phi[b] - th)
    return eA, eB, eS, cA, cB


class TwoState:
    def __init__(self, hmax=0.45, nh=1801, ntheta=24, K=P['K'], Ms=P['Ms']):
        u, wu = np.polynomial.legendre.leggauss(ntheta)
        u = 0.5 * (u + 1); wu = 0.5 * wu          # cos(theta) in (0,1)
        self.thetas = np.arccos(u); self.w = wu
        self.h = np.linspace(-hmax, hmax, nh)
        self.K = K; self.Ms = Ms
        self.HK = 2 * K / (mu0 * Ms)
        self.tab = _sw_tables(self.thetas, self.h)

    def sar(self, d, H, f, T=T, tau0=P['tau0'], rho=P['rho'], nstep=4000, ncyc=6):
        """Return SAR (W/kg) for monodisperse diameter d (scalar)."""
        eA, eB, eS, cA, cB = self.tab
        V = np.pi * d**3 / 6
        sig = self.K * V / (kB * T)
        h0 = H / self.HK
        if h0 > self.h[-1]:
            raise ValueError("field above table range")
        t = np.arange(nstep * ncyc + 1) / (nstep * f)
        hh = h0 * np.cos(2 * np.pi * f * t)
        idx = np.interp(hh, self.h, np.arange(len(self.h)))
        lo = np.floor(idx).astype(int); fr = idx - lo; hi = np.minimum(lo + 1, len(self.h) - 1)

        def g(A):
            return A[:, lo] * (1 - fr) + A[:, hi] * fr

        bA = sig * (g(eS) - g(eA)); bB = sig * (g(eS) - g(eB))
        nu0 = 1 / (2 * tau0)              # low-field limit reproduces tau_N = tau0 exp(sigma)
        rA = nu0 * np.exp(-bA); rB = nu0 * np.exp(-bB)
        mA = g(cA); mB = g(cB)
        dt = 1 / (nstep * f)
        p = np.full(len(self.thetas), 0.5)
        m = np.empty((len(self.thetas), len(t)))
        m[:, 0] = p * mA[:, 0] + (1 - p) * mB[:, 0]
        for k in range(1, len(t)):
            ra = 0.5 * (rA[:, k] + rA[:, k - 1]); rb = 0.5 * (rB[:, k] + rB[:, k - 1])
            s = ra + rb; peq = rb / s
            p = peq + (p - peq) * np.exp(-s * dt)
            m[:, k] = p * mA[:, k] + (1 - p) * mB[:, k]
        mbar = self.w @ m                 # orientation-averaged reduced magnetization
        sl = slice(nstep * (ncyc - 1), nstep * ncyc + 1)
        Hc = H * np.cos(2 * np.pi * f * t[sl]); M = self.Ms * mbar[sl]
        area = mu0 * np.abs(np.trapezoid(M, Hc))
        sl2 = slice(nstep * (ncyc - 2), nstep * (ncyc - 1) + 1)
        area2 = mu0 * np.abs(np.trapezoid(self.Ms * mbar[sl2], H * np.cos(2 * np.pi * f * t[sl2])))
        return area * f / rho, abs(area - area2) / max(area, 1e-30)




def barrier_at_peak(ts, d, H):
    """Orientation-averaged barrier (in kT) of the metastable minimum at the field maximum."""
    eA, eB, eS, cA, cB = ts.tab
    sig = ts.K * np.pi * d**3 / 6 / (kB * T)
    j = np.argmin(abs(ts.h - H / ts.HK))
    b = sig * (eS[:, j] - np.maximum(eA[:, j], eB[:, j]))
    return float(ts.w @ b / ts.w.sum())


if __name__ == '__main__':
    ts = TwoState()
    out = {}
    # low-field check: two-state model -> LRT
    out['lowfield_ratio'] = {str(dd): ts.sar(dd * 1e-9, 100.0, 300e3, nstep=8000)[0] / 1000 /
                             sar(dd * 1e-9, 100.0, 300e3, eta=np.inf)[0] for dd in (12, 14, 16, 18)}
    d = np.round(np.arange(10, 26.01, 0.25), 2)
    for tag, H, f in (('15k300', 15e3, 300e3), ('18k278', 18e3, 5e9 / 18e3)):
        lrt = sar(d * 1e-9, H, f, eta=np.inf)
        nl = [ts.sar(x * 1e-9, H, f)[0] / 1000 for x in d]
        out[tag] = dict(d=d.tolist(), lrt=lrt.tolist(), nl=nl, barrier=[barrier_at_peak(ts, x * 1e-9, H) for x in d])
        i = int(np.argmax(nl)); j = int(np.argmax(lrt))
        out[tag + '_summary'] = dict(lrt_opt=[d[j], round(float(lrt[j]), 1)], nl_opt=[d[i], round(nl[i], 1)])
    Hs = np.arange(2, 18.01, 0.5) * 1e3
    out['Hsweep'] = dict(H=(Hs / 1e3).tolist(), data={str(x): dict(
        nl=[ts.sar(x * 1e-9, h, 5e9 / 18e3)[0] / 1000 for h in Hs],
        lrt=sar(np.full(len(Hs), x * 1e-9), Hs, 5e9 / 18e3, eta=np.inf).tolist()) for x in (14, 16, 18, 20)})
    json.dump(out, open('nonlinear.json', 'w'), indent=1)
    print(json.dumps({k: v for k, v in out.items() if 'summary' in k or k == 'lowfield_ratio'}, indent=1))
