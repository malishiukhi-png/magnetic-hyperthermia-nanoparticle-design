"""Stochastic Landau-Lifshitz-Gilbert (sLLG) simulation of immobilized uniaxial magnetite nanoparticles with random
easy axes in an AC field, with explicit Gilbert damping and the free-electron gyromagnetic ratio (no calibration).
Usage: python llg.py <alpha> [H_kA_per_m f_kHz]   -> llg_a<alpha>.json   (requires numba; ~5-10 min per size)"""
import numpy as np, numba, sys, json
sys.path.insert(0, '.')
from model import P, T, kB, mu0

GAMMA = 1.760859e11            # gyromagnetic ratio (rad s-1 T-1)


@numba.njit(cache=True)
def _run(nx, ny, nz, HK, Hth_sd, H0, w, dt, nsteps, alpha, rec_every, seed):
    np.random.seed(seed)
    N = nx.shape[0]
    mx = nx.copy(); my = ny.copy(); mz = nz.copy()
    for i in range(N):
        if np.random.random() < 0.5:
            mx[i] = -mx[i]; my[i] = -my[i]; mz[i] = -mz[i]
    g = GAMMA * mu0 / (1.0 + alpha * alpha)     # prefactor with fields in A/m
    nrec = nsteps // rec_every + 1
    out = np.zeros(nrec); r = 0
    for k in range(nsteps + 1):
        if k % rec_every == 0:
            s = 0.0
            for i in range(N):
                s += mz[i]
            out[r] = s / N; r += 1
        if k == nsteps:
            break
        ha0 = H0 * np.cos(w * k * dt); ha1 = H0 * np.cos(w * (k + 1) * dt)
        for i in range(N):
            tx = Hth_sd * np.random.standard_normal()
            ty = Hth_sd * np.random.standard_normal()
            tz = Hth_sd * np.random.standard_normal()
            # Heun predictor
            mn = mx[i] * nx[i] + my[i] * ny[i] + mz[i] * nz[i]
            hx = HK * mn * nx[i] + tx; hy = HK * mn * ny[i] + ty; hz = HK * mn * nz[i] + tz + ha0
            cx = my[i] * hz - mz[i] * hy; cy = mz[i] * hx - mx[i] * hz; cz = mx[i] * hy - my[i] * hx
            dx = -g * (cx + alpha * (my[i] * cz - mz[i] * cy))
            dy = -g * (cy + alpha * (mz[i] * cx - mx[i] * cz))
            dz = -g * (cz + alpha * (mx[i] * cy - my[i] * cx))
            px = mx[i] + dx * dt; py = my[i] + dy * dt; pz = mz[i] + dz * dt
            nn = np.sqrt(px * px + py * py + pz * pz); px /= nn; py /= nn; pz /= nn
            # corrector
            mn2 = px * nx[i] + py * ny[i] + pz * nz[i]
            hx2 = HK * mn2 * nx[i] + tx; hy2 = HK * mn2 * ny[i] + ty; hz2 = HK * mn2 * nz[i] + tz + ha1
            cx2 = py * hz2 - pz * hy2; cy2 = pz * hx2 - px * hz2; cz2 = px * hy2 - py * hx2
            dx2 = -g * (cx2 + alpha * (py * cz2 - pz * cy2))
            dy2 = -g * (cy2 + alpha * (pz * cx2 - px * cz2))
            dz2 = -g * (cz2 + alpha * (px * cy2 - py * cx2))
            ax = mx[i] + 0.5 * (dx + dx2) * dt; ay = my[i] + 0.5 * (dy + dy2) * dt; az = mz[i] + 0.5 * (dz + dz2) * dt
            nn = np.sqrt(ax * ax + ay * ay + az * az)
            mx[i] = ax / nn; my[i] = ay / nn; mz[i] = az / nn
    return out


def sar_llg(d, H, f, alpha, N=1000, ncyc=4, seed=1, dt=None, Ms=P['Ms'], K=P['K'], rho=P['rho']):
    V = np.pi * d**3 / 6
    HK = 2 * K / (mu0 * Ms)
    if dt is None:
        # resolve precession in the anisotropy + applied field (about 40 steps per period)
        dt = 2 * np.pi / (GAMMA * mu0 * (HK + H)) / 40
    Hth_sd = np.sqrt(2 * alpha * kB * T / (GAMMA * mu0**2 * Ms * V * dt))   # A/m, per component (Brown)
    nper = int(np.ceil(1 / f / dt)); dt = 1 / f / nper
    rng = np.random.default_rng(seed)
    u = rng.uniform(-1, 1, N); ph = rng.uniform(0, 2 * np.pi, N); s = np.sqrt(1 - u**2)
    nx, ny, nz = s * np.cos(ph), s * np.sin(ph), u
    rec = max(nper // 2000, 1)
    mz = _run(nx, ny, nz, HK, Hth_sd, H, 2 * np.pi * f, dt, nper * ncyc, alpha, rec, seed)
    tt = np.arange(len(mz)) * rec / nper
    Hc = H * np.cos(2 * np.pi * tt)
    areas = []
    for c in range(1, ncyc):
        sl = (tt >= c - 1e-9) & (tt <= c + 1 + 1e-9)
        areas.append(-mu0 * Ms * np.trapezoid(mz[sl], Hc[sl]))
    areas = np.array(areas)
    return areas.mean() * f / rho, areas.std(ddof=1) * f / rho / np.sqrt(len(areas)), nper


def tau0_brown(d, alpha, Ms=P['Ms'], K=P['K']):
    """Effective attempt time tau_N / exp(sigma) from Brown's high-barrier formula (axially symmetric, zero field)."""
    V = np.pi * d**3 / 6
    sig = K * V / (kB * T)
    tauD = (1 + alpha**2) / (2 * alpha) * Ms * V / (GAMMA * kB * T)
    return tauD * np.sqrt(np.pi) / 2 * sig**-1.5


if __name__ == '__main__':
    alpha = float(sys.argv[1])
    H = float(sys.argv[2]) * 1e3 if len(sys.argv) > 2 else 15e3
    f = float(sys.argv[3]) * 1e3 if len(sys.argv) > 3 else 300e3
    ds = [float(x) for x in sys.argv[4].split(',')] if len(sys.argv) > 4 else [14, 16, 17, 18, 19, 20, 22]
    out = []
    for d in ds:
        v, e, nper = sar_llg(d * 1e-9, H, f, alpha, seed=int(d * 10 + alpha * 1000))
        out.append([d, v / 1000, e / 1000, tau0_brown(d * 1e-9, alpha)])
        print('llg', alpha, d, round(v / 1000, 1), '+-', round(e / 1000, 1), nper, flush=True)
        json.dump(out, open('llg_a%g.json' % alpha, 'w'))
