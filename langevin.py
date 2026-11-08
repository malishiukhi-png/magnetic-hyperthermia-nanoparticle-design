"""Usage: python langevin.py 15k300 | 18k278 | conv | hsweep   (20-60 min each on one core; requires numba)
Overdamped stochastic (Brown) dynamics of the Neel moment of immobilized uniaxial
nanoparticles with random easy axes in an AC field. Returns SAR from the dynamic loop."""
import numpy as np, numba, sys, json
sys.path.insert(0, ".")
from model import P, T, kB, mu0, sar as sar_lrt

MU0, KB = mu0, kB


@numba.njit(cache=True)
def _run(nx, ny, nz, sig, xi0, wt_per_step, nsteps, dt, seed, rec_every):
    np.random.seed(seed)
    N = nx.shape[0]
    mx = nx.copy(); my = ny.copy(); mz = nz.copy()
    # random initial up/down along easy axis
    for i in range(N):
        if np.random.random() < 0.5:
            mx[i] = -mx[i]; my[i] = -my[i]; mz[i] = -mz[i]
    nrec = nsteps // rec_every + 1
    mzbar = np.zeros(nrec)
    sq = np.sqrt(dt)
    r = 0
    for k in range(nsteps + 1):
        if k % rec_every == 0:
            s = 0.0
            for i in range(N):
                s += mz[i]
            mzbar[r] = s / N; r += 1
        if k == nsteps:
            break
        xi = xi0 * np.cos(wt_per_step * (k + 0.5))
        for i in range(N):
            # gradient of beta*E: g = -2 sig (m.n) n - xi z
            mn = mx[i] * nx[i] + my[i] * ny[i] + mz[i] * nz[i]
            gx = -2.0 * sig * mn * nx[i]
            gy = -2.0 * sig * mn * ny[i]
            gz = -2.0 * sig * mn * nz[i] - xi
            # tangential projection: drift = -(g - (m.g) m)/2
            mg = mx[i] * gx + my[i] * gy + mz[i] * gz
            dxp = -0.5 * (gx - mg * mx[i]) * dt
            dyp = -0.5 * (gy - mg * my[i]) * dt
            dzp = -0.5 * (gz - mg * mz[i]) * dt
            # noise (rotational diffusion D = 1/2 in units of tau_D)
            wx = np.random.standard_normal() * sq
            wy = np.random.standard_normal() * sq
            wz = np.random.standard_normal() * sq
            mw = mx[i] * wx + my[i] * wy + mz[i] * wz
            dxp += wx - mw * mx[i]
            dyp += wy - mw * my[i]
            dzp += wz - mw * mz[i]
            ax = mx[i] + dxp; ay = my[i] + dyp; az = mz[i] + dzp
            nn = np.sqrt(ax * ax + ay * ay + az * az)
            mx[i] = ax / nn; my[i] = ay / nn; mz[i] = az / nn
    return mzbar


TAU_D_REF = 1e-9 * 2 * (6.39**1.5) / np.sqrt(np.pi)   # tau_D at d = 15.9 nm (Brown high-barrier match)
D_REF = 15.9e-9


def sar_langevin(d, H, f, N=4000, ncyc=4, seed=1, **kw):
    return sar_langevin_dt(d, H, f, 1.0, N=N, ncyc=ncyc, seed=seed, **kw)


def sar_langevin_dt(d, H, f, dtfac, N=4000, ncyc=4, seed=1, Ms=P['Ms'], K=P['K'], T=T,
                    rho=P['rho']):
    V = np.pi * d**3 / 6
    sig = K * V / (KB * T)
    xi0 = MU0 * Ms * V * H / (KB * T)
    tauD = TAU_D_REF * (d / D_REF)**3
    period = 1 / f
    dt = dtfac * min(0.02 / (2 * sig + xi0), 0.01)
    nper = int(np.ceil(period / tauD / dt))
    dt = period / tauD / nper
    rng = np.random.default_rng(seed)
    u = rng.uniform(-1, 1, N); ph = rng.uniform(0, 2 * np.pi, N)
    s = np.sqrt(1 - u**2)
    nx, ny, nz = s * np.cos(ph), s * np.sin(ph), u
    rec = max(nper // 2000, 1)
    nsteps = nper * ncyc
    mz = _run(nx, ny, nz, sig, xi0, 2 * np.pi / nper, nsteps, dt, seed, rec)
    tt = np.arange(len(mz)) * rec / nper          # in periods
    Hc = H * np.cos(2 * np.pi * tt)
    areas = []
    for c in range(1, ncyc):
        sl = (tt >= c - 1e-9) & (tt <= c + 1 + 1e-9)
        areas.append(-MU0 * Ms * np.trapezoid(mz[sl], Hc[sl]))
    areas = np.array(areas)
    return areas.mean() * f / rho, areas.std(ddof=1) * f / rho / np.sqrt(len(areas)), nper


if __name__ == '__main__':
    which = sys.argv[1]
    res = {}
    if which == 'check':
        for d in [14e-9, 16e-9]:
            for H in [300.0]:
                vals = [sar_langevin(d, H, 300e3, N=4000, ncyc=6, seed=s)[0] for s in range(3)]
                lrt = sar_lrt(d, H, 300e3, eta=np.inf)[0] * 1000
                print(d, H, np.mean(vals) / lrt, np.std(vals) / lrt)
        sys.exit()
    if which == 'conv':
        # time-step and low-field checks at 16 nm
        out = {}
        for fac in (1.0, 0.5):
            global_dt = fac
            v = sar_langevin_dt(16e-9, 15e3, 300e3, fac, N=2000, ncyc=4, seed=7)
            out['dt%.1f' % fac] = [v[0] / 1000, v[1] / 1000]
            print('dt factor', fac, v[0] / 1000, '+-', v[1] / 1000, flush=True)
        for d in (14e-9, 16e-9, 18e-9):
            v = sar_langevin(d, 1500.0, 300e3, N=8000, ncyc=6, seed=3)
            lrt = sar_lrt(d, 1500.0, 300e3, eta=np.inf)[0] * 1000
            out['low%d' % round(d * 1e9)] = [v[0] / lrt, v[1] / lrt]
            print('low field', d, v[0] / lrt, '+-', v[1] / lrt, flush=True)
        json.dump(out, open('langevin_conv.json', 'w'))
        sys.exit()
    if which == 'hsweep':
        out = {}
        for d in (16, 20):
            for H in (6e3, 10e3, 14e3):
                v, e, _ = sar_langevin(d * 1e-9, H, 5e9 / 18e3, N=2000, ncyc=4, seed=int(d * 100 + H / 1e3))
                out['%d_%d' % (d, H / 1e3)] = [v / 1000, e / 1000]
                print('hsweep', d, H, v / 1000, '+-', e / 1000, flush=True)
        json.dump(out, open('langevin_hsweep.json', 'w'))
        sys.exit()
    H, f = {'15k300': (15e3, 300e3), '18k278': (18e3, 5e9 / 18e3)}[which]
    ds = np.arange(12, 24.01, 1.0) if which == '15k300' else np.arange(14, 24.01, 2.0)
    out = []
    for d in ds:
        v, e, nper = sar_langevin(d * 1e-9, H, f, N=2000, ncyc=4, seed=int(d * 10))
        out.append([d, v / 1000, e / 1000])
        print(which, d, round(v / 1000, 1), '+-', round(e / 1000, 1), nper, flush=True)
    json.dump(out, open(f'langevin_{which}.json', 'w'))
