"""Post-processing of the sLLG runs: field scenarios, polydispersity, anisotropy, dipolar coupling, nonlinear dose,
Monte Carlo over the magnetic and thermal inputs, and comparison with the live-cell data of Cabrera et al. -> nl_results.json"""
import json, sys, os, numpy as np
from scipy.stats import spearmanr
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import model
from model import sar, bioheat, P, kB, mu0, T as T0
RUNS = sys.argv[1] if len(sys.argv) > 1 else 'sllg_runs.jsonl'
HERE = os.path.dirname(os.path.abspath(__file__)) + '/'
rows = [json.loads(l) for l in open(RUNS)]
GAMMA = 1.760859e11
Rt, kt, rc = 10e-3, 0.5, 4.0e6


def tau0_brown(d, alpha, Ms=P['Ms'], K=P['K']):
    V = np.pi * d**3 / 6; sig = K * V / (kB * T0)
    tauD = (1 + alpha**2) / (2 * alpha) * Ms * V / (GAMMA * kB * T0)
    return tauD * np.sqrt(np.pi) / 2 * sig**-1.5


def opt(ds, ys):
    """Maximum of SAR(d): parabola through the three highest grid points in (d, ln SAR)."""
    ds, ys = np.asarray(ds, float), np.asarray(ys, float)
    i = int(np.argmax(ys))
    if i == 0 or i == len(ds) - 1:
        return float(ds[i]), float(ys[i])
    c = np.polyfit(ds[i - 1:i + 2], np.log(np.maximum(ys[i - 1:i + 2], 1e-9)), 2)
    dm = -c[1] / (2 * c[0]); return float(dm), float(np.exp(np.polyval(c, dm)))


def select(**kw):
    out = []
    for r in rows:
        if all(abs(r[k] - v) <= 1e-6 * max(1, abs(v)) if isinstance(v, float) else r.get(k) == v for k, v in kw.items()):
            out.append(r)
    return sorted(out, key=lambda r: r['d'])


def lrt_opt(H, f, tau0=1e-9):
    d = np.linspace(8e-9, 34e-9, 1041); y = sar(d, H, f, eta=np.inf, tau0=tau0); i = int(np.argmax(y)); return round(float(d[i] * 1e9), 1), round(float(y[i]), 1)


def lrt_at(ds, H, f, sigma=0.0, K=None):
    return [float(sar(np.array([x * 1e-9]), H, f, eta=np.inf, sigma=sigma, K=K)[0]) for x in ds]


def lam(d, delta, Ms=P['Ms'], T=T0):
    m = Ms * np.pi * d**3 / 6
    return float(mu0 * m**2 / (4 * np.pi * (d + 2 * delta)**3 * kB * T))


def unit(wb, R=Rt, k=kt):
    W = rc * wb; return bioheat(np.array([0.0, R, R + 5e-3]), R, 1e6, W, W, k)


def Qedge(wb=0.5e-3, R=Rt, k=kt):
    return 5.0 / unit(wb, R, k)[1] * 1e6


def dose(S, wb=0.5e-3):            # S in W/g -> mg/cm3 for the boundary at 42 C
    return Qedge(wb) / (S * 1000)


R = {}
# ---- stored reference data --------------------------------------------------------------------------------
REF = {}
for a in (0.1, 1.0):
    G = np.array(json.load(open(HERE + 'llg_a%g.json' % a)))
    REF[a] = {float(x[0]): (float(x[1]), float(x[2])) for x in G}
ODE = {float(x[0]): (float(x[1]), float(x[2])) for x in json.load(open(HERE + 'langevin_15k300.json'))}
ODE18 = {float(x[0]): (float(x[1]), float(x[2])) for x in json.load(open(HERE + 'langevin_18k278.json'))}
# ---- field scenarios --------------------------------------------------------------------------------------
FIELDS = [(18e3, 100e3), (15e3, 300e3), (18e3, 5e9 / 18e3), (12.5e3, 400e3), (10e3, 500e3), (5e3, 1000e3)]
R['fields'] = []
for H, f in FIELDS:
    e = dict(H=H / 1e3, f=round(f / 1e3, 1), Hf=H * f, lrt=lrt_opt(H, f)); e['curves'] = {}
    for a in (0.1, 1.0):
        if H == 15e3:
            ds = sorted(REF[a]); ys = [REF[a][x][0] for x in ds]; es = [REF[a][x][1] for x in ds]
        else:
            r = select(group='field', H=float(H), alpha=a, f=float(f)); ds = [x['d'] for x in r]; ys = [x['sar'] for x in r]; es = [x['se'] for x in r]
        e['curves'][str(a)] = dict(d=ds, sar=ys, se=es)
        if len(ds) >= 3:
            dm, sm = opt(ds, ys); e['opt_a%g' % a] = [round(dm, 1), round(sm, 0)]
    R['fields'].append(e)
R['overdamped'] = dict(k15_300=dict(d=sorted(ODE), sar=[ODE[x][0] for x in sorted(ODE)], se=[ODE[x][1] for x in sorted(ODE)]),
                       k18_278=dict(d=sorted(ODE18), sar=[ODE18[x][0] for x in sorted(ODE18)]))
R['overdamped']['opt15'] = [round(x, 1) for x in opt(sorted(ODE), [ODE[x][0] for x in sorted(ODE)])]
R['overdamped']['opt18'] = [round(x, 1) for x in opt(sorted(ODE18), [ODE18[x][0] for x in sorted(ODE18)])]
# ---- dipolar coupling --------------------------------------------------------------------------------------
R['lambda'] = {str(d): {str(dl): round(lam(d * 1e-9, dl * 1e-9), 2) for dl in (5, 10, 15, 20)} for d in (12, 14, 16, 18, 20, 22, 24, 26)}
# ---- shape anisotropy of a prolate spheroid --------------------------------------------------------------
def Ksh(m, Ms=P['Ms']):
    e = np.sqrt(m**2 - 1); Nz = (m / e * np.log(m + e) - 1) / e**2; Nx = (1 - Nz) / 2
    return 0.5 * mu0 * Ms**2 * (Nx - Nz)
ms = np.linspace(1.01, 2, 991); Ks = np.array([Ksh(m) for m in ms])
R['shape'] = dict(aspect_for_13=round(float(np.interp(13e3, Ks, ms)), 2), aspect_for_30=round(float(np.interp(30e3, Ks, ms)), 2), K_at_1p1=round(float(Ksh(1.1) / 1e3), 1), K_at_1p2=round(float(Ksh(1.2) / 1e3), 1), K_at_1p4=round(float(Ksh(1.4) / 1e3), 1), cubic_equiv_kJ=round(13 / 12, 2))
# ---- dose with nonlinear SAR at the reference field and in the clinical 100 kHz scenario -------------------
R['Q_edge_MW'] = {str(w): round(Qedge(w) / 1e6, 3) for w in (0.25e-3, 0.5e-3, 1e-3)}
sizes = [14.0, 16.0, 18.0, 20.0, 22.0, 24.0, 26.0]
tab = []
for d in sizes:
    e = dict(d=d, lam5=round(lam(d * 1e-9, 5e-9), 2), lam10=round(lam(d * 1e-9, 10e-9), 2), lam15=round(lam(d * 1e-9, 15e-9), 2))
    e['lrt'] = lrt_at([d], 15e3, 300e3)[0]; e['lrt_dose'] = dose(e['lrt'])
    e['od'] = ODE.get(d, (None,))[0]
    for a in (0.1, 1.0):
        if d in REF[a]:
            s = REF[a][d][0]; e['s%g' % a] = s; e['c%g' % a] = dose(s)
    tab.append(e)
R['dose_table'] = tab
R['dose_range_note'] = dict(low=dose(1.0, 0.25e-3) / dose(1.0), high=dose(1.0, 1e-3) / dose(1.0))
# clinical scenario
cl = {}
for a in (0.1, 1.0):
    r = select(group='field', H=18e3, f=100e3, alpha=a)
    if r:
        cl[str(a)] = {str(x['d']): dict(sar=x['sar'], dose=dose(max(x['sar'], 1e-6))) for x in r}
R['clinical100'] = cl
R['clinical100_lrt'] = {str(d): lrt_at([d], 18e3, 100e3)[0] for d in sizes}
# ---- polydispersity (direct ensembles) ----------------------------------------------------------------------
R['poly'] = {}
for s in (0.1, 0.2):
    for a in (0.1, 1.0):
        r = select(group='poly', sigma=s, alpha=a)
        if len(r) >= 3:
            ds = [x['d'] for x in r]; ys = [x['sar'] for x in r]; dm, sm = opt(ds, ys)
            # check against log-normal averaging of the monodisperse grid
            g = sorted(REF[a]); gy = [REF[a][x][0] for x in g]
            def av(d0):
                x = np.linspace(np.log(d0) - 4 * s, np.log(d0) + 4 * s, 401); dd = np.exp(x); w = np.exp(-(x - np.log(d0))**2 / (2 * s * s)) * dd**3
                return float(np.sum(w * np.interp(dd, g, gy, left=0.0, right=0.0)) / np.sum(w))
            R['poly']['%g_%g' % (s, a)] = dict(d=ds, sar=ys, se=[x['se'] for x in r], opt=[round(dm, 1), round(sm, 0)], interp=[round(av(x), 1) for x in ds], dose_opt=dose(sm))
R['poly_lrt'] = {str(s): lrt_opt_s for s, lrt_opt_s in ((0.0, lrt_opt(15e3, 300e3)),)}
for s in (0.1, 0.2):
    d = np.linspace(8e-9, 34e-9, 261); y = sar(d, 15e3, 300e3, eta=np.inf, sigma=s); i = int(np.argmax(y)); R['poly_lrt'][str(s)] = [round(float(d[i] * 1e9), 1), round(float(y[i]), 1)]
# ---- anisotropy ------------------------------------------------------------------------------------------
R['K'] = {}
for K in (9e3, 20e3, 30e3):
    for a in (0.1, 1.0):
        r = select(group='K', K=K, alpha=a)
        if len(r) >= 3:
            ds = [x['d'] for x in r]; ys = [x['sar'] for x in r]; dm, sm = opt(ds, ys)
            R['K']['%g_%g' % (K / 1e3, a)] = dict(d=ds, sar=ys, opt=[round(dm, 1), round(sm, 0)])
for a in (0.1, 1.0):
    ds = sorted(REF[a]); ys = [REF[a][x][0] for x in ds]; dm, sm = opt(ds, ys); R['K']['13_%g' % a] = dict(d=ds, sar=ys, opt=[round(dm, 1), round(sm, 0)])
R['K_lrt'] = {}
for K in (9e3, 13e3, 20e3, 30e3):
    d = np.linspace(8e-9, 34e-9, 521); y = sar(d, 15e3, 300e3, eta=np.inf, K=K); i = int(np.argmax(y)); R['K_lrt'][str(int(K / 1e3))] = [round(float(d[i] * 1e9), 1), round(float(y[i]), 1)]
# ---- Cabrera et al.: live cells, 100 kHz, 20 kA/m ------------------------------------------------------------
R['cabrera'] = []
for r in select(group='cabrera'):
    R['cabrera'].append(dict(sample=r['sample'], d0=r['d'], sigma=r['sigma'], alpha=r['alpha'], K=r['K'] / 1e3, sar=round(r['sar'], 1), se=round(r['se'], 1)))
# ---- low damping (alpha = 0.0015, as in the loop simulations of Cabrera et al.), 15 kA/m, 300 kHz ---------------
r = select(group='lowalpha')
if r:
    dm, sm = opt([x['d'] for x in r], [x['sar'] for x in r])
    R['lowalpha'] = dict(alpha=0.0015, d=[x['d'] for x in r], sar=[round(x['sar'], 1) for x in r], se=[round(x['se'], 1) for x in r], opt=[round(dm, 1), round(sm, 0)])
# ---- intermediate damping, same conditions: where does the optimum of 22 nm begin to move? ------------------------
R['damping'] = {}
for a in sorted({x['alpha'] for x in rows if x['group'] == 'damping'}):
    r = select(group='damping', alpha=a)
    dm, sm = opt([x['d'] for x in r], [x['sar'] for x in r])
    R['damping'][str(a)] = dict(d=[x['d'] for x in r], sar=[round(x['sar'], 1) for x in r], se=[round(x['se'], 1) for x in r], opt=[round(dm, 1), round(sm, 0)])
# ---- Monte Carlo ---------------------------------------------------------------------------------------------
mcr =[r for r in rows if r['group'] == 'mc']
S = json.load(open(os.path.dirname(os.path.abspath(RUNS)) + '/mc_inputs.json'))
by = {}
for r in mcr: by.setdefault(r['mc'], {})[r['d']] = r['sar']
G0 = (14.0, 16.0, 18.0, 20.0, 22.0, 24.0, 26.0, 28.0)
idx = sorted(i for i, v in by.items() if all(d in v for d in G0))
if len(idx) >= 10:
    rng = np.random.default_rng(7); N = len(S['Ms'])
    th = dict(k=rng.uniform(0.4, 0.6, N), wb=np.exp(rng.uniform(np.log(0.25e-3), np.log(1e-3), N)), R=rng.uniform(5e-3, 15e-3, N))
    out = {k: [] for k in ('d_opt', 'sar', 'loss_m2', 'loss_p2', 's18', 's20', 's22', 'dose_opt', 'dose18', 'dose20', 'dose22', 'center')}
    for i in idx:
        G = sorted(by[i]); y = np.array([by[i][d] for d in G]); dm, sm = opt(G, y)
        f = lambda x: float(np.exp(np.interp(x, G, np.log(np.maximum(y, 1e-3)))))
        W = rc * th['wb'][i]; Ru = th['R'][i]; u = bioheat(np.array([0.0, Ru]), Ru, 1e6, W, W, th['k'][i]); Q = 5.0 / u[1] * 1e6
        out['d_opt'].append(dm); out['sar'].append(sm); out['loss_m2'].append(100 * (1 - f(max(dm - 2, G[0])) / sm)); out['loss_p2'].append(100 * (1 - f(min(dm + 2, G[-1])) / sm))
        for d in (18.0, 20.0, 22.0):
            out['s%d' % d].append(by[i][d]); out['dose%d' % d].append(Q / (max(by[i][d], 1e-6) * 1000))
        out['dose_opt'].append(Q / (sm * 1000)); out['center'].append(37 + u[0] * Q / 1e6)
    inp = {k: np.array(S[k])[idx] for k in ('Ms', 'K', 'alpha', 'sigma')}; inp.update({k: th[k][idx] for k in th})
    MC = dict(n=len(idx))
    MC['n_edge'] = int(sum(1 for i in idx if by[i][max(by[i])] == max(by[i].values())))
    for k, v in out.items():
        v = np.array(v); MC[k] = dict(median=round(float(np.median(v)), 2), p5=round(float(np.percentile(v, 5)), 2), p95=round(float(np.percentile(v, 95)), 2))
    MC['spearman'] = {k: {p: round(float(spearmanr(inp[p], np.array(v))[0]), 2) for p in inp} for k, v in out.items()}
    MC['raw'] = {k: [float(x) for x in v] for k, v in out.items()}
    R['mc'] = MC
json.dump(R, open(HERE + 'nl_results.json', 'w'), indent=1)
for k in ('lambda', 'shape', 'Q_edge_MW', 'K_lrt', 'poly_lrt'):
    print(k, R[k])
for e in R['fields']: print(e['H'], e['f'], 'LRT', e['lrt'], 'a.1', e.get('opt_a0.1'), 'a1', e.get('opt_a1'))
print('OD', R['overdamped']['opt15'], R['overdamped']['opt18'])
for e in R['dose_table']: print({k: (round(v, 2) if isinstance(v, float) else v) for k, v in e.items()})
print(R['cabrera'])
print({k: v.get('opt') for k, v in R['K'].items()}); print({k: (v['opt'], v['sar'][:3], v['interp'][:3]) for k, v in R['poly'].items()})
if 'mc' in R: print(json.dumps({k: v for k, v in R['mc'].items() if k != 'raw'}, indent=0)[:1800])
