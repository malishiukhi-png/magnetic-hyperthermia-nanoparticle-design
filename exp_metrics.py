"""Comparison of all models with Khandhar et al. (373 kHz, 14 kA/m) and summary of the Tong et al. conditions
(325 kHz, 9.35 kA/m). Writes exp_metrics.json. Uses only stored sLLG grids (llg_khandhar_a*.json, llg_kh_K23.json,
llg_kh_c.json, llg_tong_u.json, llg_tong_c.json) and the LRT of model.py."""
import json, os, sys, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); os.chdir(HERE)
import model

F, H = 373e3, 14e3
MS_EXP = 75.0 * 5180.0
model.T = 310.15
NARROW = [(12, 0.01, 23.5), (13, 0.10, 54.1), (14, 0.21, 87.3), (15, 0.23, 100.2), (16, 0.16, 141.0), (18, 0.34, 43.3)]
BROAD = [(9, 0.29, 11.1), (10, 0.10, 24.2), (14, 0.31, 101.2), (16, 0.36, 55.1), (16.5, 0.36, 30.1)]


def lognormal(func, d0, s, n=401):
    if s < 0.02:
        return float(func(np.array([d0]))[0])
    x = np.linspace(np.log(d0) - 4 * s, np.log(d0) + 4 * s, n); d = np.exp(x)
    w = np.exp(-(x - np.log(d0))**2 / (2 * s * s)) * d**3
    return float(np.sum(w * func(d)) / np.sum(w))


def grid(fn):
    G = np.array(json.load(open(fn)))
    dg, sg = G[:, 0] * 1e-9, G[:, 1]
    return lambda d: np.interp(d, dg, sg, left=0.0, right=sg[-1])


MODELS = {
    'LRT, K = 13': lambda d: model.sar(d, H, F, eta=0.69e-3, Ms=MS_EXP, tau0=1e-9, K=13e3),
    'LRT, K = 23': lambda d: model.sar(d, H, F, eta=0.69e-3, Ms=MS_EXP, tau0=1e-9, K=23e3),
    'LRT, K = 41': lambda d: model.sar(d, H, F, eta=0.69e-3, Ms=MS_EXP, tau0=1e-9, K=41e3),
    'sLLG, K = 13, alpha = 0.1': grid('llg_khandhar_a0.1.json'),
    'sLLG, K = 13, alpha = 1': grid('llg_khandhar_a1.json'),
    'sLLG, K = 23, alpha = 1': grid('llg_kh_K23.json'),
    'sLLG, cubic, alpha = 1': grid('llg_kh_c.json'),
}
rows = {'narrow': [], 'broad': []}
for key, data in (('narrow', NARROW), ('broad', BROAD)):
    for d0, s, meas in data:
        r = dict(d0=d0, sigma=s, measured=meas)
        for name, fn in MODELS.items():
            r[name] = round(lognormal(fn, d0 * 1e-9, s), 1)
        rows[key].append(r)
summ = {}
allrows = rows['narrow'] + rows['broad']
for name in MODELS:
    nar = np.array([r[name] for r in rows['narrow']]); mn = np.array([r['measured'] for r in rows['narrow']])
    i = int(np.argmax(nar))
    scale = float(np.sum(mn * nar) / np.sum(nar**2)) if nar.any() else float('nan')
    r16 = [r for r in rows['broad'] if r['d0'] == 16][0][name] / max([r for r in rows['narrow'] if r['d0'] == 16][0][name], 1e-9)
    pa = np.array([max(r[name], 0.1) for r in allrows]); ma = np.array([r['measured'] for r in allrows])
    summ[name] = dict(peak_d0=rows['narrow'][i]['d0'], over=round(1 / scale, 2) if scale > 0 else None,
                      ratio16=round(r16, 2), log_rmse=round(float(np.sqrt(np.mean(np.log10(pa / ma)**2))), 2),
                      max_narrow=round(float(nar.max()), 1))
summ['measured'] = dict(peak_d0=16, ratio16=round(55.1 / 141.0, 2))
tong = {}
for k in ('tong_u', 'tong_c'):
    try:
        tong[k] = json.load(open('llg_%s.json' % k))
    except FileNotFoundError:
        pass
out = dict(rows=rows, summary=summ, tong=tong, tong_measured_40nm_W_per_g_Fe3O4=round(1026 * 0.7236, 0))
json.dump(out, open('exp_metrics.json', 'w'), indent=1)
print(json.dumps(summ, indent=1)); print(json.dumps(tong))
