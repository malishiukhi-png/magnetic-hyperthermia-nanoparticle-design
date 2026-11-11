"""LRT comparison with Khandhar et al. for higher effective anisotropy constants (revision) -> khandhar_K.json"""
import sys, os, json, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import model
F, H = 373e3, 14e3
MS_EXP = 75.0 * 5180.0
T_EXP = 310.15
# same data as exp_metrics.py (Khandhar et al.: Table 1 sizes/widths, Figure 7 SLP)
NARROW = [(12, 0.01, 23.5), (13, 0.10, 54.1), (14, 0.21, 87.3), (15, 0.23, 100.2), (16, 0.16, 141.0), (18, 0.34, 43.3)]
BROAD = [(9, 0.29, 11.1), (10, 0.10, 24.2), (14, 0.31, 101.2), (16, 0.36, 55.1), (16.5, 0.36, 30.1)]


def lognormal(func, d0, s, n=401):
    if s < 0.02:
        return float(func(np.array([d0]))[0])
    x = np.linspace(np.log(d0) - 4 * s, np.log(d0) + 4 * s, n); d = np.exp(x)
    w = np.exp(-(x - np.log(d0))**2 / (2 * s * s)) * d**3
    return float(np.sum(w * func(d)) / np.sum(w))
model.T = T_EXP
res = {}
for K in (13e3, 18e3, 23e3, 30e3, 41e3):
    fn = lambda d: model.sar(d, H, F, eta=0.69e-3, Ms=MS_EXP, tau0=1e-9, K=K)
    rows = {}
    for key, data in (('narrow', NARROW), ('broad', BROAD)):
        rows[key] = [[d0, s, meas, round(lognormal(fn, d0 * 1e-9, s), 1)] for d0, s, meas in data]
    nar = np.array([r[3] for r in rows['narrow']]); meas = np.array([r[2] for r in rows['narrow']])
    i = int(np.argmax(nar)); scale = float(np.sum(meas * nar) / np.sum(nar**2))
    r16 = [r for r in rows['broad'] if r[0] == 16][0][3] / [r for r in rows['narrow'] if r[0] == 16][0][3]
    # monodisperse optimum (water)
    dg = np.arange(6, 30, 0.05) * 1e-9; sg = fn(dg)
    res['%g' % (K / 1e3)] = dict(rows=rows, peak_d0=rows['narrow'][i][0], overestimate=round(1 / scale, 2),
                                 ratio16=round(r16, 2), mono_opt_nm=round(float(dg[np.argmax(sg)] * 1e9), 2),
                                 log_rmse=round(float(np.sqrt(np.mean(np.log10(nar / meas)**2))), 3))
    print(K / 1e3, res['%g' % (K / 1e3)]['peak_d0'], res['%g' % (K / 1e3)]['overestimate'], res['%g' % (K / 1e3)]['ratio16'],
          res['%g' % (K / 1e3)]['mono_opt_nm'], res['%g' % (K / 1e3)]['log_rmse'], [r[3] for r in rows['narrow']])
json.dump(res, open('khandhar_K.json', 'w'), indent=1)
