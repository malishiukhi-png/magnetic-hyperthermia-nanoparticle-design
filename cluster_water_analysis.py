"""Summarizes the runs with dipolar coupling and with Brownian rotation -> cluster_water_results.json."""
import json, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)) + '/'
kB = 1.380649e-23; MU0 = 4e-7 * np.pi; FE2OX = 0.7236
C = [json.loads(l) for l in open(HERE + 'sllg_cluster_runs.jsonl')]
W = [json.loads(l) for l in open(HERE + 'sllg_water_runs.jsonl')]


def opt(ds, ys):
    ds, ys = np.asarray(ds, float), np.asarray(ys, float); i = int(np.argmax(ys))
    if i == 0 or i == len(ds) - 1:
        return float(ds[i]), float(ys[i])
    c = np.polyfit(ds[i - 1:i + 2], np.log(np.maximum(ys[i - 1:i + 2], 1e-9)), 2); dm = -c[1] / (2 * c[0])
    return float(dm), float(np.exp(np.polyval(c, dm)))


def lam(d, t, Ms=446e3, T=310.15):
    m = Ms * np.pi * (d * 1e-9)**3 / 6
    return float(MU0 * m * m / (4 * np.pi * ((d + 2 * t) * 1e-9)**3 * kB * T))


def sel(rows, **kw):
    return sorted([r for r in rows if all(r.get(k) == v for k, v in kw.items())], key=lambda r: r['d'])


R = {'cluster': {}, 'water': {}}
ref = {r['d']: r for r in sel(C, group='design', dip=0, alpha=0.1)}
R['cluster']['reference'] = {'%.0f' % d: [round(r['sar'], 1), round(r['se'], 1)] for d, r in ref.items()}
R['cluster']['reference_opt'] = [round(x, 1) for x in opt(sorted(ref), [ref[d]['sar'] for d in sorted(ref)])]
for s in (5.0, 10.0, 15.0):
    r = sel(C, group='design', dip=1, alpha=0.1, shell=s, phi=0.3)
    e = {'%.0f' % x['d']: dict(sar=round(x['sar'], 1), se=round(x['se'], 1), ratio=round(x['sar'] / ref[x['d']]['sar'], 2), lam=round(lam(x['d'], s), 2)) for x in r}
    R['cluster']['shell%d' % s] = e
    if r:
        R['cluster']['shell%d_opt' % s] = [round(v, 1) for v in opt([x['d'] for x in r], [x['sar'] for x in r])]
for name, kw in (('phi01_22', dict(group='design', dip=1, alpha=0.1, shell=5.0, phi=0.1, d=22.0)), ('alpha1_22', dict(group='design', dip=1, alpha=1.0, shell=10.0, d=22.0))):
    r = sel(C, **kw)
    if r:
        R['cluster'][name] = dict(sar=round(r[0]['sar'], 1), se=round(r[0]['se'], 1), ratio=round(r[0]['sar'] / ref[22.0]['sar'], 2))
R['cluster']['cabrera'] = [dict(sample=r['sample'], K=r['K'] / 1e3, alpha=r['alpha'], dip=r['dip'], sar=round(r['sar'], 1), se=round(r['se'], 1)) for r in sel(C, group='cabrera')]

R['water']['tong'] = {}
for r in sel(W, group='tong'):
    e = R['water']['tong'].setdefault(r['name'], {})
    key = '%s_%s' % ('rot' if r['eta'] < 1 else 'imm', '9' if r['H'] < 1e4 else '21')
    e[key] = [round(r['sar'] / FE2OX, 0), round(r['se'] / FE2OX, 0)]
R['water']['khandhar'] = {}
MEAS = {'12': 23.5, '13': 54.1, '14': 87.3, '15': 100.2, '16': 141.0, '18': 43.3}
for r in sel(W, group='khandhar'):
    e = R['water']['khandhar'].setdefault(r['name'], dict(measured=MEAS[r['name']]))
    e['a%g' % r['alpha']] = round(r['sar'], 1)
for a in ('a0.1', 'a1', 'a0.0015'):
    k = sorted(R['water']['khandhar'], key=float); y = [R['water']['khandhar'][x].get(a, 0.0) for x in k]; m = [MEAS[x] for x in k]
    R['water']['khandhar_' + a] = dict(peak_d0=k[int(np.argmax(y))], scale=round(float(np.dot(m, y) / np.dot(y, y)), 3))
R['water']['cabrera'] = [dict(sample=r['name'], K=r['K'] / 1e3, alpha=r['alpha'], sar=round(r['sar'], 1), se=round(r['se'], 1)) for r in sel(W, group='cabrera')]
r = sel(W, group='check')
if r:
    R['water']['check_20nm'] = [round(r[0]['sar'], 1), round(r[0]['se'], 1)]
json.dump(R, open(HERE + 'cluster_water_results.json', 'w'), indent=1)
print(json.dumps(R, indent=0)[:3500])
