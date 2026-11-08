"""Low-damping check: sLLG SAR at the conditions of Khandhar et al. (373 kHz, 14 kA/m, Ms = 388.5 kA/m) with alpha = 0.0015,
the damping used in the hysteresis-loop simulations of Cabrera et al. (2018). Needs the compiled sllg program in this folder.
Writes llg_khandhar_a0.0015.json (d nm, SAR W/g, s.e.) and exp_compare_lowalpha.json."""
import json, os, subprocess, zlib
from concurrent.futures import ThreadPoolExecutor
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
BIN = os.environ.get('SLLG_BIN', os.path.join(HERE, 'sllg'))
ALPHA, H, F, MS, K, T = 0.0015, 14e3, 373e3, 75.0 * 5180.0, 13e3, 310.15
NARROW = [(12, 0.01, 23.5), (13, 0.10, 54.1), (14, 0.21, 87.3), (15, 0.23, 100.2), (16, 0.16, 141.0), (18, 0.34, 43.3)]


def run(d):
    seed = zlib.crc32(('khandhar-lowalpha-%g' % d).encode())
    v = subprocess.run([BIN, '%.6f' % d, '0', '%.8g' % ALPHA, '%.8g' % H, '%.8g' % F, '%.8g' % MS, '%.8g' % K, '1024', '4', str(seed), '%.6f' % T],
                       capture_output=True, text=True).stdout.split()
    return [d, float(v[0]) / 1000, float(v[1]) / 1000]


grid = (8, 10, 12, 13, 14, 15, 16, 17, 18, 19, 20, 22, 24, 27, 30)
with ThreadPoolExecutor(2) as ex:
    out = list(ex.map(run, grid))
json.dump(out, open(os.path.join(HERE, 'llg_khandhar_a0.0015.json'), 'w'))
L = np.array(out); dg, sg = L[:, 0] * 1e-9, L[:, 1]


def lognormal(d0, s, n=401):
    f = lambda d: np.interp(d, dg, sg, left=0.0, right=sg[-1])
    if s < 0.02:
        return float(f(np.array([d0]))[0])
    x = np.linspace(np.log(d0) - 4 * s, np.log(d0) + 4 * s, n); d = np.exp(x)
    w = np.exp(-(x - np.log(d0))**2 / (2 * s * s)) * d**3
    return float(np.sum(w * f(d)) / np.sum(w))


rows = [dict(d0=d0, sigma=s, measured=m, model=round(lognormal(d0 * 1e-9, s), 1)) for d0, s, m in NARROW]
meas = np.array([r['measured'] for r in rows]); pred = np.array([r['model'] for r in rows])
res = dict(alpha=ALPHA, narrow=rows, peak_d0=rows[int(np.argmax(pred))]['d0'], peak_value=float(pred.max()), scale_to_measured=round(float(np.sum(meas * pred) / np.sum(pred**2)), 3))
json.dump(res, open(os.path.join(HERE, 'exp_compare_lowalpha.json'), 'w'), indent=1)
print(json.dumps(out)); print(json.dumps(res))
