"""Time-step check of the sLLG simulation at 18 nm, 15 kA/m, 300 kHz, alpha = 1 -> llg_conv.json"""
import sys, json, numpy as np
sys.path.insert(0, '.')
import llg
from model import mu0, P
HK = 2 * P['K'] / (mu0 * P['Ms'])
out = {}
for fac in (1.0, 0.5):
    dt = 2 * np.pi / (llg.GAMMA * mu0 * (HK + 15e3)) / 40 * fac
    v, e, n = llg.sar_llg(18e-9, 15e3, 300e3, 1.0, N=1000, ncyc=4, seed=181, dt=dt)
    out[str(fac)] = [v / 1000, e / 1000]
    print('llgconv', fac, round(v / 1000, 1), '+-', round(e / 1000, 1), flush=True)
json.dump(out, open('llg_conv.json', 'w'), indent=1)
