"""Monte Carlo propagation of parameter uncertainty (immobilized particles, H = 15 kA/m, f = 300 kHz)."""
import json, numpy as np
from scipy.stats import spearmanr
from model import sar, bioheat
rng = np.random.default_rng(1); N = 2000
S = dict(Ms=rng.uniform(350e3, 480e3, N), K=np.exp(rng.uniform(np.log(8e3), np.log(20e3), N)),
         tau0=np.exp(rng.uniform(np.log(1e-10), np.log(1e-9), N)), sigma=rng.uniform(0.0, 0.15, N),
         k=rng.uniform(0.4, 0.6, N), wb=np.exp(rng.uniform(np.log(0.25e-3), np.log(1e-3), N)), R=rng.uniform(5e-3, 15e-3, N))
d = np.linspace(7e-9, 30e-9, 461); rc = 4.0e6
out = dict(d_opt=[], sar=[], dose=[], center=[], loss2=[])
for i in range(N):
    kw = dict(Ms=S['Ms'][i], tau0=S['tau0'][i])
    s = sar(d, 15e3, 300e3, eta=np.inf, K=S['K'][i], sigma=S['sigma'][i], **kw)
    j = int(np.argmax(s)); dopt = d[j]; smax = s[j]
    s2 = sar(np.array([dopt + 2e-9]), 15e3, 300e3, eta=np.inf, K=S['K'][i], sigma=S['sigma'][i], **kw)[0]
    W = rc * S['wb'][i]; Rt = S['R'][i]
    th = bioheat(np.array([0.0, Rt]), Rt, 1e6, W, W, S['k'][i])      # response to Q = 1 MW/m3
    Q = 5.0 / th[1] * 1e6                                              # boundary at 42 C
    out['d_opt'].append(dopt * 1e9); out['sar'].append(smax); out['dose'].append(Q / (smax * 1000)); out['center'].append(37 + th[0] * Q / 1e6)
    out['loss2'].append(100 * (1 - s2 / smax))
R = {}
for k, v in out.items():
    v = np.array(v); R[k] = dict(median=round(float(np.median(v)), 2), p5=round(float(np.percentile(v, 5)), 2), p95=round(float(np.percentile(v, 95)), 2))
# global sensitivity: Spearman rank correlation of each input with each output
R['sens'] = {k: {p: round(float(spearmanr(S[p], np.array(v))[0]), 2) for p in S} for k, v in out.items()}
R['N'] = N
json.dump(R, open('uncertainty.json', 'w'), indent=1); print(json.dumps(R, indent=1))
