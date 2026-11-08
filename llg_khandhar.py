"""sLLG SAR of immobilized magnetite on a size grid at the conditions of Khandhar et al. (373 kHz, 14 kA/m),
with their measured saturation magnetization (75 A m2 kg-1 -> 388.5 kA m-1). Usage: python llg_khandhar.py <alpha>"""
import sys, json, numpy as np
sys.path.insert(0, '.')
import llg
alpha = float(sys.argv[1])
MS_EXP = 75.0 * 5180.0
out = []
for d in (8, 10, 12, 13, 14, 15, 16, 17, 18, 19, 20, 22, 24, 27, 30):
    v, e, _ = llg.sar_llg(d * 1e-9, 14e3, 373e3, alpha, N=1000, ncyc=4, seed=int(d * 10 + alpha * 1000), Ms=MS_EXP)
    out.append([d, v / 1000, e / 1000])
    print('khandhar', alpha, d, round(v / 1000, 1), '+-', round(e / 1000, 1), flush=True)
    json.dump(out, open('llg_khandhar_a%g.json' % alpha, 'w'))
