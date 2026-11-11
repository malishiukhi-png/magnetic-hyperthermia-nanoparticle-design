"""Validation of the C sLLG kernel against published numba results (llg_a1.json, llg_a0.1.json) and timing."""
import sys, time, json, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import llg_c
for a, d in ((1.0, 18), (0.1, 20), (1.0, 16)):
    t = time.time()
    v, e, n = llg_c.sar_llg(d * 1e-9, 15e3, 300e3, a, seed=int(d * 10 + a * 1000) + 7)
    ref = [r for r in json.load(open('llg_a%g.json' % a)) if r[0] == d][0]
    print(f'alpha={a} d={d}: C {v/1000:.1f} +- {e/1000:.1f} | numba {ref[1]:.1f} +- {ref[2]:.1f} | {time.time()-t:.0f} s, nper={n}', flush=True)
