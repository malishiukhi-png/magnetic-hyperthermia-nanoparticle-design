"""Convergence of the stochastic estimate near the switching threshold (18 nm, K = 13 kJ/m3, alpha = 0.1, 15 kA/m, 300 kHz):
independent seeds, number of particles and number of field cycles."""
import json, subprocess
import os
HERE = os.path.dirname(os.path.abspath(__file__)) + '/'
BIN = HERE + 'sllg'; OUT = HERE + 'conv.jsonl'
jobs = [dict(N=1024, ncyc=4, seed=s) for s in (101, 202, 303)] + [dict(N=2048, ncyc=4, seed=404), dict(N=1024, ncyc=8, seed=505)]
for j in jobs:
    v = subprocess.run([BIN, '18', '0', '0.1', '15e3', '300e3', '446e3', '13e3', str(j['N']), str(j['ncyc']), str(j['seed']), '310.15'], capture_output=True, text=True).stdout.split()
    open(OUT, 'a').write(json.dumps(dict(j, sar=float(v[0]) / 1000, se=float(v[1]) / 1000)) + '\n')
open(OUT + '.done', 'w').write('done')
