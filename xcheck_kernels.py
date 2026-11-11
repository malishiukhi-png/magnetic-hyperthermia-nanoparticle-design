"""Cross-check of the sllg.c kernel against the separately written llg_kernel.c (this folder) for K = 23 kJ/m3,
alpha = 1, 15 kA/m, 300 kHz (16 and 18 nm). Compare with k23.jsonl. -> xcheck_k23.json"""
import sys, os, json
HERE = os.path.dirname(os.path.abspath(__file__)) + '/'
sys.path.insert(0, HERE)
from llg_c import sar_llg
out = []
for d in (16e-9, 18e-9):
    s, e, n = sar_llg(d, 15e3, 300e3, 1.0, K=23e3, seed=11)
    out.append([d * 1e9, s / 1000, e / 1000]); print(out[-1], flush=True)
json.dump(out, open(HERE + 'xcheck_k23.json', 'w'))
