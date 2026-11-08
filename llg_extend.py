"""Add core sizes to an existing sLLG result file. Usage: python llg_extend.py <alpha> <H_kA/m> <f_kHz> <sizes>"""
import sys, json
sys.path.insert(0, '.')
import llg
a, H, f = float(sys.argv[1]), float(sys.argv[2]) * 1e3, float(sys.argv[3]) * 1e3
fn = 'llg_a%g.json' % a
out = json.load(open(fn))
for d in [float(x) for x in sys.argv[4].split(',')]:
    v, e, _ = llg.sar_llg(d * 1e-9, H, f, a, seed=int(d * 10 + a * 1000))
    out.append([d, v / 1000, e / 1000, llg.tau0_brown(d * 1e-9, a)])
    out.sort(key=lambda r: r[0])
    json.dump(out, open(fn, 'w'))
    print('ext', a, d, round(v / 1000, 1), '+-', round(e / 1000, 1), flush=True)
