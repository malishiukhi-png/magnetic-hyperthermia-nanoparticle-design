"""New sLLG runs for the revision (C kernel, llg_c.py). Usage: python llg_jobs.py <job> ; results -> llg_<job>.json
Jobs:
  a001      uniaxial K = 13 kJ/m3, alpha = 0.01, 15 kA/m, 300 kHz (damping sensitivity)
  cubic1    cubic K1 = -11 kJ/m3 (bulk magnetite), alpha = 1, 15 kA/m, 300 kHz
  cubic01   cubic K1 = -11 kJ/m3, alpha = 0.1, 15 kA/m, 300 kHz
  tong_u    uniaxial K = 13, alpha = 1, 9.35 kA/m, 325 kHz (conditions of Tong et al. 2017)
  tong_c    cubic K1 = -11, alpha = 1, 9.35 kA/m, 325 kHz
  kh_c      cubic K1 = -11, alpha = 1, 14 kA/m, 373 kHz, Ms = 388.5 kA/m (conditions of Khandhar et al. 2012)
  kh_K23    uniaxial K = 23 kJ/m3, alpha = 1, 14 kA/m, 373 kHz, Ms = 388.5 kA/m"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import llg_c
MS_KH = 75.0 * 5180.0
J = {
    'a001':    dict(aniso='uniaxial', alpha=0.01, H=15e3, f=300e3, ds=[14, 16, 17, 18, 19, 20, 22, 24, 26]),
    'cubic1':  dict(aniso='cubic', alpha=1.0, H=15e3, f=300e3, ds=[14, 16, 18, 20, 22, 24, 26, 30, 35, 40]),
    'cubic01': dict(aniso='cubic', alpha=0.1, H=15e3, f=300e3, ds=[40]),
    'cubic1x': dict(aniso='cubic', alpha=1.0, H=15e3, f=300e3, ds=[45, 50], out='cubic1'),
    'tong_u':  dict(aniso='uniaxial', alpha=1.0, H=9.35e3, f=325e3, ds=[11, 15, 17, 19, 21, 23, 25, 33, 40]),
    'tong_c':  dict(aniso='cubic', alpha=1.0, H=9.35e3, f=325e3, ds=[11, 15, 19, 25, 33, 40]),
    'kh_c':    dict(aniso='cubic', alpha=1.0, H=14e3, f=373e3, Ms=MS_KH, ds=[16, 24, 30, 35]),
    'kh_K23':  dict(aniso='uniaxial', alpha=1.0, H=14e3, f=373e3, Ms=MS_KH, K=23e3, ds=[10, 12, 13, 14, 15, 16, 17, 18, 20, 24]),
}
for job in sys.argv[1:]:
    p = J[job]; fn = 'llg_%s.json' % p.get('out', job)
    out = json.load(open(fn)) if os.path.exists(fn) else []
    done = {r[0] for r in out}
    kw = {k: p[k] for k in ('Ms', 'K') if k in p}
    for d in p['ds']:
        if d in done:
            continue
        v, e, n = llg_c.sar_llg(d * 1e-9, p['H'], p['f'], p['alpha'], aniso=p['aniso'],
                                seed=int(d * 10 + p['alpha'] * 1000) + 11, **kw)
        out.append([d, v / 1000, e / 1000]); out.sort()
        json.dump(out, open(fn, 'w'))
        print(job, d, round(v / 1000, 1), '+-', round(e / 1000, 1), 'nper', n, flush=True)
