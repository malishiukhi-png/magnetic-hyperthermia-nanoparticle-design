"""sLLG runs with explicit dipolar coupling in random clusters (sllg_dip.c) -> sllg_cluster_runs.jsonl.
Finished cases are skipped, so the script can be restarted. Needs the compiled sllg_dip program in this folder."""
import json, os, subprocess, threading, queue, zlib

HERE = os.path.dirname(os.path.abspath(__file__))
BIN = os.environ.get('SLLG_DIP_BIN', os.path.join(HERE, 'sllg_dip')); OUT = os.path.join(HERE, 'sllg_cluster_runs.jsonl')
KEYS = ('group', 'd', 'sigma', 'alpha', 'H', 'f', 'Ms', 'K', 'shell', 'phi', 'ncl', 'nc', 'ncyc', 'T', 'dip')


def key(j):
    return json.dumps({k: j[k] for k in KEYS}, sort_keys=True)


def run(j):
    seed = zlib.crc32(key(j).encode())
    a = [BIN, '%.6f' % j['d'], '%.6f' % j['sigma'], '%.8g' % j['alpha'], '%.8g' % j['H'], '%.8g' % j['f'], '%.8g' % j['Ms'], '%.8g' % j['K'], '%.4f' % j['shell'],
         '%.4f' % j['phi'], str(j['ncl']), str(j['nc']), str(j['ncyc']), str(seed), '%.6f' % j['T'], str(j['dip'])]
    v = subprocess.run(a, capture_output=True, text=True).stdout.split()
    return dict(j, sar=float(v[0]) / 1000, se=float(v[1]) / 1000, nper=int(v[2]), Nreal=int(v[3]), Hdmax=float(v[4]))


D = dict(group='design', sigma=0.0, H=15e3, f=300e3, Ms=446e3, K=13e3, ncl=32, nc=32, ncyc=4, T=310.15)
SIZES = (14.0, 16.0, 18.0, 20.0, 22.0, 24.0, 26.0)
jobs = [dict(D, d=d, alpha=0.1, shell=5.0, phi=0.3, dip=0) for d in SIZES]                                  # reference without coupling
jobs += [dict(D, d=d, alpha=0.1, shell=5.0, phi=0.3, dip=1) for d in SIZES]                                 # 5 nm coating, full size range
jobs += [dict(D, d=d, alpha=0.1, shell=s, phi=0.3, dip=1) for s in (10.0, 15.0) for d in (18.0, 20.0, 22.0, 24.0)]   # thicker coatings
C = dict(group='cabrera', H=20e3, f=100e3, Ms=360e3, ncl=32, nc=16, ncyc=3, T=298.15, shell=1.0, phi=0.3)   # live-cell samples
jobs += [dict(C, sample='21nm', d=20.2, sigma=0.28, K=4.7e3, alpha=0.1, dip=x) for x in (0, 1)]
jobs += [dict(C, sample='11nm', d=10.3, sigma=0.352, K=2.9e3, alpha=0.0015, dip=1)]
jobs += [dict(D, d=22.0, alpha=0.1, shell=5.0, phi=0.1, dip=1)]                                             # looser cluster
jobs += [dict(D, d=22.0, alpha=1.0, shell=10.0, phi=0.3, dip=1)]                                            # higher damping

done = set()
if os.path.exists(OUT):
    done = {key(json.loads(l)) for l in open(OUT)}
q = queue.Queue(); [q.put(j) for j in jobs if key(j) not in done]
lock = threading.Lock()


def worker():
    while True:
        try:
            j = q.get_nowait()
        except queue.Empty:
            return
        r = run(j)
        with lock:
            open(OUT, 'a').write(json.dumps(r) + '\n')
            print(r['group'], r.get('sample', ''), r['d'], 'shell', r['shell'], 'phi', r['phi'], 'alpha', r['alpha'], 'K', r['K'], 'dip', r['dip'], '->', round(r['sar'], 1), '+-', round(r['se'], 1), flush=True)


ts = [threading.Thread(target=worker) for _ in range(int(os.environ.get('NPROC', 2)))]
[t.start() for t in ts]; [t.join() for t in ts]
print('done', flush=True)
