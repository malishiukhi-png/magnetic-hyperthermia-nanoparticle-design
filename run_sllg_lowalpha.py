"""Additional sLLG runs appended to sllg_runs.jsonl: the 11 nm sample of Cabrera et al. (2018) with the anisotropy and damping of
their loop simulations, and the size dependence at 15 kA/m and 300 kHz with alpha = 0.0015. Needs the compiled sllg program in this folder."""
import json, os, subprocess, threading, queue, zlib

HERE = os.path.dirname(os.path.abspath(__file__))
BIN = os.path.join(HERE, 'sllg'); OUT = os.path.join(HERE, 'sllg_runs.jsonl')


def key(j):
    return json.dumps({k: j[k] for k in sorted(j)}, sort_keys=True)


def run(j):
    seed = zlib.crc32(key(j).encode())
    r = subprocess.run([BIN, '%.6f' % j['d'], '%.6f' % j['sigma'], '%.8g' % j['alpha'], '%.8g' % j['H'], '%.8g' % j['f'], '%.8g' % j['Ms'], '%.8g' % j['K'],
                        str(j['N']), str(j['ncyc']), str(seed), '%.6f' % j['T']], capture_output=True, text=True)
    v = r.stdout.split()
    return dict(j, sar=float(v[0]) / 1000, se=float(v[1]) / 1000, nper=int(v[2]), Nreal=int(v[3]))


cab = dict(group='cabrera', d=10.3, sigma=0.352, sample='11nm', H=20e3, f=100e3, Ms=360e3, N=2048, ncyc=4, T=298.15)
low = dict(group='lowalpha', sigma=0.0, alpha=0.0015, H=15e3, f=300e3, Ms=446e3, K=13e3, N=1024, ncyc=4, T=310.15)
jobs = [dict(cab, alpha=a, K=K) for K in (3400.0, 2900.0) for a in (0.0015, 0.1, 1.0)] + [dict(cab, alpha=0.0015, K=4700.0), dict(cab, alpha=0.0015, K=13000.0)]
jobs += [dict(low, d=float(d)) for d in range(14, 34, 2)]
q = queue.Queue(); [q.put(j) for j in jobs]
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


ts = [threading.Thread(target=worker) for _ in range(2)]
[t.start() for t in ts]; [t.join() for t in ts]
