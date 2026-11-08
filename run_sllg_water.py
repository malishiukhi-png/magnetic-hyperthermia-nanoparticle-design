"""sLLG runs with Brownian rotation in a liquid (sllg_rot.c) and matching immobilized runs -> sllg_water_runs.jsonl.
Cases: the narrow series of Tong et al. (ACS Nano 11 (2017) 6808; 325 kHz, 9.35 and 20.7 kA/m, water), the narrow series of Khandhar et al.,
and the two samples of Cabrera et al. in water. Finished cases are skipped. Needs the compiled sllg_rot program in this folder."""
import json, os, subprocess, threading, queue, zlib

HERE = os.path.dirname(os.path.abspath(__file__))
BIN = os.environ.get('SLLG_ROT_BIN', os.path.join(HERE, 'sllg_rot')); OUT = os.path.join(HERE, 'sllg_water_runs.jsonl')
KEYS = ('group', 'name', 'd', 'sigma', 'alpha', 'H', 'f', 'Ms', 'K', 'N', 'ncyc', 'T', 'shell', 'eta')
ETA25 = 0.89e-3
FE2OX = 0.7236            # g Fe per g Fe3O4


def key(j):
    return json.dumps({k: j[k] for k in KEYS}, sort_keys=True)


def run(j):
    seed = zlib.crc32(key(j).encode())
    a = [BIN, '%.6f' % j['d'], '%.6f' % j['sigma'], '%.8g' % j['alpha'], '%.8g' % j['H'], '%.8g' % j['f'], '%.8g' % j['Ms'], '%.8g' % j['K'], str(j['N']), str(j['ncyc']), str(seed),
         '%.6f' % j['T'], '%.4f' % j['shell'], '%.8g' % j['eta']]
    v = subprocess.run(a, capture_output=True, text=True).stdout.split()
    return dict(j, sar=float(v[0]) / 1000, se=float(v[1]) / 1000, nper=int(v[2]), Nreal=int(v[3]))


jobs = []
# Tong et al., Table 1: core diameter mean +- s.d. (nm), Ms (emu/g Fe), hydrodynamic diameter (nm)
TONG = [(14.7, 1.4, 117.3, 35.8), (19.1, 1.6, 97.8, 32.4), (24.9, 2.4, 115.3, 37.2), (33.3, 4.0, 119.4, 48.2), (40.4, 6.3, 116.1, 52.4)]
import math
for m, sd, ms, dh in TONG:
    s = math.sqrt(math.log(1 + (sd / m)**2)); d0 = m / math.exp(s * s / 2); Ms = ms * FE2OX * 5180.0
    base = dict(group='tong', name='%.1f' % m, d=d0, sigma=s, alpha=0.1, f=325e3, Ms=Ms, K=13.5e3, N=512, T=298.15, shell=(dh - m) / 2)
    for H in (9.35e3, 20.7e3):
        jobs.append(dict(base, H=H, ncyc=12, eta=ETA25))
        jobs.append(dict(base, H=H, ncyc=6, eta=1e12))
# Khandhar et al.: narrow series, 373 kHz, 14 kA/m, Ms = 75 A m2/kg, 5 nm shell (as in the linear-response comparison)
for d0, s in ((12, 0.01), (13, 0.10), (14, 0.21), (15, 0.23), (16, 0.16), (18, 0.34)):
    base = dict(group='khandhar', name=str(d0), d=float(d0), sigma=s, f=373e3, H=14e3, Ms=75.0 * 5180.0, K=13e3, N=512, T=298.15, shell=5.0)
    jobs.append(dict(base, alpha=0.1, ncyc=12, eta=ETA25))
    jobs.append(dict(base, alpha=1.0, ncyc=12, eta=ETA25))
    jobs.append(dict(base, alpha=0.0015, ncyc=12, eta=ETA25))
# Cabrera et al., water: hydrodynamic diameters 50 and 67 nm
for name, m, d0, s, dh in (('11nm', 11.0, 10.3, 0.352, 50.0), ('21nm', 21.0, 20.2, 0.28, 67.0)):
    base = dict(group='cabrera', name=name, d=d0, sigma=s, f=100e3, H=20e3, Ms=360e3, N=512, T=298.15, shell=(dh - m) / 2, ncyc=10, eta=ETA25)
    for K, a in ((4.7e3, 0.1), (13e3, 0.1), (3.4e3, 0.0015)):
        if name == '21nm' and a < 0.01:
            continue
        jobs.append(dict(base, K=K, alpha=a))

# check: infinite viscosity must reproduce the non-rotating program
jobs.append(dict(group='check', name='20', d=20.0, sigma=0.0, alpha=0.1, f=300e3, H=15e3, Ms=446e3, K=13e3, N=512, T=310.15, shell=5.0, ncyc=6, eta=1e12))

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
            print(r['group'], r['name'], 'H', r['H'], 'alpha', r['alpha'], 'K', r['K'], 'eta', r['eta'], '->', round(r['sar'], 1), '+-', round(r['se'], 1), flush=True)


ts = [threading.Thread(target=worker) for _ in range(int(os.environ.get('NPROC', 2)))]
[t.start() for t in ts]; [t.join() for t in ts]
print('done', flush=True)
