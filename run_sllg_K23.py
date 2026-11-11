"""sLLG check of the v5_3 manuscript: optimum core size at clinical fields for the best-fitting effective anisotropy K = 23 kJ/m3."""
import json, subprocess, threading, queue, zlib
import os
HERE = os.path.dirname(os.path.abspath(__file__)) + '/'
BIN = HERE + 'sllg'; OUT = HERE + 'k23.jsonl'
jobs = []
for a in (1.0, 0.1):
    for d in range(14, 23):
        jobs.append(dict(d=float(d), alpha=a, H=15e3, f=300e3))
for d in range(15, 23):
    jobs.append(dict(d=float(d), alpha=1.0, H=18e3, f=278e3))
for d in range(16, 25, 1):
    jobs.append(dict(d=float(d), alpha=1.0, H=18e3, f=100e3))
q = queue.Queue(); [q.put(j) for j in jobs]; lock = threading.Lock()
def w():
    while True:
        try: j = q.get_nowait()
        except queue.Empty: return
        seed = zlib.crc32(json.dumps(j, sort_keys=True).encode())
        v = subprocess.run([BIN, '%.4f' % j['d'], '0', '%g' % j['alpha'], '%g' % j['H'], '%g' % j['f'], '446e3', '23e3', '1024', '4', str(seed), '310.15'], capture_output=True, text=True).stdout.split()
        r = dict(j, K=23e3, sar=float(v[0]) / 1000, se=float(v[1]) / 1000)
        with lock:
            open(OUT, 'a').write(json.dumps(r) + '\n')
ts = [threading.Thread(target=w) for _ in range(2)]; [t.start() for t in ts]; [t.join() for t in ts]
open(OUT + '.done', 'w').write('done')
