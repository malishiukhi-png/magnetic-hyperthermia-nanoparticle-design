"""Queue of sLLG runs (two workers). Appends one JSON line per finished job; finished jobs are skipped on restart."""
import json, subprocess, sys, os, threading, queue, zlib
import numpy as np
HERE=os.path.dirname(os.path.abspath(__file__)); BIN=os.path.join(HERE,'sllg'); OUT=os.path.join(HERE,'sllg_runs.jsonl')
Ms0=446e3; K0=13e3; T0=310.15
J=[]
def job(group,d,sig,a,H,f,Ms=Ms0,K=K0,N=1024,nc=4,T=T0,**meta):
    J.append(dict(group=group,d=float(d),sigma=float(sig),alpha=float(a),H=float(H),f=float(f),Ms=float(Ms),K=float(K),N=int(N),ncyc=int(nc),T=float(T),**meta))
# 1) field scan along the Hergt-Dutz limit / clinical point
fields=[(18e3,100e3),(18e3,5e9/18e3),(12.5e3,400e3),(10e3,500e3),(5e3,1000e3)]
for H,f in fields:
    for a in (0.1,1.0):
        for d in (12,14,16,18,20,22,24,26,28):
            job('field',d,0,a,H,f)
# 2) comparison with cell data (Cabrera et al.): 100 kHz, 20 kA/m, room temperature, TEM log-normal ensembles
for name,d0,s in (('11nm',10.3,0.352),('21nm',20.2,0.28)):
    for a in (0.1,1.0):
        for K in (13e3,4.7e3):
            job('cabrera',d0,s,a,20e3,100e3,Ms=360e3,K=K,N=2048,T=298.15,sample=name)
# 3) polydispersity at the reference field (direct log-normal ensembles)
for s in (0.1,0.2):
    for a in (0.1,1.0):
        for d in (14,16,18,20,22,24,26,28):
            job('poly',d,s,a,15e3,300e3,N=2048)
# 4) anisotropy at the reference field
for K,ds in ((9e3,(16,18,20,22,24,26,28,30)),(20e3,(12,14,16,18,20,22,24,26)),(30e3,(10,12,14,16,18,20,22,24))):
    for a in (0.1,1.0):
        for d in ds:
            job('K',d,0,a,15e3,300e3,K=K)
# 5) Monte Carlo over the magnetic inputs (nonlinear SAR); thermal inputs are added analytically afterwards
rng=np.random.default_rng(2026); NMC=120
S=dict(Ms=rng.uniform(350e3,480e3,NMC),K=np.exp(rng.uniform(np.log(8e3),np.log(20e3),NMC)),alpha=np.exp(rng.uniform(np.log(0.1),np.log(1.0),NMC)),sigma=rng.uniform(0,0.15,NMC))
json.dump({k:v.tolist() for k,v in S.items()},open(os.path.join(HERE,'mc_inputs.json'),'w'))
for i in range(NMC):
    for d in (14,16,18,20,22,24,26,28):
        job('mc',d,S['sigma'][i],S['alpha'][i],15e3,300e3,Ms=S['Ms'][i],K=S['K'][i],N=192,mc=i)
def key(j): return json.dumps({k:j[k] for k in sorted(j) if k not in('sar','se','nper','Nreal')},sort_keys=True)
done=set()
if os.path.exists(OUT):
    for l in open(OUT):
        done.add(key(json.loads(l)))
q=queue.Queue(); [q.put(j) for j in J if key(j) not in done]
print('jobs',len(J),'todo',q.qsize(),flush=True)
lock=threading.Lock()
def work():
    while True:
        try: j=q.get_nowait()
        except queue.Empty: return
        seed=zlib.crc32(key(j).encode())
        r=subprocess.run([BIN,'%.6f'%j['d'],'%.6f'%j['sigma'],'%.8g'%j['alpha'],'%.8g'%j['H'],'%.8g'%j['f'],'%.8g'%j['Ms'],'%.8g'%j['K'],str(j['N']),str(j['ncyc']),str(seed),'%.6f'%j['T']],capture_output=True,text=True)
        v=r.stdout.split()
        j2=dict(j,sar=float(v[0])/1000,se=float(v[1])/1000,nper=int(v[2]),Nreal=int(v[3]))
        with lock:
            open(OUT,'a').write(json.dumps(j2)+'\n')
ts=[threading.Thread(target=work) for _ in range(2)]
[t.start() for t in ts]; [t.join() for t in ts]
print('finished',flush=True)
