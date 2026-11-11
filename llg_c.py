"""sLLG SAR with a compiled C kernel (llg_kernel.c): same physics and protocol as llg.py (numba), plus cubic anisotropy.
Compiled automatically on first import with: gcc -O3 -march=native -shared -fPIC -o llg_kernel.so llg_kernel.c -lm
sar_llg(d, H, f, alpha, aniso='uniaxial'|'cubic', K=..., K1=...) -> (SAR W/kg, standard error W/kg, steps per period)"""
import ctypes, os, sys, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import P, T, kB, mu0

GAMMA = 1.760859e11
_HERE = os.path.dirname(os.path.abspath(__file__))
if not os.path.exists(os.path.join(_HERE, 'llg_kernel.so')):
    import subprocess
    subprocess.check_call(['gcc', '-O3', '-march=native', '-shared', '-fPIC', '-o', os.path.join(_HERE, 'llg_kernel.so'),
                           os.path.join(_HERE, 'llg_kernel.c'), '-lm'])
_lib = ctypes.CDLL(os.path.join(_HERE, 'llg_kernel.so'))
_D = ctypes.POINTER(ctypes.c_double)
_lib.run.argtypes = [ctypes.c_int, ctypes.c_int, _D, _D, ctypes.c_double, ctypes.c_double, ctypes.c_double,
                     ctypes.c_double, ctypes.c_double, ctypes.c_long, ctypes.c_double, ctypes.c_double, ctypes.c_long,
                     ctypes.c_uint64, _D]
_lib.run.restype = None


def _random_rotations(rng, N):
    """Uniformly random rotation matrices (columns = crystal axes in the lab frame)."""
    q = rng.normal(size=(N, 4)); q /= np.linalg.norm(q, axis=1)[:, None]
    w, x, y, z = q.T
    R = np.empty((N, 3, 3))
    R[:, 0, 0] = 1 - 2 * (y * y + z * z); R[:, 0, 1] = 2 * (x * y - z * w); R[:, 0, 2] = 2 * (x * z + y * w)
    R[:, 1, 0] = 2 * (x * y + z * w); R[:, 1, 1] = 1 - 2 * (x * x + z * z); R[:, 1, 2] = 2 * (y * z - x * w)
    R[:, 2, 0] = 2 * (x * z - y * w); R[:, 2, 1] = 2 * (y * z + x * w); R[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return R


def sar_llg(d, H, f, alpha, N=1000, ncyc=4, seed=1, dt=None, Ms=P['Ms'], K=P['K'], K1=-11e3, rho=P['rho'],
            aniso='uniaxial', T_K=T):
    V = np.pi * d**3 / 6
    rng = np.random.default_rng(seed)
    if aniso == 'uniaxial':
        mode = 0; HK = 2 * K / (mu0 * Ms); Hscale = HK
        u = rng.uniform(-1, 1, N); ph = rng.uniform(0, 2 * np.pi, N); s = np.sqrt(1 - u**2)
        n = np.stack([s * np.cos(ph), s * np.sin(ph), u], 1)
        Rall = np.zeros((N, 9)); Rall[:, :3] = n
        m0 = n * np.where(rng.random(N) < 0.5, -1.0, 1.0)[:, None]
    else:
        mode = 1; HK = 2 * K1 / (mu0 * Ms); Hscale = abs(HK)
        R = _random_rotations(rng, N)
        Rall = np.ascontiguousarray(np.transpose(R, (0, 2, 1)).reshape(N, 9))   # rows of Rall[i] = crystal axes
        sgn = np.where(rng.random((N, 3)) < 0.5, -1.0, 1.0) / np.sqrt(3)          # start in a random <111> minimum
        m0 = np.einsum('nij,nj->ni', R, sgn) if K1 < 0 else R[:, :, 0] * sgn[:, :1] * np.sqrt(3)
    if dt is None:
        dt = 2 * np.pi / (GAMMA * mu0 * (Hscale + H)) / 40
    nper = int(np.ceil(1 / f / dt)); dt = 1 / f / nper
    Hth_sd = np.sqrt(2 * alpha * kB * T_K / (GAMMA * mu0**2 * Ms * V * dt))
    nsteps = nper * ncyc; rec = max(nper // 2000, 1)
    out = np.zeros(nsteps // rec + 2)
    Rc = np.ascontiguousarray(Rall, dtype=float); mc = np.ascontiguousarray(m0, dtype=float).ravel()
    _lib.run(N, mode, Rc.ctypes.data_as(_D), mc.ctypes.data_as(_D), HK, Hth_sd, H, 2 * np.pi * f, dt, nsteps, alpha,
             GAMMA * mu0, rec, int(seed), out.ctypes.data_as(_D))
    mz = out[:nsteps // rec + 1]
    tt = np.arange(len(mz)) * rec / nper
    Hc = H * np.cos(2 * np.pi * tt)
    areas = []
    for c in range(1, ncyc):
        sl = (tt >= c - 1e-9) & (tt <= c + 1 + 1e-9)
        areas.append(-mu0 * Ms * np.trapezoid(mz[sl], Hc[sl]))
    areas = np.array(areas)
    return areas.mean() * f / rho, areas.std(ddof=1) * f / rho / np.sqrt(len(areas)), nper
