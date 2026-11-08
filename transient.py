"""Transient Pennes bioheat in a spherical tumor, thermal dose (CEM43) and non-uniform nanoparticle loading.
Writes transient.json and fig_transient.png (Figure 8)."""
import numpy as np, sys, json, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, '.')
from model import *
from scipy.sparse import diags
from scipy.sparse.linalg import splu, spsolve
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams.update({'font.family': 'Liberation Sans', 'font.size': 10, 'axes.labelweight': 'bold',
                     'axes.linewidth': 0.8, 'xtick.direction': 'in', 'ytick.direction': 'in',
                     'xtick.top': True, 'ytick.right': True, 'xtick.minor.visible': True, 'ytick.minor.visible': True,
                     'legend.frameon': False, 'legend.fontsize': 8, 'savefig.dpi': 300, 'lines.linewidth': 1.8})
C = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4']
INK = '#0b0b0b'; MUTED = '#898781'; CM = 1 / 2.54

Rt = 10e-3; kt = 0.5; rc = 4.0e6      # tumor radius, conductivity, rho_b c_b (as in figures.py)
RHOC = 3.6e6                          # volumetric heat capacity of tissue (J m-3 K-1), assumed
WB = 0.5e-3                           # perfusion (s-1), reference
ROUT, NR = 60e-3, 600
r = (np.arange(NR) + 0.5) * ROUT / NR
dr = ROUT / NR
rf = np.arange(NR + 1) * dr
vol = (rf[1:]**3 - rf[:-1]**3) / 3
inside = r <= Rt


def operator(wb):
    G = kt * rf[1:-1]**2 / dr
    main = np.zeros(NR); main[:-1] += G; main[1:] += G
    main[-1] += kt * rf[-1]**2 / (dr / 2)          # theta = 0 at r = ROUT
    main += rc * wb * vol
    return diags([-G, main, -G], [-1, 0, 1], format='csc')


def steady(Q, wb=WB):
    return spsolve(operator(wb), Q * vol)


def transient(Q, wb=WB, t_heat=3600.0, t_cool=1200.0, dt=2.0):
    Kop = operator(wb); Cm = diags(RHOC * vol / dt)
    lu = splu((Cm + 0.5 * Kop).tocsc()); Bm = Cm - 0.5 * Kop
    n_h = int(t_heat / dt); n_c = int(t_cool / dt)
    th = np.zeros(NR); out = [th.copy()]
    b = Q * vol
    for k in range(n_h + n_c):
        th = lu.solve(Bm @ th + (b if k < n_h else 0 * b))
        out.append(th.copy())
    return np.arange(n_h + n_c + 1) * dt, np.array(out)


def cem43(Tc, dt):
    Rb = np.where(Tc >= 43.0, 0.5, 0.25)
    return np.sum(Rb**(43.0 - Tc), axis=0) * dt / 60.0


def vfrac(mask):
    return float(np.sum(vol[inside & mask]) / np.sum(vol[inside]))


def t90(Tprofile):
    """Temperature exceeded by 90 % of the tumor volume."""
    Ti = Tprofile[inside]; vi = vol[inside]
    o = np.argsort(Ti)[::-1]; cum = np.cumsum(vi[o]) / vi.sum()
    return float(Ti[o][np.searchsorted(cum, 0.9)])


iR = np.argmin(abs(r - Rt)); iB = np.where(inside)[0][-1]
res = {}

# ---- check against the analytical steady solution (model.bioheat) --------------------------------
Qu = np.where(inside, 1e6, 0.0)
fd = steady(Qu)
an = bioheat(r, Rt, 1e6, rc * WB, rc * WB, kt)
res['fd_vs_analytic_maxrelerr'] = float(np.max(np.abs(fd - an)) / an.max())

# ---- uniform loading, dose that brings the boundary to 42 C at steady state ----------------------
Qedge = 5.0 / fd[iB] * 1e6
Q = Qu / 1e6 * Qedge
t, th = transient(Q)
T = 37 + th
dtv = t[1] - t[0]
nh = int(3600 / dtv)
def reach(i, level):
    k = np.argmax(T[:, i] >= level); return float(t[k] / 60) if T[:, i].max() >= level else None
th_ss = steady(Q)
res['uniform'] = dict(Q_MWm3=round(Qedge / 1e6, 4),
                      t90pct_center_min=round(float(t[np.argmax(th[:, 0] >= 0.9 * th_ss[0])] / 60), 1),
                      t90pct_edge_min=round(float(t[np.argmax(th[:, iB] >= 0.9 * th_ss[iB])] / 60), 1),
                      t_center_42_min=reach(0, 42.0), t_edge_41_min=reach(iB, 41.0))
for tm in (30, 60):
    n = int(tm * 60 / dtv)
    cem = cem43(T[:n + 1], dtv)
    res['uniform']['cem_%d' % tm] = dict(center=round(float(cem[0]), 1), edge=round(float(cem[iB]), 2),
                                        frac_ge10=round(100 * vfrac(cem >= 10)), frac_ge60=round(100 * vfrac(cem >= 60)),
                                        T90=round(t90(T[n]), 1))
cem_total = cem43(T, dtv)
res['uniform']['cem_60_plus_cooling'] = dict(center=round(float(cem_total[0]), 1), edge=round(float(cem_total[iB]), 2))
mass_uniform = Qedge * np.sum(vol[inside])           # proportional to total magnetite mass (same SAR)

# ---- non-uniform (Gaussian) loading with the same total mass ----------------------------------------
nonu = {}
profiles = {}
for lab, s in (('R/2', Rt / 2), ('R/3', Rt / 3)):
    shape = np.where(inside, np.exp(-r**2 / (2 * s**2)), 0.0)
    Qg = shape * mass_uniform / np.sum(shape * vol)
    Tg = 37 + steady(Qg)
    # dose needed to bring the boundary to 42 C with this distribution
    fac = 5.0 / (Tg[iB] - 37)
    Tg2 = 37 + fac * (Tg - 37)
    nonu[lab] = dict(same_mass=dict(center=round(float(Tg[0]), 1), edge=round(float(Tg[iB]), 1),
                                    frac42=round(100 * vfrac(Tg >= 42)), frac46=round(100 * vfrac(Tg >= 46)),
                                    T90=round(t90(Tg), 1)),
                     edge42=dict(dose_factor=round(float(fac), 2), center=round(float(Tg2[0]), 1),
                                 frac46=round(100 * vfrac(Tg2 >= 46)), frac50=round(100 * vfrac(Tg2 >= 50))))
    profiles[lab] = Tg
Tu = 37 + th_ss
nonu['uniform'] = dict(center=round(float(Tu[0]), 1), edge=round(float(Tu[iB]), 1), frac42=round(100 * vfrac(Tu >= 42)),
                       frac46=round(100 * vfrac(Tu >= 46)), T90=round(t90(Tu), 1))
res['nonuniform'] = nonu

# ---- perfusion sensitivity of the transient ---------------------------------------------------------
res['perfusion'] = {}
for wb in (0.25e-3, 1e-3):
    tt, thh = transient(Q, wb=wb, t_heat=3600, t_cool=0)
    TT = 37 + thh
    c = cem43(TT, tt[1] - tt[0])
    res['perfusion'][str(wb)] = dict(center_ss=round(float(TT[-1, 0]), 1), edge_ss=round(float(TT[-1, iB]), 1),
                                     cem60_center=round(float(c[0]), 1), cem60_edge=round(float(c[iB]), 2))

json.dump(res, open('transient.json', 'w'), indent=1)
print(json.dumps(res, indent=1))

# ---- Figure 8 -----------------------------------------------------------------------------------------
fig, axs = plt.subplots(1, 2, figsize=(16 * CM, 6.2 * CM))
ax = axs[0]
tm = t / 60
ax.axhspan(42, 46, color='#f0efec', zorder=0)
ax.axvline(60, color=MUTED, lw=1); ax.text(61.5, 48.0, 'field\noff', fontsize=8, color='#52514e', va='center')
ax.plot(tm, T[:, 0], color=C[0], label='Tumor center')
ax.plot(tm, T[:, iB], color=C[1], ls='--', label='Tumor boundary')
ax.plot(tm, T[:, np.argmin(abs(r - 15e-3))], color=C[2], ls='-.', label='5 mm outside')
ax.set_xlim(0, 80); ax.set_ylim(37, 51)
ax.set_xlabel('Time (min)'); ax.set_ylabel('Temperature (°C)')
ax.legend(loc='upper left', ncol=1, bbox_to_anchor=(0.12, 1.0), handlelength=2.2); ax.set_yticks(range(38, 51, 2))
ax.text(0.03, 0.95, '(a)', transform=ax.transAxes, fontweight='bold', va='top')
ax = axs[1]
ax.axhspan(42, 46, color='#f0efec', zorder=0)
ax.axvline(10, color=MUTED, lw=1)
ax.plot(r * 1e3, Tu, color=INK, label='Uniform')
ax.plot(r * 1e3, profiles['R/2'], color=C[0], ls='--', label='Gaussian, s = R/2')
ax.plot(r * 1e3, profiles['R/3'], color=C[1], ls='-.', label='Gaussian, s = R/3')
ax.set_xlim(0, 25); ax.set_ylim(37, 60)
ax.set_xlabel('Radial distance (mm)'); ax.set_ylabel('Temperature (°C)')
ax.legend(loc='upper right')
ax.text(0.03, 0.95, '(b)', transform=ax.transAxes, fontweight='bold', va='top')
fig.tight_layout(); fig.savefig('fig_transient.png'); plt.close(fig)
