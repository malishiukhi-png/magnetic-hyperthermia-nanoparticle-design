import numpy as np, sys, json, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, '.')
from model import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator, LogLocator

# Conference template: Arial 10 numbers, Arial 10 bold axis labels, inner major+minor ticks, width 8 or 16 cm, >=300 dpi
plt.rcParams.update({'font.family': 'Liberation Sans', 'font.size': 10, 'axes.labelweight': 'bold',
                     'axes.linewidth': 0.8, 'xtick.direction': 'in', 'ytick.direction': 'in',
                     'xtick.top': True, 'ytick.right': True, 'xtick.minor.visible': True, 'ytick.minor.visible': True,
                     'legend.frameon': False, 'legend.fontsize': 9, 'savefig.dpi': 300, 'lines.linewidth': 1.8})
C = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4']   # validated categorical order
INK = '#0b0b0b'; MUTED = '#898781'
CM = 1 / 2.54
H0, F0 = 15e3, 300e3
d = np.linspace(6, 26, 401) * 1e-9
dn = d * 1e9
R = {}

def panel(ax, tag):
    ax.text(0.03, 0.95, tag, transform=ax.transAxes, fontweight='bold', va='top', ha='left')

# ---------- Fig A: relaxation times and SAR fluid vs immobilized -------------------------------------------
fig, axs = plt.subplots(1, 2, figsize=(16 * CM, 6.2 * CM))
tN, tB, te = taus(d)
ax = axs[0]
ax.semilogy(dn, tN, color=C[0], label='Néel')
ax.semilogy(dn, tB, color=C[1], ls='--', label='Brownian (water)')
ax.semilogy(dn, te, color=INK, lw=1.2, ls=':', label='Effective (water)')
ax.axhline(1 / (2 * np.pi * F0), color=MUTED, lw=1)
ax.text(25.6, 1 / (2 * np.pi * F0) / 2.2, '1/(2πf), f = 300 kHz', color='#52514e', fontsize=8, ha='right', va='top')
ax.set_ylim(1e-9, 1e-2); ax.set_xlim(6, 26)
ax.set_xlabel('Core diameter (nm)'); ax.set_ylabel('Relaxation time (s)')
ax.yaxis.set_major_locator(LogLocator(numticks=8)); ax.legend(loc='upper left', bbox_to_anchor=(0.0, 0.88))
panel(ax, '(a)')
ax = axs[1]
sf = sar(d, H0, F0); si = sar(d, H0, F0, eta=np.inf)
ax.plot(dn, sf, color=C[1], ls='--', label='Water (Néel + Brownian)')
ax.plot(dn, si, color=C[0], label='Immobilized (Néel only)')
ax.set_xlim(6, 26); ax.set_ylim(0, 820)
ax.set_xlabel('Core diameter (nm)'); ax.set_ylabel('SAR (W g$^{-1}$)')
ax.legend(loc='upper right'); panel(ax, '(b)')
fig.tight_layout(); fig.savefig('fig_relax_sar.png'); plt.close(fig)
i = np.argmax(si); j = np.argmax(sf)
R['opt_immob_d'] = round(dn[i], 1); R['opt_immob_sar'] = round(float(si[i])); R['opt_fluid_d'] = round(dn[j], 1); R['opt_fluid_sar'] = round(float(sf[j]))
for dd in (17, 18, 20):
    a = sar(dd * 1e-9, H0, F0)[0]; b = sar(dd * 1e-9, H0, F0, eta=np.inf)[0]
    R['loss_%d' % dd] = round(100 * (1 - b / a)); R['fluid_%d' % dd] = round(a); R['immob_%d' % dd] = round(b)
cross = dn[np.argmin(np.abs(np.log(tN) - np.log(tB)))]; R['cross_d'] = round(cross, 1)
R['xi_opt'] = round(float(xi_of(dn[i] * 1e-9, H0)), 1); R['HK'] = round(2 * P['K'] / (mu0 * P['Ms']) / 1e3, 1)

# ---------- Fig B: polydispersity and anisotropy ----------------------------------------------------------
fig, axs = plt.subplots(1, 2, figsize=(16 * CM, 6.2 * CM))
ax = axs[0]
for k, (s, ls) in enumerate(((0, '-'), (0.1, '--'), (0.2, '-.'))):
    y = sar(d, H0, F0, eta=np.inf, sigma=s); ii = np.argmax(y)
    ax.plot(dn, y, color=C[k], ls=ls, label='σ = %.1f' % s)
    R['poly_%.1f_d' % s] = round(dn[ii], 1); R['poly_%.1f_sar' % s] = round(float(y[ii]))
    R['poly_%.1f_fwhm' % s] = [round(float(dn[y >= y[ii] / 2].min()), 1), round(float(dn[y >= y[ii] / 2].max()), 1)]
ax.set_xlim(6, 26); ax.set_ylim(0, 650)
ax.set_xlabel('Median core diameter (nm)'); ax.set_ylabel('SAR (W g$^{-1}$)'); ax.legend(loc='upper right'); panel(ax, '(a)')
ax = axs[1]
Ks = [5e3, 9e3, 13e3, 20e3, 30e3]; lss = ['-', '--', '-.', ':', (0, (5, 1, 1, 1))]
R['K'] = {}
for k, (K, ls) in enumerate(zip(Ks, lss)):
    y = sar(d, H0, F0, eta=np.inf, K=K); ii = np.argmax(y)
    ax.plot(dn, y, color=C[k], ls=ls, label='%g kJ m$^{-3}$' % (K / 1e3))
    R['K'][str(int(K / 1e3))] = [round(dn[ii], 1), round(float(y[ii]))]
ax.set_xlim(6, 26); ax.set_ylim(0, 1000)
ax.set_xlabel('Core diameter (nm)'); ax.set_ylabel('SAR (W g$^{-1}$)'); ax.legend(loc='upper right', ncol=2, fontsize=8, handlelength=2.2, columnspacing=0.8, title='Anisotropy constant, K', title_fontsize=8); panel(ax, '(b)')
fig.tight_layout(); fig.savefig('fig_poly_aniso.png'); plt.close(fig)

# ---------- Fig C: field optimisation under safety limits -------------------------------------------------
fs = np.linspace(50e3, 1000e3, 191); Hmax = 18e3
dd = np.linspace(8, 24, 161) * 1e-9
def field_map(Cl):
    M = np.zeros((len(dd), len(fs)))
    for jj, fq in enumerate(fs):
        M[:, jj] = sar(dd, min(Cl / fq, Hmax), fq, eta=np.inf)
    return M
Mhd = field_map(5e9); Mbz = field_map(4.85e8)
fig, axs = plt.subplots(1, 2, figsize=(16 * CM, 6.2 * CM), gridspec_kw={'width_ratios': [1.15, 1]})
ax = axs[0]
from matplotlib.colors import LinearSegmentedColormap
blues = LinearSegmentedColormap.from_list('b', ['#f7fbff', '#cde2fb', '#86b6ef', '#3987e5', '#256abf', '#184f95', '#0d366b'])
im = ax.pcolormesh(fs / 1e3, dd * 1e9, Mhd, cmap=blues, shading='auto', rasterized=True)
cb = fig.colorbar(im, ax=ax, pad=0.02); cb.set_label('SAR (W g$^{-1}$)', fontweight='bold'); cb.ax.tick_params(direction='in')
ii, jj = np.unravel_index(np.argmax(Mhd), Mhd.shape)
ax.plot(fs[jj] / 1e3, dd[ii] * 1e9, marker='o', ms=7, mfc='white', mec=INK)
ax.set_xlabel('Frequency (kHz)'); ax.set_ylabel('Core diameter (nm)'); ax.tick_params(which='both', color='white')
panel(ax, '(a)'); ax.texts[-1].set_bbox(dict(facecolor='white', edgecolor='none', pad=1.5))
R['hd_best'] = [round(float(Mhd[ii, jj])), round(fs[jj] / 1e3), round(min(5e9 / fs[jj], Hmax) / 1e3, 1), round(dd[ii] * 1e9, 1)]
kk, ll = np.unravel_index(np.argmax(Mbz), Mbz.shape)
R['bz_best'] = [round(float(Mbz[kk, ll])), round(fs[ll] / 1e3), round(min(4.85e8 / fs[ll], Hmax) / 1e3, 2), round(dd[kk] * 1e9, 1)]
ax = axs[1]
ax.semilogy(fs / 1e3, Mhd.max(axis=0), color=C[0], label='H·f ≤ 5 × 10$^{9}$ A m$^{-1}$ s$^{-1}$')
ax.semilogy(fs / 1e3, Mbz.max(axis=0), color=C[1], ls='--', label='H·f ≤ 4.85 × 10$^{8}$ A m$^{-1}$ s$^{-1}$')
ax.set_xlabel('Frequency (kHz)'); ax.set_ylabel('Maximum SAR (W g$^{-1}$)'); ax.set_ylim(1, 2000); ax.set_xlim(50, 1000)
ax.legend(loc='center right', fontsize=8, bbox_to_anchor=(1.0, 0.53)); panel(ax, '(b)')
R['hd_at_100'] = round(float(Mhd.max(axis=0)[np.argmin(abs(fs - 100e3))])); R['hd_at_1000'] = round(float(Mhd.max(axis=0)[-1]))
fig.tight_layout(); fig.savefig('fig_field.png'); plt.close(fig)

# ---------- Fig D: bioheat ----------------------------------------------------------------------------------
Rt = 10e-3; kt = 0.5; rc = 4.0e6
r = np.linspace(0, 30e-3, 3001)
def unit(wb):
    W = rc * wb; return bioheat(r, Rt, 1e6, W, W, kt)
fig, axs = plt.subplots(1, 2, figsize=(16 * CM, 6.2 * CM))
ax = axs[0]
u = unit(0.5e-3); iR = np.argmin(abs(r - Rt))
Qc = 6 / u[0] * 1e6; Qe = 5 / u[iR] * 1e6
tc = 37 + u * Qc / 1e6; tedge = 37 + u * Qe / 1e6
ax.axhspan(42, 46, color='#f0efec', zorder=0); ax.text(29.5, 44, '42–46 °C', ha='right', va='center', fontsize=8, color='#52514e')
ax.axvline(10, color=MUTED, lw=1); ax.text(10.4, 49.5, 'tumor boundary', fontsize=8, color='#52514e', va='top')
ax.plot(r * 1e3, tc, color=C[0], label='Center at 43 °C')
ax.plot(r * 1e3, tedge, color=C[1], ls='--', label='Boundary at 42 °C')
ax.set_xlim(0, 30); ax.set_ylim(37, 50)
ax.set_xlabel('Radial distance (mm)'); ax.set_ylabel('Temperature (°C)'); ax.legend(loc='upper right', bbox_to_anchor=(1.0, 0.9)); panel(ax, '(a)')
vin = r <= Rt
R['center_strategy'] = dict(Q=round(Qc / 1e6, 3), edge=round(float(tc[iR]), 1), frac42=round(100 * float(np.sum((tc[vin] >= 42) * r[vin]**2) / np.sum(r[vin]**2))),
                            healthy5=round(float(tc[np.argmin(abs(r - 15e-3))]), 1))
R['edge_strategy'] = dict(Q=round(Qe / 1e6, 3), center=round(float(tedge[0]), 1), healthy5=round(float(tedge[np.argmin(abs(r - 15e-3))]), 1),
                          frac46=round(100 * float(np.sum((tedge[vin] >= 46) * r[vin]**2) / np.sum(r[vin]**2))))
ax = axs[1]
scen = [('Ideal', R['opt_immob_sar']), ('σ = 0.2', R['poly_0.2_sar']), ('18 nm\nmismatch', R['immob_18'])]
x = np.arange(len(scen)); R['conc'] = {}
for k, (lab, S) in enumerate(scen):
    cs = []
    for wb in (0.25e-3, 0.5e-3, 1e-3):
        uu = unit(wb); cs.append(5 / uu[iR] * 1e6 / (S * 1000))   # dose that brings the tumor boundary to 42 C
    R['conc'][lab.replace('\n', ' ')] = [round(c, 2) for c in cs]
    ax.bar(k, cs[1], width=0.55, color=C[0], zorder=2)
    ax.errorbar(k, cs[1], yerr=[[cs[1] - cs[0]], [cs[2] - cs[1]]], color=INK, capsize=4, lw=1, zorder=3)
    ax.text(k, cs[2] + 0.08, '%.2f' % cs[1], ha='center', fontsize=8)
ax.set_xticks(x); ax.set_xticklabels([s[0] for s in scen]); ax.xaxis.set_minor_locator(plt.NullLocator())
ax.set_ylabel('Required dose (mg cm$^{-3}$)'); ax.set_ylim(0, max(max(v) for v in R['conc'].values()) * 1.25); panel(ax, '(b)')
fig.tight_layout(); fig.savefig('fig_bioheat.png'); plt.close(fig)
R['edge_center_by_perf'] = {}
for wb in (0.25e-3, 0.5e-3, 1e-3):
    uu = unit(wb); Qx = 5 / uu[iR]; R['edge_center_by_perf'][str(wb)] = round(37 + uu[0] * Qx, 1)
json.dump(R, open('results.json', 'w'), indent=1, ensure_ascii=False)
print(json.dumps(R, indent=1, ensure_ascii=False))
