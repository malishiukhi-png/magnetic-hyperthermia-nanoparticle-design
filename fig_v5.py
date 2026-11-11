"""Figures 5 and 6 of the manuscript.
Figure 5 (fig_nonlinear.png): SAR of immobilized particles at 15 kA/m and 300 kHz. (a) LRT, two-state model, overdamped
stochastic simulations; (b) uniaxial sLLG with alpha = 0.01, 0.1, 1 and the LRT with the same zero-field Neel time;
(c) sLLG with the cubic anisotropy of bulk magnetite compared with uniaxial anisotropy.
Figure 6 (fig_experiment.png): comparison with Khandhar et al. (a, b) and with the conditions of Tong et al. (c)."""
import json, os, sys, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); os.chdir(HERE)
import model
from model import P, T, kB
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

G_ = 1.760859e11


def tau0_brown(d, a, Ms=P['Ms'], K=P['K']):
    V = np.pi * d**3 / 6; sig = K * V / (kB * T); tD = (1 + a * a) / (2 * a) * Ms * V / (G_ * kB * T)
    return tD * np.sqrt(np.pi) / 2 * sig**-1.5


plt.rcParams.update({'font.family': 'Liberation Sans', 'font.size': 9.5, 'axes.labelweight': 'bold',
                     'axes.linewidth': 0.8, 'xtick.direction': 'in', 'ytick.direction': 'in',
                     'xtick.top': True, 'ytick.right': True, 'xtick.minor.visible': True, 'ytick.minor.visible': True,
                     'legend.frameon': False, 'legend.fontsize': 7.3, 'savefig.dpi': 300, 'lines.linewidth': 1.6})
C = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#7a5cd6']; INK = '#0b0b0b'; CM = 1 / 2.54
J = lambda f: np.array(json.load(open(f)))


def pts(ax, G, col, mk, lab):
    ax.errorbar(G[:, 0], G[:, 1], yerr=2 * G[:, 2], fmt=mk, ms=4, color=col, mfc='white', mew=1.2, capsize=2, lw=1, label=lab)


# ------------------------------------------------ Figure 5 ---------------------------------------------------------
NL = json.load(open('nonlinear.json'))
fig, axs = plt.subplots(1, 3, figsize=(18 * CM, 8.4 * CM))
ax = axs[0]
D = NL['15k300']; d = np.array(D['d']); bar = np.array(D['barrier'])
ax.axvspan(d.min(), d[bar < 3.0].max(), color='#f0efec', zorder=0)
ax.plot(d, D['lrt'], color=C[0], label='LRT, τ$_0$ = 1 ns')
ax.plot(d, D['nl'], color=C[1], ls='--', label='Two-state model')
pts(ax, J('langevin_15k300.json'), INK, 'o', 'Overdamped stochastic')
ax.set_xlim(10, 26); ax.set_ylim(0, 1000)
ax = axs[1]
dg = np.linspace(10e-9, 27e-9, 341)
for a, col, mk in ((0.01, C[5], 's'), (0.1, C[3], '^'), (1.0, C[2], 'D')):
    lrt = np.array([model.sar(np.array([x]), 15e3, 300e3, eta=np.inf, tau0=tau0_brown(x, a))[0] for x in dg])
    ax.plot(dg * 1e9, lrt, color=col, ls='-', lw=1.1, alpha=0.9, label='LRT, same τ$_N$ (α = %g)' % a)
    fn = 'llg_a001.json' if a == 0.01 else 'llg_a%g.json' % a
    pts(ax, J(fn), col, mk, 'sLLG, α = %g' % a)
ax.set_xlim(10, 27); ax.set_ylim(0, 1000)
ax = axs[2]
pts(ax, J('llg_a1.json'), C[2], 'D', 'Uniaxial, α = 1')
pts(ax, J('llg_cubic1.json'), C[4], 'o', 'Cubic, α = 1')
try:
    pts(ax, J('llg_cubic01.json'), C[3], '^', 'Cubic, α = 0.1')
except FileNotFoundError:
    pass
ax.set_xlim(10, 52); ax.set_ylim(0, 1000); ax.set_xticks([10, 20, 30, 40, 50])
for ax, lab in zip(axs, '(a) (b) (c)'.split()):
    ax.set_xlabel('Core diameter (nm)'); ax.set_ylabel('SAR (W g$^{-1}$)')
    ax.text(0.04, 0.96, lab, transform=ax.transAxes, fontweight='bold', va='top')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.19), ncol=1 if ax is not axs[1] else 2, fontsize=7)
fig.tight_layout(w_pad=0.6); fig.savefig('fig_nonlinear.png'); plt.close(fig)

# ------------------------------------------------ Figure 6 ---------------------------------------------------------
E = json.load(open('exp_metrics.json'))
nar = E['rows']['narrow']; d0 = np.array([r['d0'] for r in nar]); meas = np.array([r['measured'] for r in nar])
sty = {'LRT, K = 13': (C[0], '-', 's', 'LRT, τ$_0$ = 1 ns'), 'sLLG, K = 13, alpha = 0.1': (C[1], '--', '^', 'sLLG, α = 0.1'),
       'sLLG, K = 13, alpha = 1': (C[2], '-.', 'D', 'sLLG, α = 1'), 'sLLG, K = 23, alpha = 1': (C[5], '-', 'v', 'sLLG, K = 23 kJ m$^{-3}$, α = 1'),
       'sLLG, cubic, alpha = 1': (C[4], ':', 'o', 'sLLG, cubic, α = 1')}
fig, axs = plt.subplots(1, 3, figsize=(18 * CM, 8.6 * CM))
fig.subplots_adjust(left=0.085, right=0.985, top=0.97, bottom=0.40, wspace=0.42)
for name, (col, ls, mk, lab) in sty.items():
    y = np.array([r[name] for r in nar])
    axs[0].plot(d0, np.maximum(y, 0.3), color=col, ls=ls, marker=mk, ms=3.5, lw=1.3, label=lab)
    if 'cubic' not in name:
        axs[1].plot(d0, y / y.max(), color=col, ls=ls, marker=mk, ms=3.5, lw=1.3, label=lab)
axs[0].plot(d0, meas, 'o', color=INK, mfc='white', mew=1.2, ms=5.5, label='Measured (Khandhar et al.)')
axs[1].plot(d0, meas / meas.max(), 'o', color=INK, mfc='white', mew=1.2, ms=5.5)
axs[0].set_yscale('log'); axs[0].set_ylim(0.3, 2000); axs[1].set_ylim(0, 1.12)
for ax in axs[:2]:
    ax.set_xlim(11, 19); ax.set_xticks([12, 14, 16, 18]); ax.set_xlabel('Median core diameter (nm)')
axs[0].set_ylabel('SLP (W g$^{-1}$)'); axs[1].set_ylabel('Normalized SLP')
ax = axs[2]
for k, col, mk, lab in (('tong_u', C[2], 'D', 'sLLG, uniaxial, α = 1'), ('tong_c', C[4], 'o', 'sLLG, cubic, α = 1')):
    if k in E['tong']:
        G = np.array(E['tong'][k]); ax.errorbar(G[:, 0], G[:, 1], yerr=2 * G[:, 2], fmt=mk + '-', ms=4, color=col, mfc='white',
                                                mew=1.2, capsize=2, lw=1.0, label=lab)
ax.plot([40.4], [E['tong_measured_40nm_W_per_g_Fe3O4']], '*', color=INK, ms=9, label='Measured, 40 nm (Tong et al.)')
ax.set_xlim(8, 44); ax.set_ylim(0, 1000); ax.set_xticks([10, 20, 30, 40]); ax.set_xlabel('Core diameter (nm)'); ax.set_ylabel('SAR (W g$^{-1}$)')
for ax, lab in zip(axs, '(a) (b) (c)'.split()):
    ax.text(0.05, 0.96, lab, transform=ax.transAxes, fontweight='bold', va='top')
h, l = axs[0].get_legend_handles_labels()
fig.legend(h, l, loc='upper center', bbox_to_anchor=(0.36, 0.22), ncol=2, fontsize=7, columnspacing=1.2)
axs[2].legend(loc='upper center', bbox_to_anchor=(0.5, -0.26), ncol=1, fontsize=7)
fig.savefig('fig_experiment.png'); plt.close(fig)
print('figures written')
