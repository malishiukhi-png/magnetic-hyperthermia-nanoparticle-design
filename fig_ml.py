import json, numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'font.family': 'Liberation Sans', 'font.size': 10, 'axes.labelweight': 'bold', 'axes.linewidth': 0.8,
                     'xtick.direction': 'in', 'ytick.direction': 'in', 'xtick.top': True, 'ytick.right': True,
                     'xtick.minor.visible': True, 'ytick.minor.visible': True, 'legend.frameon': False, 'legend.fontsize': 9,
                     'savefig.dpi': 300, 'lines.linewidth': 1.8})
C = ['#2a78d6', '#eb6834', '#1baf7a']; INK = '#0b0b0b'; MUTED = '#898781'; CM = 1 / 2.54
R = json.load(open('ml_results.json')); T = np.load('ml_test.npz')
fig, axs = plt.subplots(1, 2, figsize=(16 * CM, 6.4 * CM))
ax = axs[0]
t, p = T['true'], T['pred']; m = (t >= 1) & (p >= 1)
ax.loglog(t[m], p[m], 'o', ms=2.2, mfc=C[0], mec='none', alpha=0.45)
ax.plot([1, 2000], [1, 2000], color=INK, lw=1)
ax.set_xlim(1, 2000); ax.set_ylim(1, 2000)
ax.set_xlabel('SAR, physics model (W g$^{-1}$)'); ax.set_ylabel('SAR, neural network (W g$^{-1}$)')
nn = R['models']['Neural network']
from sklearn.metrics import r2_score
r2_raw = r2_score(T['true'], T['pred'])  # unrounded, avoids double rounding of the stored 4-decimal value
ax.text(0.05, 0.95, '(a)', transform=ax.transAxes, fontweight='bold', va='top')
ax.text(0.95, 0.06, 'Test set (n = %d)\nR$^2$ = %.3f' % (R['n_test'], r2_raw), transform=ax.transAxes, ha='right', va='bottom', fontsize=8.5)
ax = axs[1]
inv = R['inverse']; ls = ['-', '--', '-.']; mk = ['o', 's', '^']
for k, sg in enumerate((0.0, 0.1, 0.2)):
    rows = [r for r in inv if r['sigma'] == sg]
    K = [r['K'] for r in rows]
    ax.plot(K, [r['d_phys'] for r in rows], color=C[k], ls=ls[k], label='σ = %.1f' % sg)
    ax.plot(K, [r['d_surr'] for r in rows], mk[k], color=C[k], mfc='white', mew=1.4, ms=6)
ax.plot([], [], 'o', color=MUTED, mfc='white', mew=1.4, label='Surrogate (markers)')
ax.plot([], [], color=MUTED, label='Physics model (lines)')
ax.set_xlabel('Anisotropy constant (kJ m$^{-3}$)'); ax.set_ylabel('Optimal core diameter (nm)')
ax.set_xlim(3, 32); ax.set_ylim(9, 24)
ax.text(0.05, 0.95, '(b)', transform=ax.transAxes, fontweight='bold', va='top')
ax.legend(loc='upper right', fontsize=8)
fig.tight_layout(); fig.savefig('fig_ml.png'); plt.close(fig)
