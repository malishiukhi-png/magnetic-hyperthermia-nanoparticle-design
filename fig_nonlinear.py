"""Figure 7: SAR of immobilized particles at 15 kA/m and 300 kHz. (a) LRT (tau0 = 1 ns), two-state model and overdamped
stochastic simulation (nonlinear.json, langevin_15k300.json). (b) Stochastic LLG with explicit damping (llg_a0.1.json,
llg_a1.json) compared with the LRT that uses the same zero-field Neel time (Brown attempt time)."""
import json, sys, numpy as np
sys.path.insert(0, '.')
import model, llg
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams.update({'font.family': 'Liberation Sans', 'font.size': 10, 'axes.labelweight': 'bold',
                     'axes.linewidth': 0.8, 'xtick.direction': 'in', 'ytick.direction': 'in',
                     'xtick.top': True, 'ytick.right': True, 'xtick.minor.visible': True, 'ytick.minor.visible': True,
                     'legend.frameon': False, 'legend.fontsize': 8, 'savefig.dpi': 300, 'lines.linewidth': 1.8})
C = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4']; INK = '#0b0b0b'; CM = 1 / 2.54
NL = json.load(open('nonlinear.json'))
fig, axs = plt.subplots(1, 2, figsize=(16 * CM, 8.6 * CM))
ax = axs[0]
D = NL['15k300']; d = np.array(D['d']); bar = np.array(D['barrier'])
invalid = d[bar < 3.0]
ax.axvspan(d.min(), invalid.max(), color='#f0efec', zorder=0)
ax.plot(d, D['lrt'], color=C[0], label='LRT, τ$_0$ = 1 ns')
ax.plot(d, D['nl'], color=C[1], ls='--', label='Two-state model')
L = np.array(json.load(open('langevin_15k300.json')))
ax.errorbar(L[:, 0], L[:, 1], yerr=2 * L[:, 2], fmt='o', ms=4, color=INK, mfc='white', mew=1.2, capsize=2, lw=1,
            label='Overdamped stochastic')
ax.set_xlim(10, 26); ax.set_ylim(0, 1000)
ax.set_xlabel('Core diameter (nm)'); ax.set_ylabel('SAR (W g$^{-1}$)')
ax.text(0.03, 0.95, '(a)', transform=ax.transAxes, fontweight='bold', va='top')
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=2, fontsize=7.5)
ax = axs[1]
dg = np.linspace(10e-9, 26e-9, 321)
out = {}
for a, col, mk in ((0.1, C[3], '^'), (1.0, C[2], 'D')):
    lrt = np.array([model.sar(np.array([x]), 15e3, 300e3, eta=np.inf, tau0=llg.tau0_brown(x, a))[0] for x in dg])
    ax.plot(dg * 1e9, lrt, color=col, ls='-', lw=1.4, label='LRT, same τ$_N$ (α = %g)' % a)
    try:
        G = np.array(json.load(open('llg_a%g.json' % a)))
        ax.errorbar(G[:, 0], G[:, 1], yerr=2 * G[:, 2], fmt=mk, ms=4.5, color=col, mfc='white', mew=1.3, capsize=2, lw=1,
                    label='sLLG, α = %g' % a)
        out[str(a)] = dict(lrt_opt=[round(float(dg[np.argmax(lrt)] * 1e9), 2), round(float(lrt.max()), 1)],
                           llg=G[:, :3].tolist())
    except FileNotFoundError:
        pass
ax.set_xlim(10, 27); ax.set_ylim(0, 1000)
ax.set_xlabel('Core diameter (nm)'); ax.set_ylabel('SAR (W g$^{-1}$)')
ax.text(0.03, 0.95, '(b)', transform=ax.transAxes, fontweight='bold', va='top')
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=2, fontsize=7.5)
fig.tight_layout(); fig.savefig('fig_nonlinear.png'); plt.close(fig)
json.dump(out, open('llg_summary.json', 'w'), indent=1)
print(json.dumps(out, indent=1))
