"""Comparison with the measured specific loss power of monodisperse magnetite nanoparticles of Khandhar et al.
(J. Biomed. Mater. Res. A 100 (2012) 728; 373 kHz, 14 kA/m, water). Measured values were digitized from their
Figure 7 (uncertainty about +-3 W/g); median diameters and lognormal widths are from their Table 1.
Writes exp_compare.json and fig_experiment.png (Figure 8)."""
import json, numpy as np, sys
sys.path.insert(0, '.')
import model
from model import kB, mu0
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import llg

F, H = 373e3, 14e3
MS_EXP = 75.0 * 5180.0                      # 75 A m2/kg measured by Khandhar et al. -> A/m
T_EXP = 310.15                               # same temperature as the sLLG runs
# (median diameter nm, sigma, measured SLP W/g Fe3O4)
NARROW = [(12, 0.01, 23.5), (13, 0.10, 54.1), (14, 0.21, 87.3), (15, 0.23, 100.2), (16, 0.16, 141.0), (18, 0.34, 43.3)]
BROAD = [(9, 0.29, 11.1), (10, 0.10, 24.2), (14, 0.31, 101.2), (16, 0.36, 55.1), (16.5, 0.36, 30.1)]


def lognormal(func, d0, s, n=401):
    if s < 0.02:
        return float(func(np.array([d0]))[0])
    x = np.linspace(np.log(d0) - 4 * s, np.log(d0) + 4 * s, n); d = np.exp(x)
    w = np.exp(-(x - np.log(d0))**2 / (2 * s * s)) * d**3
    return float(np.sum(w * func(d)) / np.sum(w))


def lrt_water(d, tau0=1e-9):
    """LRT in water at 25 C (Neel + Brown), measured Ms, 5 nm shell; W/g."""
    model.T = T_EXP
    return model.sar(d, H, F, eta=0.69e-3, Ms=MS_EXP, tau0=tau0)


def lrt_tau0_alpha(d, alpha):
    """LRT with the zero-field Neel time of the sLLG model (Brown high-barrier attempt time), immobilized."""
    out = []
    for dd in np.atleast_1d(d):
        t0 = llg.tau0_brown(dd, alpha, Ms=MS_EXP)
        out.append(model.sar(np.array([dd]), H, F, eta=np.inf, Ms=MS_EXP, tau0=t0)[0])
    return np.array(out)


def llg_interp(alpha):
    try:
        L = np.array(json.load(open('llg_khandhar_a%g.json' % alpha)))
    except FileNotFoundError:
        return None
    dg, sg = L[:, 0] * 1e-9, L[:, 1]

    def f(d):
        return np.interp(d, dg, sg, left=0.0, right=sg[-1] if len(sg) else 0.0)
    return f, L


res = {'narrow': [], 'broad': []}
models = {'LRT (tau0 = 1 ns)': lambda d: lrt_water(d)}
for a in (0.1, 1.0):
    li = llg_interp(a)
    if li:
        models['sLLG alpha = %g' % a] = li[0]
for key, data in (('narrow', NARROW), ('broad', BROAD)):
    for d0, s, meas in data:
        row = dict(d0=d0, sigma=s, measured=meas)
        for name, fn in models.items():
            row[name] = round(lognormal(fn, d0 * 1e-9, s), 1)
        res[key].append(row)
# summary metrics: peak position of the narrow series and the 16 nm broad/narrow ratio
summ = {}
for name in models:
    vals = [r[name] for r in res['narrow']]
    i = int(np.argmax(vals))
    r16n = [r for r in res['narrow'] if r['d0'] == 16][0][name]
    r16b = [r for r in res['broad'] if r['d0'] == 16][0][name]
    meas = np.array([r['measured'] for r in res['narrow']]); pred = np.array(vals)
    scale = float(np.sum(meas * pred) / np.sum(pred**2)) if pred.any() else None
    summ[name] = dict(peak_d0=res['narrow'][i]['d0'], peak_value=vals[i], ratio_16_broad_to_narrow=round(r16b / r16n, 2) if r16n else None,
                      scale_to_measured=round(scale, 3) if scale else None)
summ['measured'] = dict(peak_d0=16, peak_value=141.0, ratio_16_broad_to_narrow=round(55.1 / 141.0, 2))
res['summary'] = summ
json.dump(res, open('exp_compare.json', 'w'), indent=1)
print(json.dumps(summ, indent=1))

# ---- Figure: measured vs. model (narrow series), absolute and normalized ----------------------------------
plt.rcParams.update({'font.family': 'Liberation Sans', 'font.size': 10, 'axes.labelweight': 'bold',
                     'axes.linewidth': 0.8, 'xtick.direction': 'in', 'ytick.direction': 'in',
                     'xtick.top': True, 'ytick.right': True, 'xtick.minor.visible': True, 'ytick.minor.visible': True,
                     'legend.frameon': False, 'legend.fontsize': 8, 'savefig.dpi': 300, 'lines.linewidth': 1.6})
C = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4']; INK = '#0b0b0b'; CM = 1 / 2.54
fig, axs = plt.subplots(1, 2, figsize=(16 * CM, 7.0 * CM))
d0 = np.array([r['d0'] for r in res['narrow']]); meas = np.array([r['measured'] for r in res['narrow']])
styles = {'LRT (tau0 = 1 ns)': (C[0], '-', 's', 'LRT, τ$_0$ = 1 ns'), 'sLLG alpha = 0.1': (C[1], '--', '^', 'sLLG, α = 0.1'),
          'sLLG alpha = 1': (C[2], '-.', 'D', 'sLLG, α = 1')}
for name in models:
    col, ls, mk, lab = styles[name]
    y = np.array([r[name] for r in res['narrow']])
    axs[0].plot(d0, y, color=col, ls=ls, marker=mk, ms=4, label=lab)
    axs[1].plot(d0, y / y.max(), color=col, ls=ls, marker=mk, ms=4, label=lab)
axs[0].plot(d0, meas, 'o', color=INK, mfc='white', mew=1.3, ms=6, label='Measured [ref]')
axs[1].plot(d0, meas / meas.max(), 'o', color=INK, mfc='white', mew=1.3, ms=6, label='Measured')
axs[0].set_yscale('log'); axs[0].set_ylim(1, 2000)
axs[0].set_xlabel('Median core diameter (nm)'); axs[0].set_ylabel('SLP (W g$^{-1}$)')
axs[1].set_xlabel('Median core diameter (nm)'); axs[1].set_ylabel('Normalized SLP'); axs[1].set_ylim(0, 1.12)
for ax, lab in zip(axs, ('(a)', '(b)')):
    ax.set_xlim(11, 19); ax.text(0.03, 0.95, lab, transform=ax.transAxes, fontweight='bold', va='top')
h, l = axs[0].get_legend_handles_labels(); l = [x.replace(' [ref]', '') for x in l]
fig.legend(h, l, loc='lower center', ncol=4, bbox_to_anchor=(0.5, 0.0))
fig.tight_layout(rect=(0, 0.08, 1, 1)); fig.savefig('fig_experiment.png'); plt.close(fig)
