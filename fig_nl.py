"""Figures of the revised manuscript that use the sLLG results (nl_results.json): field scenarios, polydispersity and anisotropy,
and the steady-state tumor figure with the dose calculated from the nonlinear SAR.  Also the revised workflow schematic."""
import json, os, sys, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from model import sar, bioheat, P
HERE = os.path.dirname(os.path.abspath(__file__)) + '/'
R = json.load(open(HERE + 'nl_results.json'))
plt.rcParams.update({'font.family': 'Liberation Sans', 'font.size': 10, 'axes.labelweight': 'bold', 'axes.linewidth': 0.8,
                     'xtick.direction': 'in', 'ytick.direction': 'in', 'xtick.top': True, 'ytick.right': True, 'xtick.minor.visible': True,
                     'ytick.minor.visible': True, 'legend.frameon': False, 'legend.fontsize': 8, 'savefig.dpi': 300, 'lines.linewidth': 1.8})
C = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4']; INK = '#0b0b0b'; MUTED = '#898781'; CM = 1 / 2.54
def tag(ax, t, x=0.03, y=0.95): ax.text(x, y, t, transform=ax.transAxes, fontweight='bold', va='top')

# ---- field scenarios -----------------------------------------------------------------------------------------
fig, axs = plt.subplots(1, 2, figsize=(16 * CM, 7.0 * CM))
ax = axs[0]
F = {(e['H'], round(e['f'])): e for e in R['fields']}
dg = np.linspace(8e-9, 30e-9, 441)
for (key, lab, col, mk) in (((18.0, 100), '100 kHz, 18 kA m$^{-1}$', C[0], 'o'), ((15.0, 300), '300 kHz, 15 kA m$^{-1}$', C[1], 's'), ((18.0, 278), '278 kHz, 18 kA m$^{-1}$', C[2], '^')):
    e = F[key]; c = e['curves']['0.1']; c1 = e['curves']['1.0']
    ax.plot(c['d'], c['sar'], '-', marker=mk, ms=4, color=col, mfc='white', mew=1.1, label=lab)
    ax.plot(c1['d'], c1['sar'], ':', color=col, lw=1.4)
    f = e['f'] * 1e3
    ax.plot(dg * 1e9, sar(dg, e['H'] * 1e3, f, eta=np.inf), '--', lw=1.0, color=col, alpha=0.75)
ax.plot([], [], '-', color=MUTED, label='sLLG, α = 0.1'); ax.plot([], [], ':', color=MUTED, lw=1.4, label='sLLG, α = 1'); ax.plot([], [], '--', color=MUTED, lw=1.0, label='LRT, τ$_0$ = 1 ns')
ax.set_xlim(10, 30); ax.set_ylim(0, 1000); ax.set_xlabel('Core diameter (nm)'); ax.set_ylabel('SAR (W g$^{-1}$)'); tag(ax, '(a)')
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=2, fontsize=7.2, columnspacing=1.0, handlelength=2.2)
ax = axs[1]
pts = [(18.0, 100), (18.0, 278), (12.5, 400), (10.0, 500), (5.0, 1000)]
pts = [p for p in pts if F[p].get('opt_a0.1')]
fq = [F[p]['f'] for p in pts]
for a, col, mk, lab in (('0.1', C[1], 's', 'sLLG, α = 0.1'), ('1', C[2], '^', 'sLLG, α = 1')):
    y = [F[p]['opt_a%s' % a][1] for p in pts]
    ax.plot(fq, y, '-', color=col, marker=mk, ms=5, mfc='white', mew=1.2, label=lab)
ax.plot(fq, [F[p]['lrt'][1] for p in pts], '--', color=C[0], marker='o', ms=4, mfc='white', mew=1.1, lw=1.3, label='LRT, τ$_0$ = 1 ns')
for p, x in zip(pts, fq):
    d = F[p]['opt_a0.1'][0]; y = max(F[p]['opt_a0.1'][1], F[p]['opt_a1'][1])
    ax.annotate('%.0f nm' % d, (x, y), textcoords='offset points', xytext=(0, 7), ha='center', fontsize=7.5, color='#52514e')
ax.set_xscale('log'); ax.set_xticks([100, 200, 500, 1000]); ax.set_xticklabels(['100', '200', '500', '1000']); ax.set_xlim(80, 1250); ax.set_ylim(0, 1100)
ax.set_xlabel('Frequency (kHz)'); ax.set_ylabel('Maximum SAR (W g$^{-1}$)'); tag(ax, '(b)')
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=2, fontsize=7.2, columnspacing=1.0, handlelength=2.2)
fig.tight_layout(); fig.savefig(HERE + 'fig_fields.png'); plt.close(fig)

# ---- polydispersity and anisotropy ---------------------------------------------------------------------------
fig, axs = plt.subplots(1, 2, figsize=(16 * CM, 6.6 * CM))
ax = axs[0]
base = R['K']
for s, col, mk, lab in ((0.0, C[0], 'o', 'σ = 0'), (0.1, C[1], 's', 'σ = 0.1'), (0.2, C[2], '^', 'σ = 0.2')):
    for a, ls in (('0.1', '-'), ('1', ':')):
        if s == 0.0: e = base['13_%s' % a]
        else:
            e = R['poly'].get('%g_%s' % (s, a))
            if e is None: continue
        ax.plot(e['d'], e['sar'], ls, marker=mk if a == '0.1' else None, ms=4, mfc='white', mew=1.1, color=col, label=lab if a == '0.1' else None)
ax.plot([], [], ':', color=MUTED, lw=1.8, label='α = 1')
ax.set_xlim(12, 30); ax.set_ylim(0, 900); ax.set_xlabel('Median core diameter (nm)'); ax.set_ylabel('SAR (W g$^{-1}$)'); ax.legend(loc='upper right', fontsize=7.6); tag(ax, '(a)')
ax = axs[1]
for K, col, mk in (('9', C[3], 'D'), ('13', C[0], 'o'), ('20', C[1], 's'), ('30', C[2], '^')):
    e = base.get('%s_0.1' % K); e1 = base.get('%s_1' % K)
    if e: ax.plot(e['d'], e['sar'], '-', marker=mk, ms=4, mfc='white', mew=1.1, color=col, label='%s kJ m$^{-3}$' % K)
    if e1: ax.plot(e1['d'], e1['sar'], ':', color=col, lw=1.6)
ax.set_xlim(9, max(31, max(base['9_0.1']['d']) + 1)); ax.set_ylim(0, 1200); ax.set_xlabel('Core diameter (nm)'); ax.set_ylabel('SAR (W g$^{-1}$)'); ax.legend(loc='upper right', fontsize=7.4, ncol=2, columnspacing=0.9, handlelength=2.0); tag(ax, '(b)')
fig.tight_layout(w_pad=1.2); fig.savefig(HERE + 'fig_poly_K.png'); plt.close(fig)

# ---- tumor: radial profile and dose from the nonlinear SAR ------------------------------------------------------
Rt = 10e-3; kt = 0.5; rc = 4.0e6
r = np.linspace(0, 30e-3, 3001)
def unit(wb): W = rc * wb; return bioheat(r, Rt, 1e6, W, W, kt)
fig, axs = plt.subplots(1, 2, figsize=(16 * CM, 7.2 * CM))
ax = axs[0]
u = unit(0.5e-3); iR = np.argmin(abs(r - Rt)); Qc = 6 / u[0] * 1e6; Qe = 5 / u[iR] * 1e6
ax.axhspan(42, 46, color='#f0efec', zorder=0); ax.text(29.5, 44, '42–46 °C', ha='right', va='center', fontsize=8, color='#52514e')
ax.axvline(10, color=MUTED, lw=1); ax.text(10.4, 49.5, 'tumor boundary', fontsize=8, color='#52514e', va='top')
ax.plot(r * 1e3, 37 + u * Qc / 1e6, color=C[0], label='Center at 43 °C'); ax.plot(r * 1e3, 37 + u * Qe / 1e6, color=C[1], ls='--', label='Boundary at 42 °C')
ax.set_xlim(0, 30); ax.set_ylim(37, 50); ax.set_xlabel('Radial distance (mm)'); ax.set_ylabel('Temperature (°C)'); ax.legend(loc='upper right', bbox_to_anchor=(1.0, 0.9)); tag(ax, '(a)')
ax = axs[1]
T = R['dose_table']; d = [e['d'] for e in T if e['d'] >= 14]
def series(k): return [e.get(k) for e in T if e['d'] >= 14]
ax.semilogy(d, series('lrt_dose'), '--', color=C[0], marker='o', ms=4, mfc='white', mew=1.1, lw=1.3, label='LRT, 300 kHz')
ax.semilogy(d, series('c0.1'), '-', color=C[1], marker='s', ms=4, mfc='white', mew=1.1, label='sLLG, α = 0.1, 300 kHz')
ax.semilogy(d, series('c1'), '-', color=C[2], marker='^', ms=4, mfc='white', mew=1.1, label='sLLG, α = 1, 300 kHz')
Cl = R.get('clinical100', {})
if '0.1' in Cl:
    dd = sorted(float(x) for x in Cl['0.1'] if float(x) >= 14)
    ax.semilogy(dd, [Cl['0.1'][str(x)]['dose'] for x in dd], '-.', color=C[3], marker='D', ms=4, mfc='white', mew=1.1, label='sLLG, α = 0.1, 100 kHz')
ax.set_xlim(13, 29); ax.set_ylim(0.1, 30); ax.set_xlabel('Core diameter (nm)'); ax.set_ylabel('Required dose (mg cm$^{-3}$)'); tag(ax, '(b)', 0.88, 0.95)
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=2, fontsize=7.0, columnspacing=1.0, handlelength=2.2)
axs[0].legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=2, fontsize=7.0, columnspacing=1.0, handlelength=2.2)
fig.tight_layout(w_pad=1.2); fig.savefig(HERE + 'fig_dose.png'); plt.close(fig)

# ---- workflow ---------------------------------------------------------------------------------------------------
plt.rcParams.update({'font.size': 8.5})
BLUE, BLUE_BG = '#2a78d6', '#e8f1fc'; ORANGE, ORANGE_BG = '#d95926', '#fdeee7'; GRAY, GRAY_BG = '#52514e', '#f3f2ef'
fig = plt.figure(figsize=(16 * CM, 7.4 * CM)); ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 160); ax.set_ylim(0, 74); ax.axis('off')
def box(x, y, w, h, title, body, edge, face, ls='-'):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0,rounding_size=2', lw=1.2, ec=edge, fc=face, ls=ls))
    ax.text(x + w / 2, y + h - 3.0, title, ha='center', va='top', fontweight='bold', color=INK, fontsize=8.6)
    ax.text(x + w / 2, y + h - 8.2, body, ha='center', va='top', color=INK, fontsize=7.4, linespacing=1.3)
def arrow(x1, y1, x2, y2, color=INK, ls='-'):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle='-|>', color=color, lw=1.2, ls=ls, mutation_scale=10, shrinkA=0, shrinkB=0))
Y, H = 33, 28
box(1, Y, 28, H, 'Inputs', 'Core size d, width σ\nEffective K, M$_s$, α\nField H, f\nImmobilized', GRAY, GRAY_BG)
box(34, Y, 30, H, 'Nonlinear dynamics', 'sLLG (no calibration)\nOverdamped Langevin\nHysteresis loop\nSAR (W g$^{-1}$)', BLUE, BLUE_BG)
box(69, Y, 28, H, 'Tumor heating', 'Pennes bioheat\nSteady and transient\nR = 10 mm\nDose c', BLUE, BLUE_BG)
box(102, Y, 26, H, 'Thermal dose', 'CEM43, T$_{90}$\nUniform or\nGaussian loading', BLUE, BLUE_BG)
box(133, Y, 26, H, 'Outputs', 'Optimal d, H, f\nDose and\ntemperature\nwindow', GRAY, GRAY_BG)
for a, b in ((29, 34), (64, 69), (97, 102), (128, 133)): arrow(a, Y + H / 2, b, Y + H / 2)
ax.add_patch(FancyBboxPatch((34, 64.5), 94, 7.5, boxstyle='round,pad=0,rounding_size=1.5', lw=1.0, ec=GRAY, fc='white', ls='--'))
ax.text(81, 68.25, 'Field limits: H·f ≤ 5 × 10$^{9}$ A m$^{-1}$ s$^{-1}$, H ≤ 18 kA m$^{-1}$; clinical device at 100 kHz', ha='center', va='center', fontsize=7.0, color=GRAY)
arrow(49, 64.5, 49, Y + H)
Yb, Hb = 3, 22
box(1, Yb, 28, Hb, 'Linear response', 'Baseline only\nτ$_N$, τ$_B$, Rosensweig', ORANGE, ORANGE_BG, ls='--')
box(34, Yb, 30, Hb, 'Measurements', 'SLP in water\nLoops in live cells', ORANGE, ORANGE_BG)
box(69, Yb, 28, Hb, 'Monte Carlo', 'M$_s$, K, α, σ,\nk, ω$_b$, R', ORANGE, ORANGE_BG)
box(102, Yb, 57, Hb, 'Interactions', 'Dipolar coupling λ versus core size and coating', ORANGE, ORANGE_BG)
arrow(20, Yb + Hb, 42, Y, color=ORANGE, ls='--'); arrow(49, Yb + Hb, 49, Y, color=ORANGE); arrow(83, Yb + Hb, 83, Y, color=ORANGE); arrow(115, Yb + Hb, 54, Y, color=ORANGE, ls='--')
fig.savefig(HERE + 'fig_workflow2.png'); plt.close(fig)
