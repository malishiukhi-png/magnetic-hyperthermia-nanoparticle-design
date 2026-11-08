"""Figure 1: modeling workflow (physics model + machine-learning surrogate)."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

plt.rcParams.update({'font.family': 'Liberation Sans', 'font.size': 8.5, 'savefig.dpi': 300})
BLUE, BLUE_BG = '#2a78d6', '#e8f1fc'
ORANGE, ORANGE_BG = '#d95926', '#fdeee7'
GRAY, GRAY_BG = '#52514e', '#f3f2ef'
INK = '#0b0b0b'
CM = 1 / 2.54
fig = plt.figure(figsize=(16 * CM, 7.4 * CM))
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 160); ax.set_ylim(0, 74); ax.axis('off')

def box(x, y, w, h, title, body, edge, face):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0,rounding_size=2', lw=1.2, ec=edge, fc=face))
    ax.text(x + w / 2, y + h - 3.2, title, ha='center', va='top', fontweight='bold', color=INK, fontsize=8.8)
    ax.text(x + w / 2, y + h - 8.6, body, ha='center', va='top', color=INK, fontsize=7.6, linespacing=1.35)

def arrow(x1, y1, x2, y2, color=INK, ls='-', text=None, tx=0, ty=0):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='-|>', color=color, lw=1.2, ls=ls, mutation_scale=10, shrinkA=0, shrinkB=0))
    if text:
        ax.text((x1 + x2) / 2 + tx, (y1 + y2) / 2 + ty, text, ha='center', va='center', fontsize=7.2, color=color, style='italic')

# ---- physics pipeline (top row) ----
Y, H = 33, 28
box(1, Y, 28, H, 'Inputs', 'Core size d, σ\nAnisotropy K\nCoating δ\nWater or immobilized\nField H, f', GRAY, GRAY_BG)
box(34, Y, 28, H, 'Relaxation', 'Néel τ$_N$\nBrownian τ$_B$\nEffective τ\n(Eqs. 1–3)', BLUE, BLUE_BG)
box(67, Y, 28, H, 'Heat dissipation', 'Linear response\ntheory\nSAR (W g$^{-1}$)\n(Eqs. 4–6)', BLUE, BLUE_BG)
box(100, Y, 28, H, 'Tumor heating', 'Pennes bioheat\nR = 10 mm\nT(r), dose c\n(Eq. 7)', BLUE, BLUE_BG)
box(133, Y, 26, H, 'Outputs', 'Optimal d, f, H\nTemperature\nwindow\nRequired dose', GRAY, GRAY_BG)
for x in (29, 62, 95, 128):
    arrow(x, Y + H / 2, x + 5 if x != 128 else x + 5, Y + H / 2)
# safety limit feeding the field choice
ax.add_patch(FancyBboxPatch((43, 64.5), 76, 7.5, boxstyle='round,pad=0,rounding_size=1.5', lw=1.0, ec=GRAY, fc='white', ls='--'))
ax.text(81, 68.25, 'Safety limit: H·f ≤ 5 × 10$^{9}$ A m$^{-1}$ s$^{-1}$, H ≤ 18 kA m$^{-1}$', ha='center', va='center', fontsize=7.6, color=GRAY)
arrow(81, 64.5, 81, Y + H)

# ---- machine-learning branch (bottom row) ----
Yb, Hb = 3, 22
box(34, Yb, 28, Hb, 'Training data', '20,000 simulated\ncases', ORANGE, ORANGE_BG)
box(67, Yb, 28, Hb, 'Surrogate model', 'Neural network\nvs. tree ensembles', ORANGE, ORANGE_BG)
box(100, Yb, 28, Hb, 'Inverse design', 'Grid search for\nmaximum SAR', ORANGE, ORANGE_BG)
arrow(81, Y, 52, Yb + Hb, color=ORANGE, text='sampling', tx=-6, ty=1.5)
arrow(62, Yb + Hb / 2, 67, Yb + Hb / 2, color=ORANGE)
arrow(95, Yb + Hb / 2, 100, Yb + Hb / 2, color=ORANGE)
arrow(128, Yb + Hb / 2, 146, Y, color=ORANGE)
arrow(114, Yb + Hb, 84, Y, color=ORANGE, ls='--', text='verification', tx=8, ty=1.5)
fig.savefig('fig_workflow.png'); plt.close(fig)
