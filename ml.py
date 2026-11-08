"""Machine-learning surrogate of the physics model (model.py) and surrogate-based inverse design.
Run: python ml.py   -> ml_results.json, fig_ml.png
"""
import json, time, warnings
import numpy as np
warnings.filterwarnings('ignore')
from model import kB, mu0, T, P, sar
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.inspection import permutation_importance

RNG = np.random.default_rng(42)
Z = np.linspace(-4, 4, 81); WZ = np.exp(-Z**2 / 2)

def sar_vec(d0, sig, K, H, f, immob):
    """Vectorized SAR (W/g) for arrays of samples; identical physics to model.sar (log-normal, mass-weighted)."""
    d = np.exp(np.log(d0)[:, None] + sig[:, None] * Z[None, :])          # (N, nz)
    V = np.pi * d**3 / 6; VH = np.pi * (d + 2 * P['delta'])**3 / 6
    tN = P['tau0'] * np.exp(np.minimum(K[:, None] * V / (kB * T), 700))
    tB = 3 * P['eta'] * VH / (kB * T)
    tau = np.where(immob[:, None] > 0.5, tN, 1 / (1 / tN + 1 / tB))
    xi = np.maximum(mu0 * P['Ms'] * V * H[:, None] / (kB * T), 1e-6)
    chi0 = mu0 * P['Ms']**2 * V / (3 * kB * T) * 3 / xi * (1 / np.tanh(xi) - 1 / xi)
    w = 2 * np.pi * f[:, None] * tau
    pv = np.pi * mu0 * chi0 * H[:, None]**2 * f[:, None] / (w + 1 / np.maximum(w, 1e-300))
    return (WZ * pv * V).sum(1) / (WZ * P['rho'] * V).sum(1) / 1000

def sample(n):
    return dict(d0=RNG.uniform(6, 26, n) * 1e-9, sig=RNG.uniform(0, 0.3, n), K=np.exp(RNG.uniform(np.log(5e3), np.log(30e3), n)),
                H=RNG.uniform(2e3, 18e3, n), f=RNG.uniform(50e3, 1000e3, n), immob=(RNG.random(n) < 0.5).astype(float))

def features(s):
    return np.column_stack([s['d0'] * 1e9, s['sig'], s['K'] / 1e3, s['H'] / 1e3, s['f'] / 1e3, s['immob']])
FEATS = ['Core diameter', 'Size dispersion σ', 'Anisotropy K', 'Field amplitude H', 'Frequency f', 'Immobilized']

if __name__ == '__main__':
    R = {}
    # consistency check against the reference implementation
    chk = sample(20)
    ref = np.array([sar(chk['d0'][i], chk['H'][i], chk['f'][i], eta=(np.inf if chk['immob'][i] else None), K=chk['K'][i],
                        sigma=chk['sig'][i])[0] for i in range(20)])
    vec = sar_vec(**chk)
    R['max_rel_diff_vs_reference'] = float(np.max(np.abs(vec - ref) / np.maximum(ref, 1e-3)))

    N = 20000
    s = sample(N); t0 = time.perf_counter(); y_sar = sar_vec(**s); t_phys = (time.perf_counter() - t0) / N
    X = features(s); y = np.log10(y_sar + 1)
    Xtr, Xte, ytr, yte, str_, ste = train_test_split(X, y, y_sar, test_size=0.2, random_state=42)
    models = {
        'Random forest': RandomForestRegressor(n_estimators=300, min_samples_leaf=2, n_jobs=-1, random_state=42),
        'Gradient boosting': HistGradientBoostingRegressor(max_iter=800, learning_rate=0.05, max_leaf_nodes=63, random_state=42),
        'Neural network': make_pipeline(StandardScaler(), MLPRegressor(hidden_layer_sizes=(128, 128, 64), alpha=1e-4, max_iter=600,
                                                                      early_stopping=True, random_state=42)),
    }
    R['models'] = {}; preds = {}
    for name, m in models.items():
        t0 = time.perf_counter(); m.fit(Xtr, ytr); fit_t = time.perf_counter() - t0
        t0 = time.perf_counter(); p = m.predict(Xte); pred_t = (time.perf_counter() - t0) / len(Xte)
        ps = 10**p - 1; preds[name] = ps
        mask = ste > 10
        R['models'][name] = dict(R2_log=round(float(r2_score(yte, p)), 4), R2=round(float(r2_score(ste, ps)), 5),
                                 MAE=round(float(mean_absolute_error(ste, ps)), 1),
                                 MAPE_above10=round(float(np.median(np.abs(ps[mask] - ste[mask]) / ste[mask]) * 100), 1),
                                 fit_s=round(fit_t, 1), predict_us=round(pred_t * 1e6, 2))
    best = max(R['models'], key=lambda k: R['models'][k]['R2_log']); R['best'] = best
    R['n_samples'] = N; R['n_train'] = len(Xtr); R['n_test'] = len(Xte); R['phys_us_per_sample'] = round(t_phys * 1e6, 1)
    bm = models[best]
    pi = permutation_importance(bm, Xte[:2000], yte[:2000], n_repeats=5, random_state=42, n_jobs=1)
    R['importance'] = {FEATS[i]: round(float(pi.importances_mean[i]), 3) for i in np.argsort(-pi.importances_mean)}

    # ---- inverse design: best (d0, f) within H*f <= 5e9, H <= 18 kA/m, immobilized ----------------------
    dgrid = np.arange(6, 26.01, 0.1); fgrid = np.arange(50, 1001, 10)
    D, F = np.meshgrid(dgrid, fgrid, indexing='ij'); D = D.ravel(); F = F.ravel()
    Hg = np.minimum(5e9 / (F * 1e3), 18e3)
    inv = []
    t_surr = t_ph = 0.0
    for sg in (0.0, 0.1, 0.2):
        for K in (5e3, 9e3, 13e3, 20e3, 30e3):
            n = len(D); Xg = np.column_stack([D, np.full(n, sg), np.full(n, K / 1e3), Hg / 1e3, F, np.ones(n)])
            t0 = time.perf_counter(); ps = 10**bm.predict(Xg) - 1; t_surr += time.perf_counter() - t0
            t0 = time.perf_counter()
            ph = sar_vec(D * 1e-9, np.full(n, sg), np.full(n, K), Hg, F * 1e3, np.ones(n)); t_ph += time.perf_counter() - t0
            i_s, i_p = int(np.argmax(ps)), int(np.argmax(ph))
            inv.append(dict(sigma=sg, K=K / 1e3, d_surr=round(float(D[i_s]), 1), f_surr=int(F[i_s]), d_phys=round(float(D[i_p]), 1),
                            f_phys=int(F[i_p]), sar_phys_at_surr=round(float(ph[i_s])), sar_phys_opt=round(float(ph[i_p])),
                            regret_pct=round(float(100 * (1 - ph[i_s] / ph[i_p])), 1)))
    R['inverse'] = inv
    R['inverse_summary'] = dict(mean_abs_dd=round(float(np.mean([abs(r['d_surr'] - r['d_phys']) for r in inv])), 2),
                                max_abs_dd=round(float(np.max([abs(r['d_surr'] - r['d_phys']) for r in inv])), 1),
                                mean_regret=round(float(np.mean([r['regret_pct'] for r in inv])), 1),
                                max_regret=round(float(np.max([r['regret_pct'] for r in inv])), 1),
                                grid_points=len(D), t_surrogate_s=round(t_surr, 2), t_physics_s=round(t_ph, 2))
    json.dump(R, open('ml_results.json', 'w'), indent=1, ensure_ascii=False)
    np.savez('ml_test.npz', true=ste, pred=preds[best])
    print(json.dumps({k: v for k, v in R.items() if k != 'inverse'}, indent=1, ensure_ascii=False))
    for r in inv: print(r)
