"""Linear-response (Rosensweig) model of magnetic nanoparticle heating + Pennes bioheat (steady, spherical)."""
import numpy as np
kB = 1.380649e-23; mu0 = 4e-7 * np.pi
T = 310.15            # K, body temperature
P = dict(Ms=446e3,    # A/m, magnetite (Rosensweig 2002)
         K=13e3,      # J/m3, |K1| of magnetite
         rho=5180.0,  # kg/m3
         tau0=1e-9,   # s
         delta=5e-9,  # m, coating/shell thickness -> hydrodynamic diameter d + 2*delta
         eta=0.69e-3) # Pa s, water at 37 C

def taus(d, eta=None, K=None, **kw):
    p = dict(P, **kw); eta = p['eta'] if eta is None else eta; K = p['K'] if K is None else K
    V = np.pi * d**3 / 6; VH = np.pi * (d + 2 * p['delta'])**3 / 6
    tN = p['tau0'] * np.exp(np.minimum(K * V / (kB * T), 700))
    tB = 3 * eta * VH / (kB * T) if np.isfinite(eta) else np.full_like(np.asarray(d, float), np.inf)
    teff = 1 / (1 / tN + 1 / tB)
    return tN, tB, teff

def power_per_vol(d, H, f, eta=None, K=None, **kw):
    """Heating power per unit volume of magnetic material (W/m3), LRT."""
    p = dict(P, **kw)
    V = np.pi * d**3 / 6
    xi = mu0 * p['Ms'] * V * H / (kB * T)
    chi_i_over_phi = mu0 * p['Ms']**2 * V / (3 * kB * T)
    L = np.where(xi > 1e-6, (1 / np.tanh(np.maximum(xi, 1e-6)) - 1 / np.maximum(xi, 1e-6)), xi / 3)
    chi0 = chi_i_over_phi * 3 / np.maximum(xi, 1e-6) * L
    _, _, t = taus(d, eta, K, **kw)
    w = 2 * np.pi * f * t
    lor = 1.0 / (w + 1.0 / np.maximum(w, 1e-300))   # = w/(1+w^2), overflow-safe
    return np.pi * mu0 * chi0 * H**2 * f * lor, xi

def sar(d, H, f, eta=None, K=None, sigma=0.0, **kw):
    """SAR in W/g of magnetic material; sigma = lognormal width of the number-weighted size distribution."""
    p = dict(P, **kw)
    d = np.atleast_1d(np.asarray(d, float))
    if sigma <= 0:
        pv, _ = power_per_vol(d, H, f, eta, K, **kw)
        return pv / p['rho'] / 1000
    out = []
    for d0 in d:
        x = np.linspace(np.log(d0) - 4 * sigma, np.log(d0) + 4 * sigma, 401)
        di = np.exp(x); g = np.exp(-(x - np.log(d0))**2 / (2 * sigma**2))
        pv, _ = power_per_vol(di, H, f, eta, K, **kw)
        Vi = di**3
        out.append(np.sum(g * pv * Vi) / np.sum(g * p['rho'] * Vi) / 1000)
    return np.array(out)

def xi_of(d, H, **kw):
    p = dict(P, **kw); return mu0 * p['Ms'] * (np.pi * d**3 / 6) * H / (kB * T)

# ---- Pennes bioheat, steady state, sphere of radius R with uniform source Q inside -------------
def bioheat(r, R, Q, Wt, Wh, k=0.5):
    """Temperature rise theta(r) (K). Wt, Wh: perfusion terms w_b*rho_b*c_b (W/m3/K) in tumor / healthy tissue."""
    m1 = np.sqrt(Wt / k); m2 = np.sqrt(Wh / k)
    th0 = Q / Wt
    # inside: theta = A sinh(m1 r)/r + th0 ; outside: theta = B exp(-m2 r)/r
    s, c = np.sinh(m1 * R), np.cosh(m1 * R)
    # continuity of theta and of k dtheta/dr at R
    # A s/R + th0 = B e/R ;  A (m1 c R - s)/R^2 = B e (-m2 R - 1)/R^2
    e = np.exp(-m2 * R)
    a11, a12, b1 = s / R, -e / R, -th0
    a21, a22, b2 = (m1 * c * R - s) / R**2, -e * (-m2 * R - 1) / R**2, 0.0
    A, B = np.linalg.solve([[a11, a12], [a21, a22]], [b1, b2])
    r = np.asarray(r, float)
    rin = np.maximum(r, 1e-9)
    inside = np.where(r < 1e-9, A * m1 + th0, A * np.sinh(m1 * rin) / rin + th0)
    outside = B * np.exp(-m2 * rin) / rin
    return np.where(r <= R, inside, outside)
