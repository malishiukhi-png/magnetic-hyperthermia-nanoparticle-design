/* Stochastic Landau-Lifshitz-Gilbert dynamics of non-interacting single-domain particles (macrospin) in an AC field.
   Heun scheme (Stratonovich), Brown thermal field. Uniaxial (mode 0) or cubic (mode 1) magnetocrystalline anisotropy.
   C port of the numba kernel in llg.py, extended to cubic anisotropy.
   Build: gcc -O3 -march=native -ffast-math -shared -fPIC -o llg_kernel.so llg_kernel.c -lm            */
#include <math.h>
#include <stdint.h>

static uint64_t s0, s1;
static inline uint64_t xrand(void) {               /* xorshift128+ */
    uint64_t x = s0; const uint64_t y = s1; s0 = y; x ^= x << 23; s1 = x ^ y ^ (x >> 17) ^ (y >> 26); return s1 + y;
}
static inline double urand(void) { return ((xrand() >> 11) + 0.5) * (1.0 / 9007199254740992.0); }
static int have_spare = 0; static double spare;
static inline double nrand(void) {                 /* Box-Muller */
    if (have_spare) { have_spare = 0; return spare; }
    double u = urand(), v = urand(); double r = sqrt(-2.0 * log(u)); double t = 6.283185307179586 * v;
    spare = r * sin(t); have_spare = 1; return r * cos(t);
}

/* anisotropy field (A/m) in lab frame for unit moment m. R: 9 numbers, columns = crystal axes (or easy axis n in R[0..2]) */
static inline void hanis(int mode, const double *R, double HK, double mx, double my, double mz,
                         double *hx, double *hy, double *hz) {
    if (mode == 0) {
        double mn = mx * R[0] + my * R[1] + mz * R[2];
        *hx = HK * mn * R[0]; *hy = HK * mn * R[1]; *hz = HK * mn * R[2];
    } else {
        /* crystal components a_k = (column k) . m ; E = K1 (a1^2 a2^2 + ...) ; H_k = -(2K1/mu0 Ms) a_k (a_j^2 + a_l^2)
           here HK = 2 K1 / (mu0 Ms) with sign of K1 (negative for magnetite) */
        double a1 = R[0] * mx + R[1] * my + R[2] * mz;
        double a2 = R[3] * mx + R[4] * my + R[5] * mz;
        double a3 = R[6] * mx + R[7] * my + R[8] * mz;
        double q1 = a1 * a1, q2 = a2 * a2, q3 = a3 * a3;
        double h1 = -HK * a1 * (q2 + q3), h2 = -HK * a2 * (q1 + q3), h3 = -HK * a3 * (q1 + q2);
        *hx = h1 * R[0] + h2 * R[3] + h3 * R[6];
        *hy = h1 * R[1] + h2 * R[4] + h3 * R[7];
        *hz = h1 * R[2] + h2 * R[5] + h3 * R[8];
    }
}

void run(int N, int mode, const double *Rall, const double *m0, double HK, double Hth_sd, double H0, double w,
         double dt, long nsteps, double alpha, double gamma_mu0, long rec_every, uint64_t seed, double *out) {
    s0 = seed * 0x9E3779B97F4A7C15ULL + 1; s1 = (seed + 7) * 0xBF58476D1CE4E5B9ULL + 3; have_spare = 0;
    for (int i = 0; i < 20; i++) xrand();
    double mxa[N], mya[N], mza[N];
    for (int i = 0; i < N; i++) { mxa[i] = m0[3 * i]; mya[i] = m0[3 * i + 1]; mza[i] = m0[3 * i + 2]; }
    double g = gamma_mu0 / (1.0 + alpha * alpha);
    long r = 0;
    for (long k = 0; k <= nsteps; k++) {
        if (k % rec_every == 0) { double s = 0; for (int i = 0; i < N; i++) s += mza[i]; out[r++] = s / N; }
        if (k == nsteps) break;
        double ha0 = H0 * cos(w * k * dt), ha1 = H0 * cos(w * (k + 1) * dt);
        for (int i = 0; i < N; i++) {
            const double *R = Rall + 9 * i;
            double mx = mxa[i], my = mya[i], mz = mza[i];
            double tx = Hth_sd * nrand(), ty = Hth_sd * nrand(), tz = Hth_sd * nrand();
            double hx, hy, hz; hanis(mode, R, HK, mx, my, mz, &hx, &hy, &hz);
            hx += tx; hy += ty; hz += tz + ha0;
            double cx = my * hz - mz * hy, cy = mz * hx - mx * hz, cz = mx * hy - my * hx;
            double dx = -g * (cx + alpha * (my * cz - mz * cy));
            double dy = -g * (cy + alpha * (mz * cx - mx * cz));
            double dz = -g * (cz + alpha * (mx * cy - my * cx));
            double px = mx + dx * dt, py = my + dy * dt, pz = mz + dz * dt;
            double nn = sqrt(px * px + py * py + pz * pz); px /= nn; py /= nn; pz /= nn;
            double hx2, hy2, hz2; hanis(mode, R, HK, px, py, pz, &hx2, &hy2, &hz2);
            hx2 += tx; hy2 += ty; hz2 += tz + ha1;
            double cx2 = py * hz2 - pz * hy2, cy2 = pz * hx2 - px * hz2, cz2 = px * hy2 - py * hx2;
            double dx2 = -g * (cx2 + alpha * (py * cz2 - pz * cy2));
            double dy2 = -g * (cy2 + alpha * (pz * cx2 - px * cz2));
            double dz2 = -g * (cz2 + alpha * (px * cy2 - py * cx2));
            double ax = mx + 0.5 * (dx + dx2) * dt, ay = my + 0.5 * (dy + dy2) * dt, az = mz + 0.5 * (dz + dz2) * dt;
            nn = sqrt(ax * ax + ay * ay + az * az);
            mxa[i] = ax / nn; mya[i] = ay / nn; mza[i] = az / nn;
        }
    }
}
