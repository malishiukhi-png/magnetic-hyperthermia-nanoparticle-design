/* Stochastic Landau-Lifshitz-Gilbert simulation of uniaxial nanoparticles that can rotate in a viscous liquid.
 * Magnetization: same equation, Heun scheme and thermal field as sllg.c. Easy axis: overdamped rigid rotation under the magnetic torque
 * 2 K V (m.n)(n x m) and a Brownian torque, dn/dt = omega x n, omega = [torque + thermal torque] / (6 eta V_h).
 *
 * Build:  gcc -O3 -march=native -ffast-math -fno-math-errno -fopenmp-simd sllg_rot.c -o sllg_rot -lm -lmvec
 *
 * Usage:  sllg_rot D_nm SIGMA ALPHA H_A_per_m F_Hz MS_A_per_m K_J_per_m3 N NCYC SEED T_K SHELL_nm ETA_Pa_s
 *         the hydrodynamic diameter of a particle is d + 2*SHELL; the loop area is averaged over the second half of the NCYC cycles.
 * Output: sar_W_per_kg stderr_W_per_kg nper N
 *
 * sllg_rot selftest   checks the rotational diffusion against <n(t).n(0)> = exp(-2 D_r t).
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdint.h>
#include <string.h>

#define GAMMA 1.760859e11
#define MU0 (4e-7*M_PI)
#define KB 1.380649e-23
#define RHO 5180.0
#define BLK 64

typedef struct { uint64_t s[4]; } rng_t;
static inline uint64_t rotl(uint64_t x, int k) { return (x << k) | (x >> (64 - k)); }
static inline uint64_t next64(rng_t *r) {
    uint64_t *s = r->s, res = rotl(s[1] * 5, 7) * 9, t = s[1] << 17;
    s[2] ^= s[0]; s[3] ^= s[1]; s[1] ^= s[2]; s[0] ^= s[3]; s[2] ^= t; s[3] = rotl(s[3], 45);
    return res;
}
static void seed_rng(rng_t *r, uint64_t seed) {
    for (int i = 0; i < 4; i++) {
        uint64_t z = (seed += 0x9e3779b97f4a7c15ULL);
        z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL; z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
        r->s[i] = z ^ (z >> 31);
    }
}
static inline double unif(rng_t *r) { return (next64(r) >> 11) * (1.0 / 9007199254740992.0); }
static void gauss_block(rng_t *r, double *g, int n) {
    double u1[n], u2[n];
    for (int i = 0; i < n; i++) { u1[i] = 1.0 - unif(r); u2[i] = unif(r); }
#pragma omp simd
    for (int i = 0; i < n; i++) g[i] = sqrt(-2.0 * log(u1[i])) * cos(2.0 * M_PI * u2[i]);
}

static int selftest(void) {
    /* free rotational diffusion (no torque) of N axes starting along z; Heun scheme with renormalization, as in the production loop */
    int N = 64000; double dh = 40e-9, eta = 1e-3, T = 300, Vh = M_PI * dh * dh * dh / 6, xi = 6 * eta * Vh, Dr = KB * T / xi;
    double tend = 1.0 / (2 * Dr), dt = tend / 2000; double B = sqrt(2 * KB * T / (xi * dt));
    double *x = calloc(N, 8), *y = calloc(N, 8), *z = malloc(N * 8); for (int i = 0; i < N; i++) z[i] = 1;
    rng_t r; seed_rng(&r, 5); double g[3];
    for (int k = 0; k < 2000; k++)
        for (int i = 0; i < N; i++) {
            gauss_block(&r, g, 3);
            double wx = B * g[0], wy = B * g[1], wz = B * g[2];
            double dx = wy * z[i] - wz * y[i], dy = wz * x[i] - wx * z[i], dz = wx * y[i] - wy * x[i];
            double px = x[i] + dx * dt, py = y[i] + dy * dt, pz = z[i] + dz * dt, nn = 1 / sqrt(px * px + py * py + pz * pz); px *= nn; py *= nn; pz *= nn;
            double ex = wy * pz - wz * py, ey = wz * px - wx * pz, ez = wx * py - wy * px;
            double ax = x[i] + 0.5 * (dx + ex) * dt, ay = y[i] + 0.5 * (dy + ey) * dt, az = z[i] + 0.5 * (dz + ez) * dt; nn = 1 / sqrt(ax * ax + ay * ay + az * az);
            x[i] = ax * nn; y[i] = ay * nn; z[i] = az * nn;
        }
    double s = 0; for (int i = 0; i < N; i++) s += z[i]; s /= N;
    int bad = fabs(s - exp(-1.0)) > 0.01;
    printf("selftest %s: <n.n0> at t = 1/(2 Dr) is %.4f (analytic %.4f)\n", bad ? "FAILED" : "passed", s, exp(-1.0));
    return bad;
}

int main(int argc, char **argv) {
    if (argc == 2 && !strcmp(argv[1], "selftest")) return selftest();
    if (argc < 14) { fprintf(stderr, "usage: see header\n"); return 2; }
    double d0 = atof(argv[1]) * 1e-9, sigma = atof(argv[2]), alpha = atof(argv[3]), H0 = atof(argv[4]), f = atof(argv[5]);
    double Ms = atof(argv[6]), K = atof(argv[7]);
    int N = atoi(argv[8]), ncyc = atoi(argv[9]); uint64_t seed = strtoull(argv[10], NULL, 10);
    double T = atof(argv[11]), shell = atof(argv[12]) * 1e-9, eta = atof(argv[13]);
    N = (N + BLK - 1) / BLK * BLK;
    double HK = 2.0 * K / (MU0 * Ms), dt = 2.0 * M_PI / (GAMMA * MU0 * (HK + H0)) / 40.0;
    long nper = (long)ceil(1.0 / f / dt); dt = 1.0 / f / nper;
    long nsteps = nper * ncyc, rec = nper / 2000; if (rec < 1) rec = 1;
    rng_t rg; seed_rng(&rg, seed);
#define AL(n) aligned_alloc(64, (n) * 8)
    double *nx = AL(N), *ny = AL(N), *nz = AL(N), *mx = AL(N), *my = AL(N), *mz = AL(N), *Hs = AL(N), *wt = AL(N), *Ai = AL(N), *Bi = AL(N);
    double Vtot = 0;
    for (int i = 0; i < N; i++) {
        double dd = d0;
        if (sigma > 0) { double g; do gauss_block(&rg, &g, 1); while (fabs(g) > 4.0); dd = d0 * exp(sigma * g); }
        double V = M_PI * dd * dd * dd / 6.0, dh = dd + 2 * shell, xi = 6.0 * eta * M_PI * dh * dh * dh / 6.0;
        Hs[i] = sqrt(2.0 * alpha * KB * T / (GAMMA * MU0 * MU0 * Ms * V * dt));
        Ai[i] = 2.0 * K * V / xi; Bi[i] = sqrt(2.0 * KB * T / (xi * dt));
        wt[i] = V; Vtot += V;
        double u = 2.0 * unif(&rg) - 1.0, ph = 2.0 * M_PI * unif(&rg), s = sqrt(1.0 - u * u);
        nx[i] = s * cos(ph); ny[i] = s * sin(ph); nz[i] = u;
        double sg = unif(&rg) < 0.5 ? -1.0 : 1.0;
        mx[i] = sg * nx[i]; my[i] = sg * ny[i]; mz[i] = sg * nz[i];
    }
    for (int i = 0; i < N; i++) wt[i] /= Vtot;
    int nb = N / BLK; rng_t *rb = malloc(sizeof(rng_t) * nb);
    for (int b = 0; b < nb; b++) seed_rng(&rb[b], seed * 1000003ULL + 17ULL * b + 1ULL);
    long nrec = nsteps / rec + 1; double *out = malloc(nrec * 8); long r = 0;
    double g = GAMMA * MU0 / (1.0 + alpha * alpha), w = 2.0 * M_PI * f;
    for (long k = 0; k <= nsteps; k++) {
        if (k % rec == 0) {
            double s = 0.0;
#pragma omp simd reduction(+:s)
            for (int i = 0; i < N; i++) s += wt[i] * mz[i];
            out[r++] = s;
        }
        if (k == nsteps) break;
        double ha0 = H0 * cos(w * k * dt), ha1 = H0 * cos(w * (k + 1) * dt);
        for (int b = 0; b < nb; b++) {
            int i0 = b * BLK;
            double gx[BLK], gy[BLK], gz[BLK], ux[BLK], uy[BLK], uz[BLK];
            gauss_block(&rb[b], gx, BLK); gauss_block(&rb[b], gy, BLK); gauss_block(&rb[b], gz, BLK);
            gauss_block(&rb[b], ux, BLK); gauss_block(&rb[b], uy, BLK); gauss_block(&rb[b], uz, BLK);
            double *restrict Mx = mx + i0, *restrict My = my + i0, *restrict Mz = mz + i0, *restrict Nx = nx + i0, *restrict Ny = ny + i0, *restrict Nz = nz + i0;
            double *restrict Hh = Hs + i0, *restrict A = Ai + i0, *restrict Bb = Bi + i0;
#pragma omp simd
            for (int i = 0; i < BLK; i++) {
                double tx = Hh[i] * gx[i], ty = Hh[i] * gy[i], tz = Hh[i] * gz[i];
                double mn = Mx[i] * Nx[i] + My[i] * Ny[i] + Mz[i] * Nz[i];
                /* predictor: magnetization */
                double hx = HK * mn * Nx[i] + tx, hy = HK * mn * Ny[i] + ty, hz = HK * mn * Nz[i] + tz + ha0;
                double cx = My[i] * hz - Mz[i] * hy, cy = Mz[i] * hx - Mx[i] * hz, cz = Mx[i] * hy - My[i] * hx;
                double dx = -g * (cx + alpha * (My[i] * cz - Mz[i] * cy)), dy = -g * (cy + alpha * (Mz[i] * cx - Mx[i] * cz)), dz = -g * (cz + alpha * (Mx[i] * cy - My[i] * cx));
                /* predictor: easy axis, torque 2 K V (m.n) (n x m) */
                double cnx = Ny[i] * Mz[i] - Nz[i] * My[i], cny = Nz[i] * Mx[i] - Nx[i] * Mz[i], cnz = Nx[i] * My[i] - Ny[i] * Mx[i];
                double wx = A[i] * mn * cnx + Bb[i] * ux[i], wy = A[i] * mn * cny + Bb[i] * uy[i], wz = A[i] * mn * cnz + Bb[i] * uz[i];
                double ex = wy * Nz[i] - wz * Ny[i], ey = wz * Nx[i] - wx * Nz[i], ez = wx * Ny[i] - wy * Nx[i];
                double px = Mx[i] + dx * dt, py = My[i] + dy * dt, pz = Mz[i] + dz * dt, nn = 1.0 / sqrt(px * px + py * py + pz * pz); px *= nn; py *= nn; pz *= nn;
                double qx = Nx[i] + ex * dt, qy = Ny[i] + ey * dt, qz = Nz[i] + ez * dt; nn = 1.0 / sqrt(qx * qx + qy * qy + qz * qz); qx *= nn; qy *= nn; qz *= nn;
                /* corrector */
                double mn2 = px * qx + py * qy + pz * qz;
                double hx2 = HK * mn2 * qx + tx, hy2 = HK * mn2 * qy + ty, hz2 = HK * mn2 * qz + tz + ha1;
                double cx2 = py * hz2 - pz * hy2, cy2 = pz * hx2 - px * hz2, cz2 = px * hy2 - py * hx2;
                double dx2 = -g * (cx2 + alpha * (py * cz2 - pz * cy2)), dy2 = -g * (cy2 + alpha * (pz * cx2 - px * cz2)), dz2 = -g * (cz2 + alpha * (px * cy2 - py * cx2));
                double cn2x = qy * pz - qz * py, cn2y = qz * px - qx * pz, cn2z = qx * py - qy * px;
                double w2x = A[i] * mn2 * cn2x + Bb[i] * ux[i], w2y = A[i] * mn2 * cn2y + Bb[i] * uy[i], w2z = A[i] * mn2 * cn2z + Bb[i] * uz[i];
                double e2x = w2y * qz - w2z * qy, e2y = w2z * qx - w2x * qz, e2z = w2x * qy - w2y * qx;
                double ax = Mx[i] + 0.5 * (dx + dx2) * dt, ay = My[i] + 0.5 * (dy + dy2) * dt, az = Mz[i] + 0.5 * (dz + dz2) * dt; nn = 1.0 / sqrt(ax * ax + ay * ay + az * az);
                Mx[i] = ax * nn; My[i] = ay * nn; Mz[i] = az * nn;
                ax = Nx[i] + 0.5 * (ex + e2x) * dt; ay = Ny[i] + 0.5 * (ey + e2y) * dt; az = Nz[i] + 0.5 * (ez + e2z) * dt; nn = 1.0 / sqrt(ax * ax + ay * ay + az * az);
                Nx[i] = ax * nn; Ny[i] = ay * nn; Nz[i] = az * nn;
            }
        }
    }
    double mean = 0, var = 0, areas[256]; int c0 = ncyc / 2, nc = ncyc - c0; long per = nper / rec;
    for (int c = c0; c < ncyc; c++) {
        double A = 0;
        for (long j = c * per; j < (c + 1) * per; j++) {
            double t0 = (double)(j * rec) / nper, t1 = (double)((j + 1) * rec) / nper;
            A += 0.5 * (out[j] + out[j + 1]) * (H0 * cos(2 * M_PI * t1) - H0 * cos(2 * M_PI * t0));
        }
        areas[c - c0] = -MU0 * Ms * A; mean += areas[c - c0];
    }
    mean /= nc;
    for (int c = 0; c < nc; c++) var += (areas[c] - mean) * (areas[c] - mean);
    printf("%.6g %.6g %ld %d\n", mean * f / RHO, nc > 1 ? sqrt(var / (nc - 1) / nc) * f / RHO : 0.0, nper, N);
    return 0;
}
