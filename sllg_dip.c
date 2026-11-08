/* Stochastic Landau-Lifshitz-Gilbert simulation of immobilized uniaxial nanoparticles with explicit dipolar coupling.
 * Same equation, Heun scheme, Brown thermal field and cycle averaging as sllg.c. Particles are grouped in independent random clusters.
 *
 * Build:  gcc -O3 -march=native -ffast-math -fno-math-errno -fopenmp-simd sllg_dip.c -o sllg_dip -lm -lmvec
 *
 * Usage:  sllg_dip D_nm SIGMA ALPHA H_A_per_m F_Hz MS_A_per_m K_J_per_m3 SHELL_nm PHI NCL NCLUSTERS NCYC SEED T_K DIP
 *         D_nm    median core diameter; SIGMA > 0 draws a log-normal number distribution truncated at +-4 sigma
 *         SHELL_nm coating thickness: spheres of diameter d + 2*SHELL do not overlap
 *         PHI     volume fraction of the coated spheres in the spherical cluster (random sequential addition)
 *         NCL     particles per cluster; NCLUSTERS independent clusters
 *         DIP     1 = dipolar coupling on, 0 = off (same geometry, used as the reference and to check against sllg.c)
 * Output: sar_W_per_kg stderr_W_per_kg nper N Hdip_max_A_per_m
 *
 * sllg_dip selftest   checks the dipolar field of a two-particle system against the analytic values.
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

/* dipolar tensor of one cluster; H_i = Ms * sum_j T_ij m_j, T = V_j/(4 pi r^3) (3 u u - 1) */
typedef struct { double *xx, *xy, *xz, *yy, *yz, *zz; } tens_t;
static void tens_alloc(tens_t *t, int n) {
    double **p[6] = {&t->xx, &t->xy, &t->xz, &t->yy, &t->yz, &t->zz};
    for (int k = 0; k < 6; k++) { *p[k] = aligned_alloc(64, ((size_t)n * n * 8 + 63) / 64 * 64); memset(*p[k], 0, (size_t)n * n * 8); }
}
static double tens_fill(tens_t *t, int n, const double *x, const double *y, const double *z, const double *V) {
    double hmax = 0;
    for (int i = 0; i < n; i++) {
        double s = 0;
        for (int j = 0; j < n; j++) {
            if (i == j) continue;
            double rx = x[i] - x[j], ry = y[i] - y[j], rz = z[i] - z[j], r2 = rx * rx + ry * ry + rz * rz, r = sqrt(r2);
            double A = V[j] / (4.0 * M_PI * r2 * r), ux = rx / r, uy = ry / r, uz = rz / r;
            t->xx[i * n + j] = A * (3 * ux * ux - 1); t->xy[i * n + j] = A * 3 * ux * uy; t->xz[i * n + j] = A * 3 * ux * uz;
            t->yy[i * n + j] = A * (3 * uy * uy - 1); t->yz[i * n + j] = A * 3 * uy * uz; t->zz[i * n + j] = A * (3 * uz * uz - 1);
            s += 2 * A;
        }
        if (s > hmax) hmax = s;
    }
    return hmax;
}
static inline void dip_field(const tens_t *t, int n, double Ms, const double *mx, const double *my, const double *mz, double *hx, double *hy, double *hz) {
    for (int i = 0; i < n; i++) {
        const double *xx = t->xx + i * n, *xy = t->xy + i * n, *xz = t->xz + i * n, *yy = t->yy + i * n, *yz = t->yz + i * n, *zz = t->zz + i * n;
        double a = 0, b = 0, c = 0;
#pragma omp simd reduction(+:a,b,c)
        for (int j = 0; j < n; j++) {
            a += xx[j] * mx[j] + xy[j] * my[j] + xz[j] * mz[j];
            b += xy[j] * mx[j] + yy[j] * my[j] + yz[j] * mz[j];
            c += xz[j] * mx[j] + yz[j] * my[j] + zz[j] * mz[j];
        }
        hx[i] = Ms * a; hy[i] = Ms * b; hz[i] = Ms * c;
    }
}
static int selftest(void) {
    tens_t t; tens_alloc(&t, 2);
    double d = 20e-9, r = 30e-9, V = M_PI * d * d * d / 6, Ms = 446e3, Vv[2] = {V, V}, A = Ms * V / (4 * M_PI * r * r * r);
    double x[2] = {0, 0}, y[2] = {0, 0}, z[2] = {0, r};
    double mx[2] = {0, 0}, my[2] = {0, 0}, mz[2] = {1, 1}, hx[2], hy[2], hz[2];
    tens_fill(&t, 2, x, y, z, Vv); dip_field(&t, 2, Ms, mx, my, mz, hx, hy, hz);
    int bad = fabs(hz[0] - 2 * A) > 1e-9 * A || fabs(hz[1] - 2 * A) > 1e-9 * A;
    double x2[2] = {0, r}, z2[2] = {0, 0};
    tens_fill(&t, 2, x2, y, z2, Vv); dip_field(&t, 2, Ms, mx, my, mz, hx, hy, hz);
    bad |= fabs(hz[0] + A) > 1e-9 * A || fabs(hz[1] + A) > 1e-9 * A;
    printf("selftest %s: head-to-tail field %.1f A/m (analytic %.1f), side-by-side %.1f (analytic %.1f)\n", bad ? "FAILED" : "passed", 2 * A, 2 * A, hz[0], -A);
    return bad;
}

int main(int argc, char **argv) {
    if (argc == 2 && !strcmp(argv[1], "selftest")) return selftest();
    if (argc < 16) { fprintf(stderr, "usage: see header\n"); return 2; }
    double d0 = atof(argv[1]) * 1e-9, sigma = atof(argv[2]), alpha = atof(argv[3]), H0 = atof(argv[4]), f = atof(argv[5]);
    double Ms = atof(argv[6]), K = atof(argv[7]), shell = atof(argv[8]) * 1e-9, phi = atof(argv[9]);
    int n = atoi(argv[10]), nc = atoi(argv[11]), ncyc = atoi(argv[12]); uint64_t seed = strtoull(argv[13], NULL, 10);
    double T = atof(argv[14]); int dip = atoi(argv[15]);
    int N = n * nc;
    double HK = 2.0 * K / (MU0 * Ms);
    rng_t rg; seed_rng(&rg, seed);
    double *nx = aligned_alloc(64, N * 8), *ny = aligned_alloc(64, N * 8), *nz = aligned_alloc(64, N * 8);
    double *mx = aligned_alloc(64, N * 8), *my = aligned_alloc(64, N * 8), *mz = aligned_alloc(64, N * 8);
    double *Vv = aligned_alloc(64, N * 8), *wt = aligned_alloc(64, N * 8), *Hs = aligned_alloc(64, N * 8);
    tens_t *TT = malloc(sizeof(tens_t) * nc);
    double Vtot = 0, Hdmax = 0;
    for (int c = 0; c < nc; c++) {
        double dd[n], Dh[n], x[n], y[n], z[n];
        for (int i = 0; i < n; i++) {
            double g = 0;
            if (sigma > 0) { do gauss_block(&rg, &g, 1); while (fabs(g) > 4.0); }
            dd[i] = d0 * exp(sigma * g);
        }
        for (int i = 1; i < n; i++) { double v = dd[i]; int j = i - 1; while (j >= 0 && dd[j] < v) { dd[j + 1] = dd[j]; j--; } dd[j + 1] = v; }
        double sum3 = 0;
        for (int i = 0; i < n; i++) { Dh[i] = dd[i] + 2 * shell; sum3 += Dh[i] * Dh[i] * Dh[i]; }
        double R = cbrt(sum3 / (8.0 * phi));
        for (;;) {
            int ok = 1;
            for (int i = 0; i < n && ok; i++) {
                long tries = 0; int placed = 0;
                while (tries++ < 2000000 && !placed) {
                    double rmax = R - Dh[i] / 2, px, py, pz;
                    do { px = 2 * unif(&rg) - 1; py = 2 * unif(&rg) - 1; pz = 2 * unif(&rg) - 1; } while (px * px + py * py + pz * pz > 1);
                    px *= rmax; py *= rmax; pz *= rmax; placed = 1;
                    for (int j = 0; j < i; j++) {
                        double rx = px - x[j], ry = py - y[j], rz = pz - z[j], s = 0.5 * (Dh[i] + Dh[j]);
                        if (rx * rx + ry * ry + rz * rz < s * s) { placed = 0; break; }
                    }
                    if (placed) { x[i] = px; y[i] = py; z[i] = pz; }
                }
                if (!placed) ok = 0;
            }
            if (ok) break;
            R *= 1.01;
        }
        double vv[n];
        for (int i = 0; i < n; i++) {
            int k = c * n + i;
            double V = M_PI * dd[i] * dd[i] * dd[i] / 6.0; vv[i] = V; Vv[k] = V; wt[k] = V; Vtot += V;
            double u = 2.0 * unif(&rg) - 1.0, ph = 2.0 * M_PI * unif(&rg), s = sqrt(1.0 - u * u);
            nx[k] = s * cos(ph); ny[k] = s * sin(ph); nz[k] = u;
            double sg = unif(&rg) < 0.5 ? -1.0 : 1.0;
            mx[k] = sg * nx[k]; my[k] = sg * ny[k]; mz[k] = sg * nz[k];
        }
        tens_alloc(&TT[c], n);
        double h = tens_fill(&TT[c], n, x, y, z, vv) * Ms;
        if (h > Hdmax) Hdmax = h;
    }
    for (int i = 0; i < N; i++) wt[i] /= Vtot;
    double dt = 2.0 * M_PI / (GAMMA * MU0 * (HK + H0 + (dip ? Hdmax : 0.0))) / 40.0;
    long nper = (long)ceil(1.0 / f / dt); dt = 1.0 / f / nper;
    long nsteps = nper * ncyc, rec = nper / 2000; if (rec < 1) rec = 1;
    for (int i = 0; i < N; i++) Hs[i] = sqrt(2.0 * alpha * KB * T / (GAMMA * MU0 * MU0 * Ms * Vv[i] * dt));
    rng_t *rb = malloc(sizeof(rng_t) * nc);
    for (int b = 0; b < nc; b++) seed_rng(&rb[b], seed * 1000003ULL + 17ULL * b + 1ULL);
    long nrec = nsteps / rec + 1; double *out = malloc(nrec * 8); long r = 0;
    double g = GAMMA * MU0 / (1.0 + alpha * alpha), w = 2.0 * M_PI * f;
    double hdx[n], hdy[n], hdz[n], gx[n], gy[n], gz[n], dx[n], dy[n], dz[n], px[n], py[n], pz[n];
    memset(hdx, 0, sizeof hdx); memset(hdy, 0, sizeof hdy); memset(hdz, 0, sizeof hdz);
    for (long k = 0; k <= nsteps; k++) {
        if (k % rec == 0) {
            double s = 0.0;
#pragma omp simd reduction(+:s)
            for (int i = 0; i < N; i++) s += wt[i] * mz[i];
            out[r++] = s;
        }
        if (k == nsteps) break;
        double ha0 = H0 * cos(w * k * dt), ha1 = H0 * cos(w * (k + 1) * dt);
        for (int b = 0; b < nc; b++) {
            int i0 = b * n;
            double *restrict Mx = mx + i0, *restrict My = my + i0, *restrict Mz = mz + i0;
            double *restrict Nx = nx + i0, *restrict Ny = ny + i0, *restrict Nz = nz + i0, *restrict Hh = Hs + i0;
            gauss_block(&rb[b], gx, n); gauss_block(&rb[b], gy, n); gauss_block(&rb[b], gz, n);
            if (dip) dip_field(&TT[b], n, Ms, Mx, My, Mz, hdx, hdy, hdz);
#pragma omp simd
            for (int i = 0; i < n; i++) {
                double tx = Hh[i] * gx[i] + hdx[i], ty = Hh[i] * gy[i] + hdy[i], tz = Hh[i] * gz[i] + hdz[i];
                double mn = Mx[i] * Nx[i] + My[i] * Ny[i] + Mz[i] * Nz[i];
                double hx = HK * mn * Nx[i] + tx, hy = HK * mn * Ny[i] + ty, hz = HK * mn * Nz[i] + tz + ha0;
                double cx = My[i] * hz - Mz[i] * hy, cy = Mz[i] * hx - Mx[i] * hz, cz = Mx[i] * hy - My[i] * hx;
                dx[i] = -g * (cx + alpha * (My[i] * cz - Mz[i] * cy));
                dy[i] = -g * (cy + alpha * (Mz[i] * cx - Mx[i] * cz));
                dz[i] = -g * (cz + alpha * (Mx[i] * cy - My[i] * cx));
                double ax = Mx[i] + dx[i] * dt, ay = My[i] + dy[i] * dt, az = Mz[i] + dz[i] * dt;
                double nn = 1.0 / sqrt(ax * ax + ay * ay + az * az);
                px[i] = ax * nn; py[i] = ay * nn; pz[i] = az * nn;
            }
            if (dip) dip_field(&TT[b], n, Ms, px, py, pz, hdx, hdy, hdz);
#pragma omp simd
            for (int i = 0; i < n; i++) {
                double tx = Hh[i] * gx[i] + hdx[i], ty = Hh[i] * gy[i] + hdy[i], tz = Hh[i] * gz[i] + hdz[i];
                double mn2 = px[i] * Nx[i] + py[i] * Ny[i] + pz[i] * Nz[i];
                double hx2 = HK * mn2 * Nx[i] + tx, hy2 = HK * mn2 * Ny[i] + ty, hz2 = HK * mn2 * Nz[i] + tz + ha1;
                double cx2 = py[i] * hz2 - pz[i] * hy2, cy2 = pz[i] * hx2 - px[i] * hz2, cz2 = px[i] * hy2 - py[i] * hx2;
                double dx2 = -g * (cx2 + alpha * (py[i] * cz2 - pz[i] * cy2));
                double dy2 = -g * (cy2 + alpha * (pz[i] * cx2 - px[i] * cz2));
                double dz2 = -g * (cz2 + alpha * (px[i] * cy2 - py[i] * cx2));
                double ax = Mx[i] + 0.5 * (dx[i] + dx2) * dt, ay = My[i] + 0.5 * (dy[i] + dy2) * dt, az = Mz[i] + 0.5 * (dz[i] + dz2) * dt;
                double nn = 1.0 / sqrt(ax * ax + ay * ay + az * az);
                Mx[i] = ax * nn; My[i] = ay * nn; Mz[i] = az * nn;
            }
        }
    }
    double mean = 0, var = 0, areas[64]; int ncc = ncyc - 1; long per = nper / rec;
    for (int c = 1; c < ncyc; c++) {
        double A = 0;
        for (long j = c * per; j < (c + 1) * per; j++) {
            double t0 = (double)(j * rec) / nper, t1 = (double)((j + 1) * rec) / nper;
            A += 0.5 * (out[j] + out[j + 1]) * (H0 * cos(2 * M_PI * t1) - H0 * cos(2 * M_PI * t0));
        }
        areas[c - 1] = -MU0 * Ms * A; mean += areas[c - 1];
    }
    mean /= ncc;
    for (int c = 0; c < ncc; c++) var += (areas[c] - mean) * (areas[c] - mean);
    printf("%.6g %.6g %ld %d %.6g\n", mean * f / RHO, ncc > 1 ? sqrt(var / (ncc - 1) / ncc) * f / RHO : 0.0, nper, N, Hdmax);
    return 0;
}
