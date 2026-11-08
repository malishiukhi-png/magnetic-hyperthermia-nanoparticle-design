/* Stochastic Landau-Lifshitz-Gilbert simulation of immobilized, non-interacting uniaxial magnetite nanoparticles
 * with random easy axes in a harmonic field.  C port of llg.py (same equation, Heun scheme, Brown thermal field,
 * time step and cycle averaging), generalized so that every particle may have its own diameter (log-normal ensembles).
 *
 * Build:  gcc -O3 -march=native -ffast-math -fno-math-errno -fopenmp sllg.c -o sllg -lm -lmvec
 *
 * Usage:  sllg D_nm SIGMA ALPHA H_A_per_m F_Hz MS_A_per_m K_J_per_m3 N NCYC SEED [T_K]
 * Output: one line "sar_W_per_kg stderr_W_per_kg nper mean_stuff"
 * SIGMA > 0: diameters are drawn from a log-normal number distribution with median D_nm; magnetization is volume weighted.
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
#define BLK 64            /* particles processed per vector block */

/* xoshiro256** seeded through splitmix64: one independent stream per vector lane */
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

/* Box-Muller, written so that the transcendental part vectorizes through libmvec */
static void gauss_block(rng_t *r, double *g, int n) {
    double u1[n], u2[n];
    for (int i = 0; i < n; i++) { u1[i] = 1.0 - unif(r); u2[i] = unif(r); }
#pragma omp simd
    for (int i = 0; i < n; i++) g[i] = sqrt(-2.0 * log(u1[i])) * cos(2.0 * M_PI * u2[i]);
}

int main(int argc, char **argv) {
    if (argc < 11) { fprintf(stderr, "usage: see header\n"); return 2; }
    double d0 = atof(argv[1]) * 1e-9, sigma = atof(argv[2]), alpha = atof(argv[3]), H0 = atof(argv[4]), f = atof(argv[5]);
    double Ms = atof(argv[6]), K = atof(argv[7]);
    int N = atoi(argv[8]), ncyc = atoi(argv[9]); uint64_t seed = strtoull(argv[10], NULL, 10);
    double T = argc > 11 ? atof(argv[11]) : 310.15;
    N = (N + BLK - 1) / BLK * BLK;

    double HK = 2.0 * K / (MU0 * Ms);
    double dt = 2.0 * M_PI / (GAMMA * MU0 * (HK + H0)) / 40.0;
    long nper = (long)ceil(1.0 / f / dt); dt = 1.0 / f / nper;
    long nsteps = nper * ncyc;
    long rec = nper / 2000; if (rec < 1) rec = 1;

    rng_t rg; seed_rng(&rg, seed);
    double *nx = aligned_alloc(64, N * 8), *ny = aligned_alloc(64, N * 8), *nz = aligned_alloc(64, N * 8);
    double *mx = aligned_alloc(64, N * 8), *my = aligned_alloc(64, N * 8), *mz = aligned_alloc(64, N * 8);
    double *Hs = aligned_alloc(64, N * 8), *wt = aligned_alloc(64, N * 8);
    double Vtot = 0, sumV = 0;
    for (int i = 0; i < N; i++) {
        double d = d0;
        if (sigma > 0) {
            double g; gauss_block(&rg, &g, 1);
            /* truncation at +-4 sigma, as in the deterministic quadrature */
            while (fabs(g) > 4.0) gauss_block(&rg, &g, 1);
            d = d0 * exp(sigma * g);
        }
        double V = M_PI * d * d * d / 6.0;
        Hs[i] = sqrt(2.0 * alpha * KB * T / (GAMMA * MU0 * MU0 * Ms * V * dt));
        wt[i] = V; Vtot += V;
        double u = 2.0 * unif(&rg) - 1.0, ph = 2.0 * M_PI * unif(&rg), s = sqrt(1.0 - u * u);
        nx[i] = s * cos(ph); ny[i] = s * sin(ph); nz[i] = u;
        double sg = unif(&rg) < 0.5 ? -1.0 : 1.0;
        mx[i] = sg * nx[i]; my[i] = sg * ny[i]; mz[i] = sg * nz[i];
    }
    for (int i = 0; i < N; i++) wt[i] /= Vtot;
    sumV = Vtot / N; (void)sumV;

    int nb = N / BLK;
    rng_t *rb = malloc(sizeof(rng_t) * nb);
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
            double gx[BLK], gy[BLK], gz[BLK];
            gauss_block(&rb[b], gx, BLK); gauss_block(&rb[b], gy, BLK); gauss_block(&rb[b], gz, BLK);
            double *restrict Mx = mx + i0, *restrict My = my + i0, *restrict Mz = mz + i0;
            double *restrict Nx = nx + i0, *restrict Ny = ny + i0, *restrict Nz = nz + i0, *restrict Hh = Hs + i0;
#pragma omp simd
            for (int i = 0; i < BLK; i++) {
                double tx = Hh[i] * gx[i], ty = Hh[i] * gy[i], tz = Hh[i] * gz[i];
                double mn = Mx[i] * Nx[i] + My[i] * Ny[i] + Mz[i] * Nz[i];
                double hx = HK * mn * Nx[i] + tx, hy = HK * mn * Ny[i] + ty, hz = HK * mn * Nz[i] + tz + ha0;
                double cx = My[i] * hz - Mz[i] * hy, cy = Mz[i] * hx - Mx[i] * hz, cz = Mx[i] * hy - My[i] * hx;
                double dx = -g * (cx + alpha * (My[i] * cz - Mz[i] * cy));
                double dy = -g * (cy + alpha * (Mz[i] * cx - Mx[i] * cz));
                double dz = -g * (cz + alpha * (Mx[i] * cy - My[i] * cx));
                double px = Mx[i] + dx * dt, py = My[i] + dy * dt, pz = Mz[i] + dz * dt;
                double nn = 1.0 / sqrt(px * px + py * py + pz * pz); px *= nn; py *= nn; pz *= nn;
                double mn2 = px * Nx[i] + py * Ny[i] + pz * Nz[i];
                double hx2 = HK * mn2 * Nx[i] + tx, hy2 = HK * mn2 * Ny[i] + ty, hz2 = HK * mn2 * Nz[i] + tz + ha1;
                double cx2 = py * hz2 - pz * hy2, cy2 = pz * hx2 - px * hz2, cz2 = px * hy2 - py * hx2;
                double dx2 = -g * (cx2 + alpha * (py * cz2 - pz * cy2));
                double dy2 = -g * (cy2 + alpha * (pz * cx2 - px * cz2));
                double dz2 = -g * (cz2 + alpha * (px * cy2 - py * cx2));
                double ax = Mx[i] + 0.5 * (dx + dx2) * dt, ay = My[i] + 0.5 * (dy + dy2) * dt, az = Mz[i] + 0.5 * (dz + dz2) * dt;
                nn = 1.0 / sqrt(ax * ax + ay * ay + az * az);
                Mx[i] = ax * nn; My[i] = ay * nn; Mz[i] = az * nn;
            }
        }
    }
    /* loop areas of cycles 2..ncyc, trapezoidal in H(t), as in llg.py */
    double mean = 0, var = 0; int nc = ncyc - 1; double areas[64];
    long per = nper / rec;   /* samples per period */
    for (int c = 1; c < ncyc; c++) {
        double A = 0;
        for (long j = c * per; j < (c + 1) * per; j++) {
            double t0 = (double)(j * rec) / nper, t1 = (double)((j + 1) * rec) / nper;
            double h0 = H0 * cos(2 * M_PI * t0), h1 = H0 * cos(2 * M_PI * t1);
            A += 0.5 * (out[j] + out[j + 1]) * (h1 - h0);
        }
        areas[c - 1] = -MU0 * Ms * A;
        mean += areas[c - 1];
    }
    mean /= nc;
    for (int c = 0; c < nc; c++) var += (areas[c] - mean) * (areas[c] - mean);
    double se = nc > 1 ? sqrt(var / (nc - 1) / nc) : 0.0;
    printf("%.6g %.6g %ld %d\n", mean * f / RHO, se * f / RHO, nper, N);
    return 0;
}
