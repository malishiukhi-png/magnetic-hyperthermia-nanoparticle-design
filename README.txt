Code for "Magnetite Nanoparticle Design for Magnetic Hyperthermia under Clinical Field Limits: From Néel Dynamics to Tumor Thermal Dose"
Requirements: Python 3 with numpy, scipy, matplotlib, scikit-learn; numba for langevin.py and llg.py; a C compiler (gcc) for llg_c.py.

Linear-response model and tumor heating (Sections 2.1-2.6, 3.1, 3.4, 3.5; Figures 2-4 and 7; Table 5)
  model.py          Rosensweig linear-response model (Neel/Brownian relaxation, log-normal size distribution) and the
                    analytical steady-state Pennes bioheat solution for a spherical tumor.
  figures.py        Figures 2-4 and 7 (tumor) and results.json (numbers quoted in Sections 3.1 and 3.4).
  uncertainty.py    Monte Carlo uncertainty and sensitivity analysis -> uncertainty.json (Table 5; about 1 min).
  transient.py      Transient Pennes bioheat (finite volume, Crank-Nicolson), CEM43 thermal dose and Gaussian loading
                    -> transient.json and Figure 8 (a few seconds).
  fig_workflow.py   Figure 1.
  Note on Eq. (6): the size distribution is integrated over ln d on a uniform grid, which is equivalent to the number
  density g(d) = exp[-ln^2(d/d0)/(2 sigma^2)]/(d sigma sqrt(2 pi)) in d.

Nonlinear Neel dynamics (Sections 2.7, 3.2; Figure 5; Table 3)
  nonlinear.py      Thermally activated two-state model of immobilized particles vs. linear response -> nonlinear.json
  langevin.py       Overdamped stochastic simulation (numba): python langevin.py 15k300 | 18k278 | conv | hsweep
                    -> langevin_*.json (20-60 min each)
  llg.py            Stochastic Landau-Lifshitz-Gilbert (sLLG) dynamics, uniaxial anisotropy, numba version used for
                    llg_a0.1.json, llg_a1.json, llg_conv.json (time-step check, llg_conv.py) and llg_khandhar_a*.json
                    (llg_khandhar.py).
  llg_kernel.c      Compiled sLLG kernel (Heun scheme, Brown thermal field) for uniaxial or cubic anisotropy.
  llg_c.py          Python driver of llg_kernel.c; compiles it on first use (gcc -O3 -shared -fPIC). Cubic anisotropy:
                    K1 = -11 kJ/m3 with randomly oriented crystal axes, started in a random <111> minimum.
  val_c.py          Check of the compiled kernel against llg_a1.json / llg_a0.1.json (e.g., 18 nm, alpha = 1; 20 nm, alpha = 0.1).
  llg_jobs.py       Runs the new sLLG series (resumes from the stored JSON files):
                      a001     uniaxial, alpha = 0.01, 15 kA/m, 300 kHz            -> llg_a001.json
                      cubic1   cubic, alpha = 1, 15 kA/m, 300 kHz, 14-50 nm        -> llg_cubic1.json (jobs cubic1, cubic1x)
                      cubic01  cubic, alpha = 0.1, 40 nm                           -> llg_cubic01.json
                      tong_u, tong_c   uniaxial / cubic, alpha = 1, 9.35 kA/m, 325 kHz (Tong et al.) -> llg_tong_*.json
                      kh_K23   uniaxial K = 23 kJ/m3, alpha = 1, 14 kA/m, 373 kHz (Khandhar et al.) -> llg_kh_K23.json
                      kh_c     cubic, alpha = 1, Khandhar conditions                -> llg_kh_c.json
                    Example: python llg_jobs.py a001 cubic1 cubic1x cubic01 tong_u tong_c kh_K23 kh_c
                    Each size takes 3-10 min on one core. JSON rows are [diameter (nm), SAR (W/g), standard error (W/g)].
  fig_v5.py         Figures 5 and 6 from the files above and exp_metrics.json.

Comparison with measurements (Sections 2.8, 3.3; Figure 6; Table 4; Supplementary Table S2)
  khandhar_K.py     LRT comparison with Khandhar et al. for K = 13, 18, 23, 30, 41 kJ/m3 -> khandhar_K.json
  exp_metrics.py    Log-normal (mass-weighted) averaging of all models over the samples of Khandhar et al. (SLP digitized
                    from their Figure 7, sizes and widths from their Table 1), agreement metrics (Table 4), per-sample values
                    (Table S2), and the sLLG results at the conditions of Tong et al. -> exp_metrics.json
  Run after llg_jobs.py: python khandhar_K.py ; python exp_metrics.py ; python fig_v5.py

Machine-learning surrogate (Supplementary Material, Section S1; Table S1; Figure S1)
  ml.py             20,000 simulated LRT cases, random forest / gradient boosting / neural-network surrogates, permutation
                    importance and surrogate-based inverse design -> ml_results.json
  fig_ml.py         Figure S1 from ml_results.json.

All stochastic results reproduce the published values within the stated standard errors when rerun.

Extensions of the sLLG model and additional analyses of the revised manuscript (Sections 2.6, 2.7, 3.2-3.6; Tables 3, 5, 6;
Supplementary Tables S3-S9). Build the C programs with gcc and OpenMP:
  gcc -O3 -march=native -ffast-math -fno-math-errno -fopenmp sllg.c -o sllg -lm -lmvec        (also sllg_dip.c, sllg_rot.c)
  sllg.c                  Isolated immobilized particles, uniaxial anisotropy, per-particle diameters (log-normal ensembles).
                          Arguments: d(nm) sigma alpha H(A/m) f(Hz) Ms(A/m) K(J/m3) N ncycles seed T(K); prints SAR (W/kg)
                          and its standard error.
  sllg_dip.c              Random clusters of 32 particles with explicit dipolar coupling (Section 2.7).
  sllg_rot.c              Coupled Neel and Brownian rotation of isolated particles in a liquid (Section 2.7).
  run_sllg_queue.py       Main series: field scenarios, live-cell samples, polydispersity, anisotropy, Monte Carlo input sets
                          (mc_inputs.json) -> sllg_runs.jsonl
  run_sllg_lowalpha.py    Low damping (alpha = 0.0015-0.03) and the loop-simulation parameters of Cabrera et al. -> sllg_runs.jsonl
  run_sllg_K23.py         K = 23 kJ/m3 at 15 kA/m/300 kHz, 18 kA/m/278 kHz and 18 kA/m/100 kHz -> k23.jsonl (Table 3, Table S3)
  run_sllg_convergence.py Random seeds, number of particles and number of cycles -> conv.jsonl (Table S5)
  xcheck_kernels.py       Cross-check of sllg.c against llg_kernel.c -> xcheck_k23.json (Section 2.7)
  run_sllg_clusters.py    Dipolar clusters -> sllg_cluster_runs.jsonl; run_sllg_water.py: Brownian rotation -> sllg_water_runs.jsonl
  cluster_water_analysis.py  -> cluster_water_results.json (Section 3.6, Table 6, Tables S6, S7, S9)
  nl_analysis.py          Post-processing of sllg_runs.jsonl (field scenarios, damping, sLLG Monte Carlo statistics and Spearman
                          correlations, live-cell comparison) -> nl_results.json (Tables 3 and 5)
  llg_khandhar_lowalpha.py   Low damping at the conditions of Khandhar et al. -> llg_khandhar_a0.0015.json, exp_compare_lowalpha.json
  dose_coverage.py        Thermal-dose coverage of the transient tumor model -> dose_coverage.json (Section 3.4, Table S8)
  literature_data.csv     Measured values used for the comparisons (digitized or taken from the cited publications).
  The run_*.py scripts resume from their output files. Each sLLG size takes about one minute per core (1024 particles,
  four cycles); results are stochastic and reproduce the stored values within their standard errors.

Software used for the published results: Python 3.13, NumPy 2.5, SciPy 1.18, scikit-learn 1.9, Matplotlib 3.11, numba
(langevin.py, llg.py), gcc 13 (C kernels).
