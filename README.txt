Code and data for "Magnetite Nanoparticle Design for Magnetic Hyperthermia under Clinical Field Limits: From Neel Dynamics to Tumor Thermal Dose"

model.py      Rosensweig linear-response model (Neel/Brownian relaxation, log-normal size distribution) and the
              analytical steady-state Pennes bioheat solution for a spherical tumor.
figures.py    Physics results: Figures 2-5 and results.json (every number quoted in Sections 3-1 to 3-4).
ml.py         Generates 20,000 simulated cases, trains random forest / gradient boosting / neural-network surrogates,
              permutation importance and surrogate-based inverse design -> ml_results.json (Table 3, Section 3-5).
fig_ml.py     Figure 6 from ml_results.json.
fig_workflow.py  Figure 1 (modeling workflow schematic).
Run in order: python figures.py ; python ml.py ; python fig_ml.py      (numpy, scipy, matplotlib, scikit-learn)

Uncertainty and sensitivity analysis (Section 2-6, Table 4):
  python3 uncertainty.py   -> uncertainty.json (2000 Monte Carlo samples, about 1 min)

Note on Eq. (6): the size distribution is integrated over ln d on a uniform grid, which is
equivalent to the number density g(d) = exp[-ln^2(d/d0)/(2 sigma^2)]/(d sigma sqrt(2 pi)) in d.

Nonlinear dynamics and transient heating (Sections 2.8-2.9, 3.7-3.8, Figures 7-8):
  nonlinear.py      Thermally activated two-state (Stoner-Wohlfarth) model of immobilized particles vs. linear response
                    -> nonlinear.json (about 1 min)
  langevin.py       Overdamped stochastic (Langevin) simulation of the Neel moment, 2000 particles with random easy axes
                    python langevin.py 15k300 ; python langevin.py 18k278 ; python langevin.py conv ; python langevin.py hsweep
                    -> langevin_15k300.json, langevin_18k278.json, langevin_conv.json, langevin_hsweep.json (20-60 min each; requires numba)
                    Results are stochastic; reruns reproduce the published values within the stated standard errors.
  fig_nonlinear.py  Figure 7 from the files above.
  transient.py      Transient Pennes bioheat (finite volume, Crank-Nicolson), CEM43 thermal dose and Gaussian
                    (non-uniform) loading -> transient.json and Figure 8 (a few seconds).

Stochastic LLG with explicit damping and comparison with experiment (Sections 2.8, 3.7-3.8, Figures 7-8):
  llg.py            Stochastic Landau-Lifshitz-Gilbert dynamics (Heun scheme, Brown thermal field, free-electron gyromagnetic
                    ratio, no calibration). python llg.py 0.1 ; python llg.py 1   -> llg_a0.1.json, llg_a1.json
  llg_conv.py       Time-step check -> llg_conv.json
  llg_khandhar.py   sLLG at 373 kHz and 14 kA/m with the measured Ms of Khandhar et al. -> llg_khandhar_a0.1.json, _a1.json
  exp_compare.py    Log-normal averaging and comparison with the measured SLP of Khandhar et al. (values digitized from their
                    Figure 7, sizes and widths from their Table 1) -> exp_compare.json and fig_experiment.png
  fig_nonlinear.py  Figure 7 (needs nonlinear.json, langevin_15k300.json, llg_a*.json)
  Each sLLG size takes several minutes on one core; results are stochastic within the stated standard errors.

Revised analysis (nonlinear SAR as the main model; added for the submitted version):
  sllg.c            Compiled implementation of the sLLG scheme of llg.py; each particle may have its own diameter (log-normal ensembles).
                    gcc -O3 -march=native -ffast-math -fno-math-errno -fopenmp-simd sllg.c -o sllg -lmvec -lm
                    Checked against llg_a*.json (333 +- 3 vs 329 +- 3 W/g at 18 nm, alpha = 1; 688 +- 1 vs 691 +- 2 W/g at 20 nm, alpha = 0.1).
  run_sllg_queue.py Runs all new cases (field scan incl. 100 kHz/18 kA/m, log-normal ensembles, anisotropy, live-cell comparison, Monte Carlo)
                    -> sllg_runs.jsonl, mc_inputs.json  (about 2.5 h on two cores)
  nl_analysis.py    python nl_analysis.py sllg_runs.jsonl -> nl_results.json: optimum per field, dipolar coupling, nonlinear dose,
                    polydispersity, anisotropy, live-cell comparison, Monte Carlo statistics
  run_sllg_lowalpha.py      Additional cases appended to sllg_runs.jsonl (11 nm live-cell sample with the K and alpha of the cited loop simulations; alpha = 0.0015 at 15 kA/m, 300 kHz)
  llg_khandhar_lowalpha.py  alpha = 0.0015 at the conditions of the measured SLP series -> llg_khandhar_a0.0015.json, exp_compare_lowalpha.json
  sllg_dip.c, run_sllg_clusters.py   sLLG with explicit dipolar coupling in random clusters -> sllg_cluster_runs.jsonl
                    gcc -O3 -march=native -ffast-math -fno-math-errno -fopenmp-simd sllg_dip.c -o sllg_dip -lm -lmvec ; ./sllg_dip selftest
  sllg_rot.c, run_sllg_water.py      sLLG with Brownian rotation in a liquid (series of Tong et al., Khandhar et al., Cabrera et al.) -> sllg_water_runs.jsonl
                    gcc -O3 -march=native -ffast-math -fno-math-errno -fopenmp-simd sllg_rot.c -o sllg_rot -lm -lmvec ; ./sllg_rot selftest
  cluster_water_analysis.py          Summary of both sets -> cluster_water_results.json (Sections 3.4 and 3.5, Tables S5-S7)
  fig_nl.py         Figures 1, 3, 4, 6 of the manuscript from nl_results.json
  Figure 2 = fig_nonlinear.png, Figure 5 = fig_experiment.png, Figure 7 = fig_transient.png.
  The linear-response results and the machine-learning surrogate are in the Supplementary material (figures.py, uncertainty.py, ml.py, fig_ml.py).
  literature_data.csv  Measured values used for comparison (Khandhar et al. digitized; Cabrera et al. from the text).
