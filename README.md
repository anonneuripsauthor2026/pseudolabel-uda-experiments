# rebuttal-pseudolabel-uda-experiments

# Code Repository for Rebuttal Experiments

**Paper:** *Pseudo-Labeling for Unsupervised Domain Adaptation with Kernel GLMs* (ICML 2026 Submission)

This anonymous repository contains the code and reproducible scripts for the additional experiments conducted during the author rebuttal phase. These experiments empirically validate the theoretical claims made in the manuscript and address specific reviewer questions regarding baselines and hyperparameter behavior under covariate shift.

## Repository Structure

The repository is organized around three primary rebuttal experiments:

### 1. `benchmark_raisin_full.ipynb`
**Objective:** Compare our GLM Pseudo-Labeling framework against established density-ratio (Importance Weighting) and KRR pseudo-labeling baselines.
* **Methods Included:**
  * Unsupervised GLM Pseudo-Labeling (Ours)
  * KRR Pseudo-Labeling (Wang, 2023)
  * Importance-Weighted Cross-Validation via KLIEP (KLIEP-IW)
  * Importance-Weighted Cross-Validation via Kernel Mean Matching (KMM-IW)
* **Description:** This script runs the strict "split-and-fit" cross-validation pipeline across 100 random seeds on the Raisin dataset. It tracks the candidate selection process via log-loss (cross-entropy) and demonstrates the instability of density-ratio methods as well as the calibration failure of unconstrained squared-loss selection (KRR).

| Method Category | Selection Strategy / Model | Target Risk (Mean) | Standard Error (SE) |
| :--- | :--- | :--- | :--- |
| **Ours (Kernel GLM)** | **Pseudo-Labeling (Unsupervised)** | **0.383** | **0.006** |
| | Oracle (True Target Labels) | 0.376 | 0.007 |
| **Wang (2023) KRR** | Pseudo-Labeling (Unsupervised) | 0.406 | 0.009 |
| | Oracle (True Target Labels) | 0.443 | 0.007 |
| **KLIEP (Density Ratio)** | Importance-Weighted CV | 0.438 | 0.014 |
| | Oracle (True Target Labels) | 0.384 | 0.007 |
| **KMM (Density Ratio)** | Importance-Weighted CV | 0.449 | 0.009 |
| | Oracle (True Target Labels) | 0.437 | 0.008 |
| **Baseline** | Naive (Source-Only) | 0.442 | 0.014 |

### 2. `demo_covariate_shift.ipynb` part I
**Objective:** Demonstrate the necessity of target-specific adaptation, achieved through target-aware Ridge regularization's parameter selection for well-specified models.
* **Description:** This simulation generates a well-specified synthetic environment undergoing covariate shift. It compares the risk landscape of models tuned exclusively on the source distribution (Naive) versus models tuned via our target-optimal penalty selection.
* **Results:** The simulation showcases that the presence of covariate shift alters the optimal regularization path. Relying on source-optimal penalties leads to severe target risk degradation, highlighting why unsupervised target adaptation is required.

### 3. `demo_covariate_shift.ipynb` part II
**Objective:** Ablation study on the imputation model penalty (`lbd_tilde`) to validate our theoretical insight.
* **Description:** Our theoretical analysis dictates that the imputation model must be undersmoothed to ensure valid model selection. This script sweeps over a grid of `lbd_tilde` values during the pseudo-label generation phase. 
* **Results:** The output empirically demonstrates that low regularization on the imputer is necessary to achieve Oracle-level target risk, aligning with the oracle inequality Theorem 5.2 derived in the main paper.

