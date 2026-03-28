# Code Repository for Rebuttal Experiments

**Paper:** *Pseudo-Labeling for Unsupervised Domain Adaptation with Kernel GLMs* (ICML 2026 Submission)

This anonymous repository contains the code and reproducible scripts for the original and additional experiments conducted during the author rebuttal phase. These experiments empirically validate the theoretical claims made in the manuscript and address specific reviewer questions regarding baselines and hyperparameter behavior under covariate shift.

## Repository Structure

The repository is organized around three primary rebuttal experiments, plus the original synthetic experiment:

### 1. `benchmark_raisin_full.ipynb`
**Objective:** Compare our GLM Pseudo-Labeling framework against established density-ratio (Importance Weighting) and KRR pseudo-labeling baselines.
* **Methods Included:**
  * Unsupervised GLM Pseudo-Labeling (Ours)
  * KRR Pseudo-Labeling (Wang, 2023)
  * Importance-Weighted Cross-Validation via KLIEP (KLIEP-IW)
  * Importance-Weighted Cross-Validation via Kernel Mean Matching (KMM-IW)
* **Description:** This script runs the strict "split-and-fit" cross-validation pipeline across 100 random seeds on the Raisin dataset, as described in Section 6.2 of the paper. It tracks the candidate selection process via log-loss (cross-entropy) and demonstrates the instability of density-ratio methods as well as the calibration failure of unconstrained squared-loss selection (KRR).

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

**The Setup:**
* **Feature space:** $[0,1]$.
* **Sample sizes:** $n = 4000$ labeled source samples and $n_0 = n$ unlabeled target samples.
* **Response model:** Kernel logistic regression, where $y \mid x \sim \text{Bernoulli}(\sigma(f^\ast(x)))$ with the true latent function $f^\ast(x) = 1.5\cos(2\pi x)$, and $\sigma$ the sigmoid function.
* **Source covariate distribution ($\mathcal{P}$):** Concentrated on the left, $\frac{B}{B+1}\mathcal{U}[0, 1/2] + \frac{1}{B+1}\mathcal{U}[1/2, 1]$ with $B=n^{0.45}$.
* **Target covariate distribution ($\mathcal{Q}$):** Concentrated on the right, $\frac{1}{B+1}\mathcal{U}[0, 1/2] + \frac{B}{B+1}\mathcal{U}[1/2, 1]$ with $B=n^{0.45}$.
* **Kernel:** First-order Sobolev kernel, $K(z,w) = \min(z,w)$.

We split the labeled source data in half. On the first half, we run kernel logistic regression with a grid of different ridge penalty parameters ($\lambda$) to obtain a collection of candidate models. 

First, we note the necessity of adapting to covariate shift. On the left panel below, we plotted three candidate models with different penalties. We see that the optimal choice is different for the source and target distributions. On the interval $[0, 1/2]$ where source data is abundant, a large penalty (cyan) provides a great fit. However, on the interval $[1/2, 1]$ where source data is sparse but target data is heavily concentrated, this large penalty oversmooths, and a smaller penalty (red) actually performs better for the target domain.

Then on the right panel below, we compare three model selections methods based on different validation datasets:
* **Naive method (blue)**: validating on the held-out source data, it selects a suboptimal model that fails to adapt to the target distribution.
* **Oracle method (cyan)**: uses true, noiseless target responses.
* **Proposed method (red)**: using only the unlabeled target data with our generated soft pseudo-labels, it successfully selects an adaptive model, achieving performance highly comparable to the oracle.

<p align="center">
  <img src="target_source_medium.png" width="48%" alt="Candidate Models">
  <img src="pseudo_oracle_naive_imp.png" width="48%" alt="Selection Methods">
</p>

*Figure 1: Covariate shift and its adaptation in Kernel Logistic Regression. The black dashed curves show the true latent function* $f^\ast(x)$ *.*

*(Note: We also visualize the imputation model used to generate the pseudo-labels, shown in pink. While unsuitable for direct prediction, it is effective for model selection with pseudo-labels).*

For full reproducible details—including the exact grid of hyperparameters and the specific $\lambda$ penalties selected by each method—please refer directly to the `demo_covariate_shift.ipynb` notebook. The final quantitative performance of the selected models is summarized below:

| Method | Target Excess Risk | 95% condidence interval |
| :--- | :--- | :--- |
| **Naive** | 0.016300 | [0.014926, 0.017674] |
| **Pseudo-labeling (Ours)** | 0.003481 | [0.001954, 0.005008] |
| **Oracle** | 0.002855 | [0.001312, 0.004399] |


### 3. `demo_covariate_shift.ipynb` part II
**Objective:** Ablation study on the imputation model penalty (`lbd_tilde`) to validate our theoretical insight.
* **Description:** Our theoretical analysis dictates that the imputation model must be undersmoothed to ensure valid model selection. This script sweeps over a grid of `lbd_tilde` values during the pseudo-label generation phase. 
* **Results:** The output empirically demonstrates that low regularization on the imputer is necessary to achieve Oracle-level target risk, aligning with the oracle inequality Theorem 5.2 derived in the main paper.

### 4. Synthetic Data (Section 6.1)
We test our approach using logistic regression with the first-order Sobolev kernel, as explained in Section 6.1 of the paper. 
* **Run the experiment:** Use `run_experiments_logistic.ipynb`. This notebook calls `pseudo_label_experiment_general.py` (or `pseudo_label_experiment_general_KeOps.py` for the KeOps version).
* **Results:** Because the full experiment is computationally intensive, we have provided the final results in:
    * `results_logistic_torchcpu_1_5_cos_0_4_shift.zip` (covariate shift strength $B=n^{0.4}$) 
    * `results_logistic_torchcpu_1_5_cos_0_45_shift.zip` (covariate shift strength $B=n^{0.45}$) 
* **Plotting:** The results can be plotted using `plot_curves_synthetic.ipynb`, which outputs `logistic_errors_04.pdf` and `logistic_errors_045.pdf`.

## 🧮 Algorithmic Details
We implemented a generic solver for kernel GLMs in Python, using the Fisher scoring method. For full mathematical details and notes on our scalable GPU implementation with KeOps, please see our [Algorithmic Details document](ALGORITHM.md).

## 🛠️ Solvers
This repository provides a general solver for kernel ridge regression, kernel logistic regression, and kernel Poisson regression. Standard kernels are available (e.g., linear, polynomial, RBF, first-order Sobolev).

* `rkhs_glm_scaled.py`: Provides the basic solver for ridge-regularized kernel GLMs. For relatively small sample sizes ($n \le 5000$), a simple version using only Numpy and Scipy is enough. 
* `rkhs_glm_scaled_KeOps.py`: For larger problems, we implement the IRLS inner linear solves using kernel matvec oracles computed on-the-fly on the GPU, using the KeOps library.

## References & Acknowledgements

The experimental evaluation framework for the real-world dataset builds upon the open-source implementation provided by Feng et al. (2023). We adapted and significantly extended their cross-validation pipeline to integrate the proper unweighted candidate training for our GLM framework, re-using their KLIEP density ratio estimation implementation `KLIEP_importance_estimation.py`. The KRR pseudo-labeling baseline methodology follows Wang (2026).

* **Feng, X., He, X., Wang, C., Wang, C., & Zhang, J. (2023).** Towards a unified analysis of kernel-based methods under covariate shift. *Advances in Neural Information Processing Systems*, 36, 73839-73851.
* **Wang, K. (2026).** Pseudo-labeling for kernel ridge regression under covariate shift. *The Annals of Statistics*, 54(1), 252-276.

