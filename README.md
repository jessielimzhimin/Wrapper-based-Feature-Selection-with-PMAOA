# PMAOA (Peer-Modulated Arithmetic Optimization Algorithm) for Wrapper-Based Feature Selection

This repository provides a MATLAB implementation of **PMAOA**, a wrapper-based feature selection method derived from the **Arithmetic Optimization Algorithm (AOA)**. PMAOA improves the exploration–exploitation balance and reduces premature convergence by introducing peer-guided learning and an adaptive learning stage combining cross-generational memory with selective re-diversification.

## Abstract

Feature selection is a critical pre-processing step in machine learning that eliminates redundant features while retaining essential information for accurate prediction. However, most wrapper-based feature selection approaches struggle to balance exploration and exploitation effectively, often leading to premature convergence. To address this, PMAOA introduces three coordinated mechanisms.

The **Peer-Guided Stage (PGS)** introduces socially guided interactions where each candidate solution learns from a randomly chosen peer instead of relying only on a global best solution, helping maintain population diversity and prevent stagnation.

The **Adaptive Learning Stage (ALS)** alternates between two modes every generation: **Historical Difference Learning (HDL)**, which exploits a historical leader from earlier generations to refine the search direction, and **Partial Random Reinitialization (PRR)**, which reinitializes a subset of dimensions in each candidate solution to sustain exploration without discarding the solution entirely.

The **Global Survivor Selection (GSS)** mechanism merges and ranks the current and newly generated populations each generation, retaining the strongest candidates and reinforcing elitism.

Experiments on 20 UCI benchmark datasets show that PMAOA achieves superior or comparable classification accuracy with smaller feature subsets relative to 14 competing metaheuristic algorithms, at the cost of a higher average computation time attributable to its additional learning mechanisms; this trade-off is discussed in detail in the accompanying manuscript.

## Repository Contents

| File | Description |
|---|---|
| `jPMAOA.m` | Main PMAOA optimizer for wrapper-based feature selection |
| `jFitnessFunction.m` | Wrapper evaluation (KNN-based) and fitness computation |
| `Main.m` | Main script that runs PMAOA across all 20 benchmark datasets under the leakage-free nested hold-out protocol described in the manuscript, and saves the results and figures used in the comparison tables |
| `Load_UCI_Data.m` | Loads and preprocesses each of the 20 UCI benchmark datasets used in this study |
| `Dataset/` | Raw dataset files referenced by `Load_UCI_Data.m` |

## Dataset

The benchmark datasets used in the experiments are sourced from the **UCI Machine Learning Repository**. This study uses 20 datasets spanning a range of dimensionalities, sample sizes, and class counts; see `Load_UCI_Data.m` for the full list and corresponding dataset indices.

## Evaluation Protocol

Each dataset is split into an 80/20 train-test partition: the 20% outer test split is held out and never seen by the optimizer, while an inner validation split (drawn only from the training portion) is used by the wrapper fitness function during the search. The selected feature subset is evaluated once on the held-out test split using a KNN classifier. This protocol is repeated over 20 independent runs per dataset, with each run drawing a different random train-test partition; the same sequence of partitions is used consistently across PMAOA and the 14 competing metaheuristic algorithms reported in the manuscript, so that every algorithm is evaluated on the same set of splits run-for-run.

```
