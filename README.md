# PMAOA (Peer-Modulated Arithmetic Optimization Algorithm) for Wrapper-based Feature Selection

This repository provides a MATLAB implementation of **PMAOA**, a wrapper-based feature selection method derived from the **Arithmetic Optimization Algorithm (AOA)**. PMAOA is designed to improve the exploration–exploitation balance and reduce premature convergence by introducing peer-guided learning and adaptive learning mechanisms.

## Abstract

Feature selection is a critical pre-processing step in machine learning that eliminates redundant features while retaining essential information for accurate prediction. However, most wrapper-based feature selection approaches struggle to balance exploration and exploitation effectively, often leading to premature convergence. To address this, PMAOA enhances adaptability and learning diversity through two main stages.

The **Peer-Guided Stage (PGS)** introduces socially modulated interactions where each candidate solution learns from randomly chosen peers instead of relying only on a global best solution. This helps maintain population diversity and prevents stagnation.

The **Adaptive Learning Stage (ALS)** integrates cross-generational learning and selective rejuvenation through two coordinated modes: **Historical Difference Learning (HDL)**, which exploits accumulated inter-generation knowledge for refinement, and **Partial Random Reinitialization (PRI)**, which rejuvenates selected dimensions of candidate solutions to sustain exploration.

Experiments on twenty benchmark datasets show that PMAOA achieves superior or comparable classification accuracy with smaller feature subsets and moderate computational cost. This method contributes to process innovation in intelligent optimization and can support data-driven applications in smart city ecosystems.

## Dataset

The benchmark datasets used in the experiments are sourced from the **UCI Machine Learning Repository**.

## Code Overview

- `jPMAOA.m`: main PMAOA optimizer for wrapper-based feature selection
- `jFitnessFunction.m`: wrapper evaluation (KNN-based) and fitness computation

## Usage (MATLAB)

1. Open MATLAB and set the project folder (this repository) as your working directory.
2. Run:

```matlab
main
