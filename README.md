# PMAOA (Peer-Modulated Arithmetic Optimization Algorithm) for Wrapper-based Feature Selection

This repository provides a MATLAB implementation of **PMAOA**, a peer-guided variant of the Arithmetic Optimization Algorithm (AOA) designed for **wrapper-based feature selection**. PMAOA searches for an optimal subset of features by optimizing a continuous position vector and converting it to a binary feature mask using a threshold. Each candidate feature subset is evaluated using a **KNN classifier** inside a fitness function.

## Key Idea

PMAOA combines two complementary phases to balance exploration and exploitation:

1. **Peer-guided AOA update (with linear decay)**  
   Each solution learns from a randomly selected peer using a signed difference step (move toward a better peer, otherwise move away). AOA operators are controlled by:
   - `MOP` (Math Optimizer Probability)
   - `MOA` (Math Optimizer Acceleration)
   A **linear decay factor** gradually reduces the step size as the fitness evaluation budget is consumed.

2. **Mutation and Restarting phase (alternating by generation)**  
   - **Mutation:** uses two randomly selected peers and an optional delayed-leader term to promote diversity while keeping search direction.
   - **Restarting:** randomly reinitializes a portion of dimensions to escape stagnation.

Dynamic boundary handling is applied to ensure all decision variables remain in the valid range.

## Features

- Wrapper-based feature selection using **KNN**
- Continuous-to-binary conversion via thresholding
- Two-stage search strategy (peer-guided update + mutation/restart)
