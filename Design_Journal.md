# ELEC9123 Design Task E (Optimization for Green IoT) - Term T2, 2025

**Project Title:** Secrecy Rate Maximization in Beamforming-Assisted Backscatter Communication (MIMO Reader, Single-Antenna Tag/Eavesdropper)

**Author:** Yuri Yu  
**Submission File:** `z5226692_Yu_DTE_2025.zip`

---

## 1. Introduction

Backscatter communication enables ultra-low-power IoT data collection, but the open nature of RF links makes tag messages vulnerable to eavesdropping. This project focuses on maximizing the **secrecy rate (SR)** in a system with a multi-antenna reader and a single-antenna tag/eavesdropper, using beamforming and tag impedance modulation, while ensuring energy harvesting and communication reliability.

---

## 2. Objectives

- Develop a mathematical model for the multi-antenna reader ↔ single-antenna tag/eavesdropper system.
- Formulate the secrecy rate maximization problem with practical constraints.
- Implement both brute-force and efficient (AO/SCA-based) optimization algorithms.
- Simulate and analyze SR versus distance, number of antennas, etc.
- Summarize findings and propose future work.

---

## 3. System Model

### 3.1 Channel Model

- Reader-to-tag: $\mathbf{h}_{RU} \in \mathbb{C}^{N \times 1} \sim \mathcal{CN}(\mathbf{0}_{N \times 1}, \beta_{RU} \mathbf{I}_N)$  
- Path-loss: $\beta_{RU} = (\lambda/4\pi d_{RU})^2$ 
- Reader-to-eavesdropper: $\mathbf{h}_{RE} \in \mathbb{C}^{N \times 1} \sim \mathcal{CN}(\mathbf{0}_{N \times 1}, \beta_{RE} \mathbf{I}_N)$  
  > **Note:** In the secrecy rate analysis, we adopt a *worst-case assumption* where the eavesdropper is able to perfectly cancel the continuous wave (CW) signal transmitted by the reader [2].
- Tag-to-eavesdropper: $h_{UE} \in \mathbb{C} \sim \mathcal{CN}(0, \beta_{UE})$

### 3.2 Signal Model

- **Downlink (Reader to Tag):** $y_{bu} = \mathbf{h}_{RU}^T \mathbf{w} x_t + n_{bu} \approx \mathbf{h}_{RU}^T \mathbf{w} x_t$ (noise at tag is negligible [3]).
- **Tag Modulation (ASK):** $x_{b,i} = \sqrt{\eta_b} (\mathbf{h}_{RU}^T \mathbf{w} x_t) \Gamma_i$.
- **Uplink (Tag to Reader):** $y_{R,i} = \mathbf{h}_{RU}^H \mathbf{g} x_{b,i} + n_R$.
- **Eavesdropper Reception:** $y_{E,i} = h_{RE} x_{b,i} + n_E$.

### 3.3 SNR and Secrecy Rate

- **Legitimate Receiver:**
  $$
  \gamma_R = \frac{\eta_b |\mathbf{h}_{RU}^T \mathbf{w}|^2 |\mathbf{h}_{RU}^H \mathbf{g}|^2 |\Gamma_0 - \Gamma_1|^2}{4\sigma_R^2}
  $$
- **Eavesdropper:**
  $$
  \gamma_E = \frac{\eta_b |h_{UE}|^2 |\mathbf{h}_{RU}^T \mathbf{w}|^2 |\Gamma_0 - \Gamma_1|^2}{4\sigma_E^2}
  $$
- **Secrecy Rate:**
  $$
  SR = \log_2(1 + \gamma_R) - \log_2(1 + \gamma_E)
  $$

### 3.4 Optimization Problem Formulation

**Objective:** Jointly optimize the reader transmit beamforming vector $\mathbf{w}$ and tag reflection coefficients $(\Gamma_0, \Gamma_1)$ to maximize the system secrecy rate $SR$.

**Variables:**
- $\mathbf{w}$: Reader transmit beamforming vector
- $(\Gamma_0, \Gamma_1)$: Tag reflection coefficients

$$
\begin{align*}
\max_{\mathbf{w}, \mathbf{g}, \Gamma_0, \Gamma_1} \quad & SR(\mathbf{w}, \mathbf{g}, \Gamma_0, \Gamma_1) \\
\text{s.t.} \quad
& \|\mathbf{w}\|^2 \leq P_t, \|\mathbf{g}\|^2 \leq 1 \\
& \frac{|\Gamma_0 - \Gamma_1|}{2} \geq m_{th} \\
& -1 \leq \Gamma_0, \Gamma_1 \leq 1 \\
& P_{L,avg} = \eta_e \left(1 - \frac{|\Gamma_0|^2 + |\Gamma_1|^2}{2}\right) \mathbb{E}[|\mathbf{h}_{RU}^T \mathbf{w}|^2] \geq P_{th}
\end{align*}
$$

#### Optimization Problem

The resulting optimization problem is:
$$
\max_{\mathbf{w}, \mathbf{g}, \Gamma_0, \Gamma_1} \quad SR(\mathbf{w}, \mathbf{g}, \Gamma_0, \Gamma_1)
$$
where
$$
SR(\mathbf{w}, \mathbf{g}, \Gamma_0, \Gamma_1) = 
\log_2 \left( 1 + \frac{\eta_b |\mathbf{h}_{RU}^T \mathbf{w}|^2 |\mathbf{h}_{RU}^H \mathbf{g}|^2 |\Gamma_0 - \Gamma_1|^2}{4\sigma_R^2} \right) 
- 
\log_2 \left( 1 + \frac{\eta_b |\mathbf{h}_{RE}^T \mathbf{w}|^2 |\Gamma_0 - \Gamma_1|^2}{4\sigma_E^2} \right)
$$

#### Constraints

- **Average harvested power:** $P_{L,avg} \geq P_{th}$
- **Modulation depth:** $m = \frac{|\Gamma_0 - \Gamma_1|}{2} \geq m_{th}$
- **Beamforming/combining norm:** $\|\mathbf{w}\|^2 \leq P_t$, $\|\mathbf{g}\|^2 \leq 1$
- **Reflection coefficients:** $-1 \leq \Gamma_i \leq 1$

> **Note:** The derivation of the harvested power constraint is as follows:
>
> &emsp;The average harvested power at the tag is given by
> $$
> P_{L,avg} = \frac{1}{2}P_{L0} + \frac{1}{2}P_{L1} \geq P_{th}
> $$
> &emsp;where $P_{L0}$ and $P_{L1}$ are the harvested powers corresponding to reflection coefficients $\Gamma_0$ and $\Gamma_1$, respectively:
> $$
> P_{Li} = \eta_e (1 - |\Gamma_i|^2) \, \mathbb{E}\left[|\mathbf{h}_{RU}^T \mathbf{w}|^2\right], \quad i=0,1
> $$
> &emsp;Here, $\eta_e$ is the energy harvesting efficiency, and $\mathbb{E}\left[|\mathbf{h}_{RU}^T \mathbf{w}|^2\right]$ is the average received signal power at the tag.
>
> &emsp;Since the tag alternates between $\Gamma_0$ and $\Gamma_1$ with equal probability, the constraint simplifies to:
> $$
> P_{L,avg} = \eta_e \left(1 - \frac{|\Gamma_0|^2 + |\Gamma_1|^2}{2}\right) \mathbb{E}\left[|\mathbf{h}_{RU}^T \mathbf{w}|^2\right] \geq P_{th}
> $$
> &emsp;This compact form depends only on the average received power at the tag and the mean of $|\Gamma_0|^2$ and $|\Gamma_1|^2$, which facilitates optimization.

#### Simplified Optimization Problem

Moreover, It has been established in [1,2,3,4] that, for secrecy rate maximization with perfect CSI at the reader, the optimal receive combining vector is maximum ratio combining (MRC), i.e., $\mathbf{g} = \frac{\mathbf{h}_{RU}}{\|\mathbf{h}_{RU}\|}$. This is because the receive combining only affects the legitimate channel capacity and is independent of the eavesdropper’s channel. Furthermore, [5,6] demonstrate the optimality of MRC-based methods and confirm that increasing SNR monotonically improves channel capacity. Therefore, in this work, we always set $\mathbf{g}$ to the MRC solution and $\|\mathbf{g}\|^2 = 1$ to optimize SNR of legitimate channel. So the optimization problem can be simplified as:

$$
\max_{\mathbf{w}, \Gamma_0, \Gamma_1} \quad SR(\mathbf{w}, \Gamma_0, \Gamma_1)
$$
where
$$
SR(\mathbf{w}, \Gamma_0, \Gamma_1) = 
\log_2 \left( 1 + \frac{\eta_b |\mathbf{h}_{RU}^T \mathbf{w}|^2 \|\mathbf{h}_{RU}^H\|^2 |\Gamma_0 - \Gamma_1|^2}{4\sigma_R^2} \right) 
- 
\log_2 \left( 1 + \frac{\eta_b |h_{UE}|^2 |\mathbf{h}_{RU}^T \mathbf{w}|^2 |\Gamma_0 - \Gamma_1|^2}{4\sigma_E^2} \right)
$$

#### Constraints

- **Average harvested power:** $P_{L,avg} = \eta_e \left(1 - \frac{|\Gamma_0|^2 + |\Gamma_1|^2}{2}\right) \mathbb{E}\left[|\mathbf{h}_{RU}^T \mathbf{w}|^2\right] \geq P_{th}$
- **Modulation depth:** $m = \frac{|\Gamma_0 - \Gamma_1|}{2} \geq m_{th}$
- **Beamforming norm:** $\|\mathbf{w}\|^2 \leq P_t$
- **Reflection coefficients:** $-1 \leq \Gamma_0, \Gamma_1 \leq 1$

### 3.5 Default Simulation Parameters

| Parameter                              | Notation   | Value      | Unit      |
|-----------------------------------------|------------|------------|-----------|
| RF Frequency                           | $f$        | 915        | MHz       |
| Speed of Light                         | $c$        | $3 \times 10^8$ | m/s      |
| Transmission Power                     | $P_t$      | 0.5        | W         |
| Noise Power at Reader                  | $\sigma^2_R$ | -80      | dBm       |
| Noise Power at Eavesdropper            | $\sigma^2_E$ | -80      | dBm       |
| Backscattering Efficiency              | $\eta_b$   | 0.8        | -         |
| Energy Harvesting Efficiency           | $\eta_e$   | 0.8        | -         |
| Reader-Tag Distance                    | $d_{RU}$   | 10         | m         |
| Harvested Power Threshold              | $P_{th}$   | $10^{-6}$  | W         |
| Modulation Depth Threshold             | $m_{th}$   | 0.2        | -         |

---

## 4. Optimization Problem Analysis and Solution Methods

This section provides a detailed mathematical derivation and solution strategy for the secrecy rate maximization problem, based on the updated problem in Section 3.4. The main approach is based on Alternating Optimization (AO), with subproblems solved using SCA and grid search. An additional heuristic method is also introduced.

### 4.1 Problem Statement and Non-Convexity

The formulated problem is **non-convex** due to the coupling between $\mathbf{w}$ and $(\Gamma_0, \Gamma_1)$ in both the objective and the energy harvesting constraint.

### 4.2 AO-Based Solution: SCA for $\mathbf{w}$, Grid Search for $(\Gamma_0, \Gamma_1)$

#### 4.2.1 AO Framework

The main solution framework is Alternating Optimization (AO):

1. **Fix $(\Gamma_0, \Gamma_1)$, optimize $\mathbf{w}$ (Subproblem 1)**
2. **Fix $\mathbf{w}$, optimize $(\Gamma_0, \Gamma_1)$ (Subproblem 2)**
3. **Repeat until convergence**

#### 4.2.2 Subproblem 1: SCA-Based Optimization of $\mathbf{w}$

With $(\Gamma_0, \Gamma_1)$ fixed, the secrecy rate objective can be rewritten as:
$$
SR(\mathbf{w}) = \log_2\left(1 + a|\mathbf{h}_{RU}^T \mathbf{w}|^2\right) - \log_2\left(1 + b|\mathbf{h}_{RU}^T \mathbf{w}|^2\right)
$$
where $a = \frac{\eta_b \|\mathbf{h}_{RU}^H\|^2 |\Gamma_0 - \Gamma_1|^2}{4\sigma_R^2}$ and $b = \frac{\eta_b |h_{UE}|^2 |\Gamma_0 - \Gamma_1|^2}{4\sigma_E^2}$.

The constraints become:
- $\|\mathbf{w}\|^2 \leq P_t$
- $\eta_e \left(1 - \frac{|\Gamma_0|^2 + |\Gamma_1|^2}{2}\right) |\mathbf{h}_{RU}^T \mathbf{w}|^2 \geq P_{th}$

This subproblem is still non-convex due to the difference of concave functions. We use **Successive Convex Approximation (SCA)** to solve it:

- At each SCA iteration, linearize the concave part $f_2(\mathbf{w}) = \log_2(1 + b|\mathbf{h}_{RU}^T \mathbf{w}|^2)$ at the current point $\mathbf{w}^{(k)}$:
  $$
  f_2(\mathbf{w}) \approx f_2(\mathbf{w}^{(k)}) + \nabla f_2(\mathbf{w}^{(k)})^T (\mathbf{w} - \mathbf{w}^{(k)})
  $$
- The resulting problem is convex and can be efficiently solved using the CVX toolbox in MATLAB.

**SCA+CVX Implementation Steps:**
1. Initialize $\mathbf{w}^{(0)}$.
2. At each iteration $k$:
   - Linearize $f_2(\mathbf{w})$ at $\mathbf{w}^{(k)}$.
   - Solve the convex problem for $\mathbf{w}^{(k+1)}$ using CVX.
   - Check convergence; if not, repeat.

#### 4.2.3 Subproblem 2: Grid Search for $(\Gamma_0, \Gamma_1)$

With $\mathbf{w}$ fixed, the secrecy rate is a function of $(\Gamma_0, \Gamma_1)$:
$$
SR(\Gamma_0, \Gamma_1) = \log_2\left(1 + c|\Gamma_0 - \Gamma_1|^2\right) - \log_2\left(1 + d|\Gamma_0 - \Gamma_1|^2\right)
$$
where $c = \frac{\eta_b |\mathbf{h}_{RU}^T \mathbf{w}|^2 \|\mathbf{h}_{RU}^H\|^2}{4\sigma_R^2}$ and $d = \frac{\eta_b |h_{UE}|^2 |\mathbf{h}_{RU}^T \mathbf{w}|^2}{4\sigma_E^2}$.

Constraints:
- $\frac{|\Gamma_0 - \Gamma_1|}{2} \geq m_{th}$
- $-1 \leq \Gamma_0, \Gamma_1 \leq 1$
- $\eta_e \left(1 - \frac{|\Gamma_0|^2 + |\Gamma_1|^2}{2}\right) |\mathbf{h}_{RU}^T \mathbf{w}|^2 \geq P_{th}$, where $|\mathbf{h}_{RU}^T \mathbf{w}|^2 $ is a constant

**Grid Search Implementation:**
- Discretize the feasible range of $\Gamma_0$ and $\Gamma_1$ (e.g., step size $\Delta$).
- For each pair $(\Gamma_0, \Gamma_1)$, check constraints and compute $SR$.
- Select the pair with the highest feasible secrecy rate.

#### 4.2.4 AO Algorithm Summary

**Algorithm Steps:**
1. Initialize $\mathbf{w}$, $(\Gamma_0, \Gamma_1)$.
2. Repeat:
   - Fix $(\Gamma_0, \Gamma_1)$, optimize $\mathbf{w}$ via SCA+CVX.
   - Fix $\mathbf{w}$, optimize $(\Gamma_0, \Gamma_1)$ via grid search.
   - Check convergence.
3. Output the final solution.

**Advantages:**
- AO with SCA+CVX for $\mathbf{w}$ and grid search for $(\Gamma_0, \Gamma_1)$ is efficient and easy to implement.
- The method is guaranteed to converge to a stationary point.

---

### 4.3 Heuristic Optimization Method

As an alternative to the AO-based method, a heuristic approach can be used for faster, though potentially suboptimal, solutions. One such method is as follows:

#### 4.3.1 Heuristic Algorithm Outline

1. **Beamforming Heuristic:** Set $\mathbf{w}$ as the eigenvector corresponding to the largest eigenvalue of $\mathbf{h}_{RU}\mathbf{h}_{RU}^H$ (i.e., maximum ratio transmission towards the tag), scaled to satisfy the power constraint.
2. **Reflection Coefficient Heuristic:** Set $(\Gamma_0, \Gamma_1)$ to maximize modulation depth and energy harvesting, e.g., $\Gamma_0 = 1$, $\Gamma_1 = -1$, or select from a small set of candidate pairs that satisfy the constraints.
3. **Feasibility Check:** If the harvested power constraint is not satisfied, reduce $|\Gamma_0|$ and $|\Gamma_1|$ until the constraint is met.

#### 4.3.2 Implementation Steps

- Compute $\mathbf{w}_{\text{heur}} = \sqrt{P_t} \frac{\mathbf{h}_{RU}}{\|\mathbf{h}_{RU}\|}$.
- For a small set of $(\Gamma_0, \Gamma_1)$ (e.g., $(1, -1)$, $(0.8, -0.8)$, etc.), check constraints and compute $SR$.
- Select the feasible pair with the highest secrecy rate.

**Advantages:**
- Extremely low computational complexity.
- Provides a good initial point for AO or a fast suboptimal solution.

---

## 5. Algorithm Implementation Flow

1. **Brute-Force Baseline:**
   - Discretize the feasible domains of $\mathbf{w}$, $\Gamma_0$, and $\Gamma_1$, enumerate all combinations, and select the optimal solution.
   - Suitable for small-scale problems and serves as a global optimum reference.

2. **Efficient Algorithm (AO/SCA+Grid Search):**
   - Initialize $(\Gamma_0, \Gamma_1)$ and $\mathbf{w}$.
   - Alternately optimize: fix $(\Gamma_0, \Gamma_1)$ and optimize $\mathbf{w}$ using SCA+CVX; then fix $\mathbf{w}$ and optimize $(\Gamma_0, \Gamma_1)$ using grid search.
   - Iterate until convergence.

3. **Heuristic Algorithm:**
   - Use the heuristic method described above for fast approximate solutions.

4. **Performance Evaluation:**
   - Compare secrecy rate, harvested energy, variable distributions, and computational complexity between the algorithms.

---

## 6. Simulation Setup

- **Carrier Frequency:** 915 MHz ($\lambda \approx 0.328$ m)
- **Reader Tx Power ($P_t$):** 30 dBm
- **Noise PSD:** –174 dBm/Hz, BW = 200 kHz $\Rightarrow \sigma^2 \approx -121$ dBm
- **Reader-Tag Distance ($d_{RU}$):** 5–50 m (step 5 m)
- **Number of Antennas ($N$):** $\{3,4,5,6\}$
- **Monte Carlo Runs:** 10,000 per point

**Simulation Procedure:**
- For each Monte Carlo run, randomly generate channel realizations.
- For each realization, solve the secrecy rate maximization problem using brute-force, AO/SCA+grid search, and heuristic algorithms.
- Record and average the results.

---

## 7. Results (Placeholders)

- **Fig 1:** Secrecy rate vs. distance for each $N$ (brute-force vs. AO/SCA vs. heuristic)
- **Fig 2:** Optimized $(\Gamma_0, \Gamma_1)$ vs. distance
- **Fig 3:** Reader spectral efficiency $R_R$ vs. distance
- **Fig 4:** Secrecy rate convergence curve (iterations)

---

## 8. References

1. Deepak Mishra and Erik G. Larsson, "Optimal Channel Estimation for Reciprocity-Based Backscattering With a Full-Duplex MIMO Reader," *IEEE Transactions on Signal Processing*, vol. 65, no. 15, pp. 3952-3966, Aug. 2017.
2. W. Saad, X. Zhou, Z. Han, and H. V. Poor, “On the physical layer security of backscatter wireless systems,” *IEEE Trans. Wireless Commun.*, vol. 13, no. 6, pp. 3442–3451, Jun. 2014.
3. Amus Goay, Tianyi Zhang. ELEC9123: Design Task E (Optimization for Green IoT) – Secrecy Rate Maximization in Beamforming-Assisted Backscatter Communication. Term T2, 2025.
4. E. Campbell, M. Mohammadi, D. Mishra, and M. Matthaiou, "Beamforming and Power Allocation Design for Secure Backscatter Communication," *Proc. IEEE International Conference on Communications (ICC)*, 2023.
5. A. Goldsmith, “Wireless Communications,” *Cambridge University Press*, 2005.
6. D. Tse, P. Viswanath, “Fundamentals of Wireless Communication,” *Cambridge University Press*, 2005.

---

## Appendix A: Non-Convex Optimization Methods and Theoretical Analysis

This appendix provides a comprehensive overview of the non-convex optimization techniques employed in the secrecy rate maximization problem for beamforming-assisted backscatter communication. The focus is on Successive Convex Approximation (SCA), grid search, and heuristic methods, as well as theoretical analyses regarding convergence, complexity, and error bounds.

### A.1 Principles and Mathematical Formulation

The secrecy rate maximization problem is inherently non-convex due to the coupled variables and the fractional, non-linear structure of the objective function. The general form is:

$$
\max_{\mathbf{w}, \Gamma_0, \Gamma_1} \quad R_s(\mathbf{w}, \Gamma_0, \Gamma_1)
$$
$$
\text{s.t.} \quad \mathbf{w} \in \mathcal{W}, \quad (\Gamma_0, \Gamma_1) \in \mathcal{G}, \quad \text{EH and modulation constraints}
$$

where $R_s$ is the secrecy rate, $\mathcal{W}$ is the feasible set for the beamforming vector, and $\mathcal{G}$ is the feasible set for the tag reflection coefficients.

#### DC Programming

The secrecy rate can often be expressed as the difference of two concave (or convex) functions, e.g.,

$$
R_s = \left[\log_2(1+\gamma_R) - \log_2(1+\gamma_E)\right]^+
$$

where $\gamma_R$ and $\gamma_E$ are SNRs at the legitimate receiver and eavesdropper, respectively. DC programming decomposes the objective into convex and concave parts and iteratively linearizes the concave part using first-order Taylor expansion:

$$
f(x) - g(x) \approx f(x) - \left[g(x^{(k)}) + \nabla g(x^{(k)})^T (x - x^{(k)})\right]
$$

This transforms the problem into a sequence of convex subproblems.

#### SCA (Successive Convex Approximation)

SCA generalizes DC by approximating non-convex constraints and objectives with convex surrogates at each iteration, ensuring tractable optimization and convergence to a stationary point.

#### SDR (Semidefinite Relaxation)

For quadratic forms in $\mathbf{w}$, SDR lifts the problem to a higher-dimensional semidefinite matrix variable, relaxing the rank constraint. The optimal solution is then approximated by randomization or rank-1 extraction.

#### Grid Search

Grid search discretizes the feasible domains (e.g., for $\Gamma_0$, $\Gamma_1$) and exhaustively evaluates the objective for all combinations. While globally optimal for small problems, it is computationally prohibitive for high-dimensional cases.

### A.2 DC Decomposition and First-Order Approximation

The secrecy rate objective is decomposed as:

$$
R_s(\mathbf{w}, \Gamma_0, \Gamma_1) = f_1(\mathbf{w}, \Gamma_0, \Gamma_1) - f_2(\mathbf{w}, \Gamma_0, \Gamma_1)
$$

where $f_1$ and $f_2$ are convex (or concave) functions of the variables. At each iteration $k$, $f_2$ is linearized at the current point, yielding a convex approximation:

$$
f_2(\mathbf{x}) \approx f_2(\mathbf{x}^{(k)}) + \nabla f_2(\mathbf{x}^{(k)})^T (\mathbf{x} - \mathbf{x}^{(k)})
$$

This enables efficient solution via convex optimization solvers.

### A.3 Feasible Set Analysis and Constraints

- **Beamforming Vector ($\mathbf{w}$):** Typically constrained by transmit power, i.e., $\|\mathbf{w}\|^2 \leq P_t$.
- **Reflection Coefficients ($\Gamma_0$, $\Gamma_1$):** Subject to physical realizability, e.g., $|\Gamma_i| \leq 1$, and minimum modulation depth $|\Gamma_0 - \Gamma_1| \geq \Delta_{\min}$.
- **Energy Harvesting (EH):** The tag must harvest sufficient energy, leading to constraints such as $E_{\text{harvested}} \geq E_{\min}$.

The feasible set is thus the intersection of these convex and non-convex constraints.

### A.4 Convergence and Theoretical Guarantees

- **DC/SCA:** Both methods guarantee convergence to a stationary point (KKT point) under mild regularity conditions, as each iteration solves a convex approximation and the objective is non-decreasing.
- **SDR:** Provides an upper bound to the original problem. If the solution is rank-1, it is globally optimal; otherwise, randomization yields a near-optimal feasible solution.
- **Grid Search:** Guarantees global optimality within the discretization granularity, but suffers from exponential complexity.

### A.5 Complexity and Error Analysis

- **DC/SCA:** Each iteration involves solving a convex problem (e.g., SOCP or SDP), with per-iteration complexity polynomial in the number of variables. The total complexity is $O(K \cdot \text{poly}(n))$, where $K$ is the number of iterations.
- **SDR:** Complexity is dominated by the SDP solver, typically $O(n^6)$ for $n \times n$ matrices.
- **Grid Search:** Complexity is $O(M^d)$, where $M$ is the number of grid points per variable and $d$ is the number of discretized variables.
- **Error Bounds:** For grid search, the maximum error is bounded by the grid resolution. For DC/SCA, the solution is locally optimal; global optimality is not guaranteed due to non-convexity, but empirical results show near-optimal performance.

### A.6 Summary Table

| Method      | Optimality      | Complexity         | Convergence      | Error Bound         |
|-------------|----------------|--------------------|------------------|---------------------|
| DC/SCA      | Local optimum  | Polynomial/iter.   | Guaranteed       | Stationary point    |
| SDR         | Upper bound / near-optimal | High (SDP) | Guaranteed       | Gap to rank-1       |
| Grid Search | Global (discrete) | Exponential      | Trivial          | Grid resolution     |

---

This appendix provides the theoretical foundation and practical considerations for the optimization algorithms used in this project. Detailed derivations and proofs can be provided upon request.
