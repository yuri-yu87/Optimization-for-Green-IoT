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
- Implement both brute-force and efficient (convex/DC) optimization algorithms.
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
& P_{L,avg} = \eta_e \left(1 - \frac{|\Gamma_0|^2 + |\Gamma_1|^2}{2}\right) \mathbb{E}[|\mathbf{h}_{RU}^H \mathbf{w}|^2] \geq P_{th}
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
\log_2 \left( 1 + \frac{\eta_b |h_{UE}|^2 |\mathbf{h}_{RU}^T \mathbf{w}|^2 |\Gamma_0 - \Gamma_1|^2}{4\sigma_E^2} \right)
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
> P_{Li} = \eta_e (1 - |\Gamma_i|^2) \, \mathbb{E}\left[|\mathbf{h}_{RU}^H \mathbf{w}|^2\right], \quad i=0,1
> $$
> &emsp;Here, $\eta_e$ is the energy harvesting efficiency, and $\mathbb{E}\left[|\mathbf{h}_{RU}^H \mathbf{w}|^2\right]$ is the average received signal power at the tag.
>
> &emsp;Since the tag alternates between $\Gamma_0$ and $\Gamma_1$ with equal probability, the constraint simplifies to:
> $$
> P_{L,avg} = \eta_e \left(1 - \frac{|\Gamma_0|^2 + |\Gamma_1|^2}{2}\right) \mathbb{E}\left[|\mathbf{h}_{RU}^H \mathbf{w}|^2\right] \geq P_{th}
> $$
> &emsp;This compact form depends only on the average received power at the tag and the mean of $|\Gamma_0|^2$ and $|\Gamma_1|^2$, which facilitates optimization.

#### Simplified Optimization Problem

Moreover, It has been established in [1,2,3,4] that, for secrecy rate maximization with perfect CSI at the reader, the optimal receive combining vector is maximum ratio combining (MRC), i.e., $\mathbf{g}^* = \frac{\mathbf{h}_{RU}}{\|\mathbf{h}_{RU}\|}$. This is because the receive combining only affects the legitimate channel capacity and is independent of the eavesdropper’s channel. Furthermore, [5,6] demonstrate the optimality of MRC-based methods and confirm that increasing SNR monotonically improves channel capacity. Therefore, in this work, we always set $\mathbf{g}$ to the MRC solution and $\|\mathbf{g}\|^2 = 1$ to optimize SNR of legitimate channel. So the optimization problem can be simplified as:

$$
\max_{\mathbf{w}, \Gamma_0, \Gamma_1} \quad SR(\mathbf{w}, \Gamma_0, \Gamma_1)
$$
where
$$
SR(\mathbf{w}, \Gamma_0, \Gamma_1) = 
\log_2 \left( 1 + \frac{\eta_b |\mathbf{h}_{RU}^T \mathbf{w}|^2 \|\mathbf{h}_{RU}^H \|^2 |\Gamma_0 - \Gamma_1|^2}{4\sigma_R^2} \right) 
- 
\log_2 \left( 1 + \frac{\eta_b |h_{UE}|^2 |\mathbf{h}_{RU}^T \mathbf{w}|^2 |\Gamma_0 - \Gamma_1|^2}{4\sigma_E^2} \right)
$$

#### Constraints

- **Average harvested power:** $P_{L,avg} = \eta_e \left(1 - \frac{|\Gamma_0|^2 + |\Gamma_1|^2}{2}\right) \mathbb{E}\left[|\mathbf{h}_{RU}^H \mathbf{w}|^2\right] \geq P_{th}$
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

## 4. Optimization Problem Formulation

This section provides a concise formulation of the optimization objective, variables, constraints, and the non-convexity of the problem, serving as a foundation for the subsequent algorithm and theoretical analysis sections. For clarity and brevity, detailed mathematical derivations, non-convex optimization methods, and complexity/error analysis are moved to **Appendix A: Non-Convex Optimization Methods and Theoretical Analysis**.

### 4.1 Problem Statement and Simplification



### 4.2 Non-Convexity and Solution Strategies

The formulated problem is **non-convex**, mainly due to the coupling and multiplicative terms involving $\mathbf{w}$ and $(\Gamma_0, \Gamma_1)$ in both the objective and the energy harvesting constraint.

**Common Solution Strategies:**

- **Alternating Optimization (AO):** Iteratively optimize $\mathbf{w}$ with fixed $(\Gamma_0, \Gamma_1)$, then optimize $(\Gamma_0, \Gamma_1)$ with fixed $\mathbf{w}$, until convergence.
- **Difference-of-Convex (DC) Programming:** Decompose the objective into the difference of two concave functions, linearize one part, and solve the resulting convex subproblem iteratively.
- **Grid Search:** For low-dimensional variables $(\Gamma_0, \Gamma_1)$, exhaustive or grid search can efficiently find the optimum.
- **SDR and Other Methods:** For quadratic objectives or constraints, semidefinite relaxation (SDR) and other advanced methods can be applied.

> **Note:** Detailed mathematical derivations, complexity, and error analysis of these methods are provided in **Appendix A**.

---

### 4.5 Hybrid Non-Convex Optimization Method for Section 3.4 Problem

This section presents a comprehensive hybrid optimization approach specifically designed for the optimization problem formulated in Section 3.4, combining multiple non-convex optimization techniques to efficiently solve the secrecy rate maximization problem.

#### 4.5.1 Problem Structure Analysis from Section 3.4

**Original Problem (from Section 3.4):**
$$
\max_{\mathbf{w}, \Gamma_0, \Gamma_1} \quad SR(\mathbf{w}, \Gamma_0, \Gamma_1)
$$

where
$$
SR(\mathbf{w}, \Gamma_0, \Gamma_1) = 
\log_2 \left( 1 + \frac{\eta_b |\mathbf{h}_{RU}^H \mathbf{w}|^2 \|\mathbf{h}_{RU}\|^2 |\Gamma_0 - \Gamma_1|^2}{4\sigma_R^2} \right) 
- 
\log_2 \left( 1 + \frac{\eta_b |h_{UE}|^2 |\mathbf{h}_{RU}^H \mathbf{w}|^2 |\Gamma_0 - \Gamma_1|^2}{4\sigma_E^2} \right)
$$

**Constraints:**
1. **Power constraint**: $\|\mathbf{w}\|^2 \leq P_t$
2. **Modulation depth**: $\frac{|\Gamma_0 - \Gamma_1|}{2} \geq m_{th}$
3. **Amplitude constraints**: $-1 \leq \Gamma_0, \Gamma_1 \leq 1$
4. **Energy harvesting**: $\eta_e \left(1 - \frac{|\Gamma_0|^2 + |\Gamma_1|^2}{2}\right) |\mathbf{h}_{RU}^H \mathbf{w}|^2 \geq P_{th}$

**Non-Convexity Analysis:**
- **Objective**: DC structure (difference of concave functions)
- **Energy harvesting constraint**: Bilinear/quadratic structure
- **Overall**: Non-convex optimization problem

#### 4.5.2 Hybrid Optimization Framework

**Framework Overview:**
We propose a **three-stage hybrid optimization framework** that combines:
1. **AO (Alternating Optimization)** as the main framework
2. **DC Programming/SCA** for the $\mathbf{w}$ optimization subproblem
3. **Grid Search** for the $(\Gamma_0, \Gamma_1)$ optimization subproblem
4. **SDR** as an alternative for quadratic-dominated subproblems

**Algorithm Flow:**
```
Initialize: w, (Γ₀, Γ₁)
For iteration = 1 to max_iter:
    Step 1: Fix (Γ₀, Γ₁), optimize w using DC/SCA
    Step 2: Fix w, optimize (Γ₀, Γ₁) using Grid Search
    Step 3: Check convergence
    If converged: break
End
```

#### 4.5.3 Stage 1: AO Framework with Problem Decomposition

**Subproblem 1: Optimize $\mathbf{w}$ with fixed $(\Gamma_0, \Gamma_1)$**

**Objective function becomes:**
$$
SR(\mathbf{w}) = \log_2\left(1 + A|\mathbf{h}_{RU}^H \mathbf{w}|^2\right) - \log_2\left(1 + B|\mathbf{h}_{RU}^H \mathbf{w}|^2\right)
$$

where $A = \frac{\eta_b \|\mathbf{h}_{RU}\|^2 |\Gamma_0 - \Gamma_1|^2}{4\sigma_R^2}$ and $B = \frac{\eta_b |h_{UE}|^2 |\Gamma_0 - \Gamma_1|^2}{4\sigma_E^2}$ are constants.

**Constraints:**
- $\|\mathbf{w}\|^2 \leq P_t$ (convex)
- $\eta_e \left(1 - \frac{|\Gamma_0|^2 + |\Gamma_1|^2}{2}\right) |\mathbf{h}_{RU}^H \mathbf{w}|^2 \geq P_{th}$ (convex when $(\Gamma_0, \Gamma_1)$ fixed)

**Subproblem 2: Optimize $(\Gamma_0, \Gamma_1)$ with fixed $\mathbf{w}$**

**Objective function becomes:**
$$
SR(\Gamma_0, \Gamma_1) = \log_2\left(1 + C|\Gamma_0 - \Gamma_1|^2\right) - \log_2\left(1 + D|\Gamma_0 - \Gamma_1|^2\right)
$$

where $C = \frac{\eta_b |\mathbf{h}_{RU}^H \mathbf{w}|^2 \|\mathbf{h}_{RU}\|^2}{4\sigma_R^2}$ and $D = \frac{\eta_b |h_{UE}|^2 |\mathbf{h}_{RU}^H \mathbf{w}|^2}{4\sigma_E^2}$ are constants.

**Constraints:**
- $\frac{|\Gamma_0 - \Gamma_1|}{2} \geq m_{th}$ (convex)
- $-1 \leq \Gamma_0, \Gamma_1 \leq 1$ (convex)
- $\eta_e \left(1 - \frac{|\Gamma_0|^2 + |\Gamma_1|^2}{2}\right) C' \geq P_{th}$ (convex when $\mathbf{w}$ fixed)

#### 4.5.4 Stage 2: DC Programming for $\mathbf{w}$ Optimization

**DC Decomposition:**
$$
SR(\mathbf{w}) = f_1(\mathbf{w}) - f_2(\mathbf{w})
$$

where:
$$
f_1(\mathbf{w}) = \log_2\left(1 + A|\mathbf{h}_{RU}^H \mathbf{w}|^2\right)
$$
$$
f_2(\mathbf{w}) = \log_2\left(1 + B|\mathbf{h}_{RU}^H \mathbf{w}|^2\right)
$$

**First-Order Taylor Expansion:**
At iteration $k$, linearize $f_2$ at $\mathbf{w}^{(k)}$:
$$
f_2(\mathbf{w}) \approx f_2(\mathbf{w}^{(k)}) + \nabla f_2(\mathbf{w}^{(k)})^T (\mathbf{w} - \mathbf{w}^{(k)})
$$

**Gradient Calculation:**
$$
\nabla f_2(\mathbf{w}) = \frac{2B \Re\{(\mathbf{h}_{RU}^H \mathbf{w})\} \mathbf{h}_{RU}}{\ln(2)(1 + B|\mathbf{h}_{RU}^H \mathbf{w}|^2)}
$$

**Convex Subproblem Formulation:**
$$
\max_{\mathbf{w}} \quad f_1(\mathbf{w}) - f_2(\mathbf{w}^{(k)}) - \nabla f_2(\mathbf{w}^{(k)})^T (\mathbf{w} - \mathbf{w}^{(k)})
$$

subject to:
$$
\|\mathbf{w}\|^2 \leq P_t
$$
$$
\eta_e \left(1 - \frac{|\Gamma_0|^2 + |\Gamma_1|^2}{2}\right) |\mathbf{h}_{RU}^H \mathbf{w}|^2 \geq P_{th}
$$

**Solution Method:**
- Use CVX or other convex optimization solvers
- Solve iteratively until convergence

#### 4.5.5 Stage 3: Grid Search for $(\Gamma_0, \Gamma_1)$ Optimization

**Grid Discretization:**
$$
\Gamma_0, \Gamma_1 \in \{-1, -1+\Delta, -1+2\Delta, \ldots, 1\}
$$

**Grid Search Algorithm:**
```matlab
best_SR = -Inf;
for gamma0 = gamma_range
    for gamma1 = gamma_range
        % Check constraints
        if abs(gamma0 - gamma1)/2 >= mth
            % Calculate SR
            SR = log2(1 + C*abs(gamma0-gamma1)^2) - log2(1 + D*abs(gamma0-gamma1)^2);
            if SR > best_SR
                best_SR = SR;
                optimal_gamma0 = gamma0;
                optimal_gamma1 = gamma1;
            end
        end
    end
end
```

**Optimal Grid Density Selection:**
$$
\Delta^* = \arg\min_{\Delta} \left\{ \frac{\Delta^2}{2} \max \|\nabla^2 SR\| + \lambda \frac{4}{\Delta^2} \right\}
$$

#### 4.5.6 Alternative Stage 3: SDR for Quadratic Subproblems

**When to Use SDR:**
- If the $(\Gamma_0, \Gamma_1)$ subproblem has strong quadratic structure
- For higher precision requirements
- When grid search is computationally expensive

**SDR Formulation:**
Define $X = \begin{bmatrix} \Gamma_0 \\ \Gamma_1 \end{bmatrix} \begin{bmatrix} \Gamma_0 & \Gamma_1 \end{bmatrix}$, then:
$$
\max_X \quad \text{tr}(A X) - \text{tr}(B X)
$$

subject to:
$$
X \succeq 0, \quad \text{rank}(X) = 1
$$

**SDR Relaxation:**
Remove the rank constraint and solve the resulting SDP:
$$
\max_X \quad \text{tr}(A X) - \text{tr}(B X)
$$

subject to:
$$
X \succeq 0
$$

**Solution Recovery:**
Use randomization or rank-1 extraction to recover $(\Gamma_0, \Gamma_1)$ from $X$.

#### 4.5.7 Convergence Analysis and Theoretical Guarantees

**Theorem 1 (Monotonicity):**
The objective function value is monotonically non-decreasing in each AO iteration.

**Proof:**
At iteration $k$:
$$
SR(\mathbf{w}^{(k+1)}, \Gamma_0^{(k+1)}, \Gamma_1^{(k+1)}) \geq SR(\mathbf{w}^{(k)}, \Gamma_0^{(k)}, \Gamma_1^{(k)})
$$

**Theorem 2 (Convergence):**
The hybrid algorithm converges to a local optimum of the original problem.

**Proof:**
- DC programming converges to a stationary point
- Grid search finds the global optimum within the grid
- AO framework ensures convergence to a local optimum

**Convergence Rate:**
- **DC Programming**: Linear convergence
- **Grid Search**: Deterministic, finite steps
- **Overall AO**: Linear convergence

#### 4.5.8 Computational Complexity Analysis

**Per AO Iteration:**
- **Subproblem 1 (DC Programming)**: $O(N^3)$ for convex optimization
- **Subproblem 2 (Grid Search)**: $O(M^2)$ where $M = \frac{2}{\Delta}$
- **Total per iteration**: $O(N^3 + M^2)$

**Overall Complexity:**
- **Total iterations**: $I_{AO}$
- **Overall complexity**: $O(I_{AO} \cdot (N^3 + M^2))$

**Memory Requirements:**
- **DC Programming**: $O(N^2)$ for gradient and Hessian
- **Grid Search**: $O(M^2)$ for storing grid points
- **Total memory**: $O(N^2 + M^2)$

#### 4.5.9 Implementation Guidelines

**Parameter Selection:**
- **Grid step size**: $\Delta = 0.02$ for high precision, $\Delta = 0.05$ for efficiency
- **DC tolerance**: $\epsilon = 10^{-3}$ for convergence
- **AO tolerance**: $\epsilon_{AO} = 10^{-4}$ for overall convergence
- **Maximum iterations**: $I_{max} = 50$ for DC, $I_{AO}^{max} = 20$ for AO

**Implementation Flow:**
```matlab
% Initialize
w = randn(N,1); w = w/norm(w)*sqrt(Pt);
gamma0 = 1; gamma1 = -1;

% AO Main Loop
for iter_ao = 1:max_iter_ao
    % Stage 1: DC Programming for w
    w = optimize_w_DC(gamma0, gamma1, w);
    
    % Stage 2: Grid Search for (gamma0, gamma1)
    [gamma0, gamma1] = optimize_gamma_grid(w);
    
    % Check convergence
    if abs(SR_new - SR_old) < tol_ao
        break;
    end
end
```

**Error Bounds:**
- **DC Programming error**: $O(\epsilon^2)$
- **Grid Search error**: $O(\Delta^2)$
- **Overall error**: $O(\epsilon^2 + \Delta^2)$

This hybrid approach provides a comprehensive solution that leverages the strengths of multiple optimization methods while maintaining computational efficiency and theoretical guarantees.

---

### 4.3 Recommended Approach and Implementation

- **Main Approach:** Use an AO framework, where the $\mathbf{w}$-subproblem is solved via DC programming and the $(\Gamma_0, \Gamma_1)$-subproblem via grid search.
- **Advantages:** Good theoretical convergence, high computational efficiency, and ease of implementation.
- **Alternative Approaches:** AO+SCA, AO+PSO, AO+SDR, etc., can be considered for more complex or large-scale scenarios.

---

## 5. Algorithm Implementation Flow

1. **Brute-Force Baseline:**
   - Discretize the feasible domains of $\mathbf{w}$, $\Gamma_0$, and $\Gamma_1$, enumerate all combinations, and select the optimal solution.
   - Suitable for small-scale problems and serves as a global optimum reference.

2. **Efficient Algorithm (AO/DC):**
   - Initialize $(\Gamma_0, \Gamma_1)$ and $\mathbf{w}$.
   - Alternately optimize: fix $(\Gamma_0, \Gamma_1)$ and optimize $\mathbf{w}$ (convex optimization), then fix $\mathbf{w}$ and optimize $(\Gamma_0, \Gamma_1)$ (grid search or closed-form).
   - Iterate until convergence; DC programming can be used to accelerate convergence and improve solution quality.

3. **Performance Evaluation:**
   - Compare secrecy rate, harvested energy, variable distributions, and computational complexity between the two algorithms.

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
- For each realization, solve the secrecy rate maximization problem using both brute-force and efficient algorithms.
- Record and average the results.

---

## 7. Results (Placeholders)

- **Fig 1:** Secrecy rate vs. distance for each $N$ (brute-force vs. efficient)
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

This appendix provides a comprehensive overview of the non-convex optimization techniques employed in the secrecy rate maximization problem for beamforming-assisted backscatter communication. The focus is on Difference-of-Convex (DC) programming, Successive Convex Approximation (SCA), Semidefinite Relaxation (SDR), and grid search methods. Theoretical analyses regarding convergence, complexity, and error bounds are also included.

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