# ELEC9123 Design Task E (Optimization for Green IoT) - Term T2, 2025

**Project Title:** Secrecy Rate Maximization in Beamforming-Assisted Backscatter Communication (MIMO Reader, Single-Antenna Tag/Eavesdropper)

**Author:** Yuri Yu  
**Submission File:** `z5226692_Yu_DTE_2025.zip`

---

### 1. Background & Motivation

Passive backscatter systems enable ultra-low-power IoT data collection, but the open nature of RF links exposes tag messages to eavesdropping. This project investigates **multi-antenna reader beamforming** and **single-antenna tag impedance modulation** to maximize **secrecy rate (SR)**, while ensuring energy harvesting and communication reliability.

---

### 2. Project Objectives

| #  | Target                                                                 | Measurable Outcome                        |
|----|------------------------------------------------------------------------|-------------------------------------------|
| O1 | Develop mathematical model for multi-antenna reader ↔ single-antenna tag/eavesdropper | Complete, validated system model          |
| O2 | Formulate SR maximization with practical constraints                   | Optimization problem (see Section 4)      |
| O3 | Implement baseline and advanced solvers (brute-force, CVX/DC)          | `bruteSR.m`, `cvxSR.m` run successfully   |
| O4 | Simulate SR vs. distance, #antennas, etc.                              | ≥ 4 figures generated                     |
| O5 | Summarize findings & propose future work                               | Final report (6–8 pages)                  |

---

### 3. System Model

We assume flat Rayleigh block fading, where the channel remains constant during each transmission and varies independently between different channel realizations. The wireless channel from transmitter to receiver (T-to-R) is modeled as an $N \times 1$ complex Gaussian vector $\mathbf{h} \sim \mathcal{CN}(\mathbf{0}_{N \times 1}, \beta \mathbf{I}_N)$, where $\beta$ denotes the average channel power gain, incorporating both small-scale fading and large-scale path loss for the T-to-R or R-to-T link [1]. 



**Channels:**  
- Reader-to-tag: $\mathbf{h}_{RU} \in \mathbb{C}^{N \times 1} \sim \mathcal{CN}(\mathbf{0}_{N \times 1}, \beta_{RU} \mathbf{I}_N)$  
- Path-loss: $\beta_{RU} = (\lambda/4\pi d_{RU})^2$, $d_{RU} = 10$ m  
- Reader-to-eavesdropper: $\mathbf{h}_{RE} \in \mathbb{C}^{N \times 1} \sim \mathcal{CN}(\mathbf{0}_{N \times 1}, \beta_{RE} \mathbf{I}_N)$ (worst-case: eavesdropper can cancel reader’s CW interference)  
- Tag-to-eavesdropper: $h_{UE} \in \mathbb{C} \sim \mathcal{CN}(0, \beta_{UE})$

**Signals:**  
- Downlink: $y_{bu} = \mathbf{h}_{RU}^H \mathbf{w} x_t + n_{bu} \approx \mathbf{h}_{RU}^H \mathbf{w} x_t$ (ignore $n_{bu}$)  
*Note: Although the backscatter user receives noise $n_{bu}$ in the forward link, its effect on the backscattered signal at the reader is negligible due to the passive nature of the tag and double path loss. Thus, for analytical simplicity, the impact of $n_{bu}$ is omitted in the performance analysis [2].*
- Tag modulation (ASK): $x_{b,i} = \sqrt{\eta_b} (\mathbf{h}_{RU}^H \mathbf{w} x_t) \Gamma_i$  
- Uplink to reader: $y_{R,i} = \mathbf{h}_{RU}^H \mathbf{v} x_{b,i} + n_R$  
- Eavesdropper: $y_{E,i} = \mathbf{h}_{RE}^H x_{b,i} + n_E$

**SNR Expressions:**  
- Legitimate receiver:  
  $$
  \gamma_R = \frac{\eta_b P_t |\mathbf{h}_{RU}^H \mathbf{w}|^2 |\mathbf{h}_{RU}^H \mathbf{v}|^2 |\Gamma_0 - \Gamma_1|^2}{4\sigma_R^2}
  $$
- Eavesdropper:  
  $$
  \gamma_E = \frac{\eta_b P_t |h_{UE}|^2 |\mathbf{h}_{RU}^H \mathbf{w}|^2 |\Gamma_0 - \Gamma_1|^2}{4\sigma_E^2}
  $$
- Secrecy rate:  
  $$
  SR = \log_2(1 + \gamma_R) - \log_2(1 + \gamma_E)
  $$

**Constraints:**  
- Average harvested power: $P_{L,avg} \geq P_{th}$  
- Modulation depth: $m = \frac{|\Gamma_0 - \Gamma_1|}{2} \geq m_{th}$  
- Beamforming/combining norm: $\|\mathbf{w}\| \geq P_t$, $\|\mathbf{v}\| \geq 1$  
- Reflection coefficients: $-1 \leq \Gamma_i \leq 1$

---

### 4. Optimization Problem & Performance Metrics

We aim to **maximize the secrecy rate** by optimizing:  
- Reader precoding vector $\mathbf{w}$  
- Reflection coefficients $(\Gamma_0, \Gamma_1)$  

Given that the combining vector $\mathbf{v}$ is set to MRC (not an optimization variable):  
$$
\mathbf{v} = \frac{\mathbf{h}_{RU}}{\|\mathbf{h}_{RU}\|} \implies |\mathbf{h}_{RU}^H \mathbf{v}|^2 = \|\mathbf{h}_{RU}\|^2
$$

**Objective:**  
$$
\max_{\mathbf{w}, \Gamma_0, \Gamma_1} \quad SR(\mathbf{w}, \Gamma_0, \Gamma_1)
$$
where
$$
SR(\mathbf{w}, \Gamma_0, \Gamma_1) = 
\log_2 \left( 1 + \frac{\eta_b P_t |\mathbf{h}_{RU}^H \mathbf{w}|^2 \|\mathbf{h}_{RU}\|^2 |\Gamma_0 - \Gamma_1|^2}{4\sigma_R^2} \right) 
- 
\log_2 \left( 1 + \frac{\eta_b P_t |h_{UE}|^2 |\mathbf{h}_{RU}^H \mathbf{w}|^2 |\Gamma_0 - \Gamma_1|^2}{4\sigma_E^2} \right)
$$

**Subject to:**  
- Power constraint: $\|\mathbf{w}\|^2 \leq P_t$  
- Modulation depth: $m = \frac{|\Gamma_0 - \Gamma_1|}{2} \geq m_{th}$  
- Reflection bounds: $-1 \leq \Gamma_0, \Gamma_1 \leq 1$

**Remarks:**  
- The problem is **non-convex** due to bilinear terms (e.g., $|\mathbf{h}_{RU}^H \mathbf{w}|^2 |\Gamma_0 - \Gamma_1|^2$).
- Solution approach:  
  - Fix $(\Gamma_0, \Gamma_1)$, optimize $\mathbf{w}$ (convex)  
  - Alternating optimization: iterate between updating $\mathbf{w}$ and $(\Gamma_0, \Gamma_1)$  
  - Use CVX with SDR/DC programming as needed

---

### 5. Algorithm Implementation Plan

(To be detailed: includes alternating optimization, initialization, convergence criteria, etc.)

---

### 6. Simulation Set-up

- **Carrier:** 915 MHz ($\lambda \approx 0.328$ m)  
- **Reader Tx power ($P_t$):** 30 dBm  
- **Noise PSD:** –174 dBm/Hz, BW = 200 kHz $\Rightarrow \sigma^2 \approx -121$ dBm  
- **Distance ($d_{UR}$):** 5–50 m (step 5 m)  
- **Antennas ($N$):** $\{3,4,5,6\}$  
- **Monte-Carlo runs:** 10,000 channels per point

#### 6.1 Monte Carlo Simulation Method

To accurately evaluate the secrecy rate performance under random fading channels, we employ Monte Carlo simulation as follows:

- **Channel Generation:** For each simulation run, independently generate random channel realizations for all links (e.g., $\mathbf{h}_{RU}$, $\mathbf{h}_{RE}$, $h_{UE}$) according to their statistical models (e.g., Rayleigh fading with path-loss).
- **Optimization:** For each channel realization, solve the secrecy rate maximization problem using the proposed algorithm (e.g., alternating optimization).
- **Performance Recording:** Record the optimal secrecy rate and other relevant metrics for each run.
- **Averaging:** Repeat steps 1–3 for a large number of runs (e.g., 10,000), and compute the average secrecy rate and other statistics.
- **Result Presentation:** Plot the average secrecy rate versus system parameters (e.g., distance, number of antennas).

This approach ensures that the reported performance reflects the statistical nature of wireless channels, rather than a single (possibly atypical) channel realization.

---

### 7. Results (Placeholders)

- **Fig 1:** SR vs. distance for each $N$  
- **Fig 2:** Optimized $(\Gamma_0, \Gamma_1)$ vs. distance  
- **Fig 3:** Reader SE $R_R$ vs. distance  
- **Fig 4:** SR convergence curve (iterations)

---

### 8. CSI Assumptions

We assume the reader (AP) has **perfect Channel State Information (CSI)** for all channels:

- Reader–tag (forward/backward): $\mathbf{h}_{RU}$ (reciprocity: $\mathbf{h}_{RU}^{(tx)} = \mathbf{h}_{RU}^{(rx)}$)
- Reader–eavesdropper: $\mathbf{h}_{RE}$
- Tag–eavesdropper: $h_{UE}$

**Justification:**  
- Reader can estimate two-way channel via pilots (reciprocity)  
- Tag’s CSI acquisition is lightweight and energy-efficient  
- Eavesdropper’s CSI is assumed known (active: monitored; passive: inferred/statistical)  
- This is standard in physical-layer security literature ([34], [38]–[40]) and enables tractable optimization without bias from CSI acquisition cost

---

**References:**
1. Deepak Mishra and Erik G. Larsson, "Optimal Channel Estimation for Reciprocity-Based Backscattering With a Full-Duplex MIMO Reader," *IEEE Transactions on Signal Processing*, vol. 65, no. 15, pp. 3952-3966, Aug. 2017.
2. Amus Goay, Tianyi Zhang. ELEC9123: Design Task E (Optimization for Green IoT) – Secrecy Rate Maximization in Beamforming-Assisted Backscatter Communication. Term T2, 2025.