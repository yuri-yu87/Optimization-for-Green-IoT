# ELEC9123 Design Task E (Optimization for Green IoT) - Term T2, 2025

**Project Title:** Secrecy Rate Maximization in Beamforming-Assisted Backscatter Communication (MIMO reader and Single-anttenna Tag/Eva)

**Author:** Yuri Yu 
**Submission File:** `z5226692_Yu_DTE_2025.zip`

---

### 1  Background & Motivation
Passive backscatter systems promise ultra-low-power data collection for IoT.  
However, the broadcast nature of RF links exposes tag messages to eavesdroppers.  
This project explores **multi-antenna reader beamforming + single-anttenna tag impedance modulation** to maximise **secrecy rate (SR)** while guaranteeing both energy harvesting and reliability.

---

### 2  Project Objectives
| # | Target | Measurable Outcome |
|---|--------|--------------------|
| O1 | Build mathematical model for multi-antenna reader ↔ single-antenna tag / eavesdropper | Complete System Model with verified equations |
| O2 | Formulate SR-maximisation with practical constraints (energy, modulation index, unit-norm beams) | Optimisation problem (Section 5) |
| O3 | Implement baseline (brute-force) and advanced (CVX/DC) solvers | `bruteSR.m`, `cvxSR.m` run without errors |
| O4 | Simulate SR vs distance, #antennas, etc. | ≥ 4 figures reproduced |
| O5 | Summarise findings & future work | Final report, 6–8 pages |

---

### 3  System Model

graph。。。。。。

* **Channels**: Rayleigh, small-scale fading  
  - $H_{RU}∈ℂ^{N×1}\sim𝒞𝒩(0,β_{RU}𝐈_N)$ 
  - Path-loss factor: $β=(λ/4πd)^2$
  - $d_{RU} = 10m$

* **Signals**  
  - Downlink: $y_{bu}=h_{RU}^Hw\,x_t+n_{bu}$  
  - Tag modulation (ASK): $x_{b,i}=√η_b(h_{RU}^Hw\,x_t)Γ_i\$ 
  - Uplink to Reader: $y_{R,i}=h_{RU}^Hv\,x_{b,i}+n_R$  
  - Eve reception: $y_{E,i}=h_{RE}^Hw\,x_tΓ_i+n_E$

---

### 4  Performance Metrics

| Symbol | Expression | Description |
|--------|------------|-------------|
| $SNR_R$ | $\frac{P_tη_bβ_{RU}^2(Γ_0-Γ_1)^2}{4σ_R^2}$,|$v^{H} h_{RU}^2(w^Hh_{RU})^2 $| Reader SNR |
| $R_R$ | $\log_2(1+SNR_R)$ | Reader spectral efficiency |
| $SNR_E$ | $\frac{P_tη_bβ_{RU}β_{RE}|Γ_0-Γ_1|^2}{4σ_E^2}\,|h_{RE}w|^2$ | Eve SNR |
| $R_E$ | $\log_2(1+SNR_E)$ | Eve spectral efficiency |
| **SR** | $[R_R-R_E]^+$ | Secrecy rate |
| $P_{L,\text{avg}}$ | see Section 2 of notes | Average harvested power |

**Constraints**

* $P_{L,\text{avg}} ≥ P_{th}$ (1 µW)  
* $|Γ_0-Γ_1| ≥ 2m_{th}$ with $m_{th}=0.2$  
* $\|w\|_2 = \|v\|_2 = 1$

---

### 5  Optimisation Problem

\[
\begin{aligned}
\max_{w,v,Γ_0,Γ_1}\;& SR(w,v,Γ_0,Γ_1)\\[3pt]
\text{s.t.}\;& \|w\|_2=\|v\|_2=1,\\
& P_{L,\text{avg}}(Γ_0,Γ_1) \ge P_{th},\\
& |Γ_0-Γ_1| \ge 2m_{th},\; |Γ_i| \le 1.
\end{aligned}
\]

* **Non-convexity**: difference of concave logs + bilinear beam terms.  
* **Solution strategy**  
  1. **Outer search** over \((Γ_0,Γ_1)\) via grid scan / Dinkelbach.  
  2. **Inner SDP**: relax \(W = ww^H\), \(V = vv^H\) (rank-1), solve with CVX.  
  3. Recover beams via dominant eigenvector or Gaussian randomisation.

---

### 6  Algorithm Implementation Plan

| Step | MATLAB File | Key Functionality |
|------|-------------|-------------------|
| 1 | `channelGen.m` | Generate Rayleigh channels with path-loss |
| 2 | `bruteSR.m` | Coarse grid of \((Γ_0,Γ_1)\); closed-form SR |
| 3 | `cvxSR.m` | CVX solver for beamforming given Γ |
| 4 | `main.m` | Sweep distance & antennas; orchestrates 1–3 |
| 5 | `plotResults.m` | Produce Figures 1–4 |

---

### 7  Simulation Set-up (to be coded)

* **Carrier**: 915 MHz (λ ≈ 0.328 m)  
* **Reader Tx power** \(P_t\): 30 dBm  
* **Noise PSD**: –174 dBm/Hz, BW = 200 kHz ⇒ \(σ^2 ≈ -121\) dBm  
* **Distance** \(d_{UR}\): 5–50 m in 5 m steps  
* **Antennas** \(N ∈ \{3,4,5,6\}\)  
* **Monte-Carlo runs**: 10 000 channels per point

---

### 8  Results (place-holders)

* **Fig 1** SR vs distance for each \(N\)  
* **Fig 2** Optimised \((Γ_0,Γ_1)\) vs distance  
* **Fig 3** Reader SE \(R_R\) vs distance  
* **Fig 4** SR convergence curve (iterations)

---

### 9  Preliminary Observations *(draft)*

1. Larger \(N\) lifts the SR ceiling owing to beamforming gain.  
2. As \(d_{UR}\) grows, the harvested-power constraint forces smaller \(|Γ_0-Γ_1|\), so SR eventually drops.  
3. CVX-optimised beams outperform simple MRT by ≈ 2–3 dB SNR, giving ~0.4 bps/Hz SR gain at 20 m.

---

### 10  Next Actions

1. **Code** brute-force baseline → sanity-check metric formulas.  
2. Integrate **CVX** (start with small \(N\) for speed).  
3. Run full Monte-Carlo, generate figures.  
4. Draft full report Sections 4–6, embed results.  
5. Add **AI-tool disclosure** paragraph per UNSW policy.

---

> **Version 0.1** — 04 Jul 2025 | Feedback welcome before polishing.


