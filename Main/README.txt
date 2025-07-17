The `Main` folder contains the core simulation and algorithm implementation files for 
the secrecy rate maximization project in backscatter communication systems. Below is 
an overview and usage guide for the files in this directory.

---

## Folder Overview

- **main_simulation.m**  
  The main script to run Monte Carlo simulations. It evaluates the secrecy rate (SR) 
  performance of different algorithms (Brute Force, CVX-based AO, PSO) over various 
  system parameters such as the number of antennas and Tag-Eve distances. Results are 
  saved for further analysis.

- **bruteSR.m**  
  Implements the brute force search algorithm to find the global optimum secrecy rate by 
  exhaustively searching over possible beamforming vectors and reflection coefficients.

- **cvxSR.m**  
  Implements the Alternating Optimization (AO)-based secrecy rate maximization using convex 
  optimization (CVX). This method alternates between optimizing the transmit beamforming 
  vector and the tag reflection coefficients.

---

## Usage Instructions

1. **Set Parameters**  
   Open `main_simulation.m` and adjust the system parameters as needed 
   (e.g., number of antennas, transmit power, distances, Monte Carlo runs).

2. **Run Simulation**  
   Execute `main_simulation.m` in MATLAB. The script will:
   - Run Monte Carlo simulations in parallel (if available).
   - Call the different SR maximization algorithms for each parameter setting.
   - Store results in variables such as `SR_brute`, `SR_cvx`, `Gamma0_brute`, etc.

3. **Save and Analyze Results**  
   The saved data and analysis scripts are located in a separate folder, 
   `Saved_Results_analysis`. After the simulation, you can save the workspace variables as 
   a `.mat` file and use the scripts in that folder (such as `load_and_analyze_data.m`) to 
   load and analyze your results.

---

## Notes

- The code is designed for MATLAB and may require the CVX toolbox for convex optimization, 
as well as the Parallel Computing Toolbox for parallel simulations. Ensure all dependencies 
are installed before running the scripts.
- For large-scale simulations, adjust `MC_runs`