# Secrecy Rate Maximization in Beamforming-Assisted Backscatter Communication

This project investigates secrecy rate maximization in a beamforming-assisted backscatter communication system, focusing on secure and energy-efficient IoT design. The system features a multi-antenna reader, a single-antenna tag, and a single-antenna eavesdropper. The goal is to jointly optimize the reader's transmit beamforming and the tag's reflection coefficients to maximize the secrecy rate under practical constraints (energy harvesting, modulation depth, etc.).

## Project Structure

```
TaskE_Optimization_problem_of_green_iot
├── Main                       # Core simulation and algorithm scripts
│   ├── main_simulation.m       # Main simulation entry point
│   ├── bruteSR.m               # Brute force secrecy rate maximization
│   ├── cvxSR.m                 # AOSCA-CVX secrecy rate maximization
│   ├── psoSR.m                 # PSO-based algorithm (if present)
│   └── README.txt              # Usage instructions for Main
├── Saved_Results_analysis     # Post-simulation analysis and plotting
│   ├── main_post_analysis.m    # Main post-processing and plotting script
│   ├── load_and_analyze_data.m # Load and analyze simulation results
│   ├── save_simulation_data.m  # Save simulation results to .mat
│   ├── .mat                   # Saved simulation and analysis results
│   └── README.txt              # Usage instructions for analysis
├── z5226692_Yu_DTE_2025.pdf           # Full technical report and documentation
└── README.txt                   # (This file)
```

## How to Run Simulations

1. Simulation
   - Go to the `Main` directory.
   - Open and run `main_simulation.m` in MATLAB.
   - This script runs Monte Carlo simulations for different algorithms (Brute Force, AOSCA-CVX, PSO) and system parameters.
   - Results are saved as `.mat` files for further analysis.

2. Post-Analysis
   - Go to the `Saved_Results_analysis` directory.
   - Use `load_and_analyze_data.m` to load simulation results.
   - Run `main_post_analysis.m` to generate plots, perform statistical analysis, and visualize key results.

## Key Features
- Implements brute-force, AOSCA-CVX, and (optionally) PSO-based secrecy rate maximization algorithms.
- Supports energy harvesting and modulation depth constraints.
- Provides comprehensive post-simulation analysis and visualization scripts.
- All code is written in MATLAB.

## References
See `z5226692_Yu_DTE_2025.pdf` for detailed methodology, results, and references.

---
For any questions, please refer to the design journal or contact the project maintainer. 