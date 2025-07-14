# README (Updated)

## Key Implementation Principles
- **Receive combining vector g:** For all optimization methods (brute force, CVX, PSO), g is always set to the MRC solution: g = h_RU / ||h_RU||. This is theoretically optimal for maximizing the SNR at the reader (see Goldsmith, 2005; Saad et al., 2014; Design Journal Section 3.4).
- **Transmit beamforming vector w:** w is searched/optimized over the entire feasible set (0 <= ||w||^2 <= Pt), i.e., both the surface and interior of the power ball. This is necessary because the secrecy rate optimum may not be on the boundary.
- **Only w and (Gamma0, Gamma1) are optimized:** g is not optimized or searched.
- **All constraints (power, modulation depth, energy harvesting) are strictly enforced.**

## Theoretical Justification
- See Goldsmith, "Wireless Communications," 2005; Saad et al., IEEE TWC 2014; and the Design Journal for proofs and discussion.

(rest of the README unchanged) 
