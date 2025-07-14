# Optimization Improvements Summary (Updated)

## Key Theoretical Principle
- **Receive combining vector g:** For all methods (brute force and efficient), g is always set to the MRC solution: g = h_RU / ||h_RU||. This is theoretically optimal for maximizing the SNR at the reader in this system model, as shown in:
  - Goldsmith, "Wireless Communications," Cambridge University Press, 2005.
  - Saad et al., "On the physical layer security of backscatter wireless systems," IEEE TWC, 2014.
  - Your Design Journal, Section 3.4 and references therein.
- **Transmit beamforming vector w:** For true global optimality, w must be searched/optimized over the entire feasible set (the power ball): 0 <= ||w||^2 <= Pt. The secrecy rate optimum may occur in the interior, not just on the boundary, because increasing power can simultaneously benefit the eavesdropper and the legitimate receiver, so the best tradeoff may be at a lower power.

## Implementation Changes
- **g is always MRC:** All code and documentation now use g = h_RU / ||h_RU||, not optimized or searched.
- **w is globally searched/optimized:** Brute force and all efficient methods now search/optimize w over the full power ball (not just the surface), ensuring the true global optimum is found.
- **All constraints enforced:** Power, modulation depth, and energy harvesting constraints are strictly enforced in all methods.

## Theoretical Justification
- See Goldsmith (2005), Saad et al. (2014), and your Design Journal for proofs and discussion of MRC optimality in this context.
- See Design Journal Section 3.4 for the necessity of searching the full power ball for w.

## Summary Table
| Variable | Brute Force | Efficient Methods (CVX/PSO) | Theoretical Justification |
|----------|-------------|-----------------------------|--------------------------|
| g        | MRC only    | MRC only                    | Goldsmith, Saad et al.   |
| w        | Full ball   | Full ball                   | Design Journal           |
| Γ₀,Γ₁    | Grid search | Grid search/optimization    | Design Journal           |

## References
- Goldsmith, A., "Wireless Communications," Cambridge University Press, 2005.
- Saad, W., Zhou, X., Han, Z., Poor, H. V., "On the physical layer security of backscatter wireless systems," IEEE Trans. Wireless Commun., vol. 13, no. 6, pp. 3442–3451, Jun. 2014.
- Your Design Journal, Section 3.4 and references therein. 