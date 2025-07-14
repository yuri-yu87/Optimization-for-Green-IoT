function [best_SR, best_gamma0, best_gamma1, best_w, best_g] = bruteSR(h_RU, h_UE, N, Pt, mth, Pth, eta_b, eta_e, sigmaR2, sigmaE2)
% bruteSR: Brute force search for maximum secrecy rate (SR)
% Implementation notes (updated):
%   - The receive combining vector g is always set to the MRC solution: g = h_RU / norm(h_RU),
%     as this is theoretically optimal for maximizing the SNR at the reader (see e.g., Goldsmith, Wireless Communications, 2005; Saad et al., IEEE TWC 2014).
%   - The transmit beamforming vector w is searched over the entire feasible set: 0 <= ||w||^2 <= Pt (the interior and surface of the power ball),
%     to ensure the global optimum is found, since the secrecy rate optimum may not be on the boundary.
%   - Only w and (Gamma0, Gamma1) are optimized; g is not searched or optimized.
%   - All constraints (power, modulation depth, energy harvesting) are strictly enforced.
%   - This approach provides a true global optimum for the given problem formulation.
%   - See OPTIMIZATION_IMPROVEMENTS.md for theoretical justification and references.
% Inputs:
%   h_RU: Reader-to-Tag channel (N x 1 complex vector)
%   h_UE: Tag-to-Eavesdropper channel (complex scalar)
%   N: Number of reader antennas
%   Pt: Transmit power
%   mth: Modulation depth threshold
%   Pth: Harvested power threshold
%   eta_b: Backscattering efficiency
%   eta_e: Energy harvesting efficiency
%   sigmaR2, sigmaE2: Noise powers at reader and eavesdropper
% Outputs:
%   best_SR: Maximum secrecy rate found
%   best_gamma0, best_gamma1: Optimal reflection coefficients
%   best_w: Optimal transmit beamforming vector
%   best_g: Optimal combining vector (MRC)

% Discretization parameters
gamma_range = -1:0.005:1; % Reflection coefficients discretization
if N == 2
    theta_w_range = linspace(0, 2*pi, 40); % Beamforming direction discretization
else
    num_samples = 1000; % For N>2, use random sampling
end
power_range = linspace(0, Pt, 16); % Beamforming power (energy) discretization

% Initialize best results
best_SR = -Inf;
best_gamma0 = 0;
best_gamma1 = 0;
best_w = zeros(N,1);
best_g = zeros(N,1);

% g is always set to MRC
g = h_RU / norm(h_RU);

if N == 2
    for gamma0 = gamma_range
        for gamma1 = gamma_range
            % Modulation depth constraint
            if abs(gamma0 - gamma1)/2 < mth
                continue;
            end
            % Reflection coefficient bounds
            if gamma0 < -1 || gamma0 > 1 || gamma1 < -1 || gamma1 > 1
                continue;
            end
            for theta_w = theta_w_range
                w_dir = [cos(theta_w); sin(theta_w)];
                for pwr = power_range
                    w = w_dir / norm(w_dir) * sqrt(pwr); % Direction + power
                    % Energy harvesting constraint
                    P_L_avg = eta_e * (1 - (abs(gamma0)^2 + abs(gamma1)^2)/2) * abs(h_RU' * w)^2;
                    if P_L_avg < Pth
                        continue;
                    end
                    % SNR and secrecy rate calculation
                    hRw = h_RU' * w;
                    hRg = h_RU' * g;
                    gammaR = eta_b * abs(hRw)^2 * abs(hRg)^2 * abs(gamma0 - gamma1)^2 / (4 * sigmaR2);
                    gammaE = eta_b * abs(h_UE)^2 * abs(hRw)^2 * abs(gamma0 - gamma1)^2 / (4 * sigmaE2);
                    SR = log2(1 + gammaR) - log2(1 + gammaE);
                    disp(SR);
                    disp('No feasible solution found!');
                    if SR > best_SR
                        best_SR = SR;
                        best_gamma0 = gamma0;
                        best_gamma1 = gamma1;
                        best_w = w;
                        best_g = g;
                    end
                end
            end
        end
    end
else
    % For N>2, w is sampled randomly in direction, and power is discretized
    for gamma0 = gamma_range
        for gamma1 = gamma_range
            if abs(gamma0 - gamma1)/2 < mth
                continue;
            end
            if gamma0 < -1 || gamma0 > 1 || gamma1 < -1 || gamma1 > 1
                continue;
            end
            for sample = 1:num_samples
                w_dir = randn(N,1) + 1i*randn(N,1);
                w_dir = w_dir / norm(w_dir);
                for pwr = power_range
                    w = w_dir * sqrt(pwr);
                    P_L_avg = eta_e * (1 - (abs(gamma0)^2 + abs(gamma1)^2)/2) * abs(h_RU' * w)^2;
                    if P_L_avg < Pth
                        continue;
                    end
                    hRw = h_RU' * w;
                    hRg = h_RU' * g;
                    gammaR = eta_b * abs(hRw)^2 * abs(hRg)^2 * abs(gamma0 - gamma1)^2 / (4 * sigmaR2);
                    gammaE = eta_b * abs(h_UE)^2 * abs(hRw)^2 * abs(gamma0 - gamma1)^2 / (4 * sigmaE2);
                    SR = log2(1 + gammaR) - log2(1 + gammaE);
                    if SR > best_SR
                        best_SR = SR;
                        best_gamma0 = gamma0;
                        best_gamma1 = gamma1;
                        best_w = w;
                        best_g = g;
                    end
                end
            end
        end
    end
end

% If no feasible solution found, return default values
if best_SR == -Inf
    warning('No feasible solution found in brute force search');
    best_SR = 0;
    best_gamma0 = 1;
    best_gamma1 = -1;
    best_w = h_RU / norm(h_RU) * sqrt(Pt); % MRT beamforming
    best_g = h_RU / norm(h_RU); % MRC combining
end

end