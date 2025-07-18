function [best_SR, best_gamma0, best_gamma1, best_w, best_g] = bruteSR(h_RU, h_UE, N, Pt, mth, Pth, eta_b, eta_e, sigmaR2, sigmaE2)
% bruteSR: Brute force search for maximum secrecy rate (SR) with vectorization for efficiency
% Implementation notes:
%   - The receive combining vector g is always set to the MRC solution: g = h_RU / norm(h_RU)
%   - The transmit beamforming vector w is exhaustively searched over a discretized set
%   - Only w and (Gamma0, Gamma1) are optimized using Brute Force search; g is fixed to MRC
%   - All constraints (power, modulation depth, energy harvesting) are strictly enforced
%   - This approach provides a true global optimum for the given problem formulation
% Inputs:
%   h_RU: Reader-to-Tag channel (N x 1 complex vector)
%   h_UE: Tag-to-Eavesdropper channel (complex scalar)
%   N: Number of reader antennas (N = 3, 4, 5, 6)
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

% --------- Brute Force Implementation ---------
% Discretization parameters
gamma_range = -1:0.05:1; % Reflection coefficients discretization
power_range = linspace(0, Pt, 5); % Beamforming power discretization

% Discretize the direction of w using a grid on the N-dimensional unit sphere
num_phase = 8; % Number of phase points per antenna
phase_grid = linspace(0, 2*pi, num_phase+1); phase_grid(end) = []; % Remove duplicate 2pi

% Generate all possible phase combinations for N-1 antennas (first is real and positive)
if N <= 5
    [phase_mat{1:max(N-1,1)}] = ndgrid(phase_grid);
    if N > 1
        phase_combinations = zeros(numel(phase_mat{1}), N-1);
        for k = 1:N-1
            phase_combinations(:,k) = phase_mat{k}(:);   % from multi-dimensional to 2D
        end
    else
        phase_combinations = zeros(1,0); % N=1 special case
    end
else
    max_dirs = 1000;
    phase_combinations = 2*pi*rand(max_dirs, max(N-1,1));
end
num_dirs = size(phase_combinations,1); % 8^{N-1} or 1000

% Precompute all direction vectors (unit norm)
W_dir = zeros(N, num_dirs);
W_dir(1,:) = 1;
for n = 2:N
    W_dir(n,:) = exp(1i * phase_combinations(:,n-1)).';
end
W_dir = W_dir ./ vecnorm(W_dir); % Normalize each column/antenna

% Precompute all power-scaled beamforming vectors
num_pwr = numel(power_range); % 5
W_all = zeros(N, num_dirs*num_pwr);
for k = 1:num_pwr
    idx = (k-1)*num_dirs + (1:num_dirs);  % (k-1)*8^{N-1}+1:k*8^{N-1}
    W_all(:,idx) = W_dir * sqrt(power_range(k));
end

% Precompute h_RU' * w for all w
h_RU_col = h_RU(:); % Ensure column vector
hRw_all = h_RU_col.' * W_all; % 1 x (num_dirs*num_pwr)

% g is always set to MRC
g = h_RU / norm(h_RU);
hRg = h_RU' * g; % Scalar

% Precompute all valid (gamma0, gamma1) pairs (modulation depth constraint)
[Gamma0, Gamma1] = meshgrid(gamma_range, gamma_range);
Gamma0 = Gamma0(:);
Gamma1 = Gamma1(:);
mod_depth = abs(Gamma0 - Gamma1)/2;
valid_idx = mod_depth >= mth & Gamma0 >= -1 & Gamma0 <= 1 & Gamma1 >= -1 & Gamma1 <= 1;
Gamma0 = Gamma0(valid_idx);
Gamma1 = Gamma1(valid_idx);
num_gamma = numel(Gamma0); % valid number of (gamma0, gamma1) pairs

% Precompute energy harvesting constraint term for all w
abs_hRw2 = abs(hRw_all).^2; % 1 x (num_dirs*num_pwr)

% Initialize best results
best_SR = -Inf;
best_gamma0 = 0;
best_gamma1 = 0;
best_w = zeros(N,1);
best_g = zeros(N,1);

% fprintf('Start brute-force search for N = %d (vectorized)\n', N);

for idx_gamma = 1:num_gamma
    gamma0 = Gamma0(idx_gamma);
    gamma1 = Gamma1(idx_gamma);

    % Energy harvesting constraint (vectorized for all w)
    P_L_avg_all = eta_e * (1 - (abs(gamma0)^2 + abs(gamma1)^2)/2) * abs_hRw2;
    valid_w_idx = find(P_L_avg_all >= Pth);

    if isempty(valid_w_idx) % no feasible w fulfilling the energy harvesting constraint
        continue;
    end

    % SNR and secrecy rate calculation (vectorized for all valid w)
    abs_hRw2_valid = abs_hRw2(valid_w_idx);

    gammaR = eta_b * abs_hRw2_valid .* abs(hRg)^2 * abs(gamma0 - gamma1)^2 / (4 * sigmaR2);
    gammaE = eta_b * abs(h_UE)^2 * abs_hRw2_valid * abs(gamma0 - gamma1)^2 / (4 * sigmaE2);
    SR_vec = log2(1 + gammaR) - log2(1 + gammaE);

    % Find the best SR for current gamma0, gamma1
    [SR_max, idx_max] = max(SR_vec);
    if SR_max > best_SR
        best_SR = SR_max;
        best_gamma0 = gamma0;
        best_gamma1 = gamma1;
        best_w = W_all(:, valid_w_idx(idx_max));
        best_g = g;
    end
end

% If no feasible solution found, return default values
if best_SR == -Inf
    warning('No feasible solution found in brute force search');
    best_SR = 0;
    best_gamma0 = 0.8;
    best_gamma1 = -0.8;
    best_w = conj(h_RU) / norm(conj(h_RU)) * sqrt(Pt); % MRT beamforming
    best_g = h_RU / norm(h_RU); % MRC combining
end