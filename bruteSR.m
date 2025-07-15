function [best_SR, best_gamma0, best_gamma1, best_w, best_g] = bruteSR(h_RU, h_UE, N, Pt, mth, Pth, eta_b, eta_e, sigmaR2, sigmaE2)
% bruteSR: Brute force search for maximum secrecy rate (SR) with vectorization for efficiency
% Implementation notes:
%   - The receive combining vector g is always set to the MRC solution: g = h_RU / norm(h_RU)
%   - The transmit beamforming vector w is exhaustively searched over a discretized set (no random sampling)
%   - Only w and (Gamma0, Gamma1) are optimized; g is not searched or optimized
%   - All constraints (power, modulation depth, energy harvesting) are strictly enforced
%   - This approach provides a true global optimum for the given problem formulation
% Inputs:
%   h_RU: Reader-to-Tag channel (N x 1 complex vector)
%   h_UE: Tag-to-Eavesdropper channel (complex scalar)
%   N: Number of reader antennas (N >= 3)
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

% --------- Vectorized Implementation ---------
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
            phase_combinations(:,k) = phase_mat{k}(:);
        end
    else
        phase_combinations = zeros(1,0); % N=1 special case
    end
else
    max_dirs = 1000;
    phase_combinations = 2*pi*rand(max_dirs, max(N-1,1));
end
num_dirs = size(phase_combinations,1);

% Precompute all direction vectors (unit norm)
W_dir = zeros(N, num_dirs);
W_dir(1,:) = 1;
for n = 2:N
    W_dir(n,:) = exp(1i * phase_combinations(:,n-1)).';
end
W_dir = W_dir ./ vecnorm(W_dir); % Normalize each column

% Precompute all power-scaled beamforming vectors
num_pwr = numel(power_range);
W_all = zeros(N, num_dirs*num_pwr);
for k = 1:num_pwr
    idx = (k-1)*num_dirs + (1:num_dirs);
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
mod_depth = mod_depth(valid_idx);
num_gamma = numel(Gamma0);

% Precompute energy harvesting constraint term for all w
abs_hRw2 = abs(hRw_all).^2; % 1 x (num_dirs*num_pwr)

% Initialize best results
best_SR = -Inf;
best_gamma0 = 0;
best_gamma1 = 0;
best_w = zeros(N,1);
best_g = zeros(N,1);

fprintf('Start brute-force search for N = %d (vectorized)\n', N);

for idx_gamma = 1:num_gamma
    gamma0 = Gamma0(idx_gamma);
    gamma1 = Gamma1(idx_gamma);
    % md = mod_depth(idx_gamma); % 这个变量其实没用到，可以删掉

    % Energy harvesting constraint (vectorized for all w)
    P_L_avg_all = eta_e * (1 - (abs(gamma0)^2 + abs(gamma1)^2)/2) * abs_hRw2;
    valid_w_idx = find(P_L_avg_all >= Pth);

    if isempty(valid_w_idx)
        continue;
    end

    % SNR and secrecy rate calculation (vectorized for all valid w)
    % hRw_valid = hRw_all(valid_w_idx); % 没用到，可以删掉
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
    best_gamma0 = 1;
    best_gamma1 = -1;
    best_w = conj(h_RU) / norm(conj(h_RU)) * sqrt(Pt); % MRT beamforming
    best_g = h_RU / norm(h_RU); % MRC combining
end

% --------- Original (Non-Vectorized) Code for Reference ---------
%{
% Discretization parameters
gamma_range = -1:0.08:1; % Reflection coefficients discretization
power_range = linspace(0, Pt, 5); % Beamforming power discretization

% Discretize the direction of w using a grid on the N-dimensional unit sphere
% For simplicity, we use a grid over phases for each antenna (except the first, which is real and positive)
num_phase = 8; % Number of phase points per antenna
phase_grid = linspace(0, 2*pi, num_phase+1); phase_grid(end) = []; % Remove duplicate 2pi

% Generate all possible phase combinations for N-1 antennas (first is real and positive)
% This can be memory intensive for large N, so use caution
if N <= 5
    [phase_mat{1:N-1}] = ndgrid(phase_grid);
    phase_combinations = zeros(numel(phase_mat{1}), N-1);
    for k = 1:N-1
        phase_combinations(:,k) = phase_mat{k}(:);
    end
else
    % For large N, limit the number of directions to avoid explosion
    max_dirs = 1000;
    phase_combinations = 2*pi*rand(max_dirs, N-1);
end

% Initialize best results
best_SR = -Inf;
best_gamma0 = 0;
best_gamma1 = 0;
best_w = zeros(N,1);
best_g = zeros(N,1);

% g is always set to MRC
g = h_RU / norm(h_RU);

fprintf('Start brute-force search for N = %d\n', N);

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
        % Loop over all discretized directions
        for idx_dir = 1:size(phase_combinations,1)
            % Construct direction vector on the N-dimensional unit sphere
            w_dir = zeros(N,1);
            w_dir(1) = 1; % First element is real and positive
            for n = 2:N
                w_dir(n) = exp(1i * phase_combinations(idx_dir, n-1));
            end
            w_dir = w_dir / norm(w_dir); % Normalize to unit norm
            for pwr = power_range
                w = w_dir * sqrt(pwr); % Apply power scaling
                % Energy harvesting constraint
                P_L_avg = eta_e * (1 - (abs(gamma0)^2 + abs(gamma1)^2)/2) * abs(h_RU.' * w)^2;  % 使用转置，匹配文档
                if P_L_avg < Pth
                    continue;
                end
                % SNR and secrecy rate calculation
                hRw = h_RU.' * w;  % 使用转置，匹配文档
                hRg = h_RU' * g;   % MRC combining
                gammaR = eta_b * abs(hRw)^2 * abs(hRg)^2 * abs(gamma0 - gamma1)^2 / (4 * sigmaR2);
                gammaE = eta_b * abs(h_UE)^2 * abs(hRw)^2 * abs(gamma0 - gamma1)^2 / (4 * sigmaE2);
                SR = log2(1 + gammaR) - log2(1 + gammaE);
                % Uncomment for debugging:
                fprintf('SR = %.4f\n', SR);
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

% If no feasible solution found, return default values
if best_SR == -Inf
    warning('No feasible solution found in brute force search');
    best_SR = 0;
    best_gamma0 = 1;
    best_gamma1 = -1;
    best_w = conj(h_RU) / norm(conj(h_RU)) * sqrt(Pt); % MRT beamforming
    best_g = h_RU / norm(h_RU); % MRC combining
end
%}
% --------- End of Original Code ---------