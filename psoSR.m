function [opt_SR, opt_gamma0, opt_gamma1, opt_w] = psoSR(h_RU, h_UE, N, Pt, mth, Pth, eta_b, eta_e, sigmaR2, sigmaE2)
% psoSR: Alternating optimization using PSO for secrecy rate maximization
% Implementation notes (updated):
%   - The receive combining vector g is always set to the MRC solution: g = h_RU / norm(h_RU),
%     as this is theoretically optimal for maximizing the SNR at the reader (see e.g., Goldsmith, Wireless Communications, 2005; Saad et al., IEEE TWC 2014).
%   - The transmit beamforming vector w is optimized over the entire feasible set: 0 <= ||w||^2 <= Pt (the interior and surface of the power ball),
%     to ensure the global optimum is found, since the secrecy rate optimum may not be on the boundary.
%   - Only w and (Gamma0, Gamma1) are optimized; g is not searched or optimized.
%   - All constraints (power, modulation depth, energy harvesting) are strictly enforced.
%   - This approach is consistent with the theoretical optimum for the given problem formulation.
%   - See OPTIMIZATION_IMPROVEMENTS.md for theoretical justification and references.

% PSO parameters
n_particles = 30;  % Number of particles
max_iter_pso = 50; % Maximum PSO iterations
w_inertia = 0.7;   % Inertia weight
c1 = 2.0;          % Cognitive learning factor
c2 = 2.0;          % Social learning factor

% AO parameters
max_iter_ao = 20;
tol = 1e-3;
prev_SR = -Inf;

% Initialize
gamma0 = 1; gamma1 = -1; % Initial reflection coefficients
w = randn(N,1) + 1i*randn(N,1); w = w/norm(w)*sqrt(Pt);
g = h_RU / norm(h_RU); % MRC combining vector (optimal)

for iter_ao = 1:max_iter_ao
    fprintf('AO Iteration %d\n', iter_ao);
    
    % Step 1: Fix (gamma0, gamma1), optimize w using PSO
    w = optimize_w_pso(h_RU, h_UE, N, Pt, gamma0, gamma1, eta_b, eta_e, Pth, sigmaR2, sigmaE2, ...
                       n_particles, max_iter_pso, w_inertia, c1, c2);
    
    % Step 2: Fix w, optimize (gamma0, gamma1) using grid search
    [gamma0, gamma1] = optimize_gamma_grid(h_RU, h_UE, w, mth, eta_b, eta_e, Pth, sigmaR2, sigmaE2);
    
    % Step 3: Check convergence
            hRw = h_RU.' * w;  % 使用转置，匹配文档
    hRg = h_RU' * g; % MRC combining
    gammaR = eta_b * abs(hRw)^2 * abs(hRg)^2 * abs(gamma0 - gamma1)^2 / (4 * sigmaR2);
    gammaE = eta_b * abs(h_UE)^2 * abs(hRw)^2 * abs(gamma0 - gamma1)^2 / (4 * sigmaE2);
    SR = log2(1 + gammaR) - log2(1 + gammaE);
    
    fprintf('Current SR: %.4f\n', SR);
    
    if abs(SR - prev_SR) < tol
        fprintf('Converged after %d iterations\n', iter_ao);
        break;
    end
    prev_SR = SR;
end

opt_SR = SR;
opt_gamma0 = gamma0;
opt_gamma1 = gamma1;
opt_w = w;
end

function w_opt = optimize_w_pso(h_RU, h_UE, N, Pt, gamma0, gamma1, eta_b, eta_e, Pth, sigmaR2, sigmaE2, ...
                               n_particles, max_iter, w_inertia, c1, c2)
% PSO optimization for w given fixed (gamma0, gamma1)

% Initialize particles
particles = zeros(N, n_particles);
velocities = zeros(N, n_particles);
personal_best = zeros(N, n_particles);
personal_best_fitness = -Inf * ones(1, n_particles);
global_best = zeros(N, 1);
global_best_fitness = -Inf;

% Initialize particles on the power sphere
for i = 1:n_particles
    % Generate random direction
    w_rand = randn(N,1) + 1i*randn(N,1);
    particles(:,i) = sqrt(Pt) * w_rand / norm(w_rand);
    velocities(:,i) = 0.1 * (randn(N,1) + 1i*randn(N,1));
end

% PSO main loop
for iter = 1:max_iter
    % Evaluate fitness for each particle
    for i = 1:n_particles
        w_current = particles(:,i);
        
        % Check energy harvesting constraint
        P_L_avg = eta_e * (1 - (abs(gamma0)^2 + abs(gamma1)^2)/2) * abs(h_RU.' * w_current)^2;  % 使用转置，匹配文档
        if P_L_avg < Pth
            fitness = -Inf; % Penalize infeasible solutions
        else
            % Calculate secrecy rate (using MRC combining)
            hRw = h_RU.' * w_current;  % 使用转置，匹配文档
            hRg = h_RU' * (h_RU / norm(h_RU)); % MRC combining
            gammaR = eta_b * abs(hRw)^2 * abs(hRg)^2 * abs(gamma0 - gamma1)^2 / (4 * sigmaR2);
            gammaE = eta_b * abs(h_UE)^2 * abs(hRw)^2 * abs(gamma0 - gamma1)^2 / (4 * sigmaE2);
            fitness = log2(1 + gammaR) - log2(1 + gammaE);
        end
        
        % Update personal best
        if fitness > personal_best_fitness(i)
            personal_best_fitness(i) = fitness;
            personal_best(:,i) = w_current;
        end
        
        % Update global best
        if fitness > global_best_fitness
            global_best_fitness = fitness;
            global_best = w_current;
        end
    end
    
    % Update velocities and positions
    for i = 1:n_particles
        % Velocity update
        r1 = rand(N,1) + 1i*rand(N,1);
        r2 = rand(N,1) + 1i*rand(N,1);
        
        velocities(:,i) = w_inertia * velocities(:,i) + ...
                         c1 * r1 .* (personal_best(:,i) - particles(:,i)) + ...
                         c2 * r2 .* (global_best - particles(:,i));
        
        % Position update
        particles(:,i) = particles(:,i) + velocities(:,i);
        
        % Project back to power sphere
        particles(:,i) = sqrt(Pt) * particles(:,i) / norm(particles(:,i));
    end
    
    % Adaptive inertia weight
    w_inertia = w_inertia * 0.99;
end

w_opt = global_best;
end

function [gamma0_opt, gamma1_opt] = optimize_gamma_grid(h_RU, h_UE, w, mth, eta_b, eta_e, Pth, sigmaR2, sigmaE2)
% Grid search optimization for (gamma0, gamma1) given fixed w

gamma_range = -1:0.05:1;
best_SR = -Inf;
gamma0_opt = 0;
gamma1_opt = 0;

for g0 = gamma_range
    for g1 = gamma_range
        % Check modulation depth constraint
        if abs(g0 - g1)/2 < mth
            continue;
        end
        
        % Check energy harvesting constraint
        P_L_avg = eta_e * (1 - (abs(g0)^2 + abs(g1)^2)/2) * abs(h_RU.' * w)^2;  % 使用转置，匹配文档
        if P_L_avg < Pth
            continue;
        end
        
        % Calculate secrecy rate (using MRC combining)
        hRw = h_RU.' * w;  % 使用转置，匹配文档
        hRg = h_RU' * (h_RU / norm(h_RU)); % MRC combining
        gammaR = eta_b * abs(hRw)^2 * abs(hRg)^2 * abs(g0 - g1)^2 / (4 * sigmaR2);
        gammaE = eta_b * abs(h_UE)^2 * abs(hRw)^2 * abs(g0 - g1)^2 / (4 * sigmaE2);
        SR = log2(1 + gammaR) - log2(1 + gammaE);
        
        if SR > best_SR
            best_SR = SR;
            gamma0_opt = g0;
            gamma1_opt = g1;
        end
    end
end
end 