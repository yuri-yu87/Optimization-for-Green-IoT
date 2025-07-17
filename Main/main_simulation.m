% main_simulation.m
% Main simulation script: Monte Carlo simulation, calls bruteSR and cvxSR, statistics of average SR and other performance metrics
% Parallel processing of Monte Carlo

clear; clc; close all;
%% Parameter settings
N_set = [3, 4, 5, 6];
Pt = 0.5;                       % Transmit power (W)
f = 915e6;                      % Carrier frequency (Hz)
c = 3e8;                        % Speed of light (m/s)
lambda = c / f;                 % Wavelength (m)
eta_b = 0.8;                    % Backscattering efficiency
eta_e = 0.8;                    % Energy harvesting efficiency
sigmaR2 = 10^((-80-30)/10);     % Reader noise power (W)
sigmaE2 = 10^((-80-30)/10);     % Eve noise power (W)
mth = 0.2;                      % Reflection coefficient threshold
Pth = 1e-6;                     % Energy harvesting threshold (W)
d_RU = 10;                      % Reader-Tag distance (m)
d_UE_set = 5:5:50;              % Tag-Eve distance (m)
MC_runs = 100;                  % Monte Carlo runs (suggest 10000, 100 for debugging)

%% Result storage
SR_brute = zeros(length(N_set), length(d_UE_set));
SR_cvx   = zeros(length(N_set), length(d_UE_set));
SR_pso   = zeros(length(N_set), length(d_UE_set));
Gamma0_brute = zeros(length(N_set), length(d_UE_set));
Gamma1_brute = zeros(length(N_set), length(d_UE_set));
Gamma0_cvx = zeros(length(N_set), length(d_UE_set));
Gamma1_cvx = zeros(length(N_set), length(d_UE_set));
RR_brute = zeros(length(N_set), length(d_UE_set));
RR_cvx = zeros(length(N_set), length(d_UE_set));
cvxSR_convergence = cell(length(N_set), length(d_UE_set));
fprintf('start\n');

%% Main loop
for nIdx = 1:length(N_set)
    N = N_set(nIdx);
    for dIdx = 1:length(d_UE_set)
        d_UE = d_UE_set(dIdx);

        % Path loss
        beta_RU = (lambda/(4*pi*d_RU))^2;
        beta_UE = (lambda/(4*pi*d_UE))^2;

        SR_brute_mc = zeros(MC_runs,1);
        SR_cvx_mc   = zeros(MC_runs,1);
        SR_pso_mc   = zeros(MC_runs,1);
        Gamma0_brute_mc = zeros(MC_runs,1);
        Gamma1_brute_mc = zeros(MC_runs,1);
        RR_brute_mc = zeros(MC_runs,1);
        Gamma0_cvx_mc = zeros(MC_runs,1);
        Gamma1_cvx_mc = zeros(MC_runs,1);
        RR_cvx_mc = zeros(MC_runs,1);

        % Since parfor cannot write to external cell, define a cell array outside parfor first
        SR_curve_all = cell(MC_runs, 1);

        % Parallelized Monte Carlo simulation
        parfor mc = 1:MC_runs
            % Generate channel
            h_RU = sqrt(beta_RU/2) * (randn(N,1) + 1i*randn(N,1)); % Reader-Tag
            h_UE = sqrt(beta_UE/2) * (randn + 1i*randn);           % Tag-Eve

            % Brute force
            [SR1, g01, g11, w1, ~] = bruteSR(h_RU, h_UE, N, Pt, mth, Pth, eta_b, eta_e, sigmaR2, sigmaE2);
            SR_brute_mc(mc) = max(0, SR1);
            Gamma0_brute_mc(mc) = g01;
            Gamma1_brute_mc(mc) = g11;
            % RR for brute: use the same formula as in bruteSR
            hRg = h_RU' * (h_RU / norm(h_RU));
            hRw = h_RU.' * w1;
            delta_gamma = g01 - g11;
            RR_brute_mc(mc) = log2(1 + eta_b * abs(hRw)^2 * abs(hRg)^2 * abs(delta_gamma)^2 / (4 * sigmaR2));

            % CVX optimization
            [SR2, g02, g12, w2, SR_curve] = cvxSR(h_RU, h_UE, N, Pt, mth, Pth, eta_b, eta_e, sigmaR2, sigmaE2);
            SR_cvx_mc(mc) = max(0, SR2);
            Gamma0_cvx_mc(mc) = g02;
            Gamma1_cvx_mc(mc) = g12;
            hRg2 = h_RU' * (h_RU / norm(h_RU));
            hRw2 = h_RU.' * w2;
            delta_gamma2 = g02 - g12;
            RR_cvx_mc(mc) = log2(1 + eta_b * abs(hRw2)^2 * abs(hRg2)^2 * abs(delta_gamma2)^2 / (4 * sigmaR2));
            SR_curve_all{mc} = SR_curve; % Store each mc
        end

        % Outside parfor, only take the first Monte Carlo sample
        cvxSR_convergence{nIdx, dIdx} = SR_curve_all{1};

        % Statistics of average SR and parameters
        SR_brute(nIdx, dIdx) = mean(SR_brute_mc);
        SR_cvx(nIdx, dIdx)   = mean(SR_cvx_mc);
        Gamma0_brute(nIdx, dIdx) = mean(Gamma0_brute_mc);
        Gamma1_brute(nIdx, dIdx) = mean(Gamma1_brute_mc);
        Gamma0_cvx(nIdx, dIdx) = mean(Gamma0_cvx_mc);
        Gamma1_cvx(nIdx, dIdx) = mean(Gamma1_cvx_mc);
        RR_brute(nIdx, dIdx) = mean(RR_brute_mc);
        RR_cvx(nIdx, dIdx) = mean(RR_cvx_mc);

        fprintf('N=%d, d_UE=%.1f: BruteSR=%.3f, CVXSR=%.3f\n', N, d_UE, SR_brute(nIdx,dIdx), SR_cvx(nIdx,dIdx));
    end
end

%% Plotting
figure;
for nIdx = 1:length(N_set)
    plot(d_UE_set, SR_brute(nIdx,:), '--o', 'DisplayName', sprintf('Brute N=%d', N_set(nIdx)));
    hold on;
    plot(d_UE_set, SR_cvx(nIdx,:), '-s', 'DisplayName', sprintf('CVX N=%d', N_set(nIdx)));
    % plot(d_UE_set, SR_pso(nIdx,:), '-.^', 'DisplayName', sprintf('PSO N=%d', N_set(nIdx)));
end
xlabel('Tag-Eve Distance d_{UE} (m)');
ylabel('Average Secrecy Rate (bits/s/Hz)');
title('Secrecy Rate vs. Tag-Eve Distance');
legend('show');
grid on;

% Reflection coefficients Γ0, Γ1 vs. d_UE
figure;
colors = {'b', 'r', 'g', 'm'}; % Different colors
for nIdx = 1:length(N_set)
    % Brute Force Γ0
    plot(d_UE_set, Gamma0_brute(nIdx,:), '--o', 'Color', colors{nIdx}, ...
         'DisplayName', sprintf('Brute \\Gamma_0 N=%d', N_set(nIdx)), ...
         'LineWidth', 1.5, 'MarkerSize', 6);
    hold on;
    % Brute Force Γ1
    plot(d_UE_set, Gamma1_brute(nIdx,:), '--x', 'Color', colors{nIdx}, ...
         'DisplayName', sprintf('Brute \\Gamma_1 N=%d', N_set(nIdx)), ...
         'LineWidth', 1.5, 'MarkerSize', 6);
    % CVX Γ0
    plot(d_UE_set, Gamma0_cvx(nIdx,:), '-s', 'Color', colors{nIdx}, ...
         'DisplayName', sprintf('CVX \\Gamma_0 N=%d', N_set(nIdx)), ...
         'LineWidth', 1.5, 'MarkerSize', 6);
    % CVX Γ1
    plot(d_UE_set, Gamma1_cvx(nIdx,:), '-^', 'Color', colors{nIdx}, ...
         'DisplayName', sprintf('CVX \\Gamma_1 N=%d', N_set(nIdx)), ...
         'LineWidth', 1.5, 'MarkerSize', 6);
end
xlabel('Tag-Eve Distance d_{UE} (m)');
ylabel('Reflection Coefficient');
title('Reflection Coefficient vs. Tag-Eve Distance');
legend('show', 'Location', 'best');
grid on;

% RR vs. d_UE
figure;
% colors = {'b', 'r', 'g', 'm'}; % Different colors
for nIdx = 1:length(N_set)
    plot(d_UE_set, RR_brute(nIdx,:), '--o', 'DisplayName', sprintf('Brute RR N=%d', N_set(nIdx)));
    hold on;
    plot(d_UE_set, RR_cvx(nIdx,:), '-s', 'DisplayName', sprintf('CVX RR N=%d', N_set(nIdx)));
end
xlabel('Tag-Eve Distance d_{UE} (m)');
ylabel('SE at Reader R_R (bits/s/Hz)');
title('SE at Reader vs. Tag-Eve Distance');
legend('show');
grid on;

% CVX convergence curve (all N values, fixed d_UE)
figure;
sel_d = 7; % Fixed 1：d_UE=5m；2：d_UE=10m；3：d_UE=15m；4：d_UE=20m；5：d_UE=25m；6：d_UE=30m；7：d_UE=35m；8：d_UE=40m；9：d_UE=45m；10：d_UE=50m
colors = {'b', 'r', 'g', 'm'}; % Different colors
markers = {'o', 's', '^', 'd'}; % Different markers
for nIdx = 1:length(N_set)
    convergence_data = cell2mat(cvxSR_convergence(nIdx, sel_d));
    plot(1:length(convergence_data), convergence_data, '-', 'Color', colors{nIdx}, ...
         'Marker', markers{nIdx}, 'DisplayName', sprintf('N=%d', N_set(nIdx)), ...
         'LineWidth', 1.5, 'MarkerSize', 6);
    hold on;
end
xlabel('AO Iteration');
ylabel('Secrecy Rate (bits/s/Hz)');
title(sprintf('CVX AO Convergence (d_{UE}=%.1fm)', d_UE_set(sel_d)));
legend('show', 'Location', 'best');
grid on;
