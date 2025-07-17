% post_analysis.m
% Please make sure you have already run main_simulation.m and saved the data before running this script

% clear; clc; close all;

%% Step 1: Save simulation data (if not already saved)
% After running the main simulation, call the save function
% save_simulation_data('');

%% Step 2: Load data
% Load previously saved simulation data
load_and_analyze_data('tol_e-6_simu_results');

%% Step 3: Perform secondary calculations and analysis

% 1. Complete the four required plots
fprintf('\n=== Four Required Plots for the Task ===\n');

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
sel_d = 7; % Fixed 1: d_UE=5m; 2: d_UE=10m; 3: d_UE=15m; 4: d_UE=20m; 5: d_UE=25m; 6: d_UE=30m; 7: d_UE=35m; 8: d_UE=40m; 9: d_UE=45m; 10: d_UE=50m
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

% 2. Calculate performance improvement percentage
fprintf('\n=== Performance Improvement Analysis ===\n');
% eps:2.2204e-16 Avoiding division by zero errors
performance_improvement = (SR_cvx - SR_brute) ./ (SR_brute + eps) * 100; 

figure;
for nIdx = 1:length(N_set)
    plot(d_UE_set, performance_improvement(nIdx,:), '-o', 'DisplayName', sprintf('N=%d', N_set(nIdx)));
    hold on;
end
xlabel('Tag-Eve Distance d_{UE} (m)');
ylabel('Performance Improvement (%)');
title('CVX vs Brute Force Performance Improvement');
legend('show');
grid on;

% 3. Analyze best configuration
fprintf('\n=== Best Configuration Analysis ===\n');
[max_sr_brute, idx_brute] = max(SR_brute(:));
[max_sr_cvx, idx_cvx] = max(SR_cvx(:));
[n_idx_brute, d_idx_brute] = ind2sub(size(SR_brute), idx_brute);
[n_idx_cvx, d_idx_cvx] = ind2sub(size(SR_cvx), idx_cvx);

fprintf('Brute Force Best Configuration:\n');
fprintf('  - Number of antennas: N = %d\n', N_set(n_idx_brute));
fprintf('  - Tag-Eve distance: d_UE = %.1f m\n', d_UE_set(d_idx_brute));
fprintf('  - Maximum secrecy rate: %.4f bits/s/Hz\n', max_sr_brute);
fprintf('  - Corresponding reflection coefficients: Γ0 = %.4f, Γ1 = %.4f\n', ...
    Gamma0_brute(n_idx_brute, d_idx_brute), Gamma1_brute(n_idx_brute, d_idx_brute));

fprintf('\nCVX Best Configuration:\n');
fprintf('  - Number of antennas: N = %d\n', N_set(n_idx_cvx));
fprintf('  - Tag-Eve distance: d_UE = %.1f m\n', d_UE_set(d_idx_cvx));
fprintf('  - Maximum secrecy rate: %.4f bits/s/Hz\n', max_sr_cvx);
fprintf('  - Corresponding reflection coefficients: Γ0 = %.4f, Γ1 = %.4f\n', ...
    Gamma0_cvx(n_idx_cvx, d_idx_cvx), Gamma1_cvx(n_idx_cvx, d_idx_cvx));

% 4. 3D visualization
figure;
[X, Y] = meshgrid(d_UE_set, N_set);
surf(X, Y, SR_brute, 'FaceAlpha', 0.7, 'DisplayName', 'Brute Force');
hold on;
surf(X, Y, SR_cvx, 'FaceAlpha', 0.7, 'DisplayName', 'CVX');
xlabel('Tag-Eve Distance d_{UE} (m)');
ylabel('Number of Antennas N');
zlabel('Secrecy Rate (bits/s/Hz)');
title('3D View: Secrecy Rate vs Distance and Antennas');
legend('show');
colorbar;

% 5. Statistical analysis
fprintf('\n=== Statistical Analysis ===\n');
fprintf('Brute Force SR Statistics:\n');
fprintf('  - Mean: %.4f bits/s/Hz\n', mean(SR_brute(:)));
fprintf('  - Standard deviation: %.4f bits/s/Hz\n', std(SR_brute(:)));
fprintf('  - Minimum: %.4f bits/s/Hz\n', min(SR_brute(:)));
fprintf('  - Maximum: %.4f bits/s/Hz\n', max(SR_brute(:)));

fprintf('\nCVX SR Statistics:\n');
fprintf('  - Mean: %.4f bits/s/Hz\n', mean(SR_cvx(:)));
fprintf('  - Standard deviation: %.4f bits/s/Hz\n', std(SR_cvx(:)));
fprintf('  - Minimum: %.4f bits/s/Hz\n', min(SR_cvx(:)));
fprintf('  - Maximum: %.4f bits/s/Hz\n', max(SR_cvx(:)));

% 6. Distance impact analysis
fprintf('\n=== Distance Impact Analysis ===\n');
for n_idx = 1:length(N_set)
    % Calculate the impact of distance on SR (slope)
    p_brute = polyfit(d_UE_set, SR_brute(n_idx,:), 1);
    p_cvx = polyfit(d_UE_set, SR_cvx(n_idx,:), 1);
    
    fprintf('N=%d: SR change rate with distance\n', N_set(n_idx));
    fprintf('  - Brute Force: %.6f bits/s/Hz/m\n', p_brute(1));
    fprintf('  - CVX: %.6f bits/s/Hz/m\n', p_cvx(1));
end

% 7. Save analysis results
analysis_results = struct();
analysis_results.performance_improvement = performance_improvement;
analysis_results.best_config_brute = struct('N', N_set(n_idx_brute), 'd_UE', d_UE_set(d_idx_brute), 'SR', max_sr_brute);
analysis_results.best_config_cvx = struct('N', N_set(n_idx_cvx), 'd_UE', d_UE_set(d_idx_cvx), 'SR', max_sr_cvx);
analysis_results.statistics = struct('brute_mean', mean(SR_brute(:)), 'cvx_mean', mean(SR_cvx(:)), ...
    'brute_std', std(SR_brute(:)), 'cvx_std', std(SR_cvx(:)));

save('post_analysis_results.mat', 'analysis_results');
fprintf('\nAnalysis results have been saved to post_analysis_results.mat\n');