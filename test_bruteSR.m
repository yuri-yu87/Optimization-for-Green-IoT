% test_bruteSR.m
% Test script for the updated optimization functions

clear; clc;

%% Test parameters
N = 2;                          % Number of antennas
Pt = 0.5;                       % Transmit power (W)
f = 915e6;                      % Carrier frequency (Hz)
c = 3e8;                        % Speed of light (m/s)
lambda = c / f;                 % Wavelength (m)
eta_b = 0.8;                    % Backscattering efficiency
eta_e = 0.8;                    % Energy harvesting efficiency
sigmaR2 = 10^((-80-30)/10);     % Reader noise power (W)
sigmaE2 = 10^((-80-30)/10);     % Eavesdropper noise power (W)
mth = 0.2;                      % Modulation depth threshold
Pth = 1e-6;                     % Energy harvesting threshold (W)
d_RU = 10;                      % Reader-Tag distance (m)
d_UE = 15;                      % Tag-Eavesdropper distance (m)

%% Channel generation
beta_RU = (lambda/(4*pi*d_RU))^2;
beta_UE = (lambda/(4*pi/d_UE))^2;

h_RU = sqrt(beta_RU/2) * (randn(N,1) + 1i*randn(N,1)); % Reader-Tag
h_UE = sqrt(beta_UE/2) * (randn + 1i*randn);           % Tag-Eve

fprintf('Test Parameters:\n');
fprintf('N = %d, Pt = %.1f W, d_RU = %.1f m, d_UE = %.1f m\n', N, Pt, d_RU, d_UE);
fprintf('h_RU = [%.3f+%.3fi, %.3f+%.3fi]\n', real(h_RU(1)), imag(h_RU(1)), real(h_RU(2)), imag(h_RU(2)));
fprintf('h_UE = %.3f+%.3fi\n', real(h_UE), imag(h_UE));

%% Test brute force search
fprintf('\n=== Testing Brute Force Search ===\n');
tic;
[best_SR_brute, best_gamma0_brute, best_gamma1_brute, best_w_brute, best_g_brute] = bruteSR(h_RU, h_UE, N, Pt, mth, Pth, eta_b, eta_e, sigmaR2, sigmaE2);
t_brute = toc;

%% Test CVX optimization
fprintf('\n=== Testing CVX Optimization ===\n');
tic;
[best_SR_cvx, best_gamma0_cvx, best_gamma1_cvx, best_w_cvx] = cvxSR(h_RU, h_UE, N, Pt, mth, Pth, eta_b, eta_e, sigmaR2, sigmaE2);
t_cvx = toc;

%% Test PSO optimization
fprintf('\n=== Testing PSO Optimization ===\n');
tic;
[best_SR_pso, best_gamma0_pso, best_gamma1_pso, best_w_pso] = psoSR(h_RU, h_UE, N, Pt, mth, Pth, eta_b, eta_e, sigmaR2, sigmaE2);
t_pso = toc;

%% Display results
fprintf('\n=== Optimization Results Comparison ===\n');
fprintf('Method\t\tSecrecy Rate\tΓ₀\t\tΓ₁\t\tTime(s)\n');
fprintf('Brute\t\t%.4f\t\t%.3f\t\t%.3f\t\t%.3f\n', best_SR_brute, best_gamma0_brute, best_gamma1_brute, t_brute);
fprintf('CVX\t\t%.4f\t\t%.3f\t\t%.3f\t\t%.3f\n', best_SR_cvx, best_gamma0_cvx, best_gamma1_cvx, t_cvx);
fprintf('PSO\t\t%.4f\t\t%.3f\t\t%.3f\t\t%.3f\n', best_SR_pso, best_gamma0_pso, best_gamma1_pso, t_pso);

%% Verify MRC combining for CVX and PSO
g_mrc = h_RU / norm(h_RU);
fprintf('\n=== MRC Combining Verification ===\n');
fprintf('MRC combining vector g = [%.3f+%.3fi, %.3f+%.3fi]\n', real(g_mrc(1)), imag(g_mrc(1)), real(g_mrc(2)), imag(g_mrc(2)));

% Verify SNR calculations
hRg_mrc = h_RU' * g_mrc;
fprintf('h_RU^H * g = %.3f+%.3fi\n', real(hRg_mrc), imag(hRg_mrc));

% Verify secrecy rate calculations
hRw_brute = h_RU' * best_w_brute;
hRg_brute = h_RU' * best_g_brute;
gammaR_brute = eta_b * abs(hRw_brute)^2 * abs(hRg_brute)^2 * abs(best_gamma0_brute - best_gamma1_brute)^2 / (4 * sigmaR2);
gammaE_brute = eta_b * abs(h_UE)^2 * abs(hRw_brute)^2 * abs(best_gamma0_brute - best_gamma1_brute)^2 / (4 * sigmaE2);
SR_verify_brute = log2(1 + gammaR_brute) - log2(1 + gammaE_brute);

hRw_cvx = h_RU' * best_w_cvx;
hRg_cvx = h_RU' * g_mrc;
gammaR_cvx = eta_b * abs(hRw_cvx)^2 * abs(hRg_cvx)^2 * abs(best_gamma0_cvx - best_gamma1_cvx)^2 / (4 * sigmaR2);
gammaE_cvx = eta_b * abs(h_UE)^2 * abs(hRw_cvx)^2 * abs(best_gamma0_cvx - best_gamma1_cvx)^2 / (4 * sigmaE2);
SR_verify_cvx = log2(1 + gammaR_cvx) - log2(1 + gammaE_cvx);

hRw_pso = h_RU' * best_w_pso;
hRg_pso = h_RU' * g_mrc;
gammaR_pso = eta_b * abs(hRw_pso)^2 * abs(hRg_pso)^2 * abs(best_gamma0_pso - best_gamma1_pso)^2 / (4 * sigmaR2);
gammaE_pso = eta_b * abs(h_UE)^2 * abs(hRw_pso)^2 * abs(best_gamma0_pso - best_gamma1_pso)^2 / (4 * sigmaE2);
SR_verify_pso = log2(1 + gammaR_pso) - log2(1 + gammaE_pso);

fprintf('\n=== SNR and SR Verification ===\n');
fprintf('Method\t\tγR\t\tγE\t\tSR (calc)\tSR (returned)\n');
fprintf('Brute\t\t%.2f\t\t%.2f\t\t%.4f\t\t%.4f\n', gammaR_brute, gammaE_brute, SR_verify_brute, best_SR_brute);
fprintf('CVX\t\t%.2f\t\t%.2f\t\t%.4f\t\t%.4f\n', gammaR_cvx, gammaE_cvx, SR_verify_cvx, best_SR_cvx);
fprintf('PSO\t\t%.2f\t\t%.2f\t\t%.4f\t\t%.4f\n', gammaR_pso, gammaE_pso, SR_verify_pso, best_SR_pso);

%% Performance analysis
fprintf('\n=== Performance Analysis ===\n');
fprintf('Speedup CVX vs Brute: %.2fx\n', t_brute/t_cvx);
fprintf('Speedup PSO vs Brute: %.2fx\n', t_brute/t_pso);
fprintf('SR difference (CVX - Brute): %.4f\n', best_SR_cvx - best_SR_brute);
fprintf('SR difference (PSO - Brute): %.4f\n', best_SR_pso - best_SR_brute);

fprintf('\nTest completed successfully!\n'); 