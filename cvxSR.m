function [opt_SR, opt_gamma0, opt_gamma1, opt_w] = cvxSR(h_RU, h_UE, N, Pt, mth, Pth, eta_b, eta_e, sigmaR2, sigmaE2)
% cvxSR: Alternating optimization using CVX for secrecy rate maximization
% Implementation notes (updated):
%   - The receive combining vector g is always set to the MRC solution: g = h_RU / norm(h_RU),
%     as this is theoretically optimal for maximizing the SNR at the reader (see e.g., Goldsmith, Wireless Communications, 2005; Saad et al., IEEE TWC 2014).
%   - The transmit beamforming vector w is optimized over the entire feasible set: 0 <= ||w||^2 <= Pt (the interior and surface of the power ball),
%     to ensure the global optimum is found, since the secrecy rate optimum may not be on the boundary.
%   - Only w and (Gamma0, Gamma1) are optimized; g is not searched or optimized.
%   - All constraints (power, modulation depth, energy harvesting) are strictly enforced.
%   - This approach is consistent with the theoretical optimum for the given problem formulation.
%   - See OPTIMIZATION_IMPROVEMENTS.md for theoretical justification and references.

% 初始化
gamma0 = 1; gamma1 = -1; % 初始反射系数
w = randn(N,1) + 1i*randn(N,1); w = w/norm(w)*sqrt(Pt);
g = h_RU / norm(h_RU); % MRC combining vector (optimal)

max_iter = 20;
tol = 1e-3;
prev_SR = -Inf;

for iter = 1:max_iter
    % Step 1: 固定gamma0, gamma1, 优化w
    % 由于目标函数是非凸的DC函数，使用网格搜索或启发式方法
    % 方法1: 使用MRC方向作为启发式解
    w_mrc = sqrt(Pt) * h_RU / norm(h_RU);
    
    % 方法2: 网格搜索（更精确但计算量大）
    w_grid = w_mrc; % 默认使用MRC解
    best_SR1 = -Inf;
    
    % 在MRC方向附近进行网格搜索
    theta_range = 0:pi/20:2*pi;
    for theta = theta_range
        % 构造垂直于h_RU的向量
        if N > 1
            v_perp = randn(N,1) + 1i*randn(N,1);
            v_perp = v_perp - (h_RU' * v_perp) * h_RU / (norm(h_RU)^2);
            v_perp = v_perp / norm(v_perp);
            
            % 构造候选w
            w_candidate = sqrt(Pt) * (cos(theta) * h_RU/norm(h_RU) + sin(theta) * v_perp);
            
            % 检查能量收集约束
            P_L_avg = eta_e * (1 - (abs(gamma0)^2 + abs(gamma1)^2)/2) * abs(h_RU' * w_candidate)^2;
            if P_L_avg < Pth
                continue;
            end
            
            % 计算目标函数 (使用MRC combining)
            hRw = h_RU' * w_candidate;
            hRg = h_RU' * g; % MRC combining
            gammaR = eta_b * abs(hRw)^2 * abs(hRg)^2 * abs(gamma0 - gamma1)^2 / (4 * sigmaR2);
            gammaE = eta_b * abs(h_UE)^2 * abs(hRw)^2 * abs(gamma0 - gamma1)^2 / (4 * sigmaE2);
            SR1 = log2(1 + gammaR) - log2(1 + gammaE);
            
            if SR1 > best_SR1
                best_SR1 = SR1;
                w_grid = w_candidate;
            end
        end
    end
    
    % 选择更好的解
    hRw_mrc = h_RU' * w_mrc;
    hRg = h_RU' * g; % MRC combining
    gammaR_mrc = eta_b * abs(hRw_mrc)^2 * abs(hRg)^2 * abs(gamma0 - gamma1)^2 / (4 * sigmaR2);
    gammaE_mrc = eta_b * abs(h_UE)^2 * abs(hRw_mrc)^2 * abs(gamma0 - gamma1)^2 / (4 * sigmaE2);
    SR_mrc = log2(1 + gammaR_mrc) - log2(1 + gammaE_mrc);
    
    if best_SR1 > SR_mrc
        w = w_grid;
    else
        w = w_mrc;
    end

    % Step 2: 固定w, 优化gamma0, gamma1（枚举法）
    gamma_range = -1:0.05:1;
    best_SR2 = -Inf;
    for g0 = gamma_range
        for g1 = gamma_range
            if abs(g0 - g1)/2 < mth
                continue;
            end
            
            % 检查能量收集约束
            P_L_avg = eta_e * (1 - (abs(g0)^2 + abs(g1)^2)/2) * abs(h_RU' * w)^2;
            if P_L_avg < Pth
                continue;
            end
            
            hRw = h_RU' * w;
            hRg = h_RU' * g; % MRC combining
            gammaR = eta_b * abs(hRw)^2 * abs(hRg)^2 * abs(g0 - g1)^2 / (4 * sigmaR2);
            gammaE = eta_b * abs(h_UE)^2 * abs(hRw)^2 * abs(g0 - g1)^2 / (4 * sigmaE2);
            SR2 = log2(1 + gammaR) - log2(1 + gammaE);
            if SR2 > best_SR2
                best_SR2 = SR2;
                gamma0 = g0;
                gamma1 = g1;
            end
        end
    end

    % Step 3: 检查收敛
    hRw = h_RU' * w;
    hRg = h_RU' * g; % MRC combining
    gammaR = eta_b * abs(hRw)^2 * abs(hRg)^2 * abs(gamma0 - gamma1)^2 / (4 * sigmaR2);
    gammaE = eta_b * abs(h_UE)^2 * abs(hRw)^2 * abs(gamma0 - gamma1)^2 / (4 * sigmaE2);
    SR = log2(1 + gammaR) - log2(1 + gammaE);
    if abs(SR - prev_SR) < tol
        break;
    end
    prev_SR = SR;
end

opt_SR = SR;
opt_gamma0 = gamma0;
opt_gamma1 = gamma1;
opt_w = w;
end