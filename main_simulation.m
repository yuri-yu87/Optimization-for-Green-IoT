% main_simulation.m
% 主仿真脚本：Monte Carlo仿真，调用bruteSR和cvxSR，统计平均SR等性能指标

clear; clc; close all;

%% 参数设置（与Table 1一致）
N_set = [3, 4, 5, 6];           % Reader天线数
Pt = 0.5;                       % 发射功率 (W)
f = 915e6;                      % 载波频率 (Hz)
c = 3e8;                        % 光速 (m/s)
lambda = c / f;                 % 波长 (m)
eta_b = 0.8;                    % Backscattering efficiency
eta_e = 0.8;                    % Energy harvesting efficiency
sigmaR2 = 10^((-80-30)/10);     % Reader噪声功率 (W)
sigmaE2 = 10^((-80-30)/10);     % Eve噪声功率 (W)
mth = 0.2;                      % 反射系数阈值
Pth = 1e-6;                     % 能量收集阈值 (W)
d_RU = 10;                      % Reader-Tag距离 (m)
d_UE_set = 5:5:50;              % Tag-Eve距离 (m)
MC_runs = 10;                 % 蒙特卡洛次数（建议10000，调试可用1000）

%% 结果存储
SR_brute = zeros(length(N_set), length(d_UE_set));
SR_cvx   = zeros(length(N_set), length(d_UE_set));
SR_pso   = zeros(length(N_set), length(d_UE_set));

%% 主循环
for nIdx = 1:length(N_set)
    N = N_set(nIdx);
    for dIdx = 1:length(d_UE_set)
        d_UE = d_UE_set(dIdx);

        % 路损
        beta_RU = (lambda/(4*pi*d_RU))^2;
        beta_UE = (lambda/(4*pi/d_UE))^2;

        SR_brute_mc = zeros(MC_runs,1);
        SR_cvx_mc   = zeros(MC_runs,1);
        SR_pso_mc   = zeros(MC_runs,1);

        for mc = 1:MC_runs
            % 生成信道
            h_RU = sqrt(beta_RU/2) * (randn(N,1) + 1i*randn(N,1)); % Reader-Tag
            h_UE = sqrt(beta_UE/2) * (randn + 1i*randn);           % Tag-Eve

            % Brute force
            [SR1, ~, ~, ~, ~] = bruteSR(h_RU, h_UE, N, Pt, mth, Pth, eta_b, eta_e, sigmaR2, sigmaE2);
            SR_brute_mc(mc) = max(0, SR1);

            % % CVX优化
            % [SR2, ~, ~, ~] = cvxSR(h_RU, h_UE, N, Pt, mth, Pth, eta_b, eta_e, sigmaR2, sigmaE2);
            % SR_cvx_mc(mc) = max(0, SR2);
            % 
            % % PSO优化
            % [SR3, ~, ~, ~] = psoSR(h_RU, h_UE, N, Pt, mth, Pth, eta_b, eta_e, sigmaR2, sigmaE2);
            % SR_pso_mc(mc) = max(0, SR3);
        end

        % 统计平均SR
        SR_brute(nIdx, dIdx) = mean(SR_brute_mc);
        % SR_cvx(nIdx, dIdx)   = mean(SR_cvx_mc);
        % SR_pso(nIdx, dIdx)   = mean(SR_pso_mc);

        fprintf('N=%d, d_UE=%.1f: BruteSR=%.3f, CVXSR=%.3f, PSOSR=%.3f\n', N, d_UE, SR_brute(nIdx,dIdx), SR_cvx(nIdx,dIdx), SR_pso(nIdx,dIdx));
    end
end

%% 绘图
figure;
for nIdx = 1:length(N_set)
    plot(d_UE_set, SR_brute(nIdx,:), '--o', 'DisplayName', sprintf('Brute N=%d', N_set(nIdx)));
    hold on;
    % plot(d_UE_set, SR_cvx(nIdx,:), '-s', 'DisplayName', sprintf('CVX N=%d', N_set(nIdx)));
    % plot(d_UE_set, SR_pso(nIdx,:), '-.^', 'DisplayName', sprintf('PSO N=%d', N_set(nIdx)));
end
xlabel('Tag-Eve Distance d_{UE} (m)');
ylabel('Average Secrecy Rate (bits/s/Hz)');
title('Secrecy Rate vs. Tag-Eve Distance');
legend('show');
grid on;
