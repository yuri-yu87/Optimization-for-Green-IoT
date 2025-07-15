% main_simulation.m
% 主仿真脚本：Monte Carlo仿真，调用bruteSR和cvxSR，统计平均SR等性能指标
% 并行化处理蒙特卡洛

clear; clc; close all;

%% 参数设置（与Table 1一致）
% N_set = [3, 4, 5, 6];           % Reader天线数
N_set = [3, 4, 5, 6];
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
MC_runs = 100;                  % 蒙特卡洛次数（建议10000，调试可用1000）

%% 结果存储
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

%% 主循环
for nIdx = 1:length(N_set)
    N = N_set(nIdx);
    for dIdx = 1:length(d_UE_set)
        d_UE = d_UE_set(dIdx);

        % 路损
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

        % 并行化蒙特卡洛仿真
        parfor mc = 1:MC_runs
            % fprintf('mc number\t\t%.2f\n', mc); % 并行时建议注释掉，避免输出混乱
            % 生成信道
            h_RU = sqrt(beta_RU/2) * (randn(N,1) + 1i*randn(N,1)); % Reader-Tag
            h_UE = sqrt(beta_UE/2) * (randn + 1i*randn);           % Tag-Eve

            % Brute force
            [SR1, g01, g11, ~, ~] = bruteSR(h_RU, h_UE, N, Pt, mth, Pth, eta_b, eta_e, sigmaR2, sigmaE2);
            SR_brute_mc(mc) = max(0, SR1);
            Gamma0_brute_mc(mc) = g01;
            Gamma1_brute_mc(mc) = g11;
            % RR for brute: use the same formula as in bruteSR
            hRg = h_RU' * (h_RU / norm(h_RU));
            hRw = h_RU' * (h_RU / norm(h_RU));
            delta_gamma = g01 - g11;
            RR_brute_mc(mc) = log2(1 + eta_b * abs(hRw)^2 * abs(hRg)^2 * abs(delta_gamma)^2 / (4 * sigmaR2));

            % CVX优化
            [SR2, g02, g12, ~, SR_curve] = cvxSR(h_RU, h_UE, N, Pt, mth, Pth, eta_b, eta_e, sigmaR2, sigmaE2);
            SR_cvx_mc(mc) = max(0, SR2);
            Gamma0_cvx_mc(mc) = g02;
            Gamma1_cvx_mc(mc) = g12;
            hRg2 = h_RU' * (h_RU / norm(h_RU));
            hRw2 = h_RU' * (h_RU / norm(h_RU));
            delta_gamma2 = g02 - g12;
            RR_cvx_mc(mc) = log2(1 + eta_b * abs(hRw2)^2 * abs(hRg2)^2 * abs(delta_gamma2)^2 / (4 * sigmaR2));
            if mc == 1
                cvxSR_convergence{nIdx, dIdx} = SR_curve;
            end
        end

        % 统计平均SR和参数
        SR_brute(nIdx, dIdx) = mean(SR_brute_mc);
        SR_cvx(nIdx, dIdx)   = mean(SR_cvx_mc);
        Gamma0_brute(nIdx, dIdx) = mean(Gamma0_brute_mc);
        Gamma1_brute(nIdx, dIdx) = mean(Gamma1_brute_mc);
        Gamma0_cvx(nIdx, dIdx) = mean(Gamma0_cvx_mc);
        Gamma1_cvx(nIdx, dIdx) = mean(Gamma1_cvx_mc);
        RR_brute(nIdx, dIdx) = mean(RR_brute_mc);
        RR_cvx(nIdx, dIdx) = mean(RR_cvx_mc);
        % SR_pso(nIdx, dIdx)   = mean(SR_pso_mc);

        fprintf('N=%d, d_UE=%.1f: BruteSR=%.3f, CVXSR=%.3f, PSOSR=%.3f\n', N, d_UE, SR_brute(nIdx,dIdx), SR_cvx(nIdx,dIdx), SR_pso(nIdx,dIdx));
    end
end

%% 绘图
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

% 反射系数Γ0, Γ1随d_UE变化
figure;
for nIdx = 1:length(N_set)
    plot(d_UE_set, Gamma0_brute(nIdx,:), '--o', 'DisplayName', sprintf('Brute \Gamma_0 N=%d', N_set(nIdx)));
    hold on;
    plot(d_UE_set, Gamma1_brute(nIdx,:), '--x', 'DisplayName', sprintf('Brute \Gamma_1 N=%d', N_set(nIdx)));
    plot(d_UE_set, Gamma0_cvx(nIdx,:), '-s', 'DisplayName', sprintf('CVX \Gamma_0 N=%d', N_set(nIdx)));
    plot(d_UE_set, Gamma1_cvx(nIdx,:), '-^', 'DisplayName', sprintf('CVX \Gamma_1 N=%d', N_set(nIdx)));
end
xlabel('Tag-Eve Distance d_{UE} (m)');
ylabel('Reflection Coefficient');
title('Reflection Coefficient vs. Tag-Eve Distance');
legend('show');
grid on;

% RR随d_UE变化
figure;
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

% CVX收敛曲线（任选一个N和d_UE）
figure;
sel_n = 1; sel_d = 1; % 可根据需要选择
plot(cell2mat(cvxSR_convergence(sel_n, sel_d)), '-o');
xlabel('AO Iteration');
ylabel('Secrecy Rate (bits/s/Hz)');
title(sprintf('CVX AO Convergence (N=%d, d_{UE}=%.1f)', N_set(sel_n), d_UE_set(sel_d)));
grid on;


% --------- Original (Non-Vectorized) Code for Reference ---------
%{
% main_simulation.m
% 主仿真脚本：Monte Carlo仿真，调用bruteSR和cvxSR，统计平均SR等性能指标

clear; clc; close all;

%% 参数设置（与Table 1一致）
N_set = [3, 4, 5, 6];           % Reader天线数
% N_set = 3;
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
MC_runs = 1000;                 % 蒙特卡洛次数（建议10000，调试可用1000）

%% 结果存储
SR_brute = zeros(length(N_set), length(d_UE_set));
SR_cvx   = zeros(length(N_set), length(d_UE_set));
SR_pso   = zeros(length(N_set), length(d_UE_set));
fprintf('start\n');

%% 主循环
for nIdx = 1:length(N_set)
    N = N_set(nIdx);
    fprintf('anttenna number\t\t%.2f\n', N);
    for dIdx = 1:length(d_UE_set)
        d_UE = d_UE_set(dIdx);

        % 路损
        beta_RU = (lambda/(4*pi*d_RU))^2;
        beta_UE = (lambda/(4*pi*d_UE))^2;

        SR_brute_mc = zeros(MC_runs,1);
        SR_cvx_mc   = zeros(MC_runs,1);
        SR_pso_mc   = zeros(MC_runs,1);

        for mc = 1:MC_runs
            fprintf('mc number\t\t%.2f\n', mc);
            % 生成信道
            % h_RU = sqrt(beta_RU/2) * (randn(1,N) + 1i*randn(1,N)); % Reader-Tag
            % h_UE = sqrt(beta_UE/2) * (randn(1,N) + 1i*randn(1,N));           % Tag-Eve
            h_RU = sqrt(beta_RU/2) * (randn(N,1) + 1i*randn(N,1)); % Reader-Tag
            h_UE = sqrt(beta_UE/2) * (randn + 1i*randn);           % Tag-Eve

            % Brute force
            [SR1, ~, ~, ~, ~] = bruteSR(h_RU, h_UE, N, Pt, mth, Pth, eta_b, eta_e, sigmaR2, sigmaE2);
            % SR1 = brute_force_SR(h_RU, h_UE, Pt, Pth, sigmaR2, sigmaE2, eta_b, eta_e, mth);
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
%}
% --------- End of Original Code ---------
