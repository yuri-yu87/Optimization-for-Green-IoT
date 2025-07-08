%% 非凸优化MATLAB实现示例
% 该文件展示如何在MATLAB中实现各种非凸优化算法

%% 1. 梯度下降法实现
function [x_opt, f_opt, history] = gradient_descent(obj_func, grad_func, x0, opts)
    % 梯度下降法求解无约束优化问题
    % 输入:
    %   obj_func - 目标函数句柄
    %   grad_func - 梯度函数句柄
    %   x0 - 初始点
    %   opts - 选项结构体
    % 输出:
    %   x_opt - 最优解
    %   f_opt - 最优值
    %   history - 优化历史
    
    if nargin < 4
        opts = struct();
    end
    
    % 默认参数
    lr = getfield(opts, 'lr', 0.01);
    max_iter = getfield(opts, 'max_iter', 1000);
    tol = getfield(opts, 'tol', 1e-6);
    
    % 初始化
    x = x0;
    history.x = x0;
    history.f = obj_func(x0);
    
    % 主循环
    for k = 1:max_iter
        % 计算梯度
        grad = grad_func(x);
        
        % 更新
        x_new = x - lr * grad;
        
        % 记录历史
        history.x = [history.x, x_new];
        history.f = [history.f, obj_func(x_new)];
        
        % 检查收敛
        if norm(x_new - x) < tol
            break;
        end
        
        x = x_new;
    end
    
    x_opt = x;
    f_opt = obj_func(x_opt);
end

%% 2. DC规划算法
function [x_opt, f_opt] = dc_programming(g_func, h_func, grad_h_func, x0, opts)
    % DC规划算法: 最小化 f(x) = g(x) - h(x)
    % 其中g和h都是凸函数
    
    if nargin < 5
        opts = struct();
    end
    
    max_iter = getfield(opts, 'max_iter', 100);
    tol = getfield(opts, 'tol', 1e-6);
    
    x = x0;
    
    for k = 1:max_iter
        % 计算h在x_k处的梯度
        grad_h = grad_h_func(x);
        
        % 求解线性化子问题
        % min g(y) - <grad_h, y>
        cvx_begin quiet
            variable y(length(x))
            minimize(g_func(y) - grad_h' * y)
        cvx_end
        
        x_new = y;
        
        % 检查收敛
        if norm(x_new - x) < tol
            break;
        end
        
        x = x_new;
    end
    
    x_opt = x;
    f_opt = g_func(x_opt) - h_func(x_opt);
end

%% 3. 交替优化算法
function [x_opt, y_opt, f_opt] = alternating_optimization(obj_func, x0, y0, opts)
    % 交替优化算法
    % 最小化 f(x, y)
    
    if nargin < 4
        opts = struct();
    end
    
    max_iter = getfield(opts, 'max_iter', 50);
    tol = getfield(opts, 'tol', 1e-6);
    
    x = x0;
    y = y0;
    
    for k = 1:max_iter
        % 固定y，优化x
        obj_x = @(x_var) obj_func(x_var, y);
        x_new = fminunc(obj_x, x, optimoptions('fminunc', 'Display', 'off'));
        
        % 固定x，优化y
        obj_y = @(y_var) obj_func(x_new, y_var);
        y_new = fminunc(obj_y, y, optimoptions('fminunc', 'Display', 'off'));
        
        % 检查收敛
        if norm(x_new - x) + norm(y_new - y) < tol
            break;
        end
        
        x = x_new;
        y = y_new;
    end
    
    x_opt = x;
    y_opt = y;
    f_opt = obj_func(x_opt, y_opt);
end

%% 4. 半定规划松弛示例
function [W_opt, w_opt] = sdp_relaxation_beamforming(H, sigma2, opts)
    % 波束成形的SDP松弛
    % 最大化 |w^H * h|^2 / (||w||^2 + sigma2)
    % 约束: ||w||^2 <= 1
    
    [N, ~] = size(H);
    
    % SDP松弛: W = w * w^H
    cvx_begin sdp quiet
        variable W(N, N) hermitian
        
        % 目标函数（使用epigraph形式）
        maximize(real(trace(H * H' * W)))
        
        subject to
            trace(W) <= 1;  % 功率约束
            W >= 0;         % 半正定约束
    cvx_end
    
    W_opt = W;
    
    % 秩1恢复
    [U, S, ~] = svd(W_opt);
    w_opt = sqrt(S(1,1)) * U(:, 1);
    
    % 检查秩
    rank_W = sum(diag(S) > 1e-6);
    if rank_W > 1
        fprintf('警告: SDP解的秩为 %d (非秩1)\n', rank_W);
    end
end

%% 5. 实际应用示例 - 保密率最大化
function demo_secrecy_rate_optimization()
    % 保密率最大化示例
    
    % 系统参数
    N = 4;              % 天线数
    Pt = 1;             % 发射功率
    sigma_R = 0.01;     % 接收机噪声
    sigma_E = 0.01;     % 窃听者噪声
    P_th = 0.1;         % 能量阈值
    m_th = 0.2;         % 调制指数阈值
    
    % 生成信道
    h_RU = (randn(N,1) + 1i*randn(N,1))/sqrt(2);  % Reader-Tag信道
    h_RE = (randn(N,1) + 1i*randn(N,1))/sqrt(2);  % Reader-Eve信道
    
    % 定义保密率函数
    SR_func = @(w, v, Gamma0, Gamma1) compute_secrecy_rate(w, v, Gamma0, Gamma1, ...
        h_RU, h_RE, Pt, sigma_R, sigma_E);
    
    % 外层：搜索最优反射系数
    Gamma_grid = linspace(-1, 1, 20);
    best_SR = -inf;
    best_params = [];
    
    for Gamma0 = Gamma_grid
        for Gamma1 = Gamma_grid
            % 检查调制指数约束
            if abs(Gamma0 - Gamma1) < 2*m_th
                continue;
            end
            
            % 内层：优化波束成形向量
            cvx_begin quiet
                variable w(N) complex
                variable v(N) complex
                
                % 构造目标函数的近似
                SNR_R = Pt * abs(Gamma0 - Gamma1)^2 * ...
                    abs(v' * h_RU)^2 * abs(w' * h_RU)^2 / (4 * sigma_R^2);
                SNR_E = Pt * abs(Gamma0 - Gamma1)^2 * ...
                    abs(h_RE' * w)^2 / (4 * sigma_E^2);
                
                % 使用对数近似
                maximize(log(1 + SNR_R) - log(1 + SNR_E))
                
                subject to
                    norm(w) <= 1;
                    norm(v) <= 1;
            cvx_end
            
            % 计算实际保密率
            SR = SR_func(w, v, Gamma0, Gamma1);
            
            if SR > best_SR
                best_SR = SR;
                best_params = struct('w', w, 'v', v, ...
                    'Gamma0', Gamma0, 'Gamma1', Gamma1);
            end
        end
    end
    
    % 显示结果
    fprintf('=== 保密率优化结果 ===\n');
    fprintf('最大保密率: %.4f bps/Hz\n', best_SR);
    fprintf('最优反射系数: Gamma0 = %.3f, Gamma1 = %.3f\n', ...
        best_params.Gamma0, best_params.Gamma1);
    fprintf('调制指数: |Gamma0 - Gamma1| = %.3f\n', ...
        abs(best_params.Gamma0 - best_params.Gamma1));
end

function SR = compute_secrecy_rate(w, v, Gamma0, Gamma1, h_RU, h_RE, Pt, sigma_R, sigma_E)
    % 计算保密率
    SNR_R = Pt * abs(Gamma0 - Gamma1)^2 * ...
        abs(v' * h_RU)^2 * abs(w' * h_RU)^2 / (4 * sigma_R^2);
    SNR_E = Pt * abs(Gamma0 - Gamma1)^2 * ...
        abs(h_RE' * w)^2 / (4 * sigma_E^2);
    
    R_R = log2(1 + SNR_R);
    R_E = log2(1 + SNR_E);
    SR = max(0, R_R - R_E);
end

%% 6. 可视化函数
function plot_nonconvex_landscape()
    % 绘制非凸函数的3D图像
    
    % 定义一个非凸函数: f(x,y) = x^4 - 2x^2 + y^2 + xy
    f = @(x, y) x.^4 - 2*x.^2 + y.^2 + x.*y;
    
    % 创建网格
    [X, Y] = meshgrid(linspace(-2, 2, 100), linspace(-2, 2, 100));
    Z = f(X, Y);
    
    % 3D曲面图
    figure('Position', [100, 100, 1200, 500]);
    
    subplot(1, 2, 1);
    surf(X, Y, Z);
    xlabel('x');
    ylabel('y');
    zlabel('f(x,y)');
    title('非凸函数3D曲面');
    colormap('jet');
    shading interp;
    
    % 等高线图
    subplot(1, 2, 2);
    contour(X, Y, Z, 30);
    xlabel('x');
    ylabel('y');
    title('等高线图');
    colorbar;
    
    % 标记局部最优点
    hold on;
    % 使用多起点寻找局部最优
    starts = [-1.5, -1; 1.5, 1; 0, 0; -1, 1; 1, -1];
    for i = 1:size(starts, 1)
        x0 = starts(i, :)';
        [x_opt, ~] = fminunc(@(x) f(x(1), x(2)), x0, ...
            optimoptions('fminunc', 'Display', 'off'));
        plot(x_opt(1), x_opt(2), 'ro', 'MarkerSize', 10, 'LineWidth', 2);
    end
    legend('等高线', '局部最优点');
end

%% 7. 主运行脚本
function main()
    fprintf('=== MATLAB非凸优化示例 ===\n\n');
    
    % 示例1: 简单的非凸函数优化
    fprintf('1. 梯度下降法示例\n');
    obj = @(x) x(1)^4 - 2*x(1)^2 + x(2)^2 + x(1)*x(2);
    grad = @(x) [4*x(1)^3 - 4*x(1) + x(2); 2*x(2) + x(1)];
    
    x0 = [2; 1];
    [x_opt, f_opt, history] = gradient_descent(obj, grad, x0);
    fprintf('   初始点: [%.2f, %.2f]\n', x0(1), x0(2));
    fprintf('   最优解: [%.4f, %.4f]\n', x_opt(1), x_opt(2));
    fprintf('   最优值: %.4f\n\n', f_opt);
    
    % 示例2: 保密率优化
    fprintf('2. 保密率最大化示例\n');
    demo_secrecy_rate_optimization();
    
    % 示例3: 可视化
    fprintf('\n3. 生成优化景观图...\n');
    plot_nonconvex_landscape();
    
    fprintf('\n=== 示例运行完成 ===\n');
end

% 辅助函数：获取结构体字段值（带默认值）
function val = getfield(s, field, default)
    if isfield(s, field)
        val = s.(field);
    else
        val = default;
    end
end

% 运行主函数
if ~exist('MATLAB_RUNNING_AS_FUNCTION', 'var')
    main();
end