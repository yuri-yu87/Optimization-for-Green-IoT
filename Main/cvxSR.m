function [opt_SR, opt_gamma0, opt_gamma1, opt_w, SR_curve] = cvxSR(h_RU, h_UE, N, Pt, mth, Pth, eta_b, eta_e, sigmaR2, sigmaE2)
% cvxSR: Alternating Optimization (AO)-based secrecy rate maximization
% This implementation follows the AO-based solution described in Section 4.2 of the design journal.
% Key points:
%   - The receive combining vector g is always set to the MRC solution: g = h_RU / norm(h_RU).
%   - The transmit beamforming vector w and the tag reflection coefficients (gamma0, gamma1) are alternately optimized.
%   - All constraints (power, modulation depth, energy harvesting) are strictly enforced.
%   - The optimization alternates between optimizing w (for fixed gamma0, gamma1) and optimizing (gamma0, gamma1) (for fixed w), until convergence.

% Initialization
gamma0 = 0.8; gamma1 = -0.8; % Initial reflection coefficients
w = sqrt(Pt) * conj(h_RU) / norm(conj(h_RU)); % Initial transmit beamforming (MRT)
g = h_RU / norm(h_RU); % MRC combining vector (fixed)
max_iter = 30;
% tol = 1e-4;
tol = 1e-6;
prev_SR = -Inf; % Initialize previous SR as negative infinity
SR_curve = zeros(max_iter,1);

for iter = 1:max_iter
    fprintf('Start AO iter = %d \n', iter);

    % === Step 1: Optimize w for fixed (gamma0, gamma1) using SCA ===
    % SCA: secrecy rate is DC, linearize the concave part at current w, solve convex problem
    % Variables: w (complex N x 1)

    % Constraints: ||w||^2 <= Pt, EH constraint
    hRu_norm = norm(conj(h_RU));
    EH_coeff = eta_e * (1 - (abs(gamma0)^2 + abs(gamma1)^2)/2);
    if EH_coeff > 0
        min_w_power = Pth / (EH_coeff * hRu_norm^2);
        min_w_power = max(0, min_w_power);
        max_w_power = Pt;
        if min_w_power > max_w_power
            % Infeasible, return default
            opt_SR = 0;
            opt_gamma0 = gamma0;
            opt_gamma1 = gamma1;
            opt_w = zeros(N,1);
            return;
        end
    else
        % EH constraint cannot be satisfied for any w
        opt_SR = 0;
        opt_gamma0 = gamma0;
        opt_gamma1 = gamma1;
        opt_w = zeros(N,1);
        return;
    end

    % SCA parameters
    sca_max_iter = 30;
    % sca_tol = 1e-4;
    sca_tol = 1e-6;
    w_sca = w; 
    for sca_iter = 1:sca_max_iter
        % Compute constants at current w_sca
        % hRw = h_RU.' * w_sca;
        hRg = h_RU' * g;
        delta_gamma = gamma0 - gamma1;
        abs_delta_gamma2 = abs(delta_gamma)^2;

        % Signal terms
        % numR = eta_b * abs(hRw)^2 * abs(hRg)^2 * abs_delta_gamma2 / (4 * sigmaR2);
        % numE = eta_b * abs(h_UE)^2 * abs(hRw)^2 * abs_delta_gamma2 / (4 * sigmaE2);

        % The secrecy rate is:
        % SR(w) = log2(1 + a*|h_RU'*w|^2) - log2(1 + b*|h_RU'*w|^2)
        % where a = eta_b*|hRg|^2*|delta_gamma|^2/(4*sigmaR2)
        %       b = eta_b*|h_UE|^2*|delta_gamma|^2/(4*sigmaE2)
        a = eta_b * abs(hRg)^2 * abs_delta_gamma2 / (4 * sigmaR2);
        b = eta_b * abs(h_UE)^2 * abs_delta_gamma2 / (4 * sigmaE2);

        % Linearize the concave part: -log2(1 + b*|h_RU'*w|^2)
        % At w0, the first-order Taylor expansion:
        % f(w) ≈ f(w0) + real(grad'*(w-w0))
        % grad = - (2*b*conj(h_RU'*w0)*h_RU) / (ln(2)*(1 + b*|h_RU'*w0|^2))

        % Compute the objective value and gradient at the current point
        hRw0 = h_RU.' * w_sca;
        denomE = 1 + b*abs(hRw0)^2;
        gradE = - (2*b*conj(hRw0)*h_RU) / (log(2)*denomE);
        f0 = log(1 + a*square_abs(hRw0))/log(2);
        grad_f0 = (2*a*conj(hRw0)*h_RU)/(log(2)*(1 + a*abs(hRw0)^2));
        
        % The SCA subproblem:
        % maximize log2(1 + a*|h_RU'*w|^2) + real(gradE'*(w-w0)) + const
        % subject to ||w||^2 <= Pt, EH constraint
        
        cvx_begin quiet
            variable w_var(N) complex
            hRw_var = h_RU.' * w_var;
            hRw0 = h_RU.' * w_sca;
            EH_linear = abs(hRw0)^2 + 2*real(conj(hRw0) * (hRw_var - hRw0));
            obj = f0 + real(grad_f0' * (w_var - w_sca)) + real(gradE'*(w_var - w_sca));
            maximize(obj)
            subject to
                sum_square_abs(w_var) <= Pt;
                EH_coeff * EH_linear >= Pth;
        cvx_end

        if ~strcmp(cvx_status,'Solved') && ~strcmp(cvx_status,'Inaccurate/Solved')
            % Infeasible or failed
            break;
        end

        % Check convergence
        if norm(w_var - w_sca) < sca_tol
            w_sca = w_var;
            break;
        end
        w_sca = w_var;
    end
    w = w_sca;

    % === Step 2: Optimize (gamma0, gamma1) for fixed w using grid search ===
    gamma_range = -1:0.05:1;
    best_SR_gamma = -Inf;
    best_gamma0 = gamma0;
    best_gamma1 = gamma1;
    hRw = h_RU' * w;
    hRg = h_RU' * g;
    for g0 = gamma_range
        for g1 = gamma_range
            if abs(g0 - g1)/2 < mth
                continue;
            end
            % Energy harvesting constraint
            EH_coeff = eta_e * (1 - (abs(g0)^2 + abs(g1)^2)/2);
            P_L_avg = EH_coeff * abs(hRw)^2;
            if P_L_avg < Pth
                continue;
            end
            delta_gamma = g0 - g1;
            abs_delta_gamma2 = abs(delta_gamma)^2;
            gammaR = eta_b * abs(hRw)^2 * abs(hRg)^2 * abs_delta_gamma2 / (4 * sigmaR2);
            gammaE = eta_b * abs(h_UE)^2 * abs(hRw)^2 * abs_delta_gamma2 / (4 * sigmaE2);
            SR = log2(1 + gammaR) - log2(1 + gammaE);
            if SR > best_SR_gamma
                best_SR_gamma = SR;
                best_gamma0 = g0;
                best_gamma1 = g1;
            end
        end
    end
    gamma0 = best_gamma0;
    gamma1 = best_gamma1;

    % === Step 3: Check convergence ===
    SR = best_SR_gamma;
    SR_curve(iter) = SR;
    if abs(SR - prev_SR) < tol
        SR_curve = SR_curve(1:iter); % Trim unused entries
        break;
    end
    prev_SR = SR;
end

opt_SR = SR;
opt_gamma0 = gamma0;
opt_gamma1 = gamma1;
opt_w = w;
SR_curve = SR_curve(1:iter); % Ensure output is trimmed to actual iterations
end