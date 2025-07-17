    % load_and_analyze_data.m
    % Load saved simulation data and perform post-processing and analysis
    % Can be used for subsequent data processing, analysis, and visualization

    function load_and_analyze_data(filename)
        % If no filename is provided, let the user select one
        if nargin < 1
            [filename, pathname] = uigetfile('*.mat', 'Select simulation data file');
            if filename == 0
                fprintf('No file selected, exiting\n');
                return;
            end
            filename = fullfile(pathname, filename);
        end
        
        % Ensure the filename has a .mat extension
        if ~endsWith(filename, '.mat')
            filename = [filename, '.mat'];
        end
        
        % Check if the file exists
        if ~exist(filename, 'file')
            fprintf('Error: File %s does not exist\n', filename);
            return;
        end
        
        fprintf('Loading simulation data: %s\n', filename);
        
        % Load data
        data = load(filename);
        
        % Assign loaded variables to the workspace
        field_names = fieldnames(data);
        for i = 1:length(field_names)
            assignin('base', field_names{i}, data.(field_names{i}));
        end
        
        fprintf('Successfully loaded the following variables into the workspace:\n');
        for i = 1:length(field_names)
            fprintf('- %s\n', field_names{i});
        end
        
        % Display basic data information
        fprintf('\n=== Basic Data Information ===\n');
        if isfield(data, 'N_set')
            fprintf('Number of antennas: %s\n', mat2str(data.N_set));
        end
        if isfield(data, 'd_UE_set')
            fprintf('Tag-Eve distance range: %.1f - %.1f m\n', min(data.d_UE_set), max(data.d_UE_set));
        end
        if isfield(data, 'MC_runs')
            fprintf('Monte Carlo runs: %d\n', data.MC_runs);
        end
        
        % Display result statistics
        fprintf('\n=== Result Statistics ===\n');
        if isfield(data, 'SR_brute') && isfield(data, 'SR_cvx')
            fprintf('Brute Force SR range: %.4f - %.4f bits/s/Hz\n', ...
                min(data.SR_brute(:)), max(data.SR_brute(:)));
            fprintf('CVX SR range: %.4f - %.4f bits/s/Hz\n', ...
                min(data.SR_cvx(:)), max(data.SR_cvx(:)));
        end
        
        % Calculate performance comparison
        if isfield(data, 'SR_brute') && isfield(data, 'SR_cvx')
            fprintf('\n=== Performance Comparison ===\n');
            % Calculate average performance improvement
            performance_gain = (data.SR_cvx - data.SR_brute) ./ (data.SR_brute + eps) * 100;
            avg_gain = mean(performance_gain(:));
            fprintf('Average performance improvement of CVX over Brute Force: %.2f%%\n', avg_gain);
            
            % Find the best performance configuration
            [max_sr_brute, idx_brute] = max(data.SR_brute(:));
            [max_sr_cvx, idx_cvx] = max(data.SR_cvx(:));
            [n_idx_brute, d_idx_brute] = ind2sub(size(data.SR_brute), idx_brute);
            [n_idx_cvx, d_idx_cvx] = ind2sub(size(data.SR_cvx), idx_cvx);
            
            fprintf('Best Brute Force performance: %.4f bits/s/Hz (N=%d, d_UE=%.1fm)\n', ...
                max_sr_brute, data.N_set(n_idx_brute), data.d_UE_set(d_idx_brute));
            fprintf('Best CVX performance: %.4f bits/s/Hz (N=%d, d_UE=%.1fm)\n', ...
                max_sr_cvx, data.N_set(n_idx_cvx), data.d_UE_set(d_idx_cvx));
        end
        
        fprintf('\nData loading complete! You can now use these variables in the workspace for further analysis.\n');
        fprintf('Tip: Use the whos command to view all available variables\n');

        % Reflection coefficients Γ0, Γ1 vs d_UE
        figure;
        colors = {'b', 'r', 'g', 'm'}; % Different colors
        % Brute Force Γ0
        plot(data.d_UE_set, data.Gamma0_brute(1,:), '--o', 'Color', colors{1}, ...
            'DisplayName', sprintf('Brute \\Gamma_0 N=%d', data.N_set(1)), ...
            'LineWidth', 1.5, 'MarkerSize', 6);
        hold on;
        % Brute Force Γ1
        plot(data.d_UE_set, data.Gamma1_brute(1,:), '--x', 'Color', colors{1}, ...
            'DisplayName', sprintf('Brute \\Gamma_1 N=%d', data.N_set(1)), ...
            'LineWidth', 1.5, 'MarkerSize', 6);
        % CVX Γ0
        plot(data.d_UE_set, data.Gamma0_cvx(1,:), '-s', 'Color', colors{1}, ...
            'DisplayName', sprintf('CVX \\Gamma_0 N=%d', data.N_set(1)), ...
            'LineWidth', 1.5, 'MarkerSize', 6);
        % CVX Γ1
        plot(data.d_UE_set, data.Gamma1_cvx(1,:), '-^', 'Color', colors{1}, ...
            'DisplayName', sprintf('CVX \\Gamma_1 N=%d', data.N_set(1)), ...
            'LineWidth', 1.5, 'MarkerSize', 6);
        xlabel('Tag-Eve Distance d_{UE} (m)');
        ylabel('Reflection Coefficient');
        title('Reflection Coefficient vs. Tag-Eve Distance');
        legend('show', 'Location', 'best');
        grid on;

        figure;
        colors = {'b', 'r', 'g', 'm'}; % Different colors
        % Brute Force Γ0
        plot(data.d_UE_set, data.Gamma0_brute(2,:), '--o', 'Color', colors{2}, ...
            'DisplayName', sprintf('Brute \\Gamma_0 N=%d', data.N_set(2)), ...
            'LineWidth', 1.5, 'MarkerSize', 6);
        hold on;
        % Brute Force Γ1
        plot(data.d_UE_set, data.Gamma1_brute(2,:), '--x', 'Color', colors{2}, ...
            'DisplayName', sprintf('Brute \\Gamma_1 N=%d', data.N_set(2)), ...
            'LineWidth', 1.5, 'MarkerSize', 6);
        % CVX Γ0
        plot(data.d_UE_set, data.Gamma0_cvx(2,:), '-s', 'Color', colors{2}, ...
            'DisplayName', sprintf('CVX \\Gamma_0 N=%d', data.N_set(2)), ...
            'LineWidth', 1.5, 'MarkerSize', 6);
        % CVX Γ1
        plot(data.d_UE_set, data.Gamma1_cvx(2,:), '-^', 'Color', colors{2}, ...
            'DisplayName', sprintf('CVX \\Gamma_1 N=%d', data.N_set(2)), ...
            'LineWidth', 1.5, 'MarkerSize', 6);
        xlabel('Tag-Eve Distance d_{UE} (m)');
        ylabel('Reflection Coefficient');
        title('Reflection Coefficient vs. Tag-Eve Distance');
        legend('show', 'Location', 'best');
        grid on;

        figure;
        colors = {'b', 'r', 'g', 'm'}; % Different colors
        % Brute Force Γ0
        plot(data.d_UE_set, data.Gamma0_brute(3,:), '--o', 'Color', colors{3}, ...
            'DisplayName', sprintf('Brute \\Gamma_0 N=%d', data.N_set(3)), ...
            'LineWidth', 1.5, 'MarkerSize', 6);
        hold on;
        % Brute Force Γ1
        plot(data.d_UE_set, data.Gamma1_brute(3,:), '--x', 'Color', colors{3}, ...
            'DisplayName', sprintf('Brute \\Gamma_1 N=%d', data.N_set(3)), ...
            'LineWidth', 1.5, 'MarkerSize', 6);
        % CVX Γ0
        plot(data.d_UE_set, data.Gamma0_cvx(3,:), '-s', 'Color', colors{3}, ...
            'DisplayName', sprintf('CVX \\Gamma_0 N=%d', data.N_set(3)), ...
            'LineWidth', 1.5, 'MarkerSize', 6);
        % CVX Γ1
        plot(data.d_UE_set, data.Gamma1_cvx(3,:), '-^', 'Color', colors{3}, ...
            'DisplayName', sprintf('CVX \\Gamma_1 N=%d', data.N_set(3)), ...
            'LineWidth', 1.5, 'MarkerSize', 6);
        xlabel('Tag-Eve Distance d_{UE} (m)');
        ylabel('Reflection Coefficient');
        title('Reflection Coefficient vs. Tag-Eve Distance');
        legend('show', 'Location', 'best');
        grid on;

        figure;
        colors = {'b', 'r', 'g', 'm'}; % Different colors
        % Brute Force Γ0
        plot(data.d_UE_set, data.Gamma0_brute(4,:), '--o', 'Color', colors{4}, ...
            'DisplayName', sprintf('Brute \\Gamma_0 N=%d', data.N_set(4)), ...
            'LineWidth', 1.5, 'MarkerSize', 6);
        hold on;
        % Brute Force Γ1
        plot(data.d_UE_set, data.Gamma1_brute(4,:), '--x', 'Color', colors{4}, ...
            'DisplayName', sprintf('Brute \\Gamma_1 N=%d', data.N_set(4)), ...
            'LineWidth', 1.5, 'MarkerSize', 6);
        % CVX Γ0
        plot(data.d_UE_set, data.Gamma0_cvx(4,:), '-s', 'Color', colors{4}, ...
            'DisplayName', sprintf('CVX \\Gamma_0 N=%d', data.N_set(4)), ...
            'LineWidth', 1.5, 'MarkerSize', 6);
        % CVX Γ1
        plot(data.d_UE_set, data.Gamma1_cvx(4,:), '-^', 'Color', colors{4}, ...
            'DisplayName', sprintf('CVX \\Gamma_1 N=%d', data.N_set(4)), ...
            'LineWidth', 1.5, 'MarkerSize', 6);
        xlabel('Tag-Eve Distance d_{UE} (m)');
        ylabel('Reflection Coefficient');
        title('Reflection Coefficient vs. Tag-Eve Distance');
        legend('show', 'Location', 'best');
        grid on;

        figure;
        sel_d = 2; % Fixed 1:d_UE=5m;2:d_UE=10m;...10:d_UE=50m;
        colors = {'b', 'r', 'g', 'm'}; % Different colors
        markers = {'o', 's', '^', 'd'}; % Different markers
        for nIdx = 1:length(data.N_set)
            convergence_data = cell2mat(data.cvxSR_convergence(nIdx, sel_d));
            plot(1:length(convergence_data), convergence_data, '-', 'Color', colors{nIdx}, ...
                'Marker', markers{nIdx}, 'DisplayName', sprintf('N=%d', data.N_set(nIdx)), ...
                'LineWidth', 1.5, 'MarkerSize', 6);
            hold on;
        end
        xlabel('AO Iteration');
        ylabel('Secrecy Rate (bits/s/Hz)');
        title(sprintf('CVX AO Convergence (d_{UE}=%.1fm)', data.d_UE_set(sel_d)));
        legend('show', 'Location', 'best');
        grid on;

        
    end
