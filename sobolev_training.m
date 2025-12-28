%% Sobolev Training in RKHS - Experimental Reproduction
% Based on "A Theoretical Analysis of Using Gradient Data for Sobolev Training in RKHS"
% by Zain ul Abdeen et al., IFAC PapersOnLine 56-2 (2023) 3417-3422

clear all; close all; clc;

%% Experiment 1: Effect of Lipschitz Constant on RMSE
fprintf('Running Experiment 1: Effect of Lipschitz Constant on RMSE\n');

% Parameters
n_train = 30;           % Number of training samples
n_test = 20;            % Number of test samples
n_runs = 100;           % Number of runs for averaging
lipschitz_values = 10:10:500;  % Range of Lipschitz constants
gamma = 1;              % Gaussian kernel bandwidth

% Storage for results
rmse_classical = zeros(length(lipschitz_values), n_runs);
rmse_sobolev = zeros(length(lipschitz_values), n_runs);

for L_idx = 1:length(lipschitz_values)
    L = lipschitz_values(L_idx);
    fprintf('  Processing Lipschitz constant: %d\n', L);
    
    for run = 1:n_runs
        % Generate training data with Styblinski-Tang function
        [X_train, y_train, grad_train] = generate_styblinski_tang_data(n_train, L);
        [X_test, y_test, ~] = generate_styblinski_tang_data(n_test, L);
        
        % Classical training (beta = 0)
        f_classical = kernel_ridge_regression(X_train, y_train, grad_train, gamma, 0.01, 0);
        y_pred_classical = predict_kernel(f_classical, X_train, X_test, gamma);
        rmse_classical(L_idx, run) = sqrt(mean((y_test - y_pred_classical).^2));
        
        % Sobolev training (beta = 1)
        f_sobolev = kernel_ridge_regression(X_train, y_train, grad_train, gamma, 0.01, 1);
        y_pred_sobolev = predict_kernel(f_sobolev, X_train, X_test, gamma);
        rmse_sobolev(L_idx, run) = sqrt(mean((y_test - y_pred_sobolev).^2));
    end
end

% Plot results
figure('Position', [100, 100, 800, 600]);
mean_classical = mean(rmse_classical, 2);
std_classical = std(rmse_classical, 0, 2);
mean_sobolev = mean(rmse_sobolev, 2);
std_sobolev = std(rmse_sobolev, 0, 2);

hold on;
fill([lipschitz_values, fliplr(lipschitz_values)], ...
     [mean_classical' + std_classical', fliplr(mean_classical' - std_classical')], ...
     [1 0.7 0.7], 'EdgeColor', 'none', 'FaceAlpha', 0.3);
fill([lipschitz_values, fliplr(lipschitz_values)], ...
     [mean_sobolev' + std_sobolev', fliplr(mean_sobolev' - std_sobolev')], ...
     [0.7 0.7 1], 'EdgeColor', 'none', 'FaceAlpha', 0.3);
plot(lipschitz_values, mean_classical, 'r-o', 'LineWidth', 2, 'MarkerSize', 6, 'DisplayName', 'Classical Algorithm');
plot(lipschitz_values, mean_sobolev, 'b-o', 'LineWidth', 2, 'MarkerSize', 6, 'DisplayName', 'Sobolev Learning');
hold off;

xlabel('Lipschitz bound', 'FontSize', 12);
ylabel('Test Error', 'FontSize', 12);
legend('Location', 'northwest', 'FontSize', 11);
title('Effect of Lipschitz Constant on RMSE', 'FontSize', 14);
grid on;
saveas(gcf, '/mnt/user-data/outputs/lipschitz_effect.png');

%% Experiment 2: Effect of Training Data Amount on RMSE and Lipschitz Constant
fprintf('\nRunning Experiment 2: Heatmap Analysis\n');

% Parameters
sample_sizes = [10, 100, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000];
lipschitz_range = [30, 60, 80, 100, 120, 225, 535];
n_runs_heatmap = 50;  % Reduced for computational efficiency
n_test_heatmap = 50;

% Storage for results
error_sobolev = zeros(length(lipschitz_range), length(sample_sizes));
error_classical = zeros(length(lipschitz_range), length(sample_sizes));

for L_idx = 1:length(lipschitz_range)
    L = lipschitz_range(L_idx);
    fprintf('  Lipschitz constant: %d\n', L);
    
    for s_idx = 1:length(sample_sizes)
        n = sample_sizes(s_idx);
        
        temp_error_sobolev = zeros(n_runs_heatmap, 1);
        temp_error_classical = zeros(n_runs_heatmap, 1);
        
        for run = 1:n_runs_heatmap
            % Generate data
            [X_train, y_train, grad_train] = generate_styblinski_tang_data(n, L);
            [X_test, y_test, ~] = generate_styblinski_tang_data(n_test_heatmap, L);
            
            % Sobolev training
            f_sobolev = kernel_ridge_regression(X_train, y_train, grad_train, gamma, 0.01, 1);
            y_pred_sobolev = predict_kernel(f_sobolev, X_train, X_test, gamma);
            temp_error_sobolev(run) = sqrt(mean((y_test - y_pred_sobolev).^2));
            
            % Classical training
            f_classical = kernel_ridge_regression(X_train, y_train, grad_train, gamma, 0.01, 0);
            y_pred_classical = predict_kernel(f_classical, X_train, X_test, gamma);
            temp_error_classical(run) = sqrt(mean((y_test - y_pred_classical).^2));
        end
        
        error_sobolev(L_idx, s_idx) = mean(temp_error_sobolev);
        error_classical(L_idx, s_idx) = mean(temp_error_classical);
    end
end

% Plot Sobolev heatmap
figure('Position', [100, 100, 900, 600]);
imagesc(sample_sizes, lipschitz_range, log10(error_sobolev));
colorbar;
colormap('hot');
xlabel('No. of samples', 'FontSize', 12);
ylabel('Lipschitz constant', 'FontSize', 12);
title('Heatmap for Sobolev Learning Algorithm', 'FontSize', 14);
set(gca, 'YDir', 'normal');
saveas(gcf, '/mnt/user-data/outputs/heatmap_sobolev.png');

% Plot Classical heatmap
figure('Position', [100, 100, 900, 600]);
imagesc(sample_sizes, lipschitz_range, log10(error_classical));
colorbar;
colormap('hot');
xlabel('No. of samples', 'FontSize', 12);
ylabel('Lipschitz constant', 'FontSize', 12);
title('Heatmap for Classical Learning Algorithm', 'FontSize', 14);
set(gca, 'YDir', 'normal');
saveas(gcf, '/mnt/user-data/outputs/heatmap_classical.png');

fprintf('\nExperiments completed. Results saved to /mnt/user-data/outputs/\n');

%% Helper Functions

function [X, y, grad_y] = generate_styblinski_tang_data(n, L)
    % Generate data based on scaled Styblinski-Tang function
    % The function is scaled to control the Lipschitz constant
    
    X = 2 * rand(n, 1) - 1;  % Sample uniformly in [-1, 1]
    
    % Styblinski-Tang function: f(x) = (x^4 - 16*x^2 + 5*x) / 2
    % Scale factor to control Lipschitz constant
    scale = L / 50;  % Approximate scaling
    
    y = scale * (X.^4 - 16*X.^2 + 5*X) / 2;
    grad_y = scale * (4*X.^3 - 32*X + 5) / 2;
    
    % Add small noise
    y = y + 0.01 * randn(n, 1);
    grad_y = grad_y + 0.01 * randn(n, 1);
end

function model = kernel_ridge_regression(X_train, y_train, grad_train, gamma, lambda, beta)
    % Kernel ridge regression with optional gradient data (Sobolev training)
    % 
    % Inputs:
    %   X_train: Training inputs (n x d)
    %   y_train: Training outputs (n x 1)
    %   grad_train: Training gradients (n x d)
    %   gamma: Kernel bandwidth
    %   lambda: Regularization parameter
    %   beta: Gradient weight (0 for classical, >0 for Sobolev)
    
    n = size(X_train, 1);
    d = size(X_train, 2);
    
    % Compute kernel matrix
    K = gaussian_kernel(X_train, X_train, gamma);
    
    if beta == 0
        % Classical training: only use function values
        alpha = (K + lambda * eye(n)) \ y_train;
        model.alpha = alpha;
        model.alpha_grad = [];
    else
        % Sobolev training: use both function values and gradients
        
        % Compute gradient kernel matrices
        K_grad = cell(d, 1);
        for j = 1:d
            K_grad{j} = gradient_kernel(X_train, X_train, gamma, j);
        end
        
        % Build augmented system
        % [K, beta*K_grad'; beta*K_grad, beta*K_grad_grad] [alpha; alpha_grad] = [y; beta*grad_y]
        
        % For simplicity, we use an approximate formulation
        % that combines value and gradient information
        K_aug = K;
        y_aug = y_train;
        
        for j = 1:d
            K_aug = K_aug + beta * K_grad{j}' * K_grad{j};
            y_aug = y_aug + beta * K_grad{j}' * grad_train(:, j);
        end
        
        alpha = (K_aug + lambda * eye(n)) \ y_aug;
        model.alpha = alpha;
        model.alpha_grad = [];
    end
    
    model.X_train = X_train;
    model.gamma = gamma;
    model.beta = beta;
end

function y_pred = predict_kernel(model, X_train, X_test, gamma)
    % Predict using trained kernel model
    
    K_test = gaussian_kernel(X_test, X_train, gamma);
    y_pred = K_test * model.alpha;
end

function K = gaussian_kernel(X1, X2, gamma)
    % Compute Gaussian (RBF) kernel matrix
    % K(x, y) = exp(-||x - y||^2 / (2*gamma^2))
    
    n1 = size(X1, 1);
    n2 = size(X2, 1);
    
    K = zeros(n1, n2);
    for i = 1:n1
        for j = 1:n2
            K(i, j) = exp(-norm(X1(i, :) - X2(j, :))^2 / (2 * gamma^2));
        end
    end
end

function K_grad = gradient_kernel(X1, X2, gamma, dim)
    % Compute gradient of Gaussian kernel with respect to dimension dim
    % ∂K(x, y)/∂x_dim = -(x_dim - y_dim) / gamma^2 * K(x, y)
    
    n1 = size(X1, 1);
    n2 = size(X2, 1);
    
    K = gaussian_kernel(X1, X2, gamma);
    K_grad = zeros(n1, n2);
    
    for i = 1:n1
        for j = 1:n2
            K_grad(i, j) = -(X1(i, dim) - X2(j, dim)) / (gamma^2) * K(i, j);
        end
    end
end
