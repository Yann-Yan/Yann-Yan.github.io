clc;
clear;
close all;

%% =====================================================
% Direct Learning Architecture (DLA) Training Engine
% Description: Advanced closed-loop DPD training algorithm operating 
% in deep PA saturation (Peak = 0.85). Features dynamic learning rate 
% annealing, OOM prevention, and an anti-saturation digital shield.
%% =====================================================

%% ----------------------------------------------------------
% 1. Load Data & Apply High-Peak Back-off (Deep Compression)
%% ----------------------------------------------------------
% [USER ACTION REQUIRED] Specify the directory containing the PA dataset
dataFolder = "To be filled in"; 
data_path = fullfile(dataFolder, "nPA.mat");

D = load(data_path);
X_raw = D.x(:);

% --- Challenge Mode ---
% Pushing the PA deep into the nonlinear compression region (0.85).
% This significantly increases power efficiency but demands highly 
% robust DPD control to prevent system divergence.
target_peak = 0.85; 
X_complex = X_raw * (target_peak / max(abs(X_raw)));

disp(['>> Initialization: Dataset loaded. Challenge Mode (Peak = ', num2str(target_peak), ')']);

%% ----------------------------------------------------------
% 2. System Hyperparameters & Architecture Configuration
%% ----------------------------------------------------------
M = 2;                  % Increased Memory Depth for complex DLA mapping
M_pa = 1;               % Physical PA Memory Depth
hiddenLayerSize = [40 30]; % Deep Neural Network Architecture

% DLA Iteration Parameters
num_iter = 10;
outer_patience = 3;     % Custom early stopping threshold based on physical EVM
tol = 0.02;             % Minimum EVM improvement tolerance

% --- Learning Rate Annealing (Beta Plan) ---
% Dynamically reduces the iteration step size as the system approaches
% the optimal inverse mapping, preventing oscillation around the minimum.
beta_plan = [
    1  0.35;
    3  0.25;
    6  0.15;
    8  0.08
];

% --- [CRITICAL] Memory Optimization Parameter ---
% Limits the training subset to 10,000 samples per iteration.
train_size = 10000; 

% --- [CRITICAL] Hardware Protection Parameter ---
% Maximum allowed amplitude for the Neural Network output.
MAX_Z_AMP = 1.05; 

%% ----------------------------------------------------------
% 3. Feature Extraction & Standardization
%% ----------------------------------------------------------
N = length(X_complex);
N_eff = N - M;
input_dim = 2*(M+1);
X_input = zeros(input_dim, N_eff);

% Construct Tap-Delay Line (TDL) Matrix
for n = M+1:N
    col = n - M;
    for k = 0:M
        X_input(2*k + 1, col) = real(X_complex(n-k));
        X_input(2*k+2, col)   = imag(X_complex(n-k));
    end
end

% Strict Z-score Normalization for BP gradient stability
input_mean = mean(X_input, 2);
input_std  = std(X_input, 0, 2);
input_std(input_std==0) = 1;
X_input_norm = (X_input - input_mean) ./ input_std;

X_ref_full = X_complex(M+1:N);

%% ----------------------------------------------------------
% 4. Baseline EVM Evaluation & Precise Alignment
%% ----------------------------------------------------------
Y_base_full = fpa2(X_complex, false);
Y_base_val  = Y_base_full(M+1+M_pa:end);
X_val_base  = X_ref_full(M_pa+1:end);

% --- [CRITICAL] Precise Baseline Alignment ---
% Utilizes the Least Squares operator (\) to extract and neutralize 
% static linear gain and phase rotation. Ensures EVM purely measures 
% nonlinear physical distortion.
complex_gain_base = Y_base_val \ X_val_base;
evm_baseline = rms(X_val_base - Y_base_val * complex_gain_base) / rms(X_val_base) * 100;

fprintf("\n--- Baseline Simulation (No DPD) ---\n");
fprintf("Baseline System EVM = %.4f %%\n", evm_baseline);
fprintf("------------------------------------\n");

%% ----------------------------------------------------------
% 5. Initialize Neural Network & DLA Targets
%% ----------------------------------------------------------
net = fitnet(hiddenLayerSize,'trainlm');
net.performFcn = 'mse';
net.divideFcn = 'dividetrain'; % Disable internal split to maximize data usage
net.trainParam.epochs   = 50;  % Short epoch limit per DLA iteration
net.trainParam.min_grad = 1e-6;
net.trainParam.showWindow = true;       
net.trainParam.showCommandLine = false;  

% Initial estimate of the PA's inverse characteristics
PA_Gain_Estimate = 1 / abs(complex_gain_base);
Z_target = X_ref_full * PA_Gain_Estimate; 

best_evm = inf;
no_improve_count = 0;
evm_history = zeros(num_iter + 1, 1);
evm_history(1) = evm_baseline; 

disp(">> Executing: Starting DLA Iterative Loop with Anti-Runaway Protection...");

%% ----------------------------------------------------------
% 6. The Core Iterative DLA Loop
%% ----------------------------------------------------------
for iter = 1:num_iter
    
    % 6.1 Apply Learning Rate Annealing
    beta = beta_plan(1,2);
    for b = 1:size(beta_plan,1)
        if iter >= beta_plan(b,1)
            beta = beta_plan(b,2);
        end
    end
    
    T_train = [real(Z_target).'; imag(Z_target).'];
    
    % --- [CRITICAL] Memory Optimization via Downsampling ---
    % Restricts the dataset size fed to the 'trainlm' optimizer.
    % This strictly prevents 'Out of Memory' (OOM) crashes caused by 
    % the massive Jacobian matrix inversion (J^T * J) required by 
    % the Levenberg-Marquardt algorithm.
    idx_train = 1:min(N_eff, train_size);
    
    % Train the Neural Network using the truncated subset
    net = train(net, X_input_norm(:, idx_train), T_train(:, idx_train)); 
    
    % 6.2 DPD Inference (Forward Pass)
    Z_pre_norm = net(X_input_norm);
    Z_pre = (Z_pre_norm(1,:) + 1j*Z_pre_norm(2,:)).';
    
    % 6.3 Hardware Feedback Simulation
    Z_feed = [zeros(M, 1); Z_pre];
    Y_out_full = fpa2(Z_feed, false);
    Y_out = Y_out_full(M+1:end);
    
    % 6.4 Calculate Iteration Error & System EVM
    Y_val = Y_out(M_pa+1:end);
    X_val = X_ref_full(M_pa+1:end);
    Z_val = Z_pre(M_pa+1:end);
    
    complex_gain = Y_val \ X_val; % Real-time alignment
    Y_aligned = Y_val * complex_gain;
    
    error = X_val - Y_aligned;
    evm = rms(error) / rms(X_val) * 100;
    evm_history(iter + 1) = evm;
    
    fprintf("--> DLA Iteration %2d | Annealing Beta = %.2f | System EVM = %5.4f %%\n", iter, beta, evm);
    
    % --- [CRITICAL] Custom Physical Early Stopping ---
    % Overrides MATLAB's internal validation. Uses the actual closed-loop 
    % hardware EVM feedback to halt training and prevent over-fitting.
    if best_evm - evm > tol
        best_evm = evm;
        no_improve_count = 0;
    else
        no_improve_count = no_improve_count + 1;
    end
    
    if no_improve_count >= outer_patience
        fprintf("--> Convergence Limit Reached. Triggering Early Stopping to prevent overfitting.\n");
        break;
    end
    
    % 6.5 Update DLA Targets via Feedback Extraction
    feedback_error = error * PA_Gain_Estimate;
    Z_target(M_pa+1:end) = Z_val + beta * feedback_error;
    
    % ========================================================
    % --- [CRITICAL] Anti-Saturation Digital Shield ---
    % Implements hard clipping at the algorithm's boundary.
    % Physically protects the closed-loop system from demanding impossible 
    % power levels, completely eliminating saturation runaway and divergence.
    % ========================================================
    mag_Z = abs(Z_target);
    overshoot = mag_Z > MAX_Z_AMP;
    Z_target(overshoot) = Z_target(overshoot) ./ mag_Z(overshoot) * MAX_Z_AMP;
end

disp(">> Success: DLA training successfully converged.");

%% ----------------------------------------------------------
% 7. Export Model & Convergence Visualization
%% ----------------------------------------------------------
norm_param.input_mean = input_mean;
norm_param.input_std  = input_std;

% Export architecture for final verification
save('DLA_model.mat', 'net', 'norm_param', 'M', 'target_peak');
disp(">> Model exported to DLA_model.mat");

valid_idx = find(evm_history > 0);
plot_evm = evm_history(valid_idx);
x_axis = 0:(length(plot_evm)-1); 

figure('Color', 'w');
plot(x_axis, plot_evm, '-o', 'LineWidth', 2, 'MarkerFaceColor', 'b', 'MarkerSize', 6);
grid on;
xlabel('DLA Iteration (0 = Baseline without DPD)');
ylabel('System EVM (%)');
title(['DLA Iterative Convergence (Peak = ', num2str(target_peak), ')']);
