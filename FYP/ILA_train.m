clc;
clear;
close all;

%% ==========================================================
% Indirect Learning Architecture (ILA) Training Engine
% Description: Trains the baseline BP Neural Network to model the 
% inverse nonlinear characteristics of the physical Power Amplifier.
%% ==========================================================

%% ----------------------------------------------------------
% 1. Load Distorted PA Dataset
%% ----------------------------------------------------------
% [USER ACTION REQUIRED] Specify the directory containing the PA dataset
dataFolder = 'To be filled in'; 
data_path = fullfile(dataFolder, 'nPA.mat');

if ~exist(data_path, 'file')
    error('>> Fatal Error: Data file not found! Please check the dataFolder path.');
end

D = load(data_path);
X_complex = D.x(:);  % Ideal baseband signal
Y_complex = D.y(:);  % Distorted PA output
disp(">> Initialization: PA Dataset loaded successfully.");

%% ----------------------------------------------------------
% 2. [CRITICAL] Power Alignment (Anti-Numerical Explosion)
%% ----------------------------------------------------------
% Calculates a global linear gain to normalize the PA output (Y) 
% back to the baseband scale before training. 
% This strict alignment prevents the catastrophic scaling mismatch 
% (which previously caused >190,000% EVM) and avoids the "identity mapping" trap.

gain_linear = mean(abs(Y_complex)) / mean(abs(X_complex));
Y_train_sig = Y_complex / gain_linear; 

M = 1; % Memory Depth (1-tap delay line)
N = length(X_complex);
N_eff = N - M;
input_dim = 2*(M+1); % Real and imaginary components for current and past samples

%% ----------------------------------------------------------
% 3. Construct Feature Matrix (Inverse Mapping: Y -> X)
%% ----------------------------------------------------------
% Extracts the tap-delay line features to capture memory effects.
X_input_train = zeros(input_dim, N_eff);

for n = M+1:N
    col = n - M;
    for k = 0:M
        idx = 2*k + 1;
        % Input to the Inverse NN is the (normalized) PA output
        X_input_train(idx, col)   = real(Y_train_sig(n-k));
        X_input_train(idx+1,col)  = imag(Y_train_sig(n-k));
    end
end

% Target for the Inverse NN is the original pristine PA input
T_complex = X_complex(M+1:N);
T = [real(T_complex).';
     imag(T_complex).'];

% -------- [CRITICAL] Z-Score Feature Normalization --------
% Applies strict standardization to ensure the Levenberg-Marquardt 
% gradient descent updates operate within the same mathematical space,
% dramatically accelerating convergence.
input_mean = mean(X_input_train, 2);
input_std  = std(X_input_train, 0, 2);
input_std(input_std == 0) = 1; % Prevent division by zero
X_input_train_norm = (X_input_train - input_mean) ./ input_std;

%% ----------------------------------------------------------
% 4. Configure BP Neural Network Engine
%% ----------------------------------------------------------
% Deploys a Multi-Layer Perceptron as a Universal Approximator
hiddenLayerSize = [30 20]; 

% 'trainlm' (Levenberg-Marquardt) is selected for rapid 2nd-order convergence
net = fitnet(hiddenLayerSize, 'trainlm');

% Training Hyperparameters
net.performFcn = 'mse';

% Data Division to prevent over-fitting during ILA estimation
net.divideFcn = 'dividerand'; 
net.divideParam.trainRatio = 0.8;
net.divideParam.valRatio   = 0.2;
net.divideParam.testRatio  = 0;

net.trainParam.epochs   = 300;
net.trainParam.goal     = 1e-7;
net.trainParam.min_grad = 1e-7;

disp(">> Executing: Training ILA Inverse Model (Nonlinear Correction Engine)...");
[net, tr] = train(net, X_input_train_norm, T);

%% ----------------------------------------------------------
% 5. Compute Convergence Metrics (Fitting Accuracy)
%% ----------------------------------------------------------
% Evaluates the model's performance on the training set
T_pred = net(X_input_train_norm);
err_complex = (T(1,:) + 1j*T(2,:)) - (T_pred(1,:) + 1j*T_pred(2,:));
ref_rms = rms(T(1,:) + 1j*T(2,:));
train_evm_pct = (rms(err_complex) / ref_rms) * 100;

fprintf("\n=================================================\n");
fprintf(" ILA Training Phase Results (Estimation)\n");
fprintf("=================================================\n");
fprintf(" Final MSE                 : %.2e\n", tr.best_perf);
fprintf(" Training EVM (Fitting)    : %.4f %%\n", train_evm_pct);
fprintf("=================================================\n\n");

%% ----------------------------------------------------------
% 6. Export Trained Model
%% ----------------------------------------------------------
% [USER ACTION REQUIRED] Specify the output directory for the trained model
saveFolder = 'To be filled in'; 

if ~exist(saveFolder, 'dir')
    mkdir(saveFolder);
end

% Preserving "ILA_model2.mat" to reflect iterative architecture refinement
savePath = fullfile(saveFolder, 'ILA_model2.mat');

% Gain_linear and normalization parameters MUST be saved to 
% precisely synchronize the mathematical space in the closed-loop verification.
save(savePath, ...
     'net', ...
     'input_mean', ...
     'input_std', ...
     'gain_linear', ...
     'M');

disp(">> Success: Architecture exported to ILA_model2.mat");
