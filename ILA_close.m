clc;
clear;
close all;

%% ==========================================================
% FULL ILA-DPD CLOSED-LOOP VALIDATION
% Description: Evaluates the trained ILA Neural Network by placing it 
% in front of the PA model (Pre-Distortion Phase).
%% ==========================================================

%% ----------------------------------------------------------
% 1. Load Trained ILA Model
%% ----------------------------------------------------------
% [USER ACTION REQUIRED] Specify the directory containing the trained model
modelFolder = 'To be filled in'; 
modelPath = fullfile(modelFolder, 'ILA_model2.mat');

if ~exist(modelPath, 'file')
    error('>> Fatal Error: Model file not found! Please execute ILAtrain.m first.');
end

Mdl = load(modelPath);
net = Mdl.net;
input_mean  = Mdl.input_mean;
input_std   = Mdl.input_std;
gain_linear = Mdl.gain_linear; 
M = Mdl.M;
disp(">> Initialization: ILA Neural Network model loaded successfully.");

%% ----------------------------------------------------------
% 2. Load & Scale Signal [CRITICAL: Power Back-off Exploration]
%% ----------------------------------------------------------
% [USER ACTION REQUIRED] Specify the directory containing the PA dataset
dataFolder = 'To be filled in'; 
data_path = fullfile(dataFolder, 'nPA.mat');
D = load(data_path);
x_raw = D.x(:); 

% --- [CRITICAL EXPLANATION]: The Power Back-off Strategy ---
% The physical PA hard-saturates around 0.65. Attempting to force 
% linearity up to 1.0 (Peak) will result in "Fold-back" and DPD failure, 
% as the algorithm cannot create energy the hardware cannot provide.
%
% --> Try changing 'target_peak' to 1.0 to witness saturation stalling.
% --> Keep it at 0.60 to provide the necessary "Digital Headroom" for DPD.
target_peak = 0.60; 

% Scale the desired baseband signal to the back-off sweet spot
x = x_raw * (target_peak / max(abs(x_raw))); 

% Generate the baseline output WITHOUT DPD for comparison
y_before = fpa2(x, false); 

N = length(x);
N_eff = N - M;
input_dim = 2*(M+1);

%% ----------------------------------------------------------
% 3. DPD Inference (Executing Pre-Inverse Neural Network)
%% ----------------------------------------------------------
x_target = x;
X_input = zeros(input_dim, N_eff);

% Construct the tap-delay line input matrix for the inference engine
for n = M+1:N
    col = n - M;
    for k = 0:M
        idx = 2*k + 1;
        X_input(idx, col)   = real(x_target(n-k));
        X_input(idx+1,col)  = imag(x_target(n-k));
    end
end

% Standardize using the EXACT SAME parameters saved from the training phase
X_input_norm = (X_input - input_mean) ./ input_std;

% Neural Network Forward Pass (Pre-distortion mapping)
X_pre_norm = net(X_input_norm);
x_pre_complex = X_pre_norm(1,:) + 1j*X_pre_norm(2,:);
x_pre_complex = x_pre_complex.';

% --- [CRITICAL] Reverse Scaling & Digital Hardware Shield ---
% 1. Compensate for the global linear gain extracted during training
x_pre = x_pre_complex / gain_linear; 

% 2. The "Digital Shield" (Hard Clipping Constraint)
% Prevents the Neural Network from generating extreme amplitude spikes 
% that could overdrive and physically damage the simulated RF frontend.
max_allowed_amp = 1.10; 
mag = abs(x_pre);
overshoot = mag > max_allowed_amp;
x_pre(overshoot) = x_pre(overshoot) ./ mag(overshoot) * max_allowed_amp;

%% ----------------------------------------------------------
% 4. Pass Through PA Model (Physical Execution)
%% ----------------------------------------------------------
y_after = fpa2(x_pre, false);

%% ----------------------------------------------------------
% 5. System Performance Metrics Calculation
%% ----------------------------------------------------------
% Truncate the first M samples to eliminate startup transient errors
x_ref = x(M+1:N);
y_before_aligned = y_before(M+1:N);

% Calculate Error Vector Magnitude (EVM)
evm_before = rms(y_before_aligned - x_ref) / rms(x_ref) * 100;
evm_after  = rms(y_after - x_ref) / rms(x_ref) * 100;

fprintf("\n=================================================\n");
fprintf(" ILA CLOSED-LOOP EVM RESULTS (Peak = %.2f)\n", target_peak);
fprintf("=================================================\n");
fprintf(" EVM Before DPD (Baseline) = %.4f %%\n", evm_before);
fprintf(" EVM After  DPD (ILA)      = %.4f %%\n", evm_after);
fprintf("=================================================\n\n");

%% ----------------------------------------------------------
% 6. Visualization (AM-AM & AM-PM Trajectories)
%% ----------------------------------------------------------
in_amp = abs(x_ref);
out_before = abs(y_before_aligned);
out_after  = abs(y_after);

% Phase Distortion Calculation and Alignment
phase_before = unwrap(angle(y_before_aligned) - angle(x_ref));
phase_after  = unwrap(angle(y_after) - angle(x_ref));

% Align phase at low power regions (small-signal linear regime) to zero
ref_idx = in_amp < 0.1;
phase_before = phase_before - median(phase_before(ref_idx));
phase_after  = phase_after - median(phase_after(ref_idx));

figure('Color','w','Position',[100 100 1100 450]);

% --- Plot 1: AM-AM Gain Compression ---
subplot(1,2,1);
scatter(in_amp, out_before, 6, [0.7 0.7 0.7], 'filled', 'MarkerFaceAlpha', 0.15); hold on;
scatter(in_amp, out_after, 6, [0 0.5 0.8], 'filled', 'MarkerFaceAlpha', 0.15);
plot([0 target_peak], [0 target_peak], 'r--', 'LineWidth', 2);
grid on; 
xlabel('Desired Input Amplitude'); 
ylabel('Measured Output Amplitude');
title(['AM-AM Response (Target Peak = ', num2str(target_peak), ')']); 
legend('Baseline (No DPD)', 'After ILA DPD', 'Ideal Linear Trajectory', 'Location', 'northwest');

% --- Plot 2: AM-PM Phase Memory ---
subplot(1,2,2);
scatter(in_amp, phase_before, 6, [0.8 0.4 0.4], 'filled', 'MarkerFaceAlpha', 0.15); hold on;
scatter(in_amp, phase_after, 6, [0 0.5 0.8], 'filled', 'MarkerFaceAlpha', 0.15);
grid on; 
xlabel('Desired Input Amplitude'); 
ylabel('Phase Error (radians)');
title('AM-PM Phase Correction');
legend('Baseline (No DPD)', 'After ILA DPD');

sgtitle(['ILA Closed-Loop Verification | Final System EVM = ', num2str(evm_after, '%.2f'), '%']);

% End of ILA Verification Script.
% Note: Frequency Domain (ACLR) analysis is omitted here and reserved 
% for the final DLA robust architecture validation.