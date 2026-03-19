clc;
clear;
close all;

%% =====================================================
% Final Direct Learning Architecture (DLA) Closed-Loop Verification
% Description: The ultimate validation script. Generates precise 
% time-domain metrics (EVM), frequency-domain metrics (ACLR), 
% and the final AM-AM/AM-PM performance trajectories.
%% =====================================================

fprintf('=================================================\n');
disp('>> Executing: Final DLA Closed-Loop Verification...');

%% ----------------------------------------------------------
% 1. Load Data & Trained DLA Model
%% ----------------------------------------------------------
% [USER ACTION REQUIRED] Specify directories
dataFolder = "To be filled in"; 
modelFolder = "To be filled in"; 

data_path = fullfile(dataFolder, "nPA.mat");
model_path = fullfile(modelFolder, "DLA_model.mat");

if ~exist(data_path, 'file') || ~exist(model_path, 'file')
    error('>> Fatal Error: Data or Model file not found! Please verify paths.');
end

% Load Model Parameters seamlessly from the DLA training phase
load(model_path, 'net', 'norm_param', 'M', 'target_peak');
M_pa = 1; % Physical PA inherent memory depth

% Load and Scale Data to the exact target peak used during training
D = load(data_path);
X_raw = D.x(:);
X_complex = X_raw * (target_peak / max(abs(X_raw)));
N = length(X_complex);

fprintf('>> Model Loaded Successfully. Target Peak = %.2f, DPD Memory M = %d\n', target_peak, M);

%% ----------------------------------------------------------
% 2. Baseline Calculation (System Performance WITHOUT DPD)
%% ----------------------------------------------------------
Y_base_full = fpa2(X_complex, false);

% Truncate transient samples caused by memory delay
X_val      = X_complex(M + M_pa + 1 : end);
Y_base_val = Y_base_full(M + M_pa + 1 : end);

% --- [CRITICAL] Baseline Gain Alignment & EVM Calculation ---
% Employs Least Squares to isolate pure nonlinear distortion
gain_base = Y_base_val \ X_val;
Y_base_aligned = Y_base_val * gain_base;
evm_base = rms(X_val - Y_base_aligned) / rms(X_val) * 100;

%% ----------------------------------------------------------
% 3. DLA Inference (System Performance WITH DPD)
%% ----------------------------------------------------------
N_eff = N - M;
input_dim = 2*(M+1);
X_input = zeros(input_dim, N_eff);

% Construct feature matrix for the trained Neural Network
for n = M+1:N
    col = n - M;
    for k = 0:M
        X_input(2*k + 1, col) = real(X_complex(n-k));
        X_input(2*k + 2, col) = imag(X_complex(n-k));
    end
end

% Standardize using the EXACT parameters from the DLA training
X_input_norm = (X_input - norm_param.input_mean) ./ norm_param.input_std;

% Generate Pre-distorted Signal
Z_pre_norm = net(X_input_norm);
Z_pre = (Z_pre_norm(1,:) + 1j*Z_pre_norm(2,:)).';
Z_feed = [zeros(M, 1); Z_pre]; 

% Pass the Pre-distorted signal through the physical PA model
Y_dpd_full = fpa2(Z_feed, false);
Y_dpd_val = Y_dpd_full(M + M_pa + 1 : end);

% Align and compute final EVM
gain_dpd = Y_dpd_val \ X_val;
Y_dpd_aligned = Y_dpd_val * gain_dpd;
evm_dpd = rms(X_val - Y_dpd_aligned) / rms(X_val) * 100;

%% ----------------------------------------------------------
% 4. Frequency Domain Analysis: PSD & ACLR Calculation
%% ----------------------------------------------------------
window_size = 1024;
[psd_X, f_raw] = pwelch(X_val, hanning(window_size), window_size/2, window_size, 'centered');
[psd_base, ~]  = pwelch(Y_base_aligned, hanning(window_size), window_size/2, window_size, 'centered');
[psd_dpd, ~]   = pwelch(Y_dpd_aligned, hanning(window_size), window_size/2, window_size, 'centered');

% Normalized Frequency Axis (-0.5 to 0.5)
f_norm = linspace(-0.5, 0.5, length(f_raw))';

% --- Discrete Integration for ACLR Evaluation ---
% Main channel bandwidth is approximately 0.2 (-0.1 to 0.1).
% Adjacent channel offset is 0.2.
bw_main = 0.2; 
offset  = 0.2; 
calc_aclr = @(psd, freq, f_center, bw) 10*log10(sum(psd(freq >= f_center-bw/2 & freq <= f_center+bw/2)));

% Baseline ACLR (Before DPD)
p_main_base = calc_aclr(psd_base, f_norm, 0, bw_main);
p_low_base  = calc_aclr(psd_base, f_norm, -offset, bw_main);
p_up_base   = calc_aclr(psd_base, f_norm, offset, bw_main);
aclr_base   = min(p_main_base - p_low_base, p_main_base - p_up_base);

% Final System ACLR (After DPD)
p_main_dpd = calc_aclr(psd_dpd, f_norm, 0, bw_main);
p_low_dpd  = calc_aclr(psd_dpd, f_norm, -offset, bw_main);
p_up_dpd   = calc_aclr(psd_dpd, f_norm, offset, bw_main);
aclr_dpd   = min(p_main_dpd - p_low_dpd, p_main_dpd - p_up_dpd);

%% ----------------------------------------------------------
% 5. Print Final Results to Console
%% ----------------------------------------------------------
fprintf('=================================================\n');
fprintf(' VERIFICATION RESULTS (Time & Frequency Domain)\n');
fprintf('-------------------------------------------------\n');
fprintf(' EVM Before DPD   = %7.4f %%\n', evm_base);
fprintf(' EVM After  DPD   = %7.4f %%\n', evm_dpd);
fprintf('-------------------------------------------------\n');
fprintf(' ACLR Before DPD  = %7.2f dBc\n', aclr_base);
fprintf(' ACLR After  DPD  = %7.2f dBc\n', aclr_dpd);
fprintf(' ACLR Improvement = %7.2f dB\n', aclr_dpd - aclr_base);
fprintf('=================================================\n');

%% ----------------------------------------------------------
% 6. Visualization Preparation
%% ----------------------------------------------------------
amp_X = abs(X_val);
amp_Y_base = abs(Y_base_aligned);
amp_Y_dpd  = abs(Y_dpd_aligned);

% Convert phase errors to degrees for intuitive visualization
phase_base = wrapTo180(angle(Y_base_aligned ./ X_val) * (180/pi));
phase_dpd  = wrapTo180(angle(Y_dpd_aligned ./ X_val) * (180/pi));

%% ----------------------------------------------------------
% 7. Plot 1: AM-AM Gain Compression Recovery
%% ----------------------------------------------------------
figure('Name', 'AM-AM Characteristics', 'Color', 'w', 'Position', [100, 100, 600, 500]);
scatter(amp_X, amp_Y_base, 10, [0.7 0.7 0.7], 'filled', 'MarkerFaceAlpha', 0.3); hold on;
scatter(amp_X, amp_Y_dpd,  10, [0 0.4470 0.7410], 'filled', 'MarkerFaceAlpha', 0.3);
plot([0 max(amp_X)], [0 max(amp_X)], '--r', 'LineWidth', 2);
grid on; 
legend('Baseline (No DPD)', 'After DLA DPD', 'Ideal Linear Trajectory', 'Location', 'northwest');
xlabel('Input Amplitude |X|'); 
ylabel('Output Amplitude |Y|');
title(sprintf('AM-AM Response (Deep Compression Peak = %.2f)', target_peak));

%% ----------------------------------------------------------
% 8. Plot 2: AM-PM Phase Memory Correction
%% ----------------------------------------------------------
figure('Name', 'AM-PM Characteristics', 'Color', 'w', 'Position', [750, 100, 600, 500]);
scatter(amp_X, phase_base, 10, [0.7 0.7 0.7], 'filled', 'MarkerFaceAlpha', 0.3); hold on;
scatter(amp_X, phase_dpd,  10, [0 0.4470 0.7410], 'filled', 'MarkerFaceAlpha', 0.3);
plot([0 max(amp_X)], [0 0], '--r', 'LineWidth', 2);
grid on; 
legend('Baseline (No DPD)', 'After DLA DPD', 'Ideal (\Delta\phi = 0)', 'Location', 'best');
xlabel('Input Amplitude |X|'); 
ylabel('Phase Error (Degrees)');
title(sprintf('AM-PM Phase Correction (Peak = %.2f)', target_peak));
ylim([-20 20]); 

%% ----------------------------------------------------------
% 9. Plot 3: Power Spectrum Density (PSD) & ACLR Suppression
%% ----------------------------------------------------------
figure('Name', 'Power Spectrum', 'Color', 'w', 'Position', [300, 300, 700, 500]);
psd_X_dB    = 10*log10(psd_X);    psd_X_dB    = psd_X_dB - max(psd_X_dB);
psd_base_dB = 10*log10(psd_base); psd_base_dB = psd_base_dB - max(psd_base_dB);
psd_dpd_dB  = 10*log10(psd_dpd);  psd_dpd_dB  = psd_dpd_dB - max(psd_dpd_dB);

plot(f_norm, psd_base_dB, 'Color', [0.7 0.7 0.7], 'LineWidth', 1.5); hold on;
plot(f_norm, psd_dpd_dB, 'Color', [0 0.4470 0.7410], 'LineWidth', 1.5);
plot(f_norm, psd_X_dB, '--k', 'LineWidth', 1.5);
grid on; 
axis([-0.5 0.5 -80 5]);
xlabel('Normalized Frequency'); 
ylabel('Normalized PSD (dB)');
title(sprintf('Spectrum Density (ACLR Suppressed by %.2f dB)', aclr_dpd - aclr_base));
legend('Baseline Leakage', 'After DLA Suppression', 'Ideal Source Spectrum', 'Location', 'northeast');

disp('>> Verification Complete: Please review the generated performance plots and console metrics.');