%% ==========================================================
% Clean & Physically Realistic Memory Polynomial PA
% DPD-Friendly Version
% Description: Simulates a non-linear Power Amplifier (PA) with 
% memory effects to serve as the baseline testbed for DPD algorithms.
% ==========================================================
clear; clc; close all;

%% ==========================================================
% 1. Load Data
% ==========================================================
% [USER ACTION REQUIRED] Please specify the directory containing the source data
dataFolder = 'To be filled in'; 
filePath = fullfile(dataFolder, 'PAData_200kBW_1MSps.mat');

if ~exist(filePath,'file')
    error('File loading failed. Please check the dataFolder path.');
end
load(filePath);   % Assumes variable 'x' (baseband input signal) exists in the workspace
x = x(:);
N = length(x);

%% ==========================================================
% 2. Normalize Input [CRITICAL]
% ==========================================================
% Strictly normalize the input to a peak amplitude of 1.0. 
% This prevents unpredictable hard saturation and ensures the PA 
% operates within a controllable compression region.
x = x / max(abs(x));   

%% ==========================================================
% 3. Static Nonlinear Coefficients
% ==========================================================
% Defines the memoryless nonlinear characteristics (AM-AM & AM-PM).
% These coefficients yield moderate compression that is fully invertible by DPD.
beta1 = 0.9 + 0.00i;     % Linear gain
beta3 = -0.35 - 0.05i;   % 3rd-order intermodulation distortion
beta5 = -0.02 + 0.01i;   % 5th-order intermodulation distortion

%% ==========================================================
% 4. Light Memory Effect
% ==========================================================
% Simulates frequency-dependent physical behavior (Memory Depth M=1).
% This causes the characteristic scattering in the AM-PM response.
alpha1 = 0.05 - 0.01i;
alpha3 = -0.03 + 0.005i;

%% ==========================================================
% 5. Memory Polynomial PA Model
% ==========================================================
% Core loop: A truncated Volterra series combining static polynomial 
% distortion and delayed tap interactions.
y = zeros(N,1);
for n = 2:N
    
    xn  = x(n);      % Current input sample
    xm1 = x(n-1);    % Previous input sample (memory tap)
    
    % Compute instantaneous static nonlinearity
    y_static = beta1 * xn ...
             + beta3 * xn * abs(xn)^2 ...
             + beta5 * xn * abs(xn)^4;
    
    % Compute memory-induced distortion
    y_memory = alpha1 * xm1 ...
             + alpha3 * xm1 * abs(xm1)^2;
    
    % Superposition of static and memory effects
    y(n) = y_static + y_memory;
end

%% ==========================================================
% 6. Controlled Small AWGN (DPD removable floor)
% ==========================================================
% Injects Additive White Gaussian Noise to simulate realistic hardware floors.
SNR_dB = 40; % Maintained at a high 40dB SNR so that nonlinear distortion remains the dominant impairment
signal_power = mean(abs(y).^2);
noise_power  = signal_power / (10^(SNR_dB/10));
noise = sqrt(noise_power/2) * ...
       (randn(size(y)) + 1j*randn(size(y)));
y = y + noise;

%% ==========================================================
% 7. Visualization
% ==========================================================
% Extracts and plots the fundamental PA characteristics
in_amp  = abs(x);
out_amp = abs(y);
phase_dist = unwrap(angle(y) - angle(x));
phase_dist = phase_dist - median(phase_dist(in_amp < 0.1)); % Phase alignment

figure('Color','w','Position',[100 100 1000 420]);

% AM-AM Plot (Amplitude Compression)
subplot(1,2,1);
scatter(in_amp,out_amp,2,[0.2 0.2 0.2], ...
        'filled','MarkerFaceAlpha',0.15);
grid on;
xlabel('Input Amplitude');
ylabel('|Output Amplitude|');
title('AM-AM Response');
ylim([0 1.2]);

% AM-PM Plot (Phase Scattering due to Memory)
subplot(1,2,2);
scatter(in_amp,phase_dist,2,[0.8 0.2 0.2], ...
        'filled','MarkerFaceAlpha',0.15);
grid on;
xlabel('Input Amplitude');
ylabel('Phase Distortion (rad)');
title('AM-PM Response');
ylim([-0.4 0.4]);

sgtitle('Baseline PA Characteristics: Memory Polynomial Model');

%% ==========================================================
% 8. Save
% ==========================================================
% [USER ACTION REQUIRED] Please specify the destination directory for the generated PA output
saveFolder = 'To be filled in'; 

if ~exist(saveFolder,'dir')
    mkdir(saveFolder);
end
savePath = fullfile(saveFolder,'nPA.mat');
save(savePath,'x','y');
disp('>> Success: Distorted PA data exported as nPA.mat');
