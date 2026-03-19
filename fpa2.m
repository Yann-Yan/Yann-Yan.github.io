function y = fpa2(x, addNoise)
% FPA2  Forward Power Amplifier Model (Memory Polynomial)
%
% Description: Acts as the digital twin of the physical RF Power Amplifier.
% This function is utilized directly within the DLA iterative training loop
% to compute hardware feedback, and during final closed-loop verification.
%
% Input:
%   x         : Complex baseband input signal (pre-distorted signal)
%   addNoise  : Boolean flag to inject AWGN (default = false)
%
% Output:
%   y         : Distorted PA output signal

%% ----------------------------------------------------------
% 0. Default Noise Flag
%% ----------------------------------------------------------
% Noise is disabled by default to ensure clean error calculation 
% and gradient stability during the neural network's iterative training.
if nargin < 2
    addNoise = false;   
end
x = x(:);
N = length(x);

%% ----------------------------------------------------------
% 1. Normalize Input [CRITICAL EXPLANATION]
%% ----------------------------------------------------------
% Normalization is intentionally DISABLED here. In closed-loop DPD 
% operation, the absolute amplitude level (including the expanded 
% pre-distortion peaks) must be preserved and fed directly into the PA. 
% Internal rescaling here would destroy the pre-distortion mapping.

%%if max(abs(x)) > 0
    %%x = x / max(abs(x));
%%end

%% ----------------------------------------------------------
% 2. Static Nonlinear Coefficients
% Defines the memoryless AM-AM and AM-PM compression characteristics.
% Operating in a moderate compression, fully invertible region.
%% ----------------------------------------------------------
beta1 = 0.9  + 0.00i;
beta3 = -0.35 - 0.05i;
beta5 = -0.02 + 0.01i;

%% ----------------------------------------------------------
% 3. Light Memory Effect (1-Tap Delay)
% Introduces frequency-dependent phase scattering.
%% ----------------------------------------------------------
alpha1 = 0.05 - 0.01i;
alpha3 = -0.03 + 0.005i;

%% ----------------------------------------------------------
% 4. Memory Polynomial Core Logic
%% ----------------------------------------------------------
y = zeros(N,1);
for n = 2:N
    
    xn  = x(n);      % Instantaneous input
    xm1 = x(n-1);    % 1-tap delayed input (Memory depth M=1)
    
    % Compute static nonlinear contribution
    y_static = beta1 * xn ...
             + beta3 * xn * abs(xn)^2 ...
             + beta5 * xn * abs(xn)^4;
    
    % Compute memory-induced distortion contribution
    y_memory = alpha1 * xm1 ...
             + alpha3 * xm1 * abs(xm1)^2;
    
    % Superposition of effects to generate the final distorted output
    y(n) = y_static + y_memory;
end

%% ----------------------------------------------------------
% 5. Optional Controlled AWGN
%% ----------------------------------------------------------
% Used strictly during final verification to evaluate the DPD's 
% robustness against realistic hardware noise floors.
if addNoise
    
    SNR_dB = 40;   % Adjustable Signal-to-Noise Ratio
    
    signal_power = mean(abs(y).^2);
    noise_power  = signal_power / (10^(SNR_dB/10));
    
    noise = sqrt(noise_power/2) * ...
           (randn(size(y)) + 1j*randn(size(y)));
    
    y = y + noise;
end
end