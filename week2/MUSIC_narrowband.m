clc;
clear;
load("Computed_RIRs.mat");

assert(fs_RIR == 44100);
num_mics = size(RIR_sources, 2);

speech_files = ["speech1.wav", "speech2.wav"];
noise_files = ["Babble_noise1.wav"];

mic = create_micsigs(num_mics, speech_files, noise_files, 10);

%% STFT
fft_length = 1024;
[spectrogram, ~, ~] = stft(mic, 'OverlapLength', 50, 'FFTLength', fft_length, 'FrequencyRange', 'onesided');
spectrogram = permute(spectrogram, [3, 1, 2]);

power_mag = mean(mean(abs(spectrogram).^2, 3), 1);
[~, bin_idx]= max(power_mag);
freq_bin = permute(spectrogram(:, bin_idx, :), [1, 3, 2]);

[E, D] = eig(cov(freq_bin'));
E = E(:, 3);

%% define the steering matrix
v_sound = 340;
f_max = 2 * bin_idx / fft_length * (fs_RIR / 2);
omega_max = 2*pi*f_max;

theta = 0 : 0.5 : 180;
theta = theta.*(pi/180);

d = zeros(size(m_pos, 1), 1);
for idx=1:size(m_pos, 1)
    d(idx) = norm(m_pos(idx, :) - m_pos(size(m_pos, 1), :));
end

% steering matrix
G = exp(-1i * omega_max * d * cos(theta) / v_sound);

%% pseudospectrum
p = 1./diag((G'*E)*E'*G);
figure
plot(abs(p));


[~, DOA_est] = maxk(p, 2);
DOA_est = DOA_est*0.5;
save("DOA_est.mat", "DOA_est");

