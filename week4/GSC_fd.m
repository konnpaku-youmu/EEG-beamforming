clc;
clear;
load("Computed_RIRs.mat");

assert(fs_RIR == 44100);
num_mics = size(RIR_sources, 2);
num_srcs = size(RIR_sources, 3);

speech_files = ["part1_track1_dry.wav", "part1_track2_dry.wav"];
noise_files = [];

[mic, speech, noise] = create_micsigs(num_mics, speech_files, noise_files, 10, true);

%% DOA estimation
DOA_est = MUSIC_wideband(mic);

%% STFT of mic signal
dft_l = 1024;
window = hamming(dft_l);
spectro_mic = stft(mic, fs_RIR, 'Window', window, 'OverlapLength', 512, 'FFTLength', dft_l);

%% Normalized DFT of RIR
a_omega = fft(RIR_sources, dft_l, 1);
h_omega = a_omega ./ a_omega(:, 1);

%% filter and sum
w_fas = zeros(size(h_omega, 1), size(h_omega, 2));
for freq_bin=1:length(h_omega)
    w_fas(freq_bin, :) = h_omega(freq_bin, :) ./ (h_omega(freq_bin, :) * h_omega(freq_bin, :)');
end

%% adaptive filter: updating
mu = 0.2;
alpha = 1e-5;

err = zeros(dft_l, size(spectro_mic, 2));
d = zeros(dft_l, size(spectro_mic, 2));
for freq_bin=1:dft_l/2
    W = zeros(4, 1); % fixed size for testing
    %% solve for the blocking matrix
    B = null(h_omega(freq_bin, :));
    
    for k=1:size(spectro_mic, 2)
        y_omega = permute(spectro_mic(freq_bin, k, :), [3, 2, 1]);
        w_fas_omega = w_fas(freq_bin, :);
        %% NLMS filter: priori
        d(freq_bin, k) = w_fas_omega * y_omega;
        
        %% noise reference
        n_ref = B' * y_omega;
        err(freq_bin, k) = d(freq_bin, k) - W' * n_ref;
        W = W + (mu / (n_ref'*n_ref + alpha)) * n_ref * conj(err(freq_bin, k));
    end
end

err(513:end-1, :) = conj(flipud(err(1:511, :)));
err(512, :) = 0;
err(end, :) = 0;

d(513:end-1, :) = conj(flipud(d(1:511, :)));
d(512, :) = 0;
d(end, :) = 0;

fas_out = istft(d, fs_RIR, 'Window', window, 'OverlapLength', 512, 'FFTLength', dft_l);
gsc_speech = istft(err, fs_RIR, 'Window', window, 'OverlapLength', 512, 'FFTLength', dft_l);

figure
hold on
% plot(speech(:, 1), 'DisplayName', 'Speech');
plot(mic(:, 1), 'DisplayName', 'Microphone');
plot(fas_out, 'DisplayName', 'W_{fas} output');
plot(gsc_speech, 'DisplayName', 'GSC output');

xlabel('Samples');
ylabel('Amplitude');
title(sprintf('Frequency-domain GSC: t_{60} = %2.2f', rev_time));
legend
