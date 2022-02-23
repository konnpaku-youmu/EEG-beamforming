clc;
clear;
load("Computed_RIRs.mat");

assert(fs_RIR == 44100);
num_mics = size(RIR_sources, 2);
num_srcs = size(RIR_sources, 3);

speech_files = ["speech1.wav", "speech2.wav"];
noise_files = [];

mic = create_micsigs(num_mics, speech_files, noise_files, 15);

%% STFT
fft_length = 1024;
[spectrogram, ~, ~] = stft(mic, 'OverlapLength', 50, 'FFTLength', fft_length, 'FrequencyRange', 'onesided');
spectrogram = permute(spectrogram, [3, 1, 2]);

power_mag = mean(mean(abs(spectrogram).^2, 3), 1);
[~, bin_idx]= max(power_mag);
freq_bin = permute(spectrogram(:, bin_idx, :), [1, 3, 2]);

[E, D] = eig(cov(freq_bin'));
null_dim = size(E, 1)-num_srcs;
E = E(:, 1:null_dim);

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
subplot(2,1,1);
plot(abs(p));
hold on

%% CFAR
cfar_p = cfar_detection(abs(p), 1.05, 9);
subplot(2,1,2);
plot(cfar_p);

cfar_idx = find(cfar_p);

cfar_idx = sort(cfar_idx);
bound = ones(num_srcs, 1);
src_idx = 1;
for i=2:length(cfar_idx)
    if abs(cfar_idx(i) - cfar_idx(i-1)) > 2
        bound(src_idx+1) = i-1;
        src_idx = src_idx + 1;
    end
end

DOA_est = zeros(num_srcs, 1);
for i=1:num_srcs
    if(i<num_srcs)
        DOA_est(i) = mean(cfar_idx(bound(i):bound(i+1)))/2;
    else
        DOA_est(i) = mean(cfar_idx(bound(i)+1:end))/2;
    end
end

save("DOA_est.mat", "DOA_est");

function cfar = cfar_detection(series, threshold, window_size)
    assert(window_size >= 3 && mod(window_size, 2) == 1)
    start = ceil(window_size / 2);
    stop = length(series) - start + 1;
    win_size_half = floor(window_size / 2);
    win_mid_pos = ceil(window_size / 2);

    cfar = zeros(size(series));
    for i=start:stop
        window = series(i-win_size_half : i+win_size_half);
        candidate = window(win_mid_pos);
        avg_adj_pow = (sum(window) - window(win_mid_pos)) / (2*win_size_half);
        if(candidate < threshold * avg_adj_pow)
            cfar(i) = 0;
        else
            cfar(i) = 1;
        end
    end
end
