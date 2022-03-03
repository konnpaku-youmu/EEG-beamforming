clc;
clear;
load("Computed_RIRs.mat");
v_sound = 340;

assert(fs_RIR == 44100);
num_mics = size(RIR_sources, 2);
num_srcs = size(RIR_sources, 3);

speech_files = ["speech1.wav", "speech2.wav"];
noise_files = ["Babble_noise1.wav"];

mic = create_micsigs(num_mics, speech_files, noise_files, 10);

DOA_est = MUSIC_wideband(mic);

%% find the target
[~, idx] = min(abs(DOA_est - 90));
target_DoA = deg2rad(DOA_est(idx));

%% calc the delay
dist = zeros(size(m_pos, 1), 1);
for idx=1:size(m_pos, 1)
    dist(idx) = norm(m_pos(idx, :) - m_pos(size(m_pos, 1), :));
end
delays_in_samples = round(fs_RIR * (dist * cos(target_DoA) / v_sound));

mic_delay = delayseq(mic, -delays_in_samples);
mic_sum = sum(mic_delay, 2);

soundsc(mic_sum, fs_RIR);