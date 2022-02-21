clear;
load("Computed_RIRs.mat");
v_sound = 340;

speech_files = ["whitenoise_signal_1.wav", "whitenoise_signal_2.wav"];
noise_files = [];

mic = create_micsigs(5, speech_files, noise_files, 10);

t_diff = TDOA_corr(2, mic);
t_diff = t_diff / fs_RIR;

d_dist = t_diff * v_sound;

d_mic = zeros(size(t_diff, 2) - 1, 2);
for mic_idx = 2:size(m_pos, 1)
    d_mic(mic_idx-1, 1) = norm(m_pos(1, :) - m_pos(mic_idx, :));
    d_mic(mic_idx-1, 2) = d_mic(mic_idx-1, 1);
end

DOA_est = mean(rad2deg(acos(d_dist ./ d_mic')), 2);

save("DOA_est.mat", "DOA_est");
