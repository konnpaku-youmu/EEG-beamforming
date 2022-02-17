clear;
load("Computed_RIRs.mat");
v_sound = 340;

speech_files = ["part1_track1_dry.wav"];
noise_files = [];

mic = create_micsigs(2, speech_files, noise_files, 10);

t_diff = TDOA_corr(mic);
t_diff = t_diff / fs_RIR;

d_dist = t_diff * v_sound;
d_mic = norm(m_pos(1, :) - m_pos(2, :));

DOA_est = rad2deg(acos(d_dist / d_mic));

save("DOA_est.mat", "DOA_est");
