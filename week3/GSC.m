clc;
clear;
load("Computed_RIRs.mat");

assert(fs_RIR == 44100);
num_mics = size(RIR_sources, 2);
num_srcs = size(RIR_sources, 3);

speech_files = ["speech1.wav"];
noise_files = [];

[mic, speech, noise] = create_micsigs(num_mics, speech_files, noise_files, 10, true);

mic_bf = DAS_BF(mic, speech, noise, 0);

blocking_mat = zeros(num_mics-1, num_mics);
blocking_mat(:, 1) = 1;
blocking_mat(:, 2:end) = -eye(num_mics-1);

noise_reference = blocking_mat * mic';

W = zeros(num_mics, num_mics-1);



