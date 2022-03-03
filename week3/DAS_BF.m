clc;
clear;
load("Computed_RIRs.mat");

assert(fs_RIR == 44100);
num_mics = size(RIR_sources, 2);
num_srcs = size(RIR_sources, 3);

speech_files = ["speech1.wav", "speech2.wav"];
noise_files = ["white"];

mic = create_micsigs(num_mics, speech_files, noise_files, 10);

soundsc(mic(:, 1), fs_RIR);