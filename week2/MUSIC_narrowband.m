clear;
load("Computed_RIRs.mat");

assert(fs_RIR == 44100);
num_mics = size(RIR_sources, 2);

speech_files = ["speech1.wav"];
noise_files = [];

mic = create_micsigs(num_mics, speech_files, noise_files, 10);
[spectrogram, ~, ~] = stft(mic, 'OverlapLength', 50, 'FFTLength', 1024);

spectrogram = spectrogram.^2;

avg_freq = mean(mean(spectrogram, 2), 3);

plot(10 * log10(abs(avg_freq)));

