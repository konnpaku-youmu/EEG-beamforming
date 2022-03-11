clc;
clear;
load("Computed_RIRs.mat");

assert(fs_RIR == 44100);
num_mics = size(RIR_sources, 2);
num_srcs = size(RIR_sources, 3);

speech_files = ["speech1.wav"];
noise_files = [];

[mic, speech, noise] = create_micsigs(num_mics, speech_files, noise_files, 5, true);

[mic_delay, DAS_out] = DAS_BF(mic, speech, noise, 0);

blocking_mat = zeros(num_mics-1, num_mics);
blocking_mat(:, 1) = 1;
blocking_mat(:, 2:end) = -eye(num_mics-1);

noise_reference = mic_delay * blocking_mat';

%% Adaptive filter
Length = 1024;
mu = 0.1;
alpha = 1e-5;
% desired signal: DAS-BF with a delay
delta = Length / 2;
d = delayseq(delta, DAS_out)';
W = ones(Length, num_mics - 1);

t = zeros(size(d, 1), 1);
err = zeros(size(d, 1), 1);
for i=Length:size(noise_reference, 1)
    u = noise_reference(i-Length+1:i, :);
    uhu = norm(u);
    t(i) = trace(W'*u);
    err(i) = conj(d(i) - t(i));
    W = W + (mu / (alpha + uhu^2))*u*err(i);
end

plot(err);
