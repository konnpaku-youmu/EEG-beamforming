clear;

speech_files = ["speech1.wav"];
noise_files = [];

mic = create_micsigs(2, speech_files, noise_files, 5);

figure
plot(mic(:, 1), 'DisplayName', 'Mic 1');
hold on
plot(mic(:, 2), 'DisplayName', 'Mic 2');
legend

% soundsc(mic, 44100);
