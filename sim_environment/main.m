clear;

speech_files = ["speech1.wav", "speech2.wav"];
noise_files = ["Babble_noise1.wav"];

mic = create_micsigs(3, speech_files, noise_files, 3);

soundsc(mic(:, 1), 44100);
