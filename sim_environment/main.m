clear;

speech_files = ["speech1.wav", "speech2.wav"];
noise_files = ["Babble_noise1.wav"];

create_micsigs(3, speech_files, noise_files, 1.5);
