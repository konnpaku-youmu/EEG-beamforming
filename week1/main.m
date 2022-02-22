% %% Part 2: Microphone signals
% 
% clear;
% 
% speech_files = ["speech1.wav"];
% noise_files = [];
% 
% mic = create_micsigs(2, speech_files, noise_files, 5);
% 
% figure
% plot(mic(:, 1), 'DisplayName', 'Mic 1');
% hold on
% plot(mic(:, 2), 'DisplayName', 'Mic 2');
% legend

%% Part 3: TDOA

clear;

speech_files = ["part1_track1_dry.wav"];
noise_files = [];

mic = create_micsigs(2, speech_files, noise_files, 10);

t_diff = TDOA_corr(mic);
