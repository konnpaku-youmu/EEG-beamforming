function [mic, speech_rx, noise_rx] = create_micsigs(num_mics, speech_files, noise_files, duration, isNoisy)
    
    load("Computed_RIRs.mat");

    Fs = fs_RIR;
    sig_len = duration * Fs;
    
    %% create the array for audio
    speech_wav = zeros(sig_len, length(speech_files));
    noise_wav = zeros(sig_len, length(noise_files));

    %% create the array for recorded signal
    speech_rec = zeros(sig_len, num_mics);
    noise_rec = zeros(sig_len, num_mics);
    
    for i=1:length(speech_files)
        [speech, f_ss] = audioread(speech_files(i));

        %% truncate the audio
        speech = speech(1:duration * f_ss);
        
        %% resample if needed
        if(f_ss ~= Fs)
            speech = resample(speech, Fs, f_ss);
        end
        
        speech_wav(:, i) = speech;
 
        %% pass through the channel
        speech_rec = speech_rec + fftfilt(RIR_sources(:, :, i), speech_wav(:, i));
    end
    
    if(isNoisy)
        %% find active segments & calculate the power (1st microphone)
        % speech power
        VAD=abs(speech_rec(:,1))>std(speech_rec(:,1))*1e-3;
        speech_active = speech_rec(VAD==1, 1);
        speech_pow = var(speech_active);
    
        % noise
        additive_noise = wgn(size(speech_rec, 1), size(speech_rec, 2), 0.1*speech_pow, 'linear');
        
        % SNR
        SNR_mic = 10 * log10(speech_pow / var(additive_noise(:, 1)));
        fprintf('SNR@Microphone: %2.2f\n', SNR_mic)
        
        speech_rec = speech_rec + additive_noise;
    end

    for i=1:length(noise_files)
        [noise, f_sn] = audioread(noise_files(i));
        %% truncate the noise
        noise = noise(1:duration * f_sn);
        
        %% resample if needed
        if(f_sn ~= Fs)
            noise = resample(noise, Fs, f_sn);
        end
        
        noise_wav(:, i) = noise;
        
        %% pass through the channel
        noise_rec = noise_rec + fftfilt(RIR_noise(:, :, i), noise_wav(:, i));
    end
    
    speech_rx = speech_rec;
    noise_rx = additive_noise;
    %% add noise to speech
    mic = speech_rec + noise_rec;
end
