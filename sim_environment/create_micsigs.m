function mic = create_micsigs(num_mics, speech_files, noise_files, duration)
    
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
            
        end
        
        speech_wav(:, i) = speech;
        
        %% pass through the channel
        speech_rec = speech_rec + fftfilt(RIR_sources, speech_wav(:, i));
    end
    
    for i=1:length(noise_files)
        [noise, f_sn] = audioread(noise_files(i));
        %% truncate the noise
        noise = noise(1:duration * f_sn);
        
        %% resample if needed
        if(f_sn ~= Fs)
            
        end
        
        noise_wav(:, i) = noise;
        
        %% pass through the channel
        noise_rec = noise_rec + fftfilt(RIR_noise, noise_wav(:, i));
    end
    
    %% add noise to speech
    mic = speech_rec + noise_rec;
end