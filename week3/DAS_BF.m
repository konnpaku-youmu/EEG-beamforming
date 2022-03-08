function mic_bf = DAS_BF(mic, speech, noise, target)

    load("Computed_RIRs.mat");
    num_mics = size(RIR_sources, 2);
    num_srcs = size(RIR_sources, 3);
    
    DOA_est = MUSIC_wideband(mic);
    
    %% find the target
    [~, idx] = min(abs(DOA_est - 90));
    target_DoA = deg2rad(DOA_est(idx));
    
    %% calc the delay
    dist = zeros(size(m_pos, 1), 1);
    v_sound = 340;
    for idx=1:size(m_pos, 1)
        dist(idx) = norm(m_pos(idx, :) - m_pos(size(m_pos, 1), :));
    end
    delays_in_samples = round(fs_RIR * (dist * cos(target_DoA) / v_sound));
    
    %% calc the SNR after steering
    speech_delay = sum(delayseq(speech, -delays_in_samples), 2);
    noise_delay = sum(delayseq(noise, -delays_in_samples), 2);
    
    VAD=abs(speech_delay(:,1))>std(speech_delay(:,1))*1e-3;
    speech_active = speech_delay(VAD==1, 1);
    speech_pow = var(speech_active);
    
    SNR_steering = 10 * log10(speech_pow / var(noise_delay));
    fprintf("SNR after DAS: %2.2f\n", SNR_steering);
    
    mic_delay = delayseq(mic, -delays_in_samples);
    mic_bf = sum(mic_delay, 2) / num_mics;
end

