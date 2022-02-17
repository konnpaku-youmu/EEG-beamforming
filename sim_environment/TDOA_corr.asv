function t_diff = TDOA_corr(mic_sig)

    seg_length = 3000;
    test_length = 500;

    ref = mic_sig(42000:42000+seg_length, 1);
    target = mic_sig(:, 2);

    t_diff = 0;

    %% scan the second microphone signal
    for start=42000:42000+test_length
        [ccor, ~,  ~] = crosscorr(ref, target(start:start+length(ref)), 'NumLags', ceil(0.2 * length(ref)));
        [ccor_max, ind] = max(ccor);
        if(ccor_max > 0.6)
            %% terminate
            t_diff = round((length(ccor) + 1)/2) - ind;
            fprintf("TDOA: %d samples\n", t_diff);
            break;
        end
    end

end