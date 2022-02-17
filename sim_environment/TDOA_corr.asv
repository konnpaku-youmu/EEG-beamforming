function t_diff = TDOA_corr(mic_sig)

    seg_length = 500;
    test_length = 100;

    ref = mic_sig(1:seg_length, 1);
    target = mic_sig(:, 2);

    t_diff = 0;

    %% scan the second microphone signal
    for start=1:test_length
        [ccor, ~,  ~] = crosscorr(ref, target(start:start+length(ref)), 'NumLags', ceil(0.1 * length(ref)));
        [ccor_max, ind] = max(ccor);
        if(ccor_max > 0.4)
            %% terminate
            t_diff = ind - (ceil(test_length/2)+start);
            figure
            stem(ccor);
            fprintf("TDOA: %d samples\n", t_diff);
            break;
        end
    end

end