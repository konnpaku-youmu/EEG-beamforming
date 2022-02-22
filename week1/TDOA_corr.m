function t_diff = TDOA_corr(num_srcs, mic_sig)
    
    begin = 152000;

    seg_length = 3000;
    test_length = 500;

    ref = mic_sig(begin:begin+seg_length, 1);
    
    t_diff = zeros(num_srcs, size(mic_sig, 2)-1);
    
    for mic_idx = 2:size(mic_sig, 2)

        target = mic_sig(:, mic_idx);

        %% scan the second microphone signal
        for start=begin:begin+test_length
            [ccor, ~,  ~] = crosscorr(ref, target(start:start+length(ref)), 'NumLags', ceil(0.2 * length(ref)));
            [ccor_max, ind] = maxk(ccor, num_srcs);
            if(min(ccor_max) > 0.25)
                %% terminate
                for idx_src = 1:num_srcs
                    t_diff(idx_src, mic_idx-1) = round((length(ccor) + 1)/2) - ind(idx_src);
                    fprintf("TDOA: %d samples\n", t_diff(idx_src, mic_idx-1));
                end
                break;
            end
        end
    end
end
