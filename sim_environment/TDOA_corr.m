mic = generate_micsig(2, 1, 5);

ref = mic(1:500, 1);
target = mic(:, 2);

%% scan the second microphone signal
for start=1:100
    crosscorr(ref, target(start:start+length(ref)), 'NumLags', ceil(0.1 * length(ref)));
    if(max(ccor) > 0.75)
        %% terminate
        break;
    end
end

function mic = generate_micsig(num_mics, num_srcs, duration)
    
    load("Computed_RIRs.mat");

    Fs = fs_RIR;
    sig_len = duration * Fs;
    
    %% generate a white noise signal
    test_sig = wgn(sig_len, num_srcs, 0.01);

    %% create the array for recorded signal
    test_rec = zeros(sig_len, num_mics);
    
    for i=1:num_srcs
        test_rec = test_rec + fftfilt(RIR_sources(:, :, i), test_sig(:, i));
    end

    mic = test_rec;
end