The data in this folder was generated using the convention for angles in 
Figure 3 in the Phase 1 (Audio processing) - Week1 document. 

This audio data corresponds to the EEG data provided and is organized in
the same folder structure (per subject, and per trial).

Two audio sources are present. An attended ("A") speaker, reproducing the
sound from a speech track, and an unattended ("U") speaker playing another 
speech track.

Possible sources location are: left ("L"), right ("R"), or front ("F").
At every instant, one source is in one location and the other is in
another, e.g.: L & R, L & F, or F & R. The two sources are never in the 
same location.

A head-mounted microphone array with 4 microphones and a 5-microphone 
linear array are considered. The speakers can be located at any angle close
to -90 degrees ("L"), 90 degrees ("R"), or 0 degrees ("F").
Both speakers change their locations after certain time intervals.

The microphone signals for the head-mounted microphone array are the files:
 yL1.wav, yL2.wav, yR1.wav, yR2.wav
The notation for the microphone signals is the same as in in Figure 3
in the Phase 1 (Audio processing) - Week1 document.

The microphone signals for the 5-microphone linear array are in the files:
 y_LMA_M1.wav, y_LMA_M2.wav, y_LMA_M3.wav, y_LMA_M4.wav, y_LMA_M5.wav
The inter-microphone distance is equal to 5cm. 
M1 is the closest to -90 degrees, while M5 is closest to 90 degrees.

The head mounted microphone array recording of the left and right speaker 
signals are given in the multichannel wav files sL_HMA.wav and sR_HMA.wav, 
respectively. The channels in the wav files are organised as follows:
Channel 1 = Left microphone 1
Channel 2 = Left microphone 2
Channel 3 = Right microphone 1
Channel 4 = Right microphone 2

The linear microphone array recording of the left and right speaker signals 
are given in the multichannel wav files sL_LMA.wav and sR_LMA.wav, 
respectively. Each channel in the wav files matches the channel of the 
microphone array i.e. Channel 1 = M1, Channel 2 = M2, etc.

Other specifics such as reverberation time, room dimensions, and microphone
positions are given in extraspecs.txt.

This folder should contain the following files:
- y_LMA_M1.wav, y_LMA_M2.wav, y_LMA_M3.wav, y_LMA_M4.wav, y_LMA_M5.wav
- yL1.wav, yL2.wav, yR1.wav, yR2.wav
- sL_HMA.wav, sR_HMA.wav
- sL_LMA.wav, sR_LMA.wav
- extraspecs.txt
- readme.txt