The data in this folder was generated using the convention for angles in Figure 3. In the Phase 1 (Audio processing) - Week1 document. 

Two audio sources are present. One on the left reproducing the file: 'part4_track2_dry.wav'(L) and another one on the right playing 'part4_track1_dry.wav' (R).

A head-mounted microphone array with 4 microphones and a 5-microphone linear array were considered. The speakers can be located at any of the following angles (in degrees):
30, 45, 60 ,90
-30, -45, -60 , -90

Both speakers change their locations after certain time interval. Each time interval is random and they can be found in the file timeIntervals.txt in seconds. 

The location of the left and right speaker at any given time can be found in anglesL.txt and anglesR.txt in degrees, respectively. 

The microphone signals for the head-mounted microphone array can be found in the files:
 yL1.wav, yL2.wav, yR1.wav, yR2.wav
The notation for the microphone signals is the same as in in Figure 3. In the Phase 1 (Audio processing) - Week1 document. 

The microphone signals for the 5-microphone linear array can be found in the files:
 y_LMA_M1.wav, y_LMA_M2.wav, y_LMA_M3.wav, y_LMA_M4.wav, y_LMA_M5.wav
The inter-microphone distance is equal to 5cm. M1 is the microphone closest to -90, while M5 is the closest to 90.

The head mounted microphone array recording of the left and right speaker signals are given in the multichannel wav files sL_HMA.wav and sR_HMA.wav, respectively. The channels in the wav files are organised as follows:
Channel 1 = Left microphone 1
Channel 2 = Left microphone 2
Channel 3 = Right microphone 1
Channel 4 = Right microphone 2

The linear microphone array recording of the left and right speaker signals are given in the multichannel wav files sL_LMA.wav and sR_LMA.wav, respectively. Each channel in the wav files matches the channel of the microphone array i.e. Channel 1 = M1, Channel 2 = M2, etc.


This folder should contain the following files:
- anglesL.txt
- anglesR.txt
- timeIntervals.txt
- y_LMA_M1.wav, y_LMA_M2.wav, y_LMA_M3.wav, y_LMA_M4.wav, y_LMA_M5.wav
- yL1.wav, yL2.wav, yR1.wav, yR2.wav
- sL_HMA.wav, sR_HMA.wav
- sL_LMA.wav, sR_LMA.wav
- readme.txt