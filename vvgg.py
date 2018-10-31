# Some ideas inspired by this video, 
# if it is possible to use a convolutional neural network approach
# for the voice recognition application

# Transfer Learning - Use Inception V3 to Solve Any ML Problem (Tempo Detection)
# https://www.youtube.com/watch?v=jk0Qh7Iva_s

import os
import math
import numpy as np
import scipy.io.wavfile
import matplotlib.pyplot as plt

waves = [
  "good.wav",
  "morning.wav",
  "thanks.wav"
]

os.system("mkdir output")

MAX_SIZE = 20000
SPLIT_NUM = 20
for filename in waves:
  plt.clf()
  wv = scipy.io.wavfile.read(os.path.join("data", filename))
  waveLen = len(wv[1])
  paddedWave = np.concatenate((wv[1], np.zeros(MAX_SIZE-waveLen)))
  splitSize = MAX_SIZE // SPLIT_NUM
  halfSplitSize = MAX_SIZE // SPLIT_NUM // 2
  
  for i in range(SPLIT_NUM):
    # split the wave into basic unit
    currWave = paddedWave[i*splitSize:(i+1)*splitSize]
    currFft  = np.fft.fft(currWave)
    currFreq = np.fft.fftfreq(currFft.size)
    
    # get the positive part, and move the display position to next 0.5
    currFft  = currFft[:halfSplitSize] / 1e7
    currFreq = currFreq[:halfSplitSize] + i * 0.5
    
    # just plot the real part
    plt.plot(currFreq, currFft.real)

  plt.xlim(left=0, right=SPLIT_NUM * 0.5)
  plt.ylim(top=0.05, bottom=0)
  plt.savefig("output/%s.png" % filename)
