# Define some helper functions for pretty figures.
import csv
import librosa
import librosa.display
import numpy as np
from IPython.display import Audio, display
from audiomentations import *
from scipy.io.wavfile import write
import IPython.display as ipd

# Reference: https://medium.com/@keur.plkar/audio-data-augmentation-in-python-a91600613e47

# Noise Injection

'''
Noise addition using normal distribution with mean = 0 and std =1

Permissible noise factor value = x > 0.004
'''
def inject_noise(data){
    wav, sr = librosa.load(data, sr=16000)
    augmented_data = wav + 0.009 * np.random.normal(0,1,len(wav))
    # Cast back to same data type
    augmented_data = augmented_data.astype(type(data[0]))
    ipd.Audio(data=wav_n,rate=sr)
    librosa.output.write_wav('./noise_add.wav',augmented_data,sr)
    return augmented_data
}

# Shifting Time
'''
Permissible factor values = sr/10
'''
def shift_time(data){
    wav, sr = librosa.load(data, sr=16000)
    augmented_data = np.roll(wav,int(sr/10))
    ipd.Audio(wav_roll,rate=sr)
    librosa.output.write_wav('./shiftedtime.wav',wav_roll,sr)
    return augmented_data
}

# Time Stretching
'''
Permissible factor values = 0 < x < 1.0
'''
def time_stretch(data){
    wav, sr = librosa.load(data, sr=16000)
    augmented_data = librosa.effects.time_stretch(wav,0.4)
    ipd.Audio(wav_time_stch,rate=sr)
    librosa.output.write_wav('./time_stretch.wav',wav_time_stch,sr)
    return augmented_data
}

# Shifting Pitch
'''
Permissible factor values = -5 <= x <= 5
'''
def shift_pitch(data){
    wav, sr = librosa.load(data, sr=16000)
    augmented_data = librosa.effects.pitch_shift(wav,sr,n_steps=-5)
    ipd.Audio(wav_pitch_sf,rate=sr)
    librosa.output.write_wav('./pitch_shift.wav',wav_pitch_sf,sr)
    return augmented_data
}

