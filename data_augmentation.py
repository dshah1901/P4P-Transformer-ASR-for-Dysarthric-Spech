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
# https://medium.com/@makcedward/data-augmentation-for-audio-76912b01fdf6 

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
def shift_time(wav){
    wav, sr = librosa.load(data, sr=16000)
    augmented_data = np.roll(wav,int(sr/10))
    ipd.Audio(wav_roll,rate=sr)
    librosa.output.write_wav('./shiftedtime.wav',wav_roll,sr)
    return augmented_data
}

# Changing Pitch

def change_pitch(){
    return librosa.effects.pitch_shift(data, sampling_rate, pitch_factor)
}

# Time Stretching
def time_stretching(){
    augment = Compose([
        TimeStretch(min_rate=0.8, max_rate=1.25, p=0.5),
    ])
    augmented_audio = augment(samples=wav, sample_rate=16000)
}