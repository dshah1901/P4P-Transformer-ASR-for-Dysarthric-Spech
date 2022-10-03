# Define some helper functions for pretty figures.
import csv
import librosa
import librosa.display
import numpy as np
from IPython.display import Audio, display
from audiomentations import *
from scipy.io.wavfile import write
import IPython.display as ipd
import soundfile as sf


# Reference: https://medium.com/@keur.plkar/audio-data-augmentation-in-python-a91600613e47

# Noise Injection

'''
Noise addition using normal distribution with mean = 0 and std =1

Permissible noise factor value = x > 0.004
'''
def inject_noise(data):
    wav, sr = librosa.load(data, sr=16000)
    augmented_data = wav + 0.009 * np.random.rand(len(wav))
    # Cast back to same data type
    augmented_data = augmented_data.astype(type(wav[0]))
    speaker, block, word_key, mic = data.split('_')
    speaker = data.split('/')[4]
    mic, extention =  mic.split('.')
    augmented_audio = "./UASPEECH/audio/DataAugmentation/"+speaker+"/"+speaker+"_"+block+"_"+word_key+"_"+mic+"_"+"injected_noise." +extention
    sf.write(augmented_audio, augmented_data, sr, 'PCM_16')
    return augmented_audio


# Shifting Time
'''
Permissible factor values = sr/10
'''
def shift_time(data):
    wav, sr = librosa.load(data, sr=16000)
    augmented_data = np.roll(wav,int(sr/10))
    speaker, block, word_key, mic = data.split('_')
    speaker = data.split('/')[4]
    mic, extention =  mic.split('.')
    augmented_audio = "./UASPEECH/audio/DataAugmentation/"+speaker+"/"+speaker+"_"+block+"_"+word_key+"_"+mic+"_"+"shifted_time." +extention
    sf.write(augmented_audio, augmented_data, sr, 'PCM_16')
    return augmented_audio


# Time Stretching
'''
Permissible factor values = 0 < x < 1.0
'''
def time_stretch(data):
    wav, sr = librosa.load(data, sr=16000)
    augmented_data = librosa.effects.time_stretch(wav,0.4)
    speaker, block, word_key, mic = data.split('_')
    speaker = data.split('/')[4]
    mic, extention =  mic.split('.')
    augmented_audio = "./UASPEECH/audio/DataAugmentation/"+speaker+"/"+speaker+"_"+block+"_"+word_key+"_"+mic+"_"+"stretched_time." +extention
    sf.write(augmented_audio, augmented_data, sr, 'PCM_16')
    return augmented_audio


# Shifting Pitch
'''
Permissible factor values = -5 <= x <= 5
'''
def shift_pitch(data):
    wav, sr = librosa.load(data, sr=16000)
    augmented_data = librosa.effects.pitch_shift(wav,sr,n_steps=-5)
    speaker, block, word_key, mic = data.split('_')
    speaker = data.split('/')[4]
    mic, extention =  mic.split('.')
    augmented_audio = "./UASPEECH/audio/DataAugmentation/"+speaker+"/"+speaker+"_"+block+"_"+word_key+"_"+mic+"_"+"shifted_pitch." +extention
    sf.write(augmented_audio, augmented_data, sr, 'PCM_16')
    return augmented_audio
