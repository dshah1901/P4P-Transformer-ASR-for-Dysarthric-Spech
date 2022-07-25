import os
import random
from glob import glob
from tkinter import Label
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import re
import tensorflow_io as tfio
import numpy as np
import pandas as pd
import wave
from scipy.io import wavfile
import contextlib
import librosa 
import soundfile as sf

def get_audio_path(labels):
    return glob("./datasets/speech_commands_v0.02.tar/speech_commands_v0.02/{}/*.wav".format(labels), recursive=True)

def get_dataset(labels):
    data =[]
    wavs = []
    durations =[]
    testing_data =[]
    testing = []
    validation_list = open("./datasets/speech_commands_v0.02.tar/speech_commands_v0.02/testing_list.txt")
    validation_files = validation_list.read().splitlines()
    for name in validation_files:
        description = name.split("/")
        full_sen = "./datasets/speech_commands_v0.02.tar/speech_commands_v0.02/" + name
        sample_rate, audio = wavfile.read(full_sen)
        with contextlib.closing(wave.open(full_sen,'r')) as wavfiless:
            frames = wavfiless.getnframes()
            rate = wavfiless.getframerate()
            duration = frames / float(rate)
            if (sample_rate == 16000 and duration >= 1.0):
                testing_data.append(full_sen)
                testing.append({"audio": full_sen, "text": description[0]})

    for speaker in labels:
        wavs += get_audio_path(speaker)

    for wav in wavs:
            description = wav.split("/")
            label = description[4] 
            sample_rate, audio = wavfile.read(wav)
            with contextlib.closing(wave.open(wav,'r')) as wavfiles:
                frame = wavfiles.getnframes()
                rate = wavfiles.getframerate()
                duration = frame/float(rate)
            if (sample_rate == 16000 and duration >= 1.0):
                data.append({"audio": wav, "text": label})
    
    res = [d for d in data if d['audio'] not in testing_data]

    print(sum(1 for d in res if d)) #training data
    print("testing data size")
    print(sum(1 for d in testing if d)) #All Data
    return res,testing
#LABELS = ['bed']
LABELS = ['backward','bed','bird','cat','dog','down','eight','five','follow','four','go','happy','house','learn','left','marvin','nine','no','off','on','right','seven','sheila','six','stop','three','tree','two','up','visual','wow','yes','zero']
#LABELS = ['bed','bird','cat','dog','down','eight','five','four','go','happy','house','left','marvin','nine','no','off','on','one','right','seven','sheila','six','stop','three','tree','two','up','wow','yes','zero']
"""