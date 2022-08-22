from enum import unique
import math
from multiprocessing.dummy import Value
# from multiprocessing.reduction import duplicate
import os
import random
from glob import glob
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import re
import pandas as pd
import numpy as np
import librosa
import IPython.display as ipd
import random
from pydub import AudioSegment as am

def get_audio_path_UA(speaker):
    """
    Returns path to audio files belonging to specified speaker
    :param speaker: string
                       string encoding the speaker
    :return: list of strings
                the strings represent file paths.
    """
    # print(glob("./datasets/UASPEECH/audio/control/{}/*.wav".format(speaker)))
    return glob("./datasets/UASPEECH/audio/control/{}/*.wav".format(speaker), recursive=True)

def data_augmentation(wavs):
    # Create a complex augmentation pipeline 
    seed = np.random.RandomState(0)
    transform = Pipeline([
        GaussianWhiteNoise(scale=(0.001, 0.0075), p=0.65),
        ExtractLoudestSection(duration=(0.85, 0.95)),
        OneOf([
            RandomCrop(crop_size=(0.01, 0.04), n_crops=(2, 5)),
            SomeOf([
                EdgeCrop('start', crop_size=(0.05, 0.1)),
                EdgeCrop('end', crop_size=(0.05, 0.1))
            ], n=(1, 2))
        ]),
        Sometimes([
            SomeOf([
                LinearFade('in', fade_size=(0.1, 0.2)),
                LinearFade('out', fade_size=(0.1, 0.2))
            ], n=(1, 2))
        ], p=0.5),
        TimeStretch(rate=(0.8, 1.2)),
        PitchShift(n_steps=(-0.25, 0.25)),
        MedianFilter(window_size=(5, 10), p=0.5)
    ], random_state=seed)

    # Generate 25 augmentations of the signal X
    print(wavs)
    X, sr = load(wavs, mono=False)
    Xs = transform.generate(X, n=1, sr=sr)
    librosa.output.write_wav('datasets/augmentationDysarthric/test2.wav', Xs[0], 16000)
    return Xs

def white_noise(wav):
    wav, sr = librosa.load(wav,sr=16000)
    noise = np.random.randn(len(wav))
    augmented_data = wav + 0.001 * noise
    # Cast back to same data type
    augmented_data = augmented_data.astype(type(wav[0]))
    librosa.output.write_wav('datasets/augmentationDysarthric/test2.wav',augmented_data,sr)

def shifting_time(wav, shift_max, shift_direction):
    wav, sr = librosa.load(wav,sr=16000)
    shift = np.random.randint(sr * shift_max)
    if shift_direction == 'right':
        shift = -shift
    elif shift_direction == 'both':
        direction = np.random.randint(0, 2)
        if direction == 1:
            shift = -shift
    augmented_data = np.roll(wav, shift)
    # Set to silence for heading/ tailing
    if shift > 0:
        augmented_data[:shift] = 0
    else:
        augmented_data[shift:] = 0  
    
    librosa.output.write_wav('datasets/augmentationDysarthric/shift.wav',augmented_data,sr)

def changing_pitch(wav, pitch_factor):
    wav, sr = librosa.load(wav,sr=16000)
    augmented_data = librosa.effects.pitch_shift(wav, sr, pitch_factor)
    librosa.output.write_wav('datasets/augmentationDysarthric/changing_pitch.wav',augmented_data,sr)

def changing_speed(wav, speed_factor):
    wav, sr = librosa.load(wav,sr=16000)
    augmented_data = librosa.effects.time_stretch(wav, speed_factor)
    librosa.output.write_wav('datasets/augmentationDysarthric/changing_speed.wav',augmented_data,sr)



def get_audio_path_UA_speaker(speaker):
    """
    Returns path to audio files belonging to specified speaker
    :param speaker: string
                       string encoding the speaker
    :return: list of strings
                the strings represent file paths.
    """
    print(glob("./datasets/UASPEECHOLD/audioold/{}/*.wav".format(speaker)))
    return glob("./datasets/UASPEECHOLD/audioold/{}/*.wav".format(speaker), recursive=True)
    #return glob("./datasets/UASPEECH/audio/Dysarthric/{}/*.wav".format(speaker), recursive=True)


def get_word_list_UA():

    word_list_xls = pd.read_excel("./datasets/UASPEECH/speaker_wordlist.xls", sheet_name="Word_filename", header=0)
    word_dictionary = {}

    for i in range(word_list_xls.shape[0]):
        value = word_list_xls.iloc[i].values
        word_dictionary[value[1]] = value[0]

    return word_dictionary

def get_data_UA(wavs):
    """
    Returns a mapping of audio paths to text
    ---
    :param wavs: string
                string containing path to audio file,
    :param maxlen: int
                max length of word
    :return data: list of dictionaries
                each dictionary contain "audio" and "text", corresponding to the audio path and its text
            removed_files: list of files that were excluded from data
    """
    data = []
    removed_files = []
    word_dictionary = get_word_list_UA()

    for wav in wavs:
        speaker, block, word_key, mic = wav.split('_')
        # use only the 155 words shared by all speakers
        if word_key.startswith('U'):
            # word_key = '_'.join([block, word_key])
            continue
        text = word_dictionary.get(word_key, -1)
        if text == -1:
            continue
        elif block == 'B1' or block == 'B2' or block == 'B3':
            data.append({'audio': wav, 'text': text})


    return data, removed_files
def get_data_UA_speaker(wavs):
    """
    Returns a mapping of audio paths to text
    ---
    :param wavs: string
                string containing path to audio file,
    :param maxlen: int
                max length of word
    :return data: list of dictionaries
                each dictionary contain "audio" and "text", corresponding to the audio path and its text
            removed_files: list of files that were excluded from data
    """

    data_train = []
    data_test = []
    removed_files = []
    word_dictionary = get_word_list_UA()

    for wav in wavs:
        print(wav)
        filesss = wav.split('_')
        print(filesss[3])
        speaker, block, word_key, mic = wav.split('_')
        # use only the 155 words shared by all speakers
        if word_key.startswith('U'):
            # word_key = '_'.join([block, word_key])
            continue
        text = word_dictionary.get(word_key, -1)
        if text == -1:
            continue
        elif block == 'B1' or block == 'B2':
            data_train.append({'audio': wav, 'text': text})
        elif block == 'B3':
            data_test.append({'audio': wav, 'text': text})

    return data_train, data_test

def get_dataset_UA(speakers):
    """Extracts and split the data into B1, B2 and B3 as dataset objects that can be used for model training
    :param speakers: list of speakers to get data from UASpeech data_set.
    :param feature_extractor: function which extracts features from data
    :param vectorizer: text vectorizer
    :return: dataset objects for model fitting
    """
    wavs = []
    for speaker in speakers:
        wavs += get_audio_path_UA(speaker)

    data, _ = get_data_UA(wavs)

    return data

def load_audio_file(file_path):
    input_length = 16000
    data = librosa.core.load(file_path)[0] #, sr=16000
    if len(data)>input_length:
        data = data[:input_length]
    else:
        data = np.pad(data, (0, max(0, input_length - len(data))), "constant")
    return data

def get_dataset_UA_Speaker(speakers):
    """Extracts and split the data into B1, B2 and B3 as dataset objects that can be used for model training
    :param speakers: list of speakers to get data from UASpeech data_set.
    :param feature_extractor: function which extracts features from data
    :param vectorizer: text vectorizer
    :return: dataset objects for model fitting
    """
    wavs = []
    for speaker in speakers:
        wavs += get_audio_path_UA_speaker(speaker)

    data_train,data_test = get_data_UA_speaker(wavs)
    white_noise("./datasets/UASPEECHOLD/audioold/M04/M04_B3_CW12_M7.wav")

    return data_train, data_test