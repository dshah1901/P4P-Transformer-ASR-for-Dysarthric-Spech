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
import librosa
import re
import pandas as pd
import numpy as np

def get_audio_path_UA(speaker):
    """
    Returns path to audio files belonging to specified speaker
    :param speaker: string
                       string encoding the speaker
    :return: list of strings
                the strings represent file paths.
    """
    print(glob("./UASPEECH/audio/control/{}/*.wav".format(speaker)))
    return glob("./UASPEECH/audio/control/{}/*.wav".format(speaker), recursive=True)



def get_word_list_UA():

    word_list_xls = pd.read_excel("./UASPEECH/speaker_wordlist.xls", sheet_name="Word_filename", header=0)
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