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
import numpy as np

def get_audio_path(speaker):
    """
    Returns path to audio files belonging to specified speaker
    :param speaker: string
                       string encoding the speaker
    :return: list of strings
                the strings represent file paths.
    """
    #print(glob("./datasets/TORGO/Control/{}/**/wav_*/*.wav".format(speaker)))
    return glob("./datasets/TORGO/Control/{}/**/wav_*/*.wav".format(speaker))

def get_data_TORGO(wavs,maxlen=5000):
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
    unique_words = []
    data = []
    removed_files = []
    pattern = re.compile(r"\[.*\]")
    for wav in wavs:
        description = wav.split("/")
        session = description[5]
        id = description[-1].split('.')[0]
        speaker = description[4]
        # print(description)
        # print(id)
        try:
            filename = glob(f"./datasets/TORGO/Control/{speaker}/{session}/prompts/{id}.txt")[0]
        except IndexError:
            continue
        with open(filename, encoding="utf-8") as f:
            line = f.readline()
            line = line.replace("\n", "")
            if len(line) > maxlen:
                continue
            line = pattern.sub("", line)
            if line == "" or line == "xxx" or '.jpg' in line:
                removed_files.append({'file': filename, 'text': line})
                continue
            line = line.rstrip()
            data.append({"audio": wav, "text": line})
            random.shuffle(data)
    return data, removed_files

def get_dataset_TORGO(speakers):
    wavs = []
    for speaker in speakers:
        wavs += get_audio_path(speaker)

    data, _ = get_data_TORGO(wavs)
    return data

def remove_unique_words(data):
    print(sum(1 for d in data if d))
    testing_data = []
    count = 0
    texts = [_["text"] for _ in data]
    duplicate_words = [number for number in texts if texts.count(number) == 1]
    unique_duplicates = list(set(duplicate_words))
    #print(unique_duplicates)
    for x in unique_duplicates:
        unique_words = list(filter(lambda student: student.get('text')==x, data))
        for i in unique_words:
            testing_data.append({"audio": i['audio'], "text": i['text']})
            
    
    # res = list(filter(lambda i: i['text'] != 'mole', data))
    res = [d for d in data if d['text'] not in unique_duplicates]
    

    print(sum(1 for d in res if d))
    print(sum(1 for d in testing_data if d))
    return res,testing_data