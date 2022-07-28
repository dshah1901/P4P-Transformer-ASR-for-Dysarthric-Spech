import os
import random
from glob import glob
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import re
import tensorflow_io as tfio
import soundfile
import numpy


def get_data_libri():
    libre_train =  glob("./datasets/LibriSpeech/train-clean-100/*/*/*.txt", recursive=True)
    libre_test =  glob("./datasets/LibriSpeech/test-clean/*/*/*.txt", recursive=True)

    libri_data_train = list()
    for file in libre_train:
        f = open(file, "r")
        for line in f:
            number = line.split()[0]
            words = line.split()[1:]
            n_1 = number.split('-')[0]
            n_2 = number.split('-')[1]
            flac = './datasets/LibriSpeech/train-clean-100/' + n_1 + '/' + n_2 + '/' + number + '.flac'
                        #convert to wav ? 
            audio, sr = soundfile.read(flac)
            soundfile.write('./datasets/LibriSpeech/train-clean-100/' + n_1 + '/' + n_2 + '/' + number + '.wav', audio, sr, 'PCM_16')
            wav = './datasets/LibriSpeech/train-clean-100/' + n_1 + '/' + n_2 + '/' + number + '.wav'
            word_list = list()
            for word in words:
                word = word.lower()
                word = re.sub('\'s', '', word)
                word = re.sub('[^a-zA-Z0-9 \n]', '', word)
                word_list.append(word)
            full_sen = " ".join(word_list)
            libri_data_train.append({"audio": wav, "text": full_sen})
    
    libri_data_test = list()
    for file in libre_test:
        f = open(file, "r")
        for line in f:
            number = line.split()[0]
            words = line.split()[1:]
            n_1 = number.split('-')[0]
            n_2 = number.split('-')[1]
            flac = './datasets/LibriSpeech/test-clean/' + n_1 + '/' + n_2 + '/' + number + '.flac'
            #convert to wav ? 
            audio, sr = soundfile.read(flac)
            soundfile.write('./datasets/LibriSpeech/test-clean/' + n_1 + '/' + n_2 + '/' + number + '.wav', audio, sr, 'PCM_16')
            wav = './datasets/LibriSpeech/test-clean/' + n_1 + '/' + n_2 + '/' + number + '.wav'
            word_list = list()
            for word in words:
                word = word.lower()
                word = re.sub('\'s', '', word)
                word = re.sub('[^a-zA-Z0-9 \n]', '', word)
                word_list.append(word)
            full_sen = " ".join(word_list)
            libri_data_test.append({"audio": wav, "text": full_sen})
    
    return libri_data_train, libri_data_test



