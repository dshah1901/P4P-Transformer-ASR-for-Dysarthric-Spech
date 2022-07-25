import os
import random
from glob import glob
import re

def get_data_libre():
    libre_train =  glob("./LibriSpeech/train-clean-100/*/*/*.txt", recursive=True)
    libri_train_data = list()
    for file in libre_train:
        f = open(file, "r")
        for line in f:
            number = line.split()[0]
            words = line.split()[1:]
            n_1 = number.split('-')[0]
            n_2 = number.split('-')[1]
            flac = './LibriSpeech/train-clean-100/' + n_1 + '/' + n_2 + '/' + number + '.flac'
            word_list = list()
            for word in words:
                word = word.lower()
                word = re.sub('\'s', '', word)
                word = re.sub('[^a-zA-Z0-9 \n]', '', word)
                word_list.append(word)
            full_sen = " ".join(word_list)
            libri_train_data.append({"audio": flac, "text": full_sen})

    libre_test =  glob("./LibriSpeech/test-clean/*/*/*.txt", recursive=True)
    libri_test_data = list()
    for file in libre_test:
        f = open(file, "r")
        for line in f:
            number = line.split()[0]
            words = line.split()[1:]
            n_1 = number.split('-')[0]
            n_2 = number.split('-')[1]
            flac = './LibriSpeech/test-clean/' + n_1 + '/' + n_2 + '/' + number + '.flac'
            word_list = list()
            for word in words:
                word = word.lower()
                word = re.sub('\'s', '', word)
                word = re.sub('[^a-zA-Z0-9 \n]', '', word)
                word_list.append(word)
            full_sen = " ".join(word_list)
            libri_test_data.append({"audio": flac, "text": full_sen})
    return libre_test, libre_train


