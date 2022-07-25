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
from preprocessing.libre_speech import *
from transformer.model import *
from accuracy.WER import *

"""
## Callbacks to display predictions - WRA - Word Recognition Accuracy. 
"""
class DisplayOutputs(keras.callbacks.Callback):
    def __init__(
        self, batch, ds, idx_to_token, target_start_token_idx=27, target_end_token_idx=28
    ):
        """Displays a batch of outputs after every epoch
        Args:
            batch: A test batch containing the keys "source" and "target"
            idx_to_token: A List containing the vocabulary tokens corresponding to their indices
            target_start_token_idx: A start token index in the target vocabulary
            target_end_token_idx: An end token index in the target vocabulary
        """
        self.batch = batch
        self.target_start_token_idx = target_start_token_idx
        self.target_end_token_idx = target_end_token_idx
        self.idx_to_char = idx_to_token
        self.val_ds = val_ds

    
    def on_epoch_end(self, epoch, logs=None):
        # if epoch % 5 != 0:
        #
        score = 0
        source = self.batch["source"]
        target = self.batch["target"].numpy()
        bs = tf.shape(source)[0]
        preds = self.model.generate(source, self.target_start_token_idx)
        preds = preds.numpy()
        for i in range(bs):
            target_text = "".join([self.idx_to_char[_] for _ in target[i, :]])
            prediction = ""
            for idx in preds[i, :]:
                prediction += self.idx_to_char[idx]
                if idx == self.target_end_token_idx:
                    break
            print(f"target:     {target_text.replace('-','')}")
            print(f"prediction: {prediction}\n")
            target_text = target_text.replace("-","")
            score = wer(target_text, prediction)

            print('{} score of one validation batch: {:.2f}\n'.format("WER", score))
        self.model.save_weights(f'./datasets{self.model.model_name}.keras')
        return score, bs

 

    def on_train_end(self, logs=None):
        """Get the accuracy score from a dataset. The possible metrics are: BLEU score and Word Error Rate"""
        
        target = []
        word_error_rate = []
        prediction = []
        score = 0
        samples = 0
        ds_itr = iter(self.val_ds)

        for self.batch in ds_itr:
            score_per_batch, target_per_batch, prediction_per_batch,bs = self.on_epoch_end(self)
            target.append(target_per_batch)
            prediction.append(prediction_per_batch)
            word_error_rate.append(score_per_batch)
            score += score_per_batch
            samples += bs


        data = pd.DataFrame({"A":target,"B":prediction,"C":word_error_rate})
        data.to_excel('Libre_Results.xlsx', sheet_name='Sheet1',index=False)
        print('Average {} score of ds: {:.2f}\n'.format("WER", 1 - (score / float(samples))))
        return 1 - (score / float(samples))


#LABELS = ['backward','bed','bird','cat','dog','down','eight','five','follow','four','go','happy','house','learn','left','marvin','nine','no','off','on','right','seven','sheila','six','stop','three','tree','two','up','visual','wow','yes','zero']
max_target_len = 50  # all transcripts in out data are < 200 characters
data_test, data_train = get_data_libre();
vectorizer = VectorizeChar(max_target_len)

def train_test_libre():
    indexes = []
    audio_ds = create_audio_ds(data_train)  #libre_train 
    for index,val in enumerate(list(audio_ds)):
        df = pd.DataFrame(val)   
        if(df.isnull().sum().sum() > 0):
            indexes.append(index)

    indexes = sorted(indexes, reverse=True)
    # Traverse the indices list
    for index in indexes:
        if index < len(data_train):
            data_train.pop(index)

    #data_test
    test_index = []
    audio_dss = create_audio_ds(data_test)   #libre_test
    for indexx,vall in enumerate(list(audio_dss)):
        df = pd.DataFrame(vall)   
        if(df.isnull().sum().sum() > 0):
            test_index.append(indexx)

    test_index = sorted(test_index, reverse=True)
    # Traverse the indices list
    for index in test_index:
        if index < len(data_test):
            data_test.pop(index)

    print("training data size after process")
    print(sum(1 for d in data_train if d)) 
    print("testing data size after process")
    print(sum(1 for d in data_test if d)) 

    return data_test, data_train 

test_data, train_data = train_test_libre()
ds = create_tf_dataset(train_data, bs=64)
val_ds = create_tf_dataset(test_data, bs=1)

"""
## Create & train the end-to-end model
"""

batch = next(iter(val_ds))

# The vocabulary to convert predicted indices into characters
idx_to_char = vectorizer.get_vocabulary()
display_cb = DisplayOutputs(
    batch, ds, idx_to_char, target_start_token_idx=2, target_end_token_idx=3
)  # set the arguments as per vocabulary index for '<' and '>'

model = Transformer(
    num_hid=200,
    num_head=2,
    num_feed_forward=400,
    target_maxlen=max_target_len,
    num_layers_enc=4,
    num_layers_dec=1,
    num_classes=34,
)
loss_fn = tf.keras.losses.CategoricalCrossentropy(
    from_logits=True,
    label_smoothing=0.1,
)

learning_rate = CustomSchedule(
    init_lr=0.00001,
    lr_after_warmup=0.001,
    final_lr=0.00001,
    warmup_epochs=15,
    decay_epochs=85,
    steps_per_epoch=len(ds),
)
optimizer = keras.optimizers.Adam(learning_rate)
model.compile(optimizer=optimizer, loss=loss_fn, metrics=['mae'],)
history = model.fit(ds, validation_data=val_ds, callbacks=[display_cb], epochs=100, verbose=1)

