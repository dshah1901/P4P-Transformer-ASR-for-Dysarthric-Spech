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


SPEAKERS = ["FC01", "FC02", "FC03", "MC01", "MC02", "MC03", "MCO4"]
max_target_len = 200  # all transcripts in out data are < 200 characters
data = get_dataset_TORGO(SPEAKERS)
# data = get_data(wavs, id_to_text, max_target_len)
print("vocab size", len(vectorizer.get_vocabulary()))

# split = int(len(data) * 0.99)
# train_data = data[:split]
# test_data = data[split:]
train_data,test_data = remove_unique_words(data)
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
checkpoint_dir = os.path.dirname(saveto)
history = model.fit(ds, validation_data=val_ds, callbacks=[display_cb], epochs=100, verbose=1)