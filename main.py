import os
# Making only GPU 1 visible so it only trains on it
os.environ["CUDA_VISIBLE_DEVICES"]="3,4" 
import random
from glob import glob
import numpy
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from matplotlib import pyplot as plt
# from LJ_SPeech_preprocess import *
from UA_Speech_preprocess import *
from jiwer import wer


"""
## Define the Transformer Input Layer
When processing past target tokens for the decoder, we compute the sum of
position embeddings and token embeddings.
When processing audio features, we apply convolutional layers to downsample
them (via convolution stides) and process local relationships.
"""


class TokenEmbedding(layers.Layer):
    def __init__(self, num_vocab=1000, maxlen=100, num_hid=64):
        super().__init__()
        self.emb = tf.keras.layers.Embedding(num_vocab, num_hid)
        self.pos_emb = layers.Embedding(input_dim=maxlen, output_dim=num_hid)

    def call(self, x):
        maxlen = tf.shape(x)[-1]
        x = self.emb(x)
        positions = tf.range(start=0, limit=maxlen, delta=1)
        positions = self.pos_emb(positions)
        return x + positions


class SpeechFeatureEmbedding(layers.Layer):
    def __init__(self, num_hid=64, maxlen=100):
        super().__init__()
        self.conv1 = tf.keras.layers.Conv1D(
            num_hid, 11, strides=2, padding="same", activation="relu"
        )
        self.conv2 = tf.keras.layers.Conv1D(
            num_hid, 11, strides=2, padding="same", activation="relu"
        )
        self.conv3 = tf.keras.layers.Conv1D(
            num_hid, 11, strides=2, padding="same", activation="relu"
        )
        self.pos_emb = layers.Embedding(input_dim=maxlen, output_dim=num_hid)

    def call(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return self.conv3(x)


"""
## Transformer Encoder Layer
"""
     
class TransformerEncoder(layers.Layer):
    def __init__(self, embed_dim, num_heads, feed_forward_dim, rate=0.1):
        super().__init__()
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.cnn = keras.Sequential(
            [
                layers.SeparableConv1D(feed_forward_dim, 1, activation="relu"),
                layers.SeparableConv1D(embed_dim, 1),
            ]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

        #second 
        self.att_1 = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.cnn_1 = keras.Sequential(
            [
                layers.SeparableConv1D(feed_forward_dim, 1, activation="relu"),
                layers.SeparableConv1D(embed_dim, 1),
            ]
        )
        self.layernorm1_1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2_1 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1_1 = layers.Dropout(rate)
        self.dropout2_1 = layers.Dropout(rate)

    def call(self, inputs, training):
        attn_output = self.att(inputs, inputs) #multi head attention
        attn_output = self.dropout1(attn_output, training=training) #add dropout
        out1 = self.layernorm1(inputs + attn_output) #normalisation
        cnn_output = self.cnn(out1) 
        cnn_output = self.dropout2(cnn_output, training=training)
        out2 =self.layernorm2(out1 + cnn_output)
        enc_out = self.att_1(out2, out2)
        enc_out_norm = self.layernorm1_1(self.dropout1_1(enc_out) + out2)
        cnn_out = self.cnn_1(enc_out_norm)
        cnn_out_norm = self.layernorm2_1(enc_out_norm + self.dropout2_1(cnn_out))
        return cnn_out_norm


# self.ffn = keras.Sequential(
        #     [
        #         layers.Dense(feed_forward_dim, activation="relu"),
        #         layers.Dense(embed_dim),
        #     ]
        # )
"""
## Transformer Decoder Layer
"""

class TransformerDecoder(layers.Layer):
    def __init__(self, embed_dim, num_heads, feed_forward_dim, dropout_rate=0.1):
        super().__init__()
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = layers.LayerNormalization(epsilon=1e-6)
        self.self_att = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim
        )
        self.enc_att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.self_dropout = layers.Dropout(0.5)
        self.enc_dropout = layers.Dropout(0.1)
        self.ffn_dropout = layers.Dropout(0.1)
        self.ffn = keras.Sequential(
            [
                layers.Dense(feed_forward_dim, activation="relu"),
                layers.Dense(embed_dim),
            ]
        )

    def causal_attention_mask(self, batch_size, n_dest, n_src, dtype):
        """Masks the upper half of the dot product matrix in self attention.
        This prevents flow of information from future tokens to current token.
        1's in the lower triangle, counting from the lower right corner.
        """
        i = tf.range(n_dest)[:, None]
        j = tf.range(n_src)
        m = i >= j - n_src + n_dest
        mask = tf.cast(m, dtype)
        mask = tf.reshape(mask, [1, n_dest, n_src])
        mult = tf.concat(
            [tf.expand_dims(batch_size, -1), tf.constant([1, 1], dtype=tf.int32)], 0
        )
        return tf.tile(mask, mult)

    def call(self, enc_out, target):
        input_shape = tf.shape(target)
        batch_size = input_shape[0]
        seq_len = input_shape[1]
        causal_mask = self.causal_attention_mask(batch_size, seq_len, seq_len, tf.bool)
        target_att = self.self_att(target, target, attention_mask=causal_mask)
        target_norm = self.layernorm1(target + self.self_dropout(target_att))
        enc_out = self.enc_att(target_norm, enc_out)
        enc_out_norm = self.layernorm2(self.enc_dropout(enc_out) + target_norm)
        ffn_out = self.ffn(enc_out_norm)
        ffn_out_norm = self.layernorm3(enc_out_norm + self.ffn_dropout(ffn_out))
        return ffn_out_norm


"""
## Complete the Transformer model
Our model takes audio spectrograms as inputs and predicts a sequence of characters.
During training, we give the decoder the target character sequence shifted to the left
as input. During inference, the decoder uses its own past predictions to predict the
next token.
"""

class Transformer(keras.Model):
    def __init__(
        self,
        num_hid=64,
        num_head=2,
        num_feed_forward=128,
        source_maxlen=100,
        target_maxlen=100,
        num_layers_enc=4,
        num_layers_dec=1,
        num_classes=10,
    ):
        super().__init__()
        self.loss_metric = keras.metrics.Mean(name="loss")
        self.num_layers_enc = num_layers_enc
        self.num_layers_dec = num_layers_dec
        self.target_maxlen = target_maxlen
        self.num_classes = num_classes
        self.vectorizer = VectorizeChar(max_len=self.target_maxlen)
        self.idx_to_char = self.vectorizer.get_vocabulary()

        self.enc_input = SpeechFeatureEmbedding(num_hid=num_hid, maxlen=source_maxlen)
        self.dec_input = TokenEmbedding(
            num_vocab=num_classes, maxlen=target_maxlen, num_hid=num_hid
        )

        self.encoder = keras.Sequential(
            [self.enc_input]
            + [
                TransformerEncoder(num_hid, num_head, num_feed_forward)
                for _ in range(num_layers_enc)
            ]
        )

        for i in range(num_layers_dec):
            setattr(
                self,
                f"dec_layer_{i}",
                TransformerDecoder(num_hid, num_head, num_feed_forward),
            )

        self.classifier = layers.Dense(num_classes)

    def decode(self, enc_out, target):
        y = self.dec_input(target)
        for i in range(self.num_layers_dec):
            y = getattr(self, f"dec_layer_{i}")(enc_out, y)
        return y

    def call(self, inputs):
        source = inputs[0]
        target = inputs[1]
        x = self.encoder(source)
        y = self.decode(x, target)
        return self.classifier(y)

    @property
    def metrics(self):
        return [self.loss_metric]

    def train_step(self, batch):
        """Processes one batch inside model.fit()."""
        source = batch["source"]
        target = batch["target"]
        dec_input = target[:, :-1]
        dec_target = target[:, 1:]
        with tf.GradientTape() as tape:
            preds = self([source, dec_input])
            one_hot = tf.one_hot(dec_target, depth=self.num_classes)
            mask = tf.math.logical_not(tf.math.equal(dec_target, 0))
            loss = self.compiled_loss(one_hot, preds, sample_weight=mask)
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        self.loss_metric.update_state(loss)
        return {"loss": self.loss_metric.result()}

    def test_step(self, batch):
        source = batch["source"]
        target = batch["target"]
        dec_input = target[:, :-1]
        dec_target = target[:, 1:]
        preds = self([source, dec_input])
        one_hot = tf.one_hot(dec_target, depth=self.num_classes)
        mask = tf.math.logical_not(tf.math.equal(dec_target, 0))
        loss = self.compiled_loss(one_hot, preds, sample_weight=mask)
        self.loss_metric.update_state(loss)
        return {"loss": self.loss_metric.result()}

    def generate(self, source, target_start_token_idx):
        """Performs inference over one batch of inputs using greedy decoding."""
        bs = tf.shape(source)[0]
        enc = self.encoder(source)
        dec_input = tf.ones((bs, 1), dtype=tf.int32) * target_start_token_idx
        dec_logits = []
        for i in range(self.target_maxlen - 1):
            dec_out = self.decode(enc, dec_input)
            logits = self.classifier(dec_out)
            logits = tf.argmax(logits, axis=-1, output_type=tf.int32)
            last_logit = tf.expand_dims(logits[:, -1], axis=-1)
            dec_logits.append(last_logit)
            dec_input = tf.concat([dec_input, last_logit], axis=-1)
        return dec_input
   
    def predictions(self, batch, logs=None):
        score = 0
        source = batch["source"]
        target = batch["target"].numpy()
        bs = tf.shape(source)[0]
        preds = self.generate(source, 2)
        preds = preds.numpy()
        for i in range(bs):
            target_text = "".join([self.idx_to_char[_] for _ in target[i, :]])
            prediction = ""
            for idx in preds[i, :]:
                prediction += self.idx_to_char[idx]
                if idx == 3:
                    break
            print(f"target:     {target_text.replace('-','')}")
            print(f"prediction: {prediction}\n")
            target_text = target_text.replace("-","")
            score += wer(target_text, prediction)

            print('{} score of one validation batch: {:.2f}\n'.format("WER", score))
        return score, target_text, prediction, bs  

    def get_accuracy(self, val_ds,logs=None):

        target = []
        word_error_rate = []
        prediction = []
        score = 0
        samples = 0
        ds_itr = iter(val_ds)

        for batch in ds_itr:
            score_per_batch, target_per_batch, prediction_per_batch,bs = self.predictions(batch)
            target.append(target_per_batch)
            prediction.append(prediction_per_batch)
            word_error_rate.append(score_per_batch)
            score += score_per_batch
            samples += bs


        data = pd.DataFrame({"A":target,"B":prediction,"C":word_error_rate})
        data.to_excel('ASR Results.xlsx', sheet_name='Sheet1',index=False)
        
        print('Average {} score of ds: {:.2f}\n'.format("WER", (score / float(samples))))

        return (score / float(samples))
"""
## Download the dataset
Note: This requires ~3.6 GB of disk space and
takes ~5 minutes for the extraction of files.
"""

# saveto = "./datasets/LJSpeech-1.1"
# wavs = glob("{}/**/*.wav".format(saveto), recursive=True)

# id_to_text = {}
# with open(os.path.join(saveto, "metadata.csv"), encoding="utf-8") as f:
#     for line in f:
#         id = line.strip().split("|")[0]
#         text = line.strip().split("|")[2]
#         id_to_text[id] = text

"""
## Preprocess the dataset
"""

class VectorizeChar:
    def __init__(self, max_len=50):
        self.vocab = (
            ["-", "#", "<", ">"]
            + [chr(i + 96) for i in range(1, 27)]
            + [" ", ".", ",", "?"]
        )
        self.max_len = max_len
        self.char_to_idx = {}
        for i, ch in enumerate(self.vocab):
            self.char_to_idx[ch] = i

    def __call__(self, text):
        text = text.lower()
        text = text[: self.max_len - 2]
        text = "<" + text + ">"
        pad_len = self.max_len - len(text)
        return [self.char_to_idx.get(ch, 1) for ch in text] + [0] * pad_len

    def get_vocabulary(self):
        return self.vocab

# SPEAKERS_TRAIN = ['CF03', 'CF04', 'CF05', 'CF02', 'CM01', 'CM04', 'CM05', 'CM08', 'CM09', 'CM10']
# SPEAKERS_TEST = ['CM06']
SPEAKER = ['M04']
max_target_len = 200  # all transcripts in out data are < 200 characters
data_train, data_test = get_dataset_UA(SPEAKER)
#data = get_data(wavs, id_to_text, max_target_len) #Getting data for LJ Speech
vectorizer = VectorizeChar(max_target_len)
print("vocab size", len(vectorizer.get_vocabulary()))


def create_text_ds(data):
    texts = [_["text"] for _ in data]
    text_ds = [vectorizer(t) for t in texts]
    text_ds = tf.data.Dataset.from_tensor_slices(text_ds)
    return text_ds


def path_to_audio(path):
    # spectrogram using stft
    audio = tf.io.read_file(path)
    audio, _ = tf.audio.decode_wav(audio, 1)
    audio = tf.squeeze(audio, axis=-1)
    stfts = tf.signal.stft(audio, frame_length=200, frame_step=80, fft_length=256)
    x = tf.math.pow(tf.abs(stfts), 0.5)
    # normalisation
    means = tf.math.reduce_mean(x, 1, keepdims=True)
    stddevs = tf.math.reduce_std(x, 1, keepdims=True)
    x = (x - means) / stddevs
    audio_len = tf.shape(x)[0]
    # padding to 10 seconds
    pad_len = 2754
    paddings = tf.constant([[0, pad_len], [0, 0]])
    x = tf.pad(x, paddings, "CONSTANT")[:pad_len, :]
    return x


def create_audio_ds(data):
    flist = [_["audio"] for _ in data]
    audio_ds = tf.data.Dataset.from_tensor_slices(flist)
    audio_ds = audio_ds.map(path_to_audio, num_parallel_calls=tf.data.AUTOTUNE)
    return audio_ds


def create_tf_dataset(data, bs=4):
    audio_ds = create_audio_ds(data)
    text_ds = create_text_ds(data)
    ds = tf.data.Dataset.zip((audio_ds, text_ds))
    ds = ds.map(lambda x, y: {"source": x, "target": y})
    ds = ds.batch(bs)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds

indexes = []
audio_ds = create_audio_ds(data_train)   
for index,val in enumerate(list(audio_ds)):
    df = pd.DataFrame(val)   
    if(df.isnull().sum().sum() > 0):
        indexes.append(index)

indexes = sorted(indexes, reverse=True)
# Traverse the indices list
for index in indexes:
    if index < len(data_train):
        data_train.pop(index)

# Dividing LJ Speech into 80 training 20 testing as it has only one speaker.
# split = int(len(data) * 0.80)
# data_train = data[:split]
print("Size of training data")
print(sum(1 for d in data_train if d))
# # data_test = data[split:]
print("Size of testing data")
print(sum(1 for d in data_test if d))
ds = create_tf_dataset(data_train, bs=64)
val_ds = create_tf_dataset(data_test, bs =64)


"""
## Callbacks to display predictions
"""

class DisplayOutputs(keras.callbacks.Callback):
    def __init__(
        self, batch, idx_to_token, target_start_token_idx=27, target_end_token_idx=28
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
            score += wer(target_text, prediction)

            print('{} score of one validation batch: {:.2f}\n'.format("WER", float(wer(target_text, prediction))))

            self.model.save_weights(f'Augmented_M04_TE.h5')
        print('{} total score of one validation batch: {:.2f}\n'.format("WER", (score)/float(bs)))
        data = pd.DataFrame({"A":epoch,"B":(score)/float(bs)}, index=[0])
        with pd.ExcelWriter("Epoch Accuracy.xlsx",mode="a",engine="openpyxl",if_sheet_exists="overlay") as writer:
            data.to_excel(writer, sheet_name="Sheet1",header=None, startrow=writer.sheets["Sheet1"].max_row,index=False)
        return score, target_text, prediction, bs

    def on_train_end(self,logs=None): 

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
        data.to_excel('ASR Results.xlsx', sheet_name='LJ Speech',index=False)
        
        print('Final Average {} score of ds: {:.2f}\n'.format("WER",  (score / float(samples))))

        return (score / float(samples))


"""
## Learning rate schedule
"""

class CustomSchedule(keras.optimizers.schedules.LearningRateSchedule):
    def __init__(
        self,
        init_lr=0.00001,
        lr_after_warmup=0.001,
        final_lr=0.00001,
        warmup_epochs=15,
        decay_epochs=85,
        steps_per_epoch=203,
    ):
        super().__init__()
        self.init_lr = init_lr
        self.lr_after_warmup = lr_after_warmup
        self.final_lr = final_lr
        self.warmup_epochs = warmup_epochs
        self.decay_epochs = decay_epochs
        self.steps_per_epoch = steps_per_epoch

    def calculate_lr(self, epoch):
        """linear warm up - linear decay"""
        warmup_lr = (
            self.init_lr
            + ((self.lr_after_warmup - self.init_lr) / (self.warmup_epochs - 1)) * epoch
        )
        decay_lr = tf.math.maximum(
            self.final_lr,
            self.lr_after_warmup
            - (epoch - self.warmup_epochs)
            * (self.lr_after_warmup - self.final_lr)
            / (self.decay_epochs),
        )
        return tf.math.minimum(warmup_lr, decay_lr)

    def __call__(self, step):
        epoch = step // self.steps_per_epoch
        return self.calculate_lr(epoch)

"""
## Create & train the end-to-end model
"""

batch = next(iter(val_ds))

# The vocabulary to convert predicted indices into characters
idx_to_char = vectorizer.get_vocabulary()
display_cb = DisplayOutputs(
    batch, idx_to_char, target_start_token_idx=2, target_end_token_idx=3
)  # set the arguments as per vocabulary index for '<' and '>'

model = Transformer(
    num_hid=64,
    num_head=2,
    num_feed_forward=320,
    target_maxlen=max_target_len,
    num_layers_enc=5,
    num_layers_dec=3,
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
model.compile(optimizer=optimizer, loss=loss_fn)
#history = model.fit(ds, validation_data=val_ds, callbacks=[display_cb], epochs=40)

#loading weights
# quick model fit to get input shape for loading weights
model.fit(val_ds.take(1), epochs=1, verbose=0)
model.load_weights(f'LJ200_2.0_UAControl_FreezeTokenD123.h5')
model.summary(); 

# print((model.layers)[1]) #Token embeddings
# print((model.layers)[2]) #Sequential (encoder)
# print((model.layers)[3]) #Transformer decoder
# print((model.layers)[4]) #Transformer decoder
# print((model.layers)[5]) #Transformer decoder
# print((model.layers)[6]) #dense

# Encoder Layer Freezing 
# ((model.encoder.layers)[1]).trainable = False;
# ((model.encoder.layers)[2]).trainable = False;
# ((model.encoder.layers)[3]).trainable = False;
# ((model.encoder.layers)[4]).trainable = False;
# ((model.encoder.layers)[5]).trainable = False;

((model.layers)[1]).trainable = False
# ((model.layers)[3]).trainable = False
# ((model.layers)[4]).trainable = False
# ((model.layers)[5]).trainable = False
# ((model.layers)[6]).trainable = False

model.summary(); 
model.compile(optimizer=optimizer, loss=loss_fn)
history = model.fit(ds, callbacks=[display_cb], epochs=100)

# Plot 
# Get training and test loss histories
training_loss = history.history['loss']
# test_loss = history.history['val_loss']

# Create count of the number of epochs
epoch_count = range(1, len(training_loss) + 1)

# Visualize loss history
plt.plot(epoch_count, training_loss, 'r--')
# plt.plot(epoch_count, test_loss, 'b-')
plt.legend(['Training Loss'])
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.savefig("training loss.png")
plt.show()