import keras.callbacks
import keras_tuner as kt 
from Modely import *


LABELS = ['backward','bed','bird','cat','dog','down','eight','five','follow','four','go','happy','house','learn','left','marvin','nine','no','off','on','right','seven','sheila','six','stop','three','tree','two','up','visual','wow','yes','zero']
max_target_len = 50  # all transcripts in out data are < 200 characters
data_train, data_test = get_dataset(LABELS)
vectorizer = VectorizeChar(max_target_len)
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

#data_test
test_index = []
audio_dss = create_audio_ds(data_test)   
for indexx,vall in enumerate(list(audio_dss)):
    df = pd.DataFrame(vall)   
    if(df.isnull().sum().sum() > 0):
        test_index.append(indexx)

test_index = sorted(test_index, reverse=True)
# Traverse the indices list
for index in test_index:
    if index < len(data_test):
        data_test.pop(index)

print("training data size after porcess")
print(sum(1 for d in data_train if d)) 
print("testing data size after porcess")
print(sum(1 for d in data_test if d)) 

train_data = data_train
test_data = data_test
ds = create_tf_dataset(train_data, bs=64)
val_ds = create_tf_dataset(test_data, bs=1)

def model_builder(hp):
    """
    Passed as a parameter to the KerasTuner search function
    ---
    :param hp: HyperParameters object.
                  see more at https://keras.io/api/keras_tuner/hyperparameters/
    :return: returns a compiled Transformer Model with randomly selected hyperparameters
    """
    # specify the hyperparameters for tuning, and the value range
    num_hid = hp.Int(name="num_hid", min_value=64, max_value=512, step=64)
    num_head = hp.Int(name="num_head", min_value=2, max_value=8, step=2)
    num_feed_forward = hp.Int(name="num_feed_forward", min_value=64, max_value=512, step=64)
    num_layers_enc = hp.Int(name="num_layers_enc", min_value=1, max_value=5, step=1)
    num_layers_dec = hp.Int(name="num_layers_dec", min_value=1, max_value=5, step=1)

    model = Transformer(
        num_hid=num_hid,
        num_head=num_head,
        num_feed_forward=num_feed_forward,
        target_maxlen=max_target_len,
        num_layers_enc=num_layers_enc,
        num_layers_dec=num_layers_dec,
        num_classes=10,
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

    return model

if __name__ == "__main__":
    # tuner = kt.Hyperband(model_builder,
    #                  objective='val_loss',
    #                  max_epochs=50,
    #                  factor=3,
    #                  directory='./datasets',
    #                  project_name='hyperparameter_tuning_UA')
    tuner = kt.BayesianOptimization(
                model_builder,
                objective="val_loss",
                max_trials=10,
                directory='./datasets',
                project_name='_Bayesian_hyperparam_UA',
                
            )

    stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)

    tuner.search(ds, validation_data=val_ds, epochs=20, callbacks=[stop_early])


    # Get the optimal hyperparameters
    best_hps=tuner.get_best_hyperparameters(num_trials=1)[0]


    model = tuner.hypermodel.build(best_hps) 
    history = model.fit(ds, validation_data=val_ds, epochs=20, callbacks=[stop_early])
    val_loss_per_epoch = history.history['val_loss']
    best_epoch = val_loss_per_epoch.index(max(val_loss_per_epoch)) + 1
    print('Best epoch: %d' % (best_epoch,))

    hypermodel = tuner.hypermodel.build(best_hps)

    # Retrain the model
    hypermodel.fit(ds, validation_data = val_ds, epochs=best_epoch)
    eval_result = hypermodel.evaluate(val_ds)
    print("[test loss]:", eval_result)