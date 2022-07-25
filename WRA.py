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
            if target_text == prediction :
                score += 1

            print('{} score of one validation batch: {:.2f}\n'.format("WRA", 1 - score / float(bs)))
        tf.keras.callbacks.ModelCheckpoint(filepath=saveto,
                                           save_weights_only=True,
                                           verbose=1)
        return score, bs

    def on_train_end(self, logs=None):
        """Get the accuracy score from a dataset. The possible metrics are: BLEU score and Word Error Rate"""
        print("In accuracy function")
        score = 0
        samples = 0
        ds_itr = iter(self.val_ds)

        for self.batch in ds_itr:
            score_per_batch, bs = self.on_epoch_end(self)
            score += score_per_batch
            samples += bs


        print('Average {} score of ds: {:.2f}\n'.format("WER", 1 - (score / float(samples))))
        return 1 - (score / float(samples))