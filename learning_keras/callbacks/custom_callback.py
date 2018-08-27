import keras
import numpy as np


class ActivationLogger(keras.callbacks.Callback):
    def set_model(self, model):
        # Called by the parent model before training, to inform
        # the callback of what model will be calling it
        self.model = model
        # Model instance that returns the activations of every layer
        layer_outputs = [layer.output for layer in model.layers]
        self.activations_model = keras.models.Model(model.input,
                                                    layer_outputs)

    def on_epoch_end(self, epoch, logs=None):
        if self.validation_data is None:
            raise RuntimeError('Requires validation_data.')
        # Obtains the first input sample of the validation data
        validation_sample = self.validation_data[0][0:1]
        activations = self.activations_model.predict(validation_sample)
        # Saves arrays to disk
        f = open('activations_at_epoch_' + str(epoch) + '.npz', 'w')
        np.savez(f, activations)
        f.close()
