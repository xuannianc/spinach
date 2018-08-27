import keras

callbacks_list = [
    keras.callbacks.ReduceLROnPlateau(
        #  Monitors the model’s validation loss
        monitor='val_loss',
        #  Divides the learning rate by 10 when triggered
        factor=0.1,
        # The callback is triggered after the validation
        # loss has stopped improving for 10 epochs.
        patience=10,
    ),
]
# Because the callback will monitor the validation loss, you
# need to pass validation_data to the call to fit.
model.fit(x, y,
          epochs=10,
          batch_size=32,
          callbacks=callbacks_list,
          validation_data=(x_val, y_val))
