import numpy as np
from learning_keras.text_and_sequence.weather_data import val_gen, val_steps, std


def evaluate_naive_method():
    batch_maes = []
    for step in range(val_steps):
        samples, targets = next(val_gen)
        # batch 中每一个 sample 的最后一个 timestep 的第一个 feature 温度作为 24h 后的预测值
        preds = samples[:, -1, 1]
        mae = np.mean(np.abs(preds - targets))
        batch_maes.append(mae)
    mae = np.mean(batch_maes)
    print('mae={}'.mae)
    celsius_mae = mae * std[1]
    print('celsius_mae={}'.format(celsius_mae))


evaluate_naive_method()
