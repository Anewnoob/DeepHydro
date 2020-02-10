import numpy as np
def get_MSE(pred, real):
    return np.mean(np.power(real - pred, 2))


def get_MAE(pred, real):
    return np.mean(np.abs(real - pred))


def get_MAPE(pred, real):
    ori_real = real.copy()
    epsilon = 1
    real[real == 0] = epsilon
    return np.mean(np.abs((ori_real - pred) / real))