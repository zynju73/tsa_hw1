import numpy as np
from fontTools.misc.bezierTools import epsilon


def naive_forecast(y:np.array, season:int=1):
  "naive forecast: season-ahead step forecast, shift by season step ahead"
  return y[:-season]

def mse(predict, target):
    return np.mean((target - predict) ** 2)


def mae(predict, target):
    return np.mean(np.abs(target-predict))


def mape(predict, target):
    epsilon = 1e-6
    non_zero_target = np.where(np.abs(target) > epsilon, target, epsilon)

    return np.mean(np.abs((target - predict) / non_zero_target)) * 100


def smape(predict, target):
    numerator = np.abs(predict - target)
    denominator = (np.abs(target) + np.abs(predict)) / 2
    epsilon = 1e-6
    non_zero_denominator = np.where(denominator > epsilon, denominator, epsilon)
    return np.mean(numerator/non_zero_denominator)*100


def mase(predict, target, season=24):

    model_mae = np.mean(np.abs(target - predict))

    target_minus_previous = target[season:] - target[:-season]
    naive_mae = np.mean(np.abs(target_minus_previous))
    epsilon=1e-6
    safe_naive_mae = np.where(naive_mae > epsilon, naive_mae, epsilon)

    return model_mae / safe_naive_mae
