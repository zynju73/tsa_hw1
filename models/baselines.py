import numpy as np
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.api import SimpleExpSmoothing



class MLForecastModel:

    def __init__(self) -> None:
        self.fitted = False

    def fit(self, X: np.ndarray) -> None:
        """
        :param X: history timesteps
        :param Y: future timesteps to predict
        """
        self._fit(X)
        self.fitted = True

    def _fit(self, X: np.ndarray):
        raise NotImplementedError

    def _forecast(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def forecast(self, X: np.ndarray) -> np.ndarray:
        """
        :param X: history timesteps
        :return: predicted future timesteps
        """
        if not self.fitted:
            raise ValueError("Model has not been trained.")
        pred = self._forecast(X)
        return pred

class ZeroForecast(MLForecastModel):
    def __init__(self, args) -> None:
        super().__init__()
        self.pred_len = args.pred_len
        self.channels = args.channels if hasattr(args, 'channels') else 1

    def _fit(self, X: np.ndarray) -> None:
        pass

    def _forecast(self, X: np.ndarray) -> np.ndarray:
        return np.zeros((X.shape[0], self.pred_len, self.channels))


class MeanForecast(MLForecastModel):
    def __init__(self, args) -> None:
        super().__init__()
        self.pred_len = args.pred_len
        self.channels = args.channels if hasattr(args, 'channels') else 1

    def _fit(self, X: np.ndarray) -> None:
        pass

    def _forecast(self, X: np.ndarray) -> np.ndarray:
        if len(X.shape) == 2:
            # (num, pred_len)
            mean = np.mean(X, axis=1).reshape(X.shape[0], 1)
            forecast = np.repeat(mean, self.pred_len, axis=1)
            if self.channels > 1:
                forecast = np.repeat(forecast[:, :, np.newaxis], self.channels, axis=2)
            return forecast
        else:
            # (num, pred_len, channels)
            mean = np.mean(X, axis=1).reshape(X.shape[0], 1, X.shape[2])
            return np.repeat(mean, self.pred_len, axis=1)

class LastPeriodForecast(MLForecastModel):
    def __init__(self, args) -> None:
        super().__init__()
        self.period = args.period
        self.pred_len = args.pred_len
        self.channels = args.channels if hasattr(args, 'channels') else 1

    def _fit(self, X: np.ndarray) -> None:
        pass

    def _forecast(self, X: np.ndarray) -> np.ndarray:
        repeat_cycles = (self.pred_len - 1) // self.period + 1
        start_idx = -repeat_cycles * self.period

        if len(X.shape) == 2:
            last_period = X[:, start_idx: start_idx + self.period]
            repeated = np.tile(last_period, (1, repeat_cycles))
            forecast = repeated[:, :self.pred_len]
            if self.channels > 1:
                forecast = np.repeat(forecast[:, :, np.newaxis], self.channels, axis=2)
            return forecast
        else:
            # (num, pred_len, channels)
            last_period = X[:, start_idx: start_idx + self.period, :]
            repeated = np.tile(last_period, (1, repeat_cycles, 1))
            return repeated[:, :self.pred_len, :]


class LinearRegressionModel(MLForecastModel):
    def __init__(self, args) -> None:
        super().__init__()
        self.pred_len = args.pred_len
        self.channels = args.channels if hasattr(args, 'channels') else 1
        self.target_channel=args.target
        # 时间序列历史长度
        #self.seq_len = None
        self.model=None

    def _fit(self, X: np.ndarray) -> None:
        pass
        # X=X[0]
        # if len(X.shape) == 1:
        #     y = X
        # else:
        #     y = X[:, -1]
        #
        #
        # X_len=X.shape[0]
        # X_fit = np.arange(1, X_len + 1).reshape(-1, 1)
        #
        # self.model = LinearRegression()
        # self.model.fit(X_fit, y)

    def _forecast(self, X: np.ndarray) -> np.ndarray:
        pass
        # seq_len=self.seq_len
        # X_pred = np.arange(seq_len + 1, seq_len + 1 + self.pred_len).reshape(-1, 1)
        #
        # pred_flat = self.model.predict(X_pred)
        #
        # # 返回形状 (pred_len, 1)
        # return pred_flat[:, np.newaxis]

class ExponentialSmoothing(MLForecastModel):
    def __init__(self, args) -> None:
        super().__init__()
        self.pred_len = args.pred_len
        #self.target_name = getattr(args, 'target', 'OT')

        self.model = None

    def _fit(self, X: np.ndarray) -> None:
        pass
        """
        X=X[0]
        if len(X.shape) == 1:
            y = X
        else:
            y = X[:, -1]

        self.model = SimpleExpSmoothing(y, initialization_method="estimated").fit()
        """
    def _forecast(self, X: np.ndarray) -> np.ndarray:
        pass
        """
        使用拟合的模型预测未来 pred_len 步。
        
        # len(self.model.data.endog) 是序列的长度 T
        # 预测范围从 T 开始，到 T + pred_len - 1 结束
        end_idx = len(self.model.data.endog) + self.pred_len - 1

        pred_flat = self.model.predict(start=len(self.model.data.endog), end=end_idx)

        # 结果必须返回 (pred_len, 1) 形状
        return pred_flat[:, np.newaxis]
        """







