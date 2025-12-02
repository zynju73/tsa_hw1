import numpy as np

def get_transform(args):
    transform_dict = {
        'IdentityTransform': IdentityTransform,
        'NormalizationTransform': NormalizationTransform,
        'StandardizationTransform': StandardizationTransform,
        'MeanNormalization':MeanTransform,
        'BoxCoxTransform':BoxCoxTransform,
    }
    return transform_dict[args.transform](args)

class Transform:
    """
    Preprocess time series
    """

    def transform(self, data):
        """
        :param data: raw timeseries
        :return: transformed timeseries
        """
        raise NotImplementedError

    def inverse_transform(self, data):
        """
        :param data: raw timeseries
        :return: inverse_transformed timeseries
        """
        raise NotImplementedError


class IdentityTransform(Transform):
    def __init__(self, args):
        pass

    def transform(self, data, update=False):
        return data

    def inverse_transform(self, data):
        return data

class NormalizationTransform(Transform):
    def __init__(self, args):
        self.min_val = None
        self.max_val = None
        self.epsilon = 1e-5

    def transform(self, data, update=False):
        if update or self.min_val is None or self.max_val is None:
            self.min_val = np.min(data)
            self.max_val = np.max(data)

        if self.max_val - self.min_val<self.epsilon:
            return (data-self.min_val)/self.epsilon

        scaled_data = (data - self.min_val) / (self.max_val - self.min_val)
        return scaled_data

    def inverse_transform(self, data):
        range_val = self.max_val - self.min_val

        if range_val < self.epsilon:
            return data*self.epsilon+self.min_val

        return data * range_val + self.min_val

class StandardizationTransform(Transform):
    def __init__(self, args):
        self.mean = None
        self.std = None
        self.epsilon = 1e-5

    def transform(self, data, update=False):
        if update or self.mean is None or self.std is None:
            self.mean = np.mean(data)
            self.std = np.std(data)
            if self.std < self.epsilon:
                self.std = self.epsilon
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return data * self.std + self.mean


class MeanTransform(Transform):
    def __init__(self, args):
        self.mean=None
        self.min_val=None
        self.max_val=None
        self.epsilon=1e-5

    def transform(self, data, update=False):
        if update or self.mean is None or self.min_val is None or self.max_val is None:
            self.mean = np.mean(data)
            self.min_val = np.min(data)
            self.max_val = np.max(data)
        if (self.max_val-self.min_val)<self.epsilon:
            return (data-self.mean)/self.epsilon
        else:
            return (data-self.mean)/(self.max_val-self.min_val)

    def inverse_transform(self, data):
        if (self.max_val - self.min_val) < self.epsilon:
            return data * self.epsilon+self.mean
        else:
            return data * (self.max_val-self.min_val)+self.mean



class BoxCoxTransform(Transform):
    def __init__(self, args):
        self.lmbda = args.box_cox_lambda

    def transform(self, data, update=False):
        lmbda = self.lmbda
        y_transformed = np.empty_like(data)


        mask_pos = data >= 0
        y_pos = data[mask_pos]
        if lmbda != 0:
            y_transformed[mask_pos] = (np.float_power(y_pos + 1, lmbda) - 1) / lmbda
        else:  # lambda = 0
            #  log(y_t + 1)
            y_transformed[mask_pos] = np.log(y_pos + 1)


        mask_neg = data < 0
        y_neg = data[mask_neg]
        neg_term = -y_neg + 1
        if lmbda != 2:
            denominator = (2.0 - lmbda)

            y_transformed[mask_neg] = - (np.float_power(neg_term, denominator) - 1) / denominator
        else:
            y_transformed[mask_neg] = - np.log(neg_term)

        return y_transformed

    def inverse_transform(self, data):
        lmbda = self.lmbda
        y_t = np.empty_like(data)

        #y_t transformed 之后符号不变
        #y_t >=0
        mask_pos_trans = data >= 0
        y_prime_pos = data[mask_pos_trans]

        if lmbda != 0:
            y_t[mask_pos_trans] = np.float_power(y_prime_pos * lmbda + 1, 1 / lmbda) - 1
        else:
            y_t[mask_pos_trans] = np.exp(y_prime_pos) - 1

        #y_t<0
        mask_neg_trans = data < 0
        y_prime_neg = data[mask_neg_trans]

        if lmbda != 2:
            denominator = (2.0 - lmbda)
            y_t[mask_neg_trans] = 1 - np.float_power(1-y_prime_neg * denominator , 1 / denominator)
        else:
            y_t[mask_neg_trans] = 1 - np.exp(-y_prime_neg)

        return y_t