import numpy as np
import pandas as pd
import argparse

class DatasetBase:
    def __init__(self, args):
        self.ratio_train = args.ratio_train
        self.ratio_val = args.ratio_val
        self.ratio_test = args.ratio_test
        self.split = False
        self.read_data()
        self.split_data()

    def read_data(self):
        raise NotImplementedError

    def split_data(self):
        pass


class M4Dataset(DatasetBase):
    def __init__(self, args):
        self.train_data_path = args.train_data_path
        self.test_data_path = args.test_data_path
        self.type = 'm4'
        super().__init__(args)

    def read_data(self):
        """
        read_data function for M4 dataset(https://github.com/Mcompetitions/M4-methods/tree/master/Dataset).

        :param
            self.data_path: list, [train_data_path: str, test_data_path: str]
        :return
            self.train_data: np.ndarray, shape=(n_samples, timesteps)
            self.test_data: np.ndarray, shape=(n_samples, timesteps)
        """
        train_data_path = self.train_data_path
        train_data = pd.read_csv(train_data_path)
        train_data.set_index('V1', inplace=True)  # 将第一列作为索引，但是这里的第一列是时间，所以这里的索引是时间
        self.train_data = np.array([v[~np.isnan(v)] for v in train_data.values], dtype=object)  # 将数据转换为numpy数组

        test_data_path = self.test_data_path
        test_data = pd.read_csv(test_data_path)
        test_data.set_index('V1', inplace=True)
        self.test_data = np.array([v[~np.isnan(v)] for v in test_data.values], dtype=object)

        self.val_data = None



class ETTDataset(DatasetBase):
    def __init__(self, args):
        self.data_path = args.data_path
        self.target = args.target
        self.type = 'ETT'
        super(ETTDataset, self).__init__(args)

    def read_data(self):
        '''
        read_data function for ETT dataset(https://github.com/zhouhaoyi/ETDataset).

        :param
            self.data_path: str
        :return
            self.data_stamp: data timestamps
            self.data_cols: data columns(features/targets)
            self.data: np.ndarray, shape=(n_samples, timesteps, channels)
        '''
        data = pd.read_csv(self.data_path)
        cols = list(data.columns)
        cols.remove(self.target)
        cols.remove('date')
        data = data[['date'] + cols + [self.target]]
        self.data_stamp = pd.to_datetime(data.date)
        self.data_cols = cols + [self.target]
        self.data = np.expand_dims(data[self.data_cols].values, axis=0)#在第0维度上扩展

    def split_data(self):
        self.split = True
        self.num_train = int(self.ratio_train * self.data.shape[1])
        self.num_val = int(self.ratio_val * self.data.shape[1])
        self.train_data = self.data[:, :self.num_train, :]
        if self.num_val == 0:
            self.val_data = None
        else:
            self.val_data = self.data[:, self.num_train: self.num_train + self.num_val, :]
        self.test_data = self.data[:, self.num_train + self.num_val:, :]


def seasonal_fill_zeros(series: pd.Series, P: int, k: int, fill_fallback_value: float = 0.0) -> pd.Series:
    """
    使用前后相邻 k 个周期同一时间点的非零均值来填充时间序列中的零值。
    """

    X = series.values.astype(float)
    T = len(X)
    is_zero = (X == 0)
    k_half = k // 2

    for i in range(T):
        if is_zero[i]:
            neighbor_indices = []

            # 历史邻居 (前 k_half 个周期)
            for j in range(1, k_half + 1):
                idx = i - j * P
                if idx >= 0:
                    neighbor_indices.append(idx)
                else:
                    break

            # 未来邻居 (后 k_half 个周期)
            for j in range(1, k_half + 1):
                idx = i + j * P
                if idx < T:
                    neighbor_indices.append(idx)
                else:
                    break

            neighbors = X[neighbor_indices]
            non_zero_neighbors = neighbors[neighbors != 0]
            if len(non_zero_neighbors) > 0:
                fill_value = non_zero_neighbors.mean()
            else:
                # 回退策略：如果所有邻居都是零或不存在，使用回退值
                fill_value = fill_fallback_value
            X[i] = fill_value

    return pd.Series(X, index=series.index)

class CustomDataset(DatasetBase):
    def __init__(self, args):
        self.data_path = args.data_path
        self.target = args.target
        self.type = 'Custom'
        super().__init__(args)

    def get_seasonal_params(self):
        """根据数据集名称获取周期P"""
        path_name = self.data_path.split('/')[-1].split('.')[0].lower()

        if 'weather' in path_name:
            P = 144
        elif 'electricity' in path_name or 'traffic' in path_name:
            P = 24
        elif 'exchange' in path_name:
            P = 7
        elif 'illness' in path_name:
            P = 48
        else:
            print("Warning: Unknown dataset name. Using P=1.")
            P = 1
        return P

    def read_data(self):
        '''
        read_data function for other datasets:
            all the .csv files can be found here (https://box.nju.edu.cn/d/b33a9f73813048b8b00f/)
            Traffic (http://pems.dot.ca.gov)
            Electricity (https://archive.ics.uci.edu/ml/datasets/ElectricityLoadDiagrams20112014)
            Exchange-Rate (https://github.com/laiguokun/multivariate-time-series-data)
            Weather (https://www.bgc-jena.mpg.de/wetter/)
            ILI(illness) (https://gis.cdc.gov/grasp/fluview/fluportaldashboard.html)

        :param
            self.data_path: str
        :return
            self.data_stamp: data timestamps
            self.data_cols: data columns(features/targets)
            self.data: np.ndarray, shape=(n_samples, timesteps, channels), where the last channel is the target
        '''
        data = pd.read_csv(self.data_path)
        cols = list(data.columns)
        cols.remove(self.target)
        cols.remove('date')
        data = data[['date'] + cols + [self.target]]
        self.data_stamp = pd.to_datetime(data.date)
        self.data_cols = cols + [self.target]
        numeric_data = data[self.data_cols].copy()

        # k 取相邻邻居的均值填充缺失值 (前后各 3 天/周/年)
        k = 6
        #根据数据集名称获取周期
        P = self.get_seasonal_params()

        # 零值填充
        print(f"Applying seasonal zero-filling (P={P}, k={k})...")
        for col in cols:
            numeric_data[col] = seasonal_fill_zeros(numeric_data[col], P=P, k=k)

        self.data = np.expand_dims(numeric_data.values.astype(float), axis=0)

    def split_data(self):
        self.split = True
        self.num_train = int(self.ratio_train * self.data.shape[1])
        self.num_val = int(self.ratio_val * self.data.shape[1])
        self.train_data = self.data[:, :self.num_train, :]
        if self.num_val == 0:
            self.val_data = None
        else:
            self.val_data = self.data[:, self.num_train: self.num_train + self.num_val, :]
        self.test_data = self.data[:, self.num_train + self.num_val:, :]


def get_dataset(args):
    dataset_dict = {
        'M4': M4Dataset,
        'ETT': ETTDataset,
        'Custom': CustomDataset,
    }
    return dataset_dict[args.dataset](args)



