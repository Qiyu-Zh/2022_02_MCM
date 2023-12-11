import warnings
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from sklearn.preprocessing import MinMaxScaler

__all__ = [
    'Account',
    'PriceDataSet',
    'resLSTM',
    'LSTM',
    'create_seq_data',
    'create_seq_data_by_df_list',
    'train_test',
    'tranform_data',
    'create_seq_data_for_day_2',
    'create_seq_data_for_day_3',
    'normalize_1D',
    'MACD_method'
    ]


class Account():
    def __init__(self, df, a_g = 0.01, a_b = 0.02):
        self.C = 1000
        self.G = 0
        self.B = 0
        self.mask = df['Mask'].values
        self.gp = df['Org Gold Price'].values
        self.bp = df['Org Bit Price'].values
        self.day = -1
        self.alpha_g = a_g
        self.alpha_b = a_b
        self.track = {'Cash': [], 'Bitcoin': [], 'Gold': [], 'Total': []}

    def setDay(self, day):
        if day < self.day:
            raise ValueError('You cannot do the time travel')
        self.day = day
    
    def sellGold(self, num):
        if num <= 0:
            raise ValueError('Trade number should be positive')
        if not self.mask[self.day]:
            raise ValueError("Cannot sell gold when market is closed")
        if self.G < num:
            warnings.warn('Attemp to sell ' + str(num) + ' gold but only remain ' + str(self.G))
            num = self.G
        self.G -= num
        self.C += num * self.gp[self.day] * (1 - self.alpha_g)

    def sellBit(self, num):
        if num <= 0:
            raise ValueError("Trade number should be positive")
        if self.B < num:
            warnings.warn('Attemp to sell ' + str(num) + ' bitcoin but only remain ' + str(self.B))
            num = self.B
        self.B -= num
        self.C += num * self.bp[self.day] * (1 - self.alpha_b)

    def buyGold(self, num):
        if num <= 0:
            raise ValueError("Trade number should be positive")
        if not self.mask[self.day]:
            raise ValueError("Cannot buy gold when market is closed")
        if self.C - num * self.gp[self.day] * (1 + self.alpha_g) < 0:
            num = self.C / (self.gp[self.day] * (1 + self.alpha_g))
            warnings.warn("No enough cash to buy the gold")
        self.C -= num * self.gp[self.day] * (1 + self.alpha_g)
        self.G += num

    def buyBit(self, num):
        if num <= 0:
            raise ValueError("Trade number should be positive")
        if self.C - num * self.bp[self.day] * (1 + self.alpha_b) < 0:
            num = self.C / (self.bp[self.day] * (1 + self.alpha_b))
            warnings.warn("No enough cash to buy the bitcoin")
        self.C -= num * self.bp[self.day] * (1 + self.alpha_b)
        self.B += num

    def isGoldOpen(self):
        return self.mask[self.day]

    def getState(self):
        return {'Cash': self.C, 'Bitcoin': self.B, 'Gold': self.G, 'Total': self.C + self.gp[self.day] * self.G * (1 - self.alpha_g) + self.bp[self.day] * self.B* (1 - self.alpha_b)}

    def writeState(self):
        self.track['Cash'].append(self.C)
        self.track['Bitcoin'].append(self.B)
        self.track['Gold'].append(self.G)
        self.track['Total'].append(self.C + self.gp[self.day] * self.G * (1 - self.alpha_g) + self.bp[self.day] * self.B* (1 - self.alpha_b))

class PriceDataSet(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float)
        self.y = torch.tensor(y, dtype=torch.float)

    def __getitem__(self, index):
        return self.X[index], self.y[index]

    def __len__(self):
        return len(self.X)

class resLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(resLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)

        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out + x[:, -1, 0].reshape(-1, 1)
        
class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(LSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)

        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

def create_seq_data_by_df_list(df_list, arg):
    data_feat, data_target = [],[]
    for df in tqdm(df_list):
        for index in range(len(df) - arg.seq_size):
            data_feat.append(df[[arg.label] * arg.input_size][index: index + arg.seq_size].values)
            data_target.append(df[arg.label][index + arg.seq_size])
    data_feat = np.array(data_feat)
    data_target = np.array(data_target)
    return data_feat, data_target

def create_seq_data(df, arg):
    data_feat, data_target = [],[]
    for index in range(len(df) - arg.seq_size):
        data_feat.append(df[[arg.label] * arg.input_size][index: index + arg.seq_size].values)
        data_target.append(df[arg.label][index + arg.seq_size])
    data_feat = np.array(data_feat)
    data_target = np.array(data_target)
    return data_feat, data_target

def train_test(data_feat, data_target, test_set_size, seq, input_dim = 8):
    train_size = data_feat.shape[0] - (test_set_size) 
    trainX = data_feat[:train_size].reshape(-1, seq, input_dim)
    testX  = data_feat[train_size:].reshape(-1, seq, input_dim)
    trainY = data_target[:train_size].reshape(-1, 1)
    testY  = data_target[train_size:].reshape(-1, 1)
    return trainX, trainY, testX, testY

def tranform_data(x_label, y_label, seq, input_dim = 8):
    X = x_label.reshape(-1, seq, input_dim)
    y  = y_label.reshape(-1, 1)
    return X, y

def create_seq_data_for_day_2(labels, day1, arg):
    data_feat = []
    for index in range(len(labels) - arg.seq_size - 1):
        x = [[i] * arg.input_size for i in labels[index + 1: index + arg.seq_size]]
        x.append([day1[index]] * arg.input_size)
        data_feat.append(x)
    data_feat = np.array(data_feat)
    return data_feat.astype(np.float32)

def create_seq_data_for_day_3(labels, day1, day2, arg):
    data_feat = []
    for index in range(len(labels) - arg.seq_size - 2):
        x = [[i] * arg.input_size for i in labels[index + 2: index + arg.seq_size]]
        x.append([day1[index]] * arg.input_size)
        x.append([day2[index]] * arg.input_size)
        data_feat.append(x)
    data_feat = np.array(data_feat)
    return data_feat.astype(np.float32)

def normalize_1D(arr):
    return MinMaxScaler(feature_range=(-1, 1)).fit_transform(arr.reshape(-1, 1)).flatten()

def _ema(arr):
    N = len(arr)
    a = 2/(N+1)
    data = np.zeros(len(arr))
    for i in range(len(data)):
        data[i] = arr[i] if i==0 else a*arr[i]+(1-a)*data[i-1]
    return data[-1]
    
def EMA(arr,period):
    data = np.full(arr.shape,np.nan)
    for i in range(period-1,len(arr)):
        data[i] = _ema(arr[i+1-period:i+1])
    return data

def MA(arr, N):
    data = np.zeros(len(arr))
    for i in range(len(arr)):
        data[i] = np.mean(arr[i-N:i])
    return data

class MACD_method:
    def __init__(self, df, col_name = 'Price'):
        self. df = df
        self. S = 12
        self.L = 26
        self.position = 0
        self.name = col_name

    def derivative(self, array):
        data = np.zeros(len(array))
        for i in range(len(array)):
            data[i] = array[i] if i==0 else array[i]-array[i-1]
        return data

    def processing(self):
        self.df['EMA_S'] = EMA(self.df[self.name].values, period=12)
        self.df['EMA_L'] = EMA(self.df[self.name].values, period=26)
        self.df['dif'] = self.df['EMA_S'] - self.df['EMA_L']

        DEA = MA(self.df['dif'].values,12)
        MACD = 2*(self.df['dif'].values - DEA)
        self.df['DEA'] = DEA
        self.df['MACD'] = MACD
        self.df['Signal'] = EMA(MACD, 9)

    def get_strategy(self):
        self.df['Choice'] = 0
        choice = 0
        for i in range(3, self.df.shape[0]):
            if self.df['MACD'][i-3] <0 and self.df['MACD'][i-1] > 0:
                choice = 1
            if self.df['MACD'][i-3] >0 and self.df['MACD'][i-1] < 0:
                choice = -1
            self.df['Choice'][i] = choice
