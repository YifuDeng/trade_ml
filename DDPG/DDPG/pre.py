import argparse
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from torchvision import transforms
import numpy as np
from datetime import date
import datetime
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from torch import nn
import torch
from torch.autograd import Variable
import matplotlib.pyplot as plt
import psycopg2

class preprocessing():
    def __init__(self, symbol):
        self.symbol = symbol


    def get_bar_data(self, start_date, end_date,interval):
        conn = psycopg2.connect(database='yifu_db', user='postgres', password='1Change3.',
                                host='localhost', port='5432')
        curs = conn.cursor()

        sql = """
            select
                *
            from binance_data
            where (symbol = '{}')
                and (date_time between '{}' and '{}')
                and time_interval = '{}'
            order by timestamp_ms
        """.format(self.symbol, start_date, end_date, interval)

        curs.execute(sql)

        data = curs.fetchall()
        data = pd.DataFrame(data, columns=['symbol', 'timestamp_ms', 'date_time', 'volume', 'open_price', 'high_price', 'low_price', 'close_price', 'interval'])

        curs.close()
        conn.close()
        return data

    def Evaluate(self, y_test, y_pre):
        # MAPE = 100 * np.mean(np.abs((y_pre - y_test) / y_test))
        MAE = mean_absolute_error(y_test, y_pre)
        R2 = r2_score(y_test, y_pre)
        MSE = mean_squared_error(y_test, y_pre)
        RMSE = np.sqrt(mean_squared_error(y_test, y_pre))
        M = [MAE, R2, MSE, RMSE]
        return M

    def get_standart_data(self, start_date, end_date,interval):
        bar_data = self.get_bar_data(start_date, end_date, interval)
        bar_data['up'] = bar_data['high_price'] - bar_data['open_price']
        bar_data['down'] = bar_data['open_price'] - bar_data['low_price']
        bar_data = bar_data[['open_price', 'close_price','up','down','volume']]

        # self.close_min_up = bar_data['up'].min()
        # self.close_max_up = bar_data["up"].max()
        # self.close_min_down = bar_data['down'].min()
        # self.close_max_down = bar_data["down"].max()
        # self.bar_data = bar_data.apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)))
        return bar_data

    def create_dataset(self, bar_data, sequence_length):
        total_len = len(bar_data)
        X = []
        Y = []
        for i in range(total_len - sequence_length):
            X.append(np.array(bar_data.iloc[i:(i + sequence_length), ].values, dtype=np.float32))
            Y.append(np.array(bar_data.iloc[(i + sequence_length), [2, 3]], dtype=np.float32))
        # print(len(X))
        # print(len(Y))
        # print(X[0])
        # print(Y[0])

        trainx, trainy = X[: int(0.7 * total_len)], Y[: int(0.7 * total_len)]
        testx, testy = X[int(0.7 * total_len):], Y[int(0.7 * total_len):]
        train_loader = DataLoader(dataset=Mydataset(trainx, trainy, transform=transforms.ToTensor()), batch_size=12,
                                  shuffle=True)
        test_loader = DataLoader(dataset=Mydataset(testx, testy), batch_size=12, shuffle=True)
        return train_loader,test_loader




class Mydataset(Dataset):

    def __init__(self, xx, yy, transform=None):
        self.x = xx
        self.y = yy
        self.tranform = transform

    def __getitem__(self, index):
        x1 = self.x[index]
        y1 = self.y[index]
        if self.tranform != None:
            return self.tranform(x1), y1
        return x1, y1

    def __len__(self):
        return len(self.x)
