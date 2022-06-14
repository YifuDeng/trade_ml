
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
# Note: 现在的包含所有币种的数据库中的数据时间是用的上海时间

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

        self.close_min_up = bar_data['up'].min()
        self.close_max_up = bar_data["up"].max()
        self.close_min_down = bar_data['down'].min()
        self.close_max_down = bar_data["down"].max()
        self.bar_data = bar_data.apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)))
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
        return train_loader, test_loader


def turn_around(model, array, seq, num):
    position = 0.0
    profit = 0.0
    total_profit = 0.0
    trade_volume = 0.0
    total_profit_list = [total_profit]
    position_list = [position]
    for i in range(len(array) - seq):
        x = array[i: i+seq]
        x = x.unsqueeze(0)

        pred = model(x)
        #
        # pre_up = pred[0][0].item() * (1-3*position)
        # pre_down = pred[0][1].item() * (1+3*position)
        pre_up = pred[0][0].item()
        pre_down = pred[0][1].item()
        real_up = array[i + seq][2].item()
        real_down = array[i + seq][3].item()
        open_price = array[i + seq][0].item()
        close_price = array[i + seq][1].item()
        if pre_up < real_up:
            position -= num
            profit += (open_price + pre_up) * num
            trade_volume += (open_price + pre_up) * num
        if pre_down < real_down:
            position += num
            profit -= (open_price - pre_down) * num
            trade_volume += (open_price - pre_down) * num
        total_profit = position * close_price + profit + trade_volume*0.0001
        total_profit_list.append(total_profit)
        position_list.append(position)
    print(position_list[-100:])
    plt.plot(position_list)
    plt.show()
    print(total_profit_list[-100:])
    plt.plot(total_profit_list)
    plt.show()


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




class lstm(nn.Module):

    def __init__(self, input_size=5, hidden_size=32, output_size=2):
        super(lstm, self).__init__()
        # lstm的输入 #batch,seq_len, input_size
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.output_size = output_size
        self.rnn = nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_size, batch_first=True)
        self.linear = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, x):
        out, (hidden, cell) = self.rnn(x)  # x.shape : batch,seq_len,hidden_size , hn.shape and cn.shape : num_layes * direction_numbers,batch,hidden_size
        a, b, c = hidden.shape
        out = self.linear(hidden.reshape(a * b, c))
        return out

if __name__ == "__main__":


    pd.set_option('display.max_columns', None)
    da = preprocessing('ETHBUSD')
    bar_data = da.get_standart_data('2022-05-01', '2022-06-10', '1m')
    print(bar_data)
    # train_loader, test_loader = da.create_dataset(bar_data, sequence_length=20)

    model = torch.load('lstm_price.pt')

    test_array = np.array(bar_data, dtype=np.float32)[int(0.7 * len(bar_data)):]
    test_tensor = torch.from_numpy(test_array)
    print(test_tensor)
    # criterion = nn.MSELoss()
    # optimizer = optim.Adam(model.parameters(), lr=0.001)

    # preds = []
    # labels = []
    # for idx, (x, label) in enumerate(test_loader):
    #     x = x.squeeze(1)  # batch_size,seq_len,input_size
    #     pred = model(x)
    #     preds.extend(pred.data.squeeze(1).tolist())
    #     label = label
    #     labels.extend(label.tolist())
    # # print(da.Evaluate(np.array(labels), np.array(preds)))
    #
    #
    # close_max_up = da.close_max_up
    # close_min_up = da.close_min_up
    # close_max_down = da.close_max_down
    # close_min_down = da.close_min_down
    #
    # p1 = np.array([close_max_up - close_min_up, close_max_down-close_min_down], dtype=float)
    # p2 = np.array([close_min_up, close_min_down], dtype=float)
    # # print(p1)
    # # print(p2)
    # # print(np.array(preds[0:50]) * p1 + p2)
    # # print(np.array(labels[0:50]) * p1 + p2)
    # print(da.Evaluate(np.array(labels) * p1 + p2, np.array(preds) * p1 + p2))
    #
    # for idx, (x, label) in enumerate(train_loader):
    #     x = x.squeeze(1)  # batch_size,seq_len,input_size
    #     pred = model(x)
    #     preds.extend(pred.data.squeeze(1).tolist())
    #     label = label
    #     labels.extend(label.tolist())
    # # print(da.Evaluate(np.array(labels), np.array(preds)))
    #
    #
    # close_max_up = da.close_max_up
    # close_min_up = da.close_min_up
    # close_max_down = da.close_max_down
    # close_min_down = da.close_min_down
    #
    # p1 = np.array([close_max_up - close_min_up, close_max_down-close_min_down], dtype=float)
    # p2 = np.array([close_min_up, close_min_down], dtype=float)
    # # print(p1)
    # # print(p2)
    # # print(np.array(preds[0:50]) * p1 + p2)
    # # print(np.array(labels[0:50]) * p1 + p2)
    # print(da.Evaluate(np.array(labels) * p1 + p2, np.array(preds) * p1 + p2))


    # plt.plot([ele * p1 + p2 for ele in np.array(preds[0:50])], "r", label="pred")
    # plt.plot([ele * p1 + p2 for ele in np.array(labels[0:50])], "b", label="real")
    # plt.savefig('b.png')
    # plt.show()
    # ele = preds[0]
    # ele1 = labels[0]
    # print(ele1 * p1 + p2)
    # print(ele * p1 + p2)
    turn_around(model, test_tensor, 20, 0.01)