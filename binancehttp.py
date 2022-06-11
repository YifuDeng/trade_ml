# coding:utf-8
# Author : klein
# CONTACT: zhangchy2@shanghaitech.edu.cn
# SOFTWARE: PyCharm
# FILE : get_trading_num.py
# DATE : 2022/3/26 21:52

import matplotlib.pyplot as plt
import requests
import time
import hmac
import hashlib
from enum import Enum
from threading import Thread, Lock
import pandas as pd
from collections import defaultdict


class RequestMethod(Enum):
    """
    请求的方法.
    """
    GET = 'GET'
    POST = 'POST'
    PUT = 'PUT'
    DELETE = 'DELETE'


class BinanceHttp(object):

    def __init__(self, key=None, secret=None, host=None, timeout=5):
        self.key = key
        self.secret = secret
        self.host = host if host else "https://api.binance.com"
        self.recv_window = 5000
        self.timeout = timeout
        self.order_count_lock = Lock()
        self.order_count = 1_000_000

    def build_parameters(self, params: dict):
        keys = list(params.keys())
        keys.sort()
        return '&'.join([f"{key}={params[key]}" for key in params.keys()])

    def request(self, req_method: RequestMethod, path: str, requery_dict=None, verify=False):
        url = self.host + path

        if verify:
            query_str = self._sign(requery_dict)
            url += '?' + query_str
        elif requery_dict:
            url += '?' + self.build_parameters(requery_dict)
        # print(url)
        headers = {"X-MBX-APIKEY": self.key}

        content = requests.request(req_method.value, url=url, headers=headers, timeout=self.timeout)
        return content.json()

    def server_time(self):
        path = '/fapi/v1/time'
        return self.request(req_method=RequestMethod.GET, path=path)

    def _timestamp(self):
        return int(time.time() * 1000)

    def _sign(self, params):
        requery_string = self.build_parameters(params)
        hexdigest = hmac.new(self.secret.encode('utf8'), requery_string.encode("utf-8"), hashlib.sha256).hexdigest()
        return requery_string + '&signature=' + str(hexdigest)

    def get_k_lines(self, symbol, interval="15m"):
        params = {"symbol": symbol, "interval": interval, "limit": 30}
        k_lines = self.request(RequestMethod.GET, "/fapi/v1/klines", params, verify=True)
        return k_lines

    def get_trading_pairs(self):
        exchangeinfo = self.request(RequestMethod.GET, "/api/v3/exchangeInfo")
        symbols = []
        for symbol in exchangeinfo["symbols"]:
            if symbol['symbol'][-4:] == 'USDT':
                symbols.append(symbol["symbol"])
        return symbols

    def process(self, k_lines):
        trading_volumes = []
        for k in k_lines:
            trading_volume = float(k[5])
            trading_volumes.append(trading_volume)

        trading_volumes = trading_volumes[:-1]
        # 如果当前交易量大于之前所有交易量和，则返回true
        # print(f"Last three minites: {trading_volumes[-1]}, {trading_volumes[-2]}")
        if trading_volumes[-1] > 10*trading_volumes[-2]:
            print(f"Trading Volume: {trading_volumes[-1]}, {trading_volumes[-2]}")
            return True
        else:
            return False

    def detect(self):
        symbols = self.get_trading_pairs()
        while True:
            print(f"Time: {time.asctime(time.localtime(time.time()))}")
            for symbol in symbols:
                k_lines = self.get_k_lines(symbol)
                mode = self.process(k_lines)
                if mode:
                    print("*"*50, symbol, "*"*50)
            time.sleep(900)

    def plot_trading_volume(self):
        k_lines = self.get_k_lines("BTCUSDT")
        # get trading volume
        trading_volumes = []
        for k in k_lines:
            trading_volume = float(k[5])
            print(f"Timestamp: {time.localtime(k[0]//1000)}, trading_volume: {trading_volume}")
            trading_volumes.append(trading_volume)

        trading_volumes = trading_volumes[:-1]
        print(trading_volumes)
        plt.plot(trading_volumes)
        plt.savefig("trading_volume.png")
        # plt.show()


if __name__ == '__main__':
    # import pandas as pd
    pd.set_option('display.max_columns', None)  # 设置列不限制数量
    pd.set_option('display.max_rows', None)

    key = "gJmd7PYotUABeUJudlCdPP1i9uOa5etxof8AfzASjFZ5YxLxwGtdY5XyPHj27fuS"
    secret = "FUmdcL3K8gAFgmVVFDzqUEhlK9UTM4sJEBIggta6uccds1PaIlS3YRt34qAQVWm1"
    binance_future = BinanceHttp(key=key, secret=secret)
    trading_pairs = ["ETHUSDT", "ETHBUSD"]
    a = binance_future.get_trading_pairs()
    print(len(a))
    # print(binance_future.get_account_mode())
    # 1. 取余额 usdt+busd+bnb
    # print(binance_future.is_valid())
    # a = binance_future.get_initial_state()
    # # print(a)
    # # 钱包余额、保证金余额、已使用保证金
    # print(a)
    # 注意： 统计指标中，除了总收益和当日收益是绝对数字，其他都要加百分号 %
    # 初始钱包余额，昨天保证金余额，初始的时间戳，交易对
    # a = binance_future.get_current_state(init_walletBalance=660, last_day_totalMarginBalance=700, init_timestamp=1647100819240, trading_pairs=trading_pairs)
    # print(a)

    # 2. 查询划转历史
    # binance_spot = BinanceFutureHttp(key=key, secret=secret, host="https://api.binance.com")
    # b = binance_future.get_transfer('BUSD')
    # print(b)