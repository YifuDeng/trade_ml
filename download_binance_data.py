"""
    author: Yifu
    我们使用币安原生的api进行数据爬取.
"""

import pandas as pd
import time
from datetime import datetime
import requests
import pytz
from binancehttp import BinanceHttp
from db_manager import DataManager
pd.set_option('expand_frame_repr', False)  #
from object import BarData, Interval, Source

BINANCE_SPOT_LIMIT = 1000
BINANCE_FUTURE_LIMIT = 1500
from threading import Thread


def generate_datetime(timestamp: float) -> datetime:
    """
    :param timestamp:
    :return:
    """
    dt = datetime.utcfromtimestamp(timestamp/1000)
    return dt


def get_binance_data(symbol: str, exchanges: str, start_time: str, end_time: str):
    """
    爬取币安交易所的数据
    :param symbol: BTCUSDT.
    :param exchanges: 现货、USDT合约, 或者币币合约.
    :param start_time: 格式如下:2020-1-1 或者2020-01-01
    :param end_time: 格式如下:2020-1-1 或者2020-01-01
    :return:
    """

    api_url = ''
    save_symbol = symbol
    gate_way = 'BINANCES'

    if exchanges == 'spot':
        print("spot")
        limit = BINANCE_SPOT_LIMIT
        save_symbol = symbol.lower()
        gate_way = 'BINANCE'
        api_url = f'https://api.binance.com/api/v3/klines?symbol={symbol}&interval=15m&limit={limit}'

    elif exchanges == 'future':
        print('future')
        limit = BINANCE_FUTURE_LIMIT
        api_url = f'https://fapi.binance.com/fapi/v1/klines?symbol={symbol}&interval=1m&limit={limit}'

    elif exchanges == 'coin_future':
        print("coin_future")
        limit = BINANCE_FUTURE_LIMIT
        f'https://dapi.binance.com/dapi/v1/klines?symbol={symbol}&interval=1m&limit={limit}'

    else:
        raise Exception('交易所名称请输入以下其中一个：spot, future, coin_future')

    start_time = int(datetime.strptime(start_time, '%Y-%m-%d').timestamp() * 1000)
    end_time = int(datetime.strptime(end_time, '%Y-%m-%d').timestamp() * 1000)

    while True:
        try:
            print(start_time)
            url = f'{api_url}&startTime={start_time}'
            print(url)
            data = requests.get(url=url, timeout=10, proxies=proxies).json()
            if not data:
                return
            """
            [
                [
                    1591258320000,      // 开盘时间
                    "9640.7",           // 开盘价
                    "9642.4",           // 最高价
                    "9640.6",           // 最低价
                    "9642.0",           // 收盘价(当前K线未结束的即为最新价)
                    "206",              // 成交量
                    1591258379999,      // 收盘时间
                    "2.13660389",       // 成交额(标的数量)
                    48,                 // 成交笔数
                    "119",              // 主动买入成交量
                    "1.23424865",      // 主动买入成交额(标的数量)
                    "0"                 // 请忽略该参数
                ]

            """



            for l in data:
                # bar = BarData(
                #     symbol=save_symbol,
                #     source=Source.BINANCE.value,
                #     datetime=generate_datetime(l[0]),
                #     interval=Interval.MINUTE_15m.value,
                #     # volume=float(l[5]),
                #     # open_price=float(l[1]),
                #     # high_price=float(l[2]),
                #     # low_price=float(l[3]),
                #     close_price=float(l[4])
                # )

                # id = match[save_symbol]
                timestamp = l[0]
                date = generate_datetime(l[0])
                volume = float(l[5])
                open_price = float(l[1])
                high_price = float(l[2])
                low_price = float(l[3])
                close_price= float(l[4])
                interval = Interval.MINUTE.value
                bar = [save_symbol, timestamp, date, volume, open_price, high_price, low_price, close_price, interval]

                buf.append(bar)


            # 到结束时间就退出, 后者收盘价大于当前的时间.
            if (data[-1][0] > end_time) or data[-1][6] >= (int(time.time() * 1000) - 60 * 1000):
                break

            if data[-1][0] != start_time:
                start_time = data[-1][0]
            else:
                break

        except Exception as error:
            print(error)
            time.sleep(10)




def download_spot(symbol):
    """
    下载现货数据的方法.
    :return:
    """
    # t1 = Thread(target=get_binance_data, args=(symbol, 'spot', "2022-1-1", "2022-1-10"))
    # t1.start()
    # t1.join()
    t1 = Thread(target=get_binance_data, args=(symbol, 'spot', "2020-1-1", "2020-6-30"))

    t2 = Thread(target=get_binance_data, args=(symbol, 'spot', "2020-7-1", "2020-12-31"))

    t3 = Thread(target=get_binance_data, args=(symbol, 'spot', "2021-1-1", "2021-6-30"))

    t4 = Thread(target=get_binance_data, args=(symbol, 'spot', "2021-7-1", "2021-12-31"))

    t5 = Thread(target=get_binance_data, args=(symbol, 'spot', "2022-1-1", "2022-6-10"))

    t1.start()
    t2.start()
    t3.start()
    t4.start()
    t5.start()

    t1.join()
    t2.join()
    t3.join()
    t4.join()
    t5.join()



def download_future(symbol):
    """
    下载合约数据的方法。
    :return:
    """
    t1 = Thread(target=get_binance_data, args=(symbol, 'future', "2022-1-1", "2022-3-1"))
    t2 = Thread(target=get_binance_data, args=(symbol, 'future', "2022-3-2", "2022-6-10"))

    t1.start()
    t2.start()

    t1.join()
    t2.join()


if __name__ == '__main__':

    # 如果你有代理你就设置，如果没有你就设置为 None 或者空的字符串 "",
    # 但是你要确保你的电脑网络能访问币安交易所，你可以通过 ping api.binance.com 看看过能否ping得通
    proxy_host = ""  # 如果没有就设置为"", 如果有就设置为你的代理主机如：127.0.0.1
    proxy_port = "1087"  # 设置你的代理端口号如: 1087, 没有你修改为0,但是要保证你能访问api.binance.com这个主机。

    proxies = None
    if proxy_host and proxy_port:
        proxy = f'http://{proxy_host}:{proxy_port}'
        proxies = {'http': proxy, 'https': proxy}
    table_name = 'binance_data'
    # symbol_bi = BinanceHttp().get_trading_pairs()[0:3]
    # symbol_list = []
    # for item in symbol_bi:
    #     symbol_list.append(item.replace('USDT', ''))
    #init datamanager and create table symbolprice
    db = DataManager()
    cur = db.connect()
    # match = db.match_id(symbol_list)
    # errorlist = []
    # for s in symbol_bi:
    #     symbol_ = s.replace('USDT', '')
    #
    #     try:
    #         id = match[symbol_]
    #     except:
    #         errorlist.append(s)
    #         continue

    # print(errorlist)
    db.create_table(table_name)
    symbol = 'ETHBUSD'
    buf = []
    download_future(symbol) # 下载现货的数据.
    buf = pd.DataFrame(buf, columns=['symbol', 'timestamp_ms', 'date_time', 'volume', 'open_price', 'high_price', 'low_price', 'close_price', 'interval'])
    buf.drop_duplicates('timestamp_ms', inplace=True, ignore_index=True)
    print(buf)
    db.insert_multiple_data_trade_history(buf, table_name)
    db.commit()
    #download data
    # succeed = []
    # fail = []
    # for symbol in symbol_bi:
    #     if symbol not in errorlist:
    #         buf = []
    #         download_spot(symbol) # 下载现货的数据.
    #         buf = pd.DataFrame(buf, columns=['symbol', 'timestamp', 'date', 'open_price', 'high_price', 'low_price', 'close_price', 'interval'])
    #         buf.drop_duplicates('timestamp', inplace=True, ignore_index=True)
    #         if len(buf) != 0:
    #             try:
    #                 db.insert_multiple_data_trade_history(buf, 'test3')
    #                 succeed.append(symbol)
    #                 db.commit()
    #             except:
    #                 print('fail')
    #                 print(symbol)
    #                 fail.append(symbol)



    db.close()
    #
    # print(fail)
    # print('------')
    # print(succeed)
