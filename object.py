from dataclasses import dataclass
from enum import Enum
import datetime
from requests import Session
import json
class Interval(Enum):
    """
    Interval of bar data.
    """
    MINUTE = "1m"
    MINUTE_5m = "5m"
    MINUTE_15m = "15m"
    MINUTE_30m = "30m"
    HOUR = "1h"
    HOUR2 = "2h"
    DAILY = "d"
    WEEKLY = "w"
    TICK = "tick"

class Source(Enum):
    BINANCE = "BINANCE"
    COINGECKO = "COINGECKO"

@dataclass
class BarData(object):
    """
    Candlestick bar data of a certain trading period.
    """

    symbol: str
    datetime: datetime
    source: Source = None
    interval: Interval = None
    close_price: float = 0


def get_contract_id(contract_address):
    url = 'https://pro-api.coingecko.com/api/v3/coins/ethereum/contract/{}?'.format(contract_address)
    parameters = {
        'x_cg_pro_api_key': 'CG-SHrqFbXTRczQXXoWmg7DmaGZ'
    }
    headers = {
        'cache - control': 'max-age=30,public,must-revalidate,s-maxage=30',
        'content-type': 'application/json; charset=utf-8',
    }
    session = Session()
    session.headers.update(headers)
    try:
        response = session.get(url, params=parameters)
        token_data = json.loads(response.text)
        id = token_data['id']
        return id
    # except ConnectionError as e:
    #     print("error")
    #     return (e)
    except:
        return None
print('')