import pandas as pd
import psycopg2
import datetime
# from object import BarData, Interval, Source
from io import StringIO

class DataManager(object):
    def __init__(self,database = 'yifu_db', username = 'postgres', password = '1Change3.'):
        self.database = database
        self.username = username
        self.password = password

    def connect(self):
        self.conn = psycopg2.connect(database=self.database, user=self.username, password=self.password, host="localhost", port="5432")
        self.cur = self.conn.cursor()
        return self.cur

    def create_table(self, table: str ='price'):
        self.cur.execute("CREATE TABLE if not exists {} ("
                         "symbol Text,"
                         "timestamp_ms NUMERIC,"
                         "date_time TIMESTAMPTZ,"
                         'volume NUMERIC,'
                         "open_price NUMERIC,"
                         "high_price NUMERIC,"
                         "low_price NUMERIC,"
                         "close_price NUMERIC,"
                         "time_interval Text,"
                         "UNIQUE (symbol, timestamp_ms)"
                         ");".format(table))

    def create_table_cg_market_data(self,table:str ='cg_market_data'):
        self.cur.execute("CREATE TABLE if not exists {} ("
                         "id varchar,"
                         "symbol varchar,"
                         "name varchar,"
                         "market_cap_rank int,"
                         "market_cap float,"
                         "primary key (id)"
                         ");".format(table))

    def create_table_trade_history(self,table:str ='trade_history'):
        self.cur.execute("CREATE TABLE if not exists {} ("
                         "address varchar,"
                         "id varchar,"
                         "symbol varchar,"
                         "coin_contract varchar,"
                         "block NUMERIC,"
                         "timestamp_s NUMERIC,"
                         "date_time TIMESTAMPTZ,"
                         "direction varchar,"
                         "price NUMERIC,"
                         "amount NUMERIC,"
                         "ret NUMERIC,"
                         "hash varchar"
                         ");".format(table))

    def create_table_final(self,table):
        self.cur.execute("CREATE TABLE if not exists {} ("
                         "address varchar,"
                         "id varchar,"
                         "symbol varchar,"
                         "coin_contract varchar,"
                         "block NUMERIC,"
                         "timestamp_s NUMERIC,"
                         "date_time TIMESTAMPTZ,"
                         "direction varchar,"
                         "amount NUMERIC,"
                         "hash varchar"
                         ");".format(table))

    def create_table_trade(self,table):
        self.cur.execute("CREATE TABLE if not exists {} ("
                         "address varchar,"
                         "symbol varchar,"
                         "coin_contract varchar,"
                         "block NUMERIC,"
                         "timestamp_s NUMERIC,"
                         "date_time TIMESTAMPTZ,"
                         "direction varchar,"
                         "amount NUMERIC,"
                         "hash varchar"
                         ");".format(table))

    def get_price(self, contract):
        sql = '''
                select
                    timestamp,
                    open_price
                from cg_price_contract
                where contract = '{}'
            '''.format(contract)
        self.cur.execute(sql)
        data = self.cur.fetchall()
        l = []
        for i in data:
            l.append([int(i[0]), float(i[1])])


        df = pd.DataFrame(l)
        df.set_index([0], inplace=True)
        df = df.sort_index(ascending=True)

        return df

    def get_price_id(self, id):
        sql = '''
                select
                    timestamp,
                    open_price
                from cg_price_contract
                where id = '{}'
                order by timestamp
            '''.format(id)
        self.cur.execute(sql)
        data = self.cur.fetchall()
        l = []
        for i in data:
            l.append([i[0], i[1]])

        df = pd.DataFrame(l)
        return df

    def get_distinct_address(self,table:str):
        self.cur.execute("select distinct address from {}".format(table))
        data = self.cur.fetchall()
        a = []
        for i in data:
            a.append(i[0])
        return a

    def get_distinct(self,table:str,column:str):
        self.cur.execute("select distinct {} from {} where {} is not null".format(column, table, column))
        data = self.cur.fetchall()
        a = []
        for i in data:
            a.append(i[0])
        return a

    def get_contract_info(self):
        sql = '''
                select
                    contract,
                    id,
                    min(timestamp) as min_time,
                    max(timestamp) as max_time
                from cg_price_contract
                group by contract,id
            '''
        self.cur.execute(sql)
        data = self.cur.fetchall()
        a = pd.DataFrame(columns=['id', 'min_time', 'max_time'])
        for i in data:
            a.loc[i[0]] = [i[1], i[2], i[3]]
        return a


    def get_contract_id(self,table:str):
        self.cur.execute("select contract,id from {} group by contract, id".format(table))
        data = self.cur.fetchall()
        a = {}
        for i in data:
            a[i[0]] = i[1]
        return a


    def insert_multiple_data_cg_market_data(self, data: set, table:str='cg_market_data'):
        data_ = str(data)[1:-1].replace('None', '0').replace('nan','0').replace("JPEG'd","JPEG-d")
        self.cur.execute(("INSERT INTO {} (id,symbol,name,market_cap_rank,market_cap) VALUES "+data_).format(table))

    def delete_table(self,table):
        self.cur.execute("DROP TABLE "+table)

    def max_timestamp(self, table:str = 'cg_price'):
        self.cur.execute("select id,symbol,max(timestamp) from {} group by id,symbol".format(table))
        data = self.cur.fetchall()
        a = []
        for i in data:
            a.append([i[0],i[1],int(i[2])])
        return(a)

    def min_timestamp(self, table:str = 'cg_price'):
        self.cur.execute("select id,symbol,min(timestamp) from {} group by id,symbol".format(table))
        data = self.cur.fetchall()
        a = []
        for i in data:
            a.append([i[0],i[1],int(i[2])])
        return(a)
    # def insert_data(self, data: BarData, table:str='price'):
    #     self.cur.execute("INSERT INTO %s (symbol, data_source, datetime, time_interval, close_price) VALUES (%s, %s,%s, %s,%s)",
    #              (table, data.symbol, data.source, str(data.datetime),data.interval,data.close_price))

    def insert_df(self, data: pd.DataFrame, table):
        # data_ = str(data)[1:-1]
        # self.cur.execute(("INSERT INTO {} (id, symbol, timestamp, date, open_price) VALUES "+data_).format(table))

        output = StringIO()
        data.to_csv(output, sep='\t', index=False, header=False)
        output.getvalue()
        # jump to start of stream
        output.seek(0)

        self.cur.copy_from(output, table)


    def insert_multiple_data_trade_history(self, data: pd.DataFrame, table:str='trade_history'):
        # data_ = str(data)[1:-1]
        # self.cur.execute(("INSERT INTO {} (id, symbol, timestamp, date, open_price) VALUES "+data_).format(table))

        output = StringIO()
        data.to_csv(output, sep='\t', index=False, header=False)
        output.getvalue()
        # jump to start of stream
        output.seek(0)

        self.cur.copy_from(output, table)

    def insert_nansen_data(self, data: pd.DataFrame, table):
        # data_ = str(data)[1:-1]
        # self.cur.execute(("INSERT INTO {} (id, symbol, timestamp, date, open_price) VALUES "+data_).format(table))

        output = StringIO()
        data.to_csv(output, sep='\t', index=False, header=False)
        output.getvalue()
        # jump to start of stream
        output.seek(0)

        self.cur.copy_from(output, table)

    def select_all_symbol(self):
        sql = 'select distinct symbol from cg_price_contract'
        self.cur.execute(sql)
        data = self.cur.fetchall()
        a = []
        for i in data:
            a.append(i[0])
        return a

    def select_all_id(self):
        sql = 'select distinct id from cg_market_data'
        self.cur.execute(sql)
        data = self.cur.fetchall()
        a = []
        for i in data:
            a.append(i[0])
        return a

    def select_id_symbol_contract(self):
        sql = '''
                select
                    a.id,
                    b.symbol,
                    b.contract
                from
                (
                select
                    distinct id
                from cg_price
                )a
                left join
                 (
                     select id,
                            symbol,
                            contract
                     from cg_data_with_ethcontract
                 )b
                on a.id = b.id
            '''
        self.cur.execute(sql)
        data = self.cur.fetchall()
        a = []
        for i in data:
            a.append([i[0],i[1],i[2]])
        return a

    def select_all_id(self, table:str = 'price'):
        self.cur.execute("select distinct id from {}".format(table))
        data = self.cur.fetchall()
        a = []
        for i in data:
            a.append(i[0])
        return(a)

    def match_id(self, symbols:list = None):
        self.cur.execute("select id,symbol from cg_market_data")
        data = self.cur.fetchall()
        match = {}
        if symbols:
            for symbol in symbols:
                for pair in data:
                    if pair[1] == symbol:
                        match[symbol] = pair[0]
        else:
            for pair in data:
                match[pair[1]] = pair[0]

        return match

    def match_symbol(self, ids:list = None):
        self.cur.execute("select id,symbol from cg_market_data")
        data = self.cur.fetchall()
        match = {}
        if ids:
            for id in ids:
                for pair in data:
                    if pair[0] == id:
                        match[id] = pair[1]
        else:
            for pair in data:
                match[pair[0]] = pair[1]

        return match

    def commit(self):
        self.conn.commit()

    def close(self):
        self.conn.close()


if __name__ == '__main__':
    db = DataManager()

    db.connect()
    a = db.get_price('BTC')
    print(a)



    db.close()