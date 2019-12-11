# -*- coding:utf-8 -*-
import happybase
import time

class HbaseUtil(object):
    #测试库hbase使用
    def __init__(self):
        pass

    # 获取一个连接
    def getHbaseConnection(self):
        conn = happybase.Connection(host='192.168.200.11', port=9090, timeout=None, autoconnect=True,
                                    table_prefix=None, table_prefix_separator=b'_', compat='0.98',
                                    transport='buffered', protocol='binary')
        return conn

    # 返回单行数据，返回tuple
    def querySingleLine(self, table, rowkey):
        conn = self.getHbaseConnection()
        t = happybase.Table(table, conn)
        return t.row(rowkey)

    # 返回多行数据，返回dict
    def queryMultilLines(self, table, list):
        conn = self.getHbaseConnection()
        t = happybase.Table(table, conn)
        return dict(t.rows(list))

    # 批量插入数据
    def batchPut(self, table):
        conn = self.getHbaseConnection()
        t = happybase.Table(table, conn)
        batch = t.batch(batch_size=10)
        return batch

    # exmple
    # batch_put = batchPut(table)
    #     person1 = {'info:name': 'lianglin', 'info:age': '30', 'info:addr': 'hubei'}
    #     person2 = {'info:name': 'jiandong', 'info:age': '22', 'info:addr': 'henan', 'info:school': 'henandaxue'}
    #     person3 = {'info:name': 'laowei', 'info:age': '29'}
    #     with batch_put as bat:
    #         bat.put('lianglin', person1)
    #         bat.put('jiandong', person2)
    #         bat.put('laowei', person3)

    # 插入单条数据
    def singlePut(self, table, rowkey, data):
        conn = self.getHbaseConnection()
        t = happybase.Table(table, conn)
        t.put(rowkey, data=data)

    # 批量删除数据
    def batchDelete(self, table, rowkeys):
        conn = self.getHbaseConnection()
        t = happybase.Table(table, conn)
        with t.batch() as bat:
            for rowkey in rowkeys:
                bat.delete(rowkey)

    # 删除单行数据
    def singleDelete(self, table, rowkey):
        conn = self.getHbaseConnection()
        t = happybase.Table(table, conn)
        t.delete(rowkey)

    # 删除多个列族的数据
    def deleteColumns(self, table, rowkey, columns):
        conn = self.getHbaseConnection()
        t = happybase.Table(table, conn)
        t.delete(rowkey, columns=columns)

    # 删除一个列族中的几个列的数据
    def deleteDetailColumns(self, table, rowkey, detailColumns):
        conn = self.getHbaseConnection()
        t = happybase.Table(table, conn)
        t.delete(rowkey, columns=detailColumns)

    # 清空表
    def truncatTable(self, table, name, families):
        conn = self.getHbaseConnection()
        conn.disable_table(table)
        conn.delete_table(table)
        conn.create_table(table, name, families)

    # 删除hbase中的表
    def deletTable(self, table):
        conn = self.getHbaseConnection()
        conn.disable_table(table)
        conn.delete_table(table)

    # 扫描一张表
    def scanTable(self, table): #, row_start, row_stop, row_prefix
        conn = self.getHbaseConnection()
        t = happybase.Table(table, conn)
        scan = t.scan(limit=10,filter="SingleColumnValueFilter('info', 'mmsi', =, 'substring:100704002')") #row_start=row_start, row_stop=row_stop, row_prefix=row_prefix
        for key, value in scan:
            for i,j in value.items():
                print(key.decode('utf8'),i.decode('utf8'),j.decode('utf8'))
        # rows = t.rows('mmsi:538004459')
        # print(rows)
        # for key,value in rows:
        #     for i,j in value.items():
        #         print(key.decode('utf8'),i.decode('utf8'),j.decode('utf8'))

    def scanfilter(self,mmsi,table,query):
        conn = self.getHbaseConnection()
        t = happybase.Table(table, conn)
        ed = int(time.time())
        bg = ed - 8640000
        # query_str = "ColumnPrefixFilter('your_prsifx_str') AND TimestampsFilter(your_timestamp)"
        # filter = "SingleColumnValueFilter('f', 'id', =, 'substring:852223')", limit = 10
        # for k, v in t.scan(filter=query,columns=["motion:mmsi","motion:rot","motion:sog"]):
        res = list()
        row_start = "{0}{1}".format(mmsi, bg)
        row_stop = "{0}{1}".format(mmsi, ed)
        for k, v in t.scan(filter=query,columns=["info:mmsi","info:source","motion:cog","motion:latitude",\
                "motion:longitude","motion:rot","motion:sog","motion:time","motion:trueHeading"],row_start=row_start,row_stop=row_stop):
            res.append([v[b"info:mmsi"].decode('utf8'),v[b"info:source"].decode('utf8'),v[b"motion:cog"].decode('utf8'),v[b"motion:latitude"].decode('utf8'),\
                v[b"motion:longitude"].decode('utf8'),v[b"motion:rot"].decode('utf8'),v[b"motion:sog"].decode('utf8'),v[b"motion:time"].decode('utf8'),
                v[b"motion:trueHeading"].decode('utf8')])

        # for k,v in t.scan(filter=query,):
        #     print(v[b"info:mmsi"])
            # for i,j in v.items():
            #     print(j.decode('utf8'))
            # for i,j in v.items():
            #     res.append([])
                # print(j.decode('utf8'))
        return res

if __name__ == '__main__':
    hbase_out = HbaseUtil()
    # hbase_out.scanTable('ship_historical_trace')
    mmsi_list = [100885990,100704002,477752100, 477752200, 477181700, 477942400, 477454300, 477548400, 477167300, 414436000, 477150500, 413828000, 565003000, 574375000,
        419001327, 477686500,
        372632000, 636015455, 538004459, 564290000, 538004367, 538004243, 477264400, 477435100, 353242000, 563077300, 419001333]
    for i in mmsi_list:
        res = hbase_out.scanfilter(i,'ship_historical_trace',"SingleColumnValueFilter('info', 'mmsi', =, 'substring:%s')"%i)
        if len(res) > 3:
            print('*'*30)
            print(res)
            print('-'*30)
    # hbase_out.scanTable('st_ship_historical_trace')

    # for i in mmsi_list:
    #     res = hbase_out.scanfilter(i,'st_ship_historical_trace',"SingleColumnValueFilter('info', 'mmsi', =, 'substring:%s')"%i)
    #     if res:
    #         print('*'*30)
    #         print(res)
    #         print('-'*30)