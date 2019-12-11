from svm_classification.predict import *
from hbase_out.hbase_outport import *
from hbase_out.hbase_zhengshi import *
from drawmap.map import *
import os
model_path = os.path.join(os.getcwd(),r"svm_classification\2.model")
m = Map()
hbase_out = HbaseZS()
svc = SvcClassifition(model_path)
# hbase_out.scanTable('ship_historical_trace')
mmsi_list = [477752100, 477752200, 477181700, 477942400, 477454300, 477548400, 477167300, 414436000, 477150500, 413828000, 565003000, 574375000,
    419001327, 477686500,
    372632000, 636015455, 538004459, 564290000, 538004367, 538004243, 477264400, 477435100, 353242000, 563077300, 419001333]




for i in mmsi_list:
    res = hbase_out.get_data_from_zs(i)
    if len(res) > 3:
        res_list = list()
        point_list = list()
        for row in res:
            res_list.append([row['mmsi'],row['source'],row['cog'],row['latitude'],row['longitude'],row['rot'],row['sog'],row['time'],row['trueHeading']])
            point_list.append([float(row['longitude']),float(row['latitude'])])
        # res_list = [['477752100', '9046', '124.0', '3.8937933', '-43.48028', '-3', '10.4', '1575860520', '129'],
        #         ['477752100', '300', '124.6', '3.6898334', '-43.18454', '7', '10.0', '1575869280', '132'],
        #         ['477752100', '300', '125.6', '3.6324966', '-43.105392', '-1', '9.8', '1575871440', '132'],
        #         ['477752100', '300', '127.9', '3.5951467', '-43.05704', '-3', '9.7', '1575871560', '132']]
        point_dic = svc.predict_list(res_list)
        # print(point_dic['predict'])

        # print(point_dic['predict'])

        # point_list = sorted(point_dic['time'])
        # print(point_list)
        start = 55
        end = 60
        index_list = list(i[0]-start for i in list(enumerate(point_dic['predict'])) if i[1] == 1)
        zhixin_list = list()
        for i in index_list:
            zhixin_list.append(point_dic['zhixin'][i])
        print(zhixin_list)
        m.draw_point(point_list[start:end],index_list)
        for i in res[start:end]:
            print('time:%s  lon:%s  lat:%s' % (i['time'],i['longitude'],i['latitude']))
        print([i-start for i in  index_list if i in range(start,end)])
        # print(res_list[0:4])
        m.draw_line(point_list[start:end],index_list)
        m.show()
        break
        # m.show()
        # x = input('111')
        # if x == '\n':
        #     continue