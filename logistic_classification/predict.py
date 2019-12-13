from svm_classification.predict import *
from hbase_out.hbase_outport import *
from hbase_out.hbase_zhengshi import *
from drawmap.map import *
import os


from sklearn.externals import joblib
import  re
import math
import time


class LogisticClassifition:
    #svm模型加载且预处理数据预测
    def __init__(self,ssl_path,lrl_path):
        self.ss = joblib.load(ssl_path)  # 加载模型,会保存该model文件
        self.lr = joblib.load(lrl_path)

        # 预测
    def predict(self,X_test):
        X_test = self.ss.transform(X_test)  # 数据标准化
        Y_predict = self.lr.predict(X_test)  # 预测
        Y_predict_proba = self.lr.predict_proba(X_test)
        return Y_predict,Y_predict_proba

    def predict_proba(self,X_test):
        X_test = self.ss.transform(X_test)  # 数据标准化

        Y_predict = self.lr.predict_proba(X_test)  # 预测
        return Y_predict

    # def load_model(self):
    #     joblib.load("logistic_ss1.model")  # 加载模型,会保存该model文件
    #     joblib.load("logistic_lr1.model")
    #     self.clf = joblib.load(self.model_path)

    def get_sign(self,content):
        timepattern = re.compile(r'time:(.*?),source')
        sourcepattern = re.compile(r'source:(.*?),lon')
        lonpattern = re.compile(r'lon:(.*?),lat')
        latpattern = re.compile(r'lat:(.*?),thead')
        theadpattern = re.compile(r'thead:(.*?),sog')
        sogpattern = re.compile(r'sog:(.*?),cog')
        cogpattern = re.compile(r'cog:(.*?),status')
        time_date = re.findall(timepattern, content)[0]
        source = int(re.findall(sourcepattern, content)[0])
        lon = float(re.findall(lonpattern, content)[0])
        lat = float(re.findall(latpattern, content)[0])
        thead = float(re.findall(theadpattern, content)[0])
        sog = float(re.findall(sogpattern, content)[0])
        cog = float(re.findall(cogpattern, content)[0])
        return {'time':time_date,'source':source,'lon':lon,'lat':lat,'thead':thead,'sog':sog,'cog':cog}


    @staticmethod
    def angles(llon, llat, rlon, rlat):
        angle = 0
        dy = rlat - llat
        dx = rlon - llon
        if dx == 0 and dy > 0:
            angle = 0
        if dx == 0 and dy < 0:
            angle = 180
        if dy == 0 and dx > 0:
            angle = 90
        if dy == 0 and dx < 0:
            angle = 270
        if dx > 0 and dy > 0:
            angle = math.atan(dx / dy) * 180 / math.pi
        elif dx < 0 and dy > 0:
            angle = 360 + math.atan(dx / dy) * 180 / math.pi
        elif dx < 0 and dy < 0:
            angle = 180 + math.atan(dx / dy) * 180 / math.pi
        elif dx > 0 and dy < 0:
            angle = 180 + math.atan(dx / dy) * 180 / math.pi
        return angle

    @staticmethod
    def get_angle(a1, a2):
        return min(abs(a1 - a2), 360 - abs(a1 - a2))



    def get_feature(self,dic1,dic2,dic3,flag):
        try:
            time1 = self.get_second(dic2['time']) - self.get_second(dic1['time'])
            time2 = self.get_second(dic3['time']) - self.get_second(dic2['time'])
        except:
            time1 = int(dic2['time']) - int(dic1['time'])
            time2 = int(dic3['time']) - int(dic2['time'])

        source1 = 1 if dic1['source'] != '300' else -1
        source2 = 1 if dic2['source'] != '300' else -1
        lon1 = (dic2['lon'] - dic1['lon']) * 10000/time1
        lat1 = (dic2['lat'] - dic1['lat']) * 10000/time1
        lon2 = (dic3['lon'] - dic2['lon']) * 10000/time2
        lat2 = (dic3['lat'] - dic2['lat']) * 10000/time2
        ang1 = self.angles(dic1['lon'],dic1['lat'],dic2['lon'],dic2['lat'])
        ang2 = self.angles(dic2['lon'],dic2['lat'],dic3['lon'],dic3['lat'])
        thead1 = self.get_angle(dic1['thead'],ang1)
        thead2 = self.get_angle(dic2['thead'], ang1)  # 目标点航首向和角度差
        thead3 = self.get_angle(dic2['thead'], ang2)  # 目标点航首向和角度差
        thead4 = self.get_angle(dic3['thead'], ang2)  # 目标点后一个点航首向和角度差
        sog1 = dic2['sog'] - dic1['sog'] # 目标点前一个点的航速
        sog2 = dic3['sog'] - dic2['sog']  # 目标点的航速
        # sog3 = dic3['sog']  # 目标点后一个点的航速
        last_is_ture = 1 if flag == 1 else -1
        # if
        # [time1, time2, source, lon1, lat1, lon2, lat2, ang1, ang2, thead1, thead2, thead3, thead4, sog1, sog2, sog3, cog1, cog2, cog3, last_is_ture]
        return [source1,source2,lon1,lat1,lon2,lat2,ang1,ang2,thead1,thead2,thead3,thead4,sog1,sog2,last_is_ture]


    @staticmethod
    def get_second(string):
        timeArray = time.strptime(string, "%Y-%m-%d %H:%M:%S")
        timeStamp = int(time.mktime(timeArray))
        return timeStamp


    def predict_string(self,contents):
        content_list = []
        pattern = re.compile(r'(mmsi:.*?status:\d)')
        index1 = re.findall(pattern, contents)
        for i in index1:
            content_list.append(i)
        res_dic = dict()
        res_dic['feature'] = list()
        count = len(content_list)
        for i in range(count):
            res_dic['feature'].append(self.get_sign(content_list[i]))

        for i in range(count):
            if i == 0:
                res_dic['predict'] = [-1, ]
            elif i == count-1:
                res_dic['predict'].append(-1)
            else:
                features = self.get_feature(res_dic['feature'][i-1],res_dic['feature'][i],res_dic['feature'][i+1],res_dic['predict'][i-1])
                print(np.array([features]))
                res_dic['predict'].append(int(self.lr.predict([features])))
        return res_dic

    def predict_list(self,lis):
        res_dic = dict()
        res_dic['feature'] = list()
        count = len(lis)
        for i in range(count):
            for j in range(len(lis[i])):
                if j in [1,2,3,4,6,7,8] and not lis[i][j]:
                    if i != 0:
                        lis[i][j] = lis[i-1][j]
                    else:
                        lis[i][j] = lis[i+1][j]
            time_date = int(lis[i][7])
            source = int(lis[i][1])
            lon = float(lis[i][4])
            lat = float(lis[i][3])
            thead = float(lis[i][8])
            sog = float(lis[i][6])
            cog = float(lis[i][2])
            res_dic['feature'].append({'time': time_date, 'source': source, 'lon': lon, 'lat': lat, 'thead': thead, 'sog': sog, 'cog': cog})
        for i in range(count):
            if i == 0:
                res_dic['predict'] = [-1, ]
            elif i == count-1:
                res_dic['predict'].append(-1)
            else:
                features = self.get_feature(res_dic['feature'][i-1],res_dic['feature'][i],res_dic['feature'][i+1],res_dic['predict'][i-1])
                print(i,features)
                a,b = self.predict(np.array([features]))
                a = int(a[0])
                if i == 500:
                    print(b)
                if a == 1 and b[0][1] <= 0.5:
                    a = -1

                res_dic['predict'].append(a if int(res_dic['feature'][i]['sog']) > 3 and features[0] < 7200 else -1)
                # print('%s   %s  %s' % (i,features,self.predict_proba(np.array([features]))))
                # if a == 1 and int(res_dic['feature'][i]['sog']) > 3 and features[0] < 7200:
                #     print("%s  %s" % (i,features))
                # res_dic['zhixin'].append(self.clf.predict_proba([features]))
                # if i in [126, 272, 274, 301, 304, 310, 314, 319, 321, 331, 343, 344, 345, 347, 367, 370, 383, 391, 393]:
                #     print(i,a, b)
        return res_dic



if __name__ == '__main__':
    m = Map()
    hbase_out = HbaseZS()
    log = LogisticClassifition('logistic_ss4.model','logistic_lr4.model')
    # hbase_out.scanTable('ship_historical_trace')
    mmsi_list = [477752100, 477752200, 477181700, 477942400, 477454300, 477548400, 477167300, 414436000, 477150500, 413828000, 565003000, 574375000,
        419001327, 477686500,
        372632000, 636015455, 538004459, 564290000, 538004367, 538004243, 477264400, 477435100, 353242000, 563077300, 419001333]




    for i in mmsi_list:
        i = 538004459
        # i = 477181700
        print(i)
        res = hbase_out.get_data_from_zs(i)
        if len(res) < 3:
            continue
        res_list = list()
        point_list = list()
        print(len(res))
        for row in res:
            res_list.append([row['mmsi'],row['source'],row['cog'],row['latitude'],row['longitude'],row['rot'],row['sog'],row['time'],row['trueHeading']])
            point_list.append([float(row['longitude']),float(row['latitude'])])
        point_dic = log.predict_list(res_list)

        start = 200
        end = 245
        index_list = list(i[0] for i in list(enumerate(point_dic['predict'])) if i[1] == 1)
        zhixin_list = list()
        c = []
        print(index_list)
        for k in range(start,end):
            if k in index_list:
                c.append(k-start)
        print(c)

        #-----------
        # for j in list(enumerate(res[start:end])):
        #     if j[0] in c:
        #         print('%s time:%s  lon:%s  lat:%s  sog:%s  source:%s' % (j[0], j[1]['time'], j[1]['longitude'], j[1]['latitude'], j[1]['sog'], j[1]['source']))
        #         timeStamp = int(j[1]['time'])
        #         timeArray = time.localtime(timeStamp)
        #         otherStyleTime = time.strftime("%Y-%m-%d %H:%M:%S", timeArray)
        #         print("%s  %s" % (j[0],otherStyleTime))
        #-----------
        m.draw_point(point_list[start:end],c)
        for j in list(enumerate(res[start:end])):
            print('%s time:%s  lon:%s  lat:%s  sog:%s  source:%s' % (j[0],j[1]['time'],j[1]['longitude'],j[1]['latitude'],j[1]['sog'],j[1]['source']))
        m.draw_line(point_list[start:end],c)
        m.show()
        break
