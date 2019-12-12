from sklearn.externals import joblib
import  re
import math
import os
import time


class SvcClassifition:
    #svm模型加载且预处理数据预测
    def __init__(self,model_path):
        self.model_path = model_path
        self.load_model()

    def load_model(self):
        self.clf = joblib.load(self.model_path)

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
        lon1 = (dic2['lon'] - dic1['lon']) * 10000
        lat1 = (dic2['lat'] - dic1['lat']) * 10000
        lon2 = (dic3['lon'] - dic2['lon']) * 10000
        lat2 = (dic3['lat'] - dic2['lat']) * 10000
        ang1 = self.angles(dic1['lon'],dic1['lat'],dic2['lon'],dic2['lat'])
        ang2 = self.angles(dic2['lon'],dic2['lat'],dic3['lon'],dic3['lat'])
        thead1 = self.get_angle(dic1['thead'],ang1)
        thead2 = self.get_angle(dic2['thead'], ang1)  # 目标点航首向和角度差
        thead3 = self.get_angle(dic2['thead'], ang2)  # 目标点航首向和角度差
        thead4 = self.get_angle(dic3['thead'], ang2)  # 目标点后一个点航首向和角度差
        sog1 = dic1['sog'] # 目标点前一个点的航速
        sog2 = dic2['sog']  # 目标点的航速
        sog3 = dic3['sog']  # 目标点后一个点的航速
        cog1 = dic1['cog']  # 目标点前一个点的航迹向
        cog2 = dic2['cog']  # 目标点的航迹向
        cog3 = dic3['cog']  # 目标点后一个点的航迹向
        last_is_ture = 1 if flag == 1 else -1
        # if
        # [time1, time2, source, lon1, lat1, lon2, lat2, ang1, ang2, thead1, thead2, thead3, thead4, sog1, sog2, sog3, cog1, cog2, cog3, last_is_ture]
        return [time1,time2,source1,source2,lon1,lat1,lon2,lat2,ang1,ang2,thead1,thead2,thead3,thead4,sog1,sog2,sog3,cog1,cog2,cog3,last_is_ture]

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
                print([features])
                res_dic['predict'].append(int(self.clf.predict([features])))
        return res_dic

    def predict_list(self,lis):
        res_dic = dict()
        res_dic['feature'] = list()
        count = len(lis)
        for i in range(count):
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
                res_dic['zhixin'] = [1,]
            elif i == count-1:
                res_dic['predict'].append(-1)
                res_dic['zhixin'].append(1)
            else:
                features = self.get_feature(res_dic['feature'][i-1],res_dic['feature'][i],res_dic['feature'][i+1],res_dic['predict'][i-1])
                res_dic['predict'].append(int(self.clf.predict([features])) if res_dic['feature'][i]['sog'] < 5 or features[0] < 7200 else -1)
                # res_dic['zhixin'].append(self.clf.predict_proba([features]))
        return res_dic


if __name__ == '__main__':
    model_path = os.path.join(os.getcwd(),"2.model")
    m = SvcClassifition(model_path)
    a = '''mmsi: 477752100,time:2019-12-09 11:02:00,source: 9046,lon: -43.48028,lat: 3.8937933,thead: 95,sog: 10.4,cog:124.0,status:0
mmsi: 477752100,time:2019-12-09 13:28:00,source: 300,lon: -43.18454,lat: 3.6898334,thead: 90,sog: 10.0,cog:124.6,status:0
mmsi: 477752100,time:2019-12-09 14:04:00,source: 300,lon: -43.105392,lat: 3.6324966,thead: 20,sog: 9.8,cog:125.6,status:0
mmsi: 477752100,time:2019-12-09 14:06:00,source: 300,lon: -43.05704,lat: 3.5951467,thead: 38,sog: 9.7,cog:127.9,status:0'''
    print(m.predict_string(a)['predict'])
    print(list(i[0] for i in list(enumerate(m.predict_string(a)['predict'])) if i[1] == 1))

    # [['477752100', '9046', '124.0', '3.8937933', '-43.48028', '-3', '10.4', '1575860520', '129'],
    #     ['477752100', '300', '124.6', '3.6898334', '-43.18454', '7', '10.0', '1575869280', '132'],
    #     ['477752100', '300', '125.6', '3.6324966', '-43.105392', '-1', '9.8', '1575871440', '132'],
    #     ['477752100', '300', '127.9', '3.5951467', '-43.05704', '-3', '9.7', '1575871560', '132']]


