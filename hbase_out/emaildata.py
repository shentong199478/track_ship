import numpy as np
import re
import time
import math
from excel_use import *
import pandas as pd

class EmailData(object):
    def __init__(self,data,del_one):
        self.data = data
        self.del_one = del_one


    def get_data(self):
        pattern = re.compile(r'(mmsi:.*?status:\d)')
        index1 = re.findall(pattern, self.data)
        timelist1 = list()
        sourcelist = list()
        lonlist = list()
        latlist = list()
        theadlist = list()
        soglist = list()
        coglist = list()

        timepattern = re.compile(r'time:(.*?),source')
        sourcepattern = re.compile(r'source:(.*?),lon')
        lonpattern = re.compile(r'lon:(.*?),lat')
        latpattern = re.compile(r'lat:(.*?),thead')
        theadpattern = re.compile(r'thead:(.*?),sog')
        sogpattern = re.compile(r'sog:(.*?),cog')
        cogpattern = re.compile(r'cog:(.*?),status')
        for i in index1:
            timelist1.append(re.findall(timepattern, i)[0])
            sourcelist.append(int(re.findall(sourcepattern,i)[0]))
            lonlist.append(float(re.findall(lonpattern, i)[0]))
            latlist.append(float(re.findall(latpattern, i)[0]))
            theadlist.append(float(re.findall(theadpattern, i)[0]))
            soglist.append(float(re.findall(sogpattern, i)[0]))
            coglist.append(float(re.findall(cogpattern, i)[0]))
        timelist = list()
        for i in timelist1:
            timelist.append(self.get_second(i))
        return index1,timelist,sourcelist,lonlist,latlist,theadlist,soglist,coglist



    def get_second(self,string):
        timeArray = time.strptime(string, "%Y-%m-%d %H:%M:%S")
        timeStamp = int(time.mktime(timeArray))
        return timeStamp





    def angles(self,llon, llat, rlon, rlat):
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

    def get_angle(self,a1, a2):
        return min(abs(a1 - a2), 360 - abs(a1 - a2))

    def get_feature(self,point_list,index,timelist,sourcelist,lonlist,latlist,theadlist,soglist,coglist,del_one):
        for i in range(len(timelist)-2):
            time1 = timelist[i+1] - timelist[i] #标记点和前一个点的时间差
            time2 = timelist[i+2] - timelist[i+1] #标记点后一个点和标记点的时间差
            if sourcelist[i] == sourcelist[i+1]:
                source = 1 #同源为1，不同源为-1
            else:
                source = -1
            lon1 = (lonlist[i+1] - lonlist[i])*10000   #标记点和前一个点的经度差
            lat1 = (latlist[i+1] - latlist[i])*10000   #标记点和前一个点的纬度差
            lon2 = (lonlist[i+2] - lonlist[i+1])*10000   #标记点和后一个点的经度差
            lat2 = (latlist[i+2] - latlist[i+1])*10000   #标记点和后一个点的纬度差
            ang1 = self.angles(lonlist[i], latlist[i], lonlist[i + 1], latlist[i + 1])  # 目标点和前一个点的坐标角度
            ang2 = self.angles(lonlist[i+1], latlist[i+1], lonlist[i+2], latlist[i+2])  # 目标点后一个点和目标点的坐标角度
            thead1 = self.get_angle(theadlist[i],ang1) #目标点前一个点航首向和角度差
            thead2 = self.get_angle(theadlist[i+1],ang1) #目标点航首向和角度差
            thead3 = self.get_angle(theadlist[i+1], ang2)  # 目标点航首向和角度差
            thead4 = self.get_angle(theadlist[i+2], ang2)  # 目标点后一个点航首向和角度差
            sog1 = soglist[i]  #目标点前一个点的航速
            sog2 = soglist[i+1] #目标点的航速
            sog3 = soglist[i+2] #目标点后一个点的航速
            cog1 = coglist[i]      #目标点前一个点的航迹向
            cog2 = coglist[i+1]    #目标点的航迹向
            cog3 = coglist[i+2]    #目标点后一个点的航迹向
            if str(i) in self.del_one:
                last_is_ture = 1
            else:
                last_is_ture = -1
            if str(i+1) in self.del_one:
                l = 1
            else:
                l = -1
            point_list.append([time1,time2,source,lon1,lat1,lon2,lat2,ang1,ang2,thead1,thead2,thead3,thead4,sog1,sog2,sog3,cog1,cog2,cog3,last_is_ture,l,index[i+1]])
        return point_list

    def get_point(self):
        index,timelist, sourcelist,lonlist, latlist, theadlist, soglist, coglist = self.get_data()
        point_list = list()
        res = self.get_feature(point_list,index,timelist,sourcelist,lonlist,latlist,theadlist,soglist,coglist,del_one)
        return res

def read_excel():
    df = pd.read_excel('标记数据源.xls')
    return df['labeldata'],df['del']
if __name__ == '__main__':
    res1,res2 = read_excel()
    a = ExcelTools()
    for i in range(len(res1)):
        data = res1[i]
        del_one = str(res2[i]).split()

#     labeldata = '''mmsi: 477548400,time:2019-10-07 21:38:00,source: 9046,lon: 121.189384,lat: 39.795235,thead: 37,sog: 12.9,cog:38.2,status:0
# mmsi: 477548400,time:2019-10-07 21:50:00,source: 300,lon: 121.23019,lat: 39.83526,thead: 38,sog: 12.3,cog:39.600002,status:0
# mmsi: 477548400,time:2019-10-07 22:14:00,source: 9046,lon: 121.28492,lat: 39.887085,thead: 37,sog: 10.1,cog:38.2,status:0
# mmsi: 477548400,time:2019-10-07 22:38:00,source: 300,lon: 121.12112,lat: 39.729233,thead: 38,sog:12.900001,cog:39.3,status:0
# mmsi: 477548400,time:2019-10-08 01:50:00,source: 300,lon: 121.5032,lat: 40.17391,thead: 8,sog: 0.3,cog:93.8,status:1
# mmsi: 477548400,time:2019-10-08 02:56:00,source: 300,lon: 121.50308,lat: 40.173965,thead: 11,sog: 0.2,cog:114.9,status:1'''
#     del_one = [3,]
        del_str = ''
        for i in del_one:
            del_str += str(i)
            del_str += ' '
        datafrom = [[data,del_str],]
        emaildata = EmailData(data,del_one)
        res = emaildata.get_point()
        book_name_xls    = '邮箱标签数据111.xls'
        sheet_name_xls = '邮箱数据111'
        value_title = [["time", "time2","source", "lon1","lat1", "lon2","lat2", "ang1","ang2","thead1","thead2","thead3","thead4","sog1",\
            "sog2","sog3","cog1","cog2","cog3","last_is_ture","predict","index"], ]

        # a.write_excel_xls(book_name_xls, sheet_name_xls, value_title)
        data_name_xls = "标记数据源1111.xls"
        sheet_xls = '1'
        value_title1 = [["labeldata","del"],]
        # a.write_excel_xls(data_name_xls, sheet_xls, value_title1)
        a.write_excel_xls_append(book_name_xls, res)
        a.write_excel_xls_append(data_name_xls, datafrom)






