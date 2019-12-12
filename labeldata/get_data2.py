import pandas as pd
import numpy as np
import random

import numpy as np
import os,sys
sys.path.append(r"D:\shentong\track_ship\labeldata")




def get_labeldata():
    x = pd. read_excel(r"D:\shentong\track_ship\labeldata\正样本2.xlsx")
    zheng = x[['time','time2','source1','source2','lon1','lat1','lon2','lat2','thead1','thead2',\
        'thead3','thead4','sog1','sog2','sog3','last_is_ture','predict']].values
    count = range(367)
    num = random.sample(count, 133)

    for i in num:
        zheng = np.vstack((zheng,zheng[i]))
    zheng = zheng[1:501]


    x = pd.read_excel(r'D:\shentong\track_ship\labeldata\负样本2.xlsx')
    b = x[['time','time2','source1','source2','lon1','lat1','lon2','lat2','thead1','thead2',\
        'thead3','thead4','sog1','sog2','sog3','last_is_ture','predict']].values

    count = range(1033)
    num = random.sample(count, 500)
    fu = b[num[0]]
    for i in num[1:]:
        fu = np.vstack((fu,b[i]))
    fu = fu[1:501]

    # count = range(1001)
    # num = random.sample(count, 1001)


    yangben = np.vstack((zheng,fu))
    permutation = np.random.permutation(yangben.shape[0])
    shuffled_dataset = yangben[permutation]
    data = shuffled_dataset[:,:16]
    label =shuffled_dataset[:,-1]
    return data,label

if __name__ == '__main__':
    data,label = get_labeldata()