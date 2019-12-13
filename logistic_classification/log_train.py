import sklearn
from sklearn.linear_model import LogisticRegressionCV, LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
# from sklearn.linear_model.coordinate_descent import ConvergenceWarning
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import warnings
import os,sys
last_path = os.path.dirname(os.getcwd())
sys.path.append(os.path.join(last_path,'labeldata'))
from get_data2 import *
# 解决中文显示问题
mpl.rcParams["font.sans-serif"] = [u"SimHei"]
mpl.rcParams["axes.unicode_minus"] = False

# 拦截异常
# warnings.filterwarnings(action='ignore', category=ConvergenceWarning)

# 导入数据并对异常数据进行清除
# path = "datas/breast-cancer-wisconsin.data"
# names = ["id", "Clump Thickness", "Uniformity of Cell Size", "Uniformity of Cell Shape"
#     , "Marginal Adhesion", "Single Epithelial Cell Size", "Bare Nuclei", "Bland Chromatin"
#     , "Normal Nucleoli", "Mitoses", "Class"]

# df = pd.read_csv(path, header=None, names=names)

# datas = df.replace("?", np.nan).dropna(how="any")  # 只要列中有nan值，进行行删除操作
# print(datas.head())     #默认显示前五行

# 数据提取与数据分割
# X = datas[names[1:10]]
# Y = datas[names[10]]


X, Y = get_labeldata1()
# 划分训练集与测试集
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, random_state=0)

# 对数据的训练集进行标准化
ss = StandardScaler()
X_train = ss.fit_transform(X_train)  # 先拟合数据在进行标准化

# 构建并训练模型
##  multi_class:分类方式选择参数，有"ovr(默认)"和"multinomial"两个值可选择，在二元逻辑回归中无区别
##  cv:几折交叉验证
##  solver:优化算法选择参数，当penalty为"l1"时，参数只能是"liblinear(坐标轴下降法)"
##  "lbfgs"和"cg"都是关于目标函数的二阶泰勒展开
##  当penalty为"l2"时，参数可以是"lbfgs(拟牛顿法)","newton_cg(牛顿法变种)","seg(minibactch随机平均梯度下降)"
##  维度<10000时，选择"lbfgs"法，维度>10000时，选择"cs"法比较好，显卡计算的时候，lbfgs"和"cs"都比"seg"快
##  penalty:正则化选择参数，用于解决过拟合，可选"l1","l2"
##  tol:当目标函数下降到该值是就停止，叫：容忍度，防止计算的过多
lr = LogisticRegressionCV(multi_class="ovr", fit_intercept=True, Cs=np.logspace(-2, 2, 20), cv=2, penalty="l2", solver="lbfgs", tol=0.01)
re = lr.fit(X_train, Y_train)

# 模型效果获取
r = re.score(X_train, Y_train)
print("R值(准确率):", r)
print("参数:", re.coef_)
print("截距:", re.intercept_)
print("稀疏化特征比率:%.2f%%" % (np.mean(lr.coef_.ravel() == 0) * 100))
print("=========sigmoid函数转化的值，即：概率p=========")
print(re.predict_proba(X_test))  # sigmoid函数转化的值，即：概率p

# 模型的保存与持久化
from sklearn.externals import joblib

joblib.dump(ss, "logistic_ss4.model")  # 将标准化模型保存
joblib.dump(lr, "logistic_lr4.model")  # 将训练后的线性模型保存
joblib.load("logistic_ss4.model")  # 加载模型,会保存该model文件
joblib.load("logistic_lr4.model")

# 预测
print(X_test)
X_test = ss.transform(X_test)  # 数据标准化
Y_predict = lr.predict(X_test)  # 预测

# 画图对预测值和实际值进行比较
x = range(len(X_test))
plt.figure(figsize=(14, 7), facecolor="w")
plt.ylim(0, 6)
plt.plot(x, Y_test, "ro", markersize=8, zorder=3, label=u"真实值")
plt.plot(x, Y_predict, "go", markersize=14, zorder=2, label=u"预测值,$R^2$=%.3f" % lr.score(X_test, Y_test))
plt.legend(loc="upper left")
plt.xlabel(u"数据编号", fontsize=18)
plt.ylabel(u"信息类型", fontsize=18)
plt.title(u"Logistic算法对数据进行分类", fontsize=20)
plt.savefig("Logistic算法对数据进行分类.png")
plt.show()

print("=============Y_test==============")
print(Y_test.ravel())
print("============Y_predict============")
print(Y_predict)

# if __name__ == '__main__':
#     joblib.dump(ss, "logistic_ss1.model")  # 将标准化模型保存
#     joblib.dump(lr, "logistic_lr1.model")  # 将训练后的线性模型保存
#     joblib.load("logistic_ss1.model")  # 加载模型,会保存该model文件
#     joblib.load("logistic_lr1.model")
#     X_test = np.array([[7080, 3960, 1, 1, 2552.3699999999394, -1837.249999999999, 1006.5500000000327, -692.1500000000003, 4.747139251462329, 7.747139251462329,
#         6.514199795444398, 7.514199795444398, 9.5, 9.400001, 9.1, -1]])
#
#     X_test = ss.transform(X_test)  # 数据标准化
#     Y_predict = lr.predict(X_test)
