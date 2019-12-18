"""
demo08_lr.py  线性回归
"""
import numpy as np
import matplotlib.pyplot as mp

train_x = np.array([0.5, 0.6, 0.8, 1.1, 1.4])
train_y = np.array([5.0, 5.5, 6.0, 6.8, 7.0])

times = 1000
w0, w1 = 1, 1
lrate = 0.01
for i in range(times):
	# 求得w0方向上的偏导数用于更新w0
	d0 = (w0 + w1*train_x - train_y).sum()
	# 求得w1方向上的偏导数用于更新w1
	d1 = (train_x*(w0 + w1*train_x - train_y)).sum()
	# 更新w0与w1
	w0 = w0 - d0*lrate
	w1 = w1 - d1*lrate
print('w0:', w0, '  w1:', w1)
# 画图
mp.figure('Linear Regression', facecolor='lightgray')
mp.title('Linear Regression', fontsize=18)
mp.grid(linestyle=':')
mp.scatter(train_x, train_y, color='dodgerblue',
	s=80, marker='o', label='Samples')
# 绘制回归线
pred_train_y = w0 + w1*train_x
mp.plot(train_x, pred_train_y, color='orangered',
	label='Regression Line', linewidth=2)
mp.legend()
mp.show()
