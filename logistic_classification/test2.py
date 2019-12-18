"""
demo05_lr.py  逻辑回归
"""
import numpy as np
import matplotlib.pyplot as mp
import sklearn.linear_model as lm

x = np.array([ [3, 1],
		       [2, 5],
		       [1, 8],
		       [6, 4],
		       [5, 2],
		       [3, 5],
		       [4, 7],
		       [4, -1]])
y = np.array([0, 1, 1, 0, 0, 1, 1, 0])

# 训练模型
model = lm.LogisticRegression(
	solver='liblinear', C=1)
model.fit(x, y)
# 模拟预测
r = model.predict([[1,8], [5,8], [8,3]])
print(r)

# 画图
mp.figure('Simple Classification', facecolor='lightgray')
mp.title('Simple Classification', fontsize=16)
# 绘制分类边界线
n = 500
l, r = x[:,0].min()-1, x[:,0].max()+1
b, t = x[:,1].min()-1, x[:,1].max()+1
grid_x, grid_y = np.meshgrid(np.linspace(l, r, n),
	        				 np.linspace(b, t, n))
# 根据业务，模拟预测
mesh_x = np.column_stack(
	(grid_x.ravel(), grid_y.ravel()))
grid_z = model.predict(mesh_x)
# 把grid_z 变维：(500,500)
grid_z = grid_z.reshape(grid_x.shape)

mp.pcolormesh(grid_x, grid_y, grid_z, cmap='gray')
mp.scatter(x[:,0], x[:,1], s=80,
	c=y, cmap='brg_r', label='Samples')
mp.legend()
mp.show()




