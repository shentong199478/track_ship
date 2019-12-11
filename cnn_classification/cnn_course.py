import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
# 下载minist数据集
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('mnist_data', one_hot=True)
# 定义输入输出形状
input_x = tf.placeholder(tf.float32, [None, 28 * 28]) / 255
output_y = tf.placeholder(tf.int32, [None, 10])
# 输入层，输入数据reshape成四维数据，其中第一维的数据代表了图片数量，因为现在还不清楚图片的数量，因此用-1代替，当传输进数据时python会自动识别出数据的第一维的数值。
input_x_images = tf.reshape(input_x, [-1, 28, 28, 1])
test_x = mnist.test.images[:3000]  # 读取mnist数据集的测试集图片，读取3000组
test_y = mnist.test.labels[:3000]  # 读取测试集数据的标签
# 第一层结构
# 使用2维的卷积层tf.layers.conv2d，使用32个5X5的滤波器，使用relu激活函数
conv1 = tf.layers.conv2d(
    inputs=input_x_images,
    filters=32,
    kernel_size=[5, 5],
    strides=1,
    padding='same',
    activation=tf.nn.relu
)
# 最大值pooling操作，数据量减半
print(conv1)  # 输出为[28,28，32]
pool1 = tf.layers.max_pooling2d(
    inputs=conv1,
    pool_size=[2, 2],
    strides=2
)
print(pool1)  # 输出为[14,14,32]
# 第二层结构
# 使用64个5X5的滤波器
conv2 = tf.layers.conv2d(
    inputs=pool1,
    filters=64,
    kernel_size=[5, 5],
    strides=1,
    padding='same',
    activation=tf.nn.relu
)
print(conv2)  # 输出为[14,14,64]
pool2 = tf.layers.max_pooling2d(
    inputs=conv2,
    pool_size=[2, 2],
    strides=2
)
print(pool2)  # 输出为[7,7,64]
# 平坦化操作，将数据变成3136个数据，为全连接层做准备
flat = tf.reshape(pool2, [-1, 7 * 7 * 64])
# 全连接层tf.layers.dense
dense = tf.layers.dense(
    inputs=flat,
    units=1024,
    activation=tf.nn.relu
)
print(dense)  # 输出为1024个数据
# dropout操作，丢弃率设置为0.5，即一半的神经元丢弃不工作，防止过拟合
dropout = tf.layers.dropout(
    inputs=dense,
    rate=0.5
)
print(dropout)  # 输出仍为1024个数据，dropout会对输出数据进行scale up
# 输出层，就是一个简单的全连接层，没有使用激活函数
outputs = tf.layers.dense(
    inputs=dropout,
    units=10
)
print(outputs)  # 输出为[10,1]
# 使用交叉熵定义损失函数
loss = tf.losses.softmax_cross_entropy(onehot_labels=output_y, logits=outputs)
print(loss)
# 训练操作，学习率设置为0.001
train_op = tf.train.GradientDescentOptimizer(0.001).minimize(loss)
# 定义精确率，输出预测值与图片标签符合的概率
accuracy_op = tf.metrics.accuracy(
    labels=tf.argmax(output_y, axis=1),  # 返回张量维度上最大值的索引
    predictions=tf.argmax(outputs, axis=1)
)
print(accuracy_op)
sess = tf.Session()
# 初始化所有变量
init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
sess.run(init)
for i in range(20000):
    # 取训练的小批次数据，50组数据（图片+标签）
    batch = mnist.train.next_batch(50)
    # 训练操作，求训练损失
    train_loss, train_op_ = sess.run([loss, train_op], {input_x: batch[0], output_y: batch[1]})
    # 每训练迭代100次就输出观察训练的损失函数和测试数据（3000组）的精确度
    if i % 100 == 0:
        # 用测试集的数据监测模型训练的效果
        test_accuracy = sess.run(accuracy_op, {input_x: test_x, output_y: test_y})
        print("Step=%d, Train loss=%.4f,Test accuracy=%.2f" % (i, train_loss, test_accuracy[0]))

# 测试
test_output = sess.run(outputs, {input_x: test_x[:20]})
inferenced_y = np.argmax(test_output, 1)
print(inferenced_y, 'Inferenced numbers')  # 打印输出预测数字
print(np.argmax(test_y[:20], 1), 'Real numbers')  # 打印输出标签数字
sess.close()