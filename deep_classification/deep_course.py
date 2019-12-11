from __future__ import absolute_import, division, print_function, unicode_literals

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

import tensorflow as tf

from tensorflow import feature_column
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split

tf.enable_eager_execution()

URL = 'https://storage.googleapis.com/applied-dl/heart.csv'
dataframe = pd.read_csv(URL)

train1, test1 = train_test_split(dataframe, test_size=0.2)
# train, val = train_test_split(train, test_size=0.2)
train = list()
train_label = list()
for index, row in train1.iterrows():
    train.append([row["age"], row['trestbps'], row['chol'], row['thalach'], row['oldpeak'], row['slope'], row['ca']])
    train_label.append(row['target'])
train = np.array(train)
train_label = np.array(train_label)
test = list()
test_label = list()
for index, row in test1.iterrows():
    test.append([row["age"], row['trestbps'], row['chol'], row['thalach'], row['oldpeak'], row['slope'], row['ca']])
    test_label.append(row['target'])
test = np.array(test)
test_label = np.array(test_label)


# 一种从 Pandas Dataframe 创建 tf.labeldata 数据集的实用程序方法（utility method）
# def df_to_dataset(dataframe, shuffle=True, batch_size=32):
#   dataframe = dataframe.copy()
#   labels = dataframe.pop('target')
#   ds = tf.labeldata.Dataset.from_tensor_slices((dict(dataframe), labels))
#   if shuffle:
#     ds = ds.shuffle(buffer_size=len(dataframe))
#   ds = ds.batch(batch_size)
#   return ds


# batch_size = 5 # 小批量大小用于演示
# train_ds = df_to_dataset(train, batch_size=batch_size)
# val_ds = df_to_dataset(val, shuffle=False, batch_size=batch_size)
# test_ds = df_to_dataset(test, shuffle=False, batch_size=batch_size)


# for feature_batch, label_batch in train_ds.take(3):
#   print('Every feature:', list(feature_batch.keys()))
#   print('A batch of ages:', feature_batch['age'])
#   print('A batch of targets:', label_batch )


# 我们将使用该批数据演示几种特征列
# example_batch = next(iter(train_ds))[0]

# 用于创建一个特征列
# 并转换一批次数据的一个实用程序方法
# def demo(feature_column):
#   feature_layer = layers.DenseFeatures(feature_column)
#   print(feature_layer(example_batch).numpy())


# age = feature_column.numeric_column("age")
# demo(age)

feature_columns = []

# 数值列
for header in ['age', 'trestbps', 'chol', 'thalach', 'oldpeak', 'slope', 'ca']:
    feature_columns.append(feature_column.numeric_column(header))





model = tf.keras.Sequential([
  layers.Dense(128, activation='relu'),
  layers.Dense(128, activation='relu'),
  layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'],
              run_eagerly=True)

model.fit(train,
          train_label,
          epochs=5)


loss, accuracy = model.evaluate(test,test_label)
print("Accuracy", accuracy)




# a = {'age':[63,67], 'sex':[1,1],'cp':[1,4],'trestbps':[145,160], 'chol':[233,286], 'fbs':[1,0],'restecg':[2,2],\
#     'thalach':[150,108],'exang':[0,1], 'oldpeak':[2.3,1.5], 'slope':[3,2], 'ca':[0,3],'thal':['fixed','normal']}
# a1 = pd.DataFrame(a)

a = [[63,145,233,150,2.3,3,0],[67,160,286,108,1.5,2,3]]
a = np.array(a)



# saver = tf.train.Saver()
# sess = tf.Session()
# sess.run(tf.global_variables_initializer())
# saver.save(sess,'my_tet_model')


b = model.predict(a)
print(np.round(b))
# print(b)