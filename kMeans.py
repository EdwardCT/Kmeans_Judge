# -*- coding: utf-8 -*-
"""
Created on Mon Feb 27 20:15:54 2023

@author: Administrator
"""


import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import pandas as pd
from  sklearn.cluster import KMeans
import time


data = pd.read_excel(r'C:\Users\Administrator\Desktop\聚睿\普罗米修\模型测试\初始数据 - 副本-kmeans.xlsx')

# labels = data["案件结果"].unique().tolist()
# data["案件结果"] = data["案件结果"].apply(lambda x: labels.index(x))

labels = data["学历"].unique().tolist()
data["学历"] = data["学历"].apply(lambda x: labels.index(x))

col = list(data.columns)

data[col] = data[col].apply(pd.to_numeric, errors='coerce').fillna(0.0)

data = pd.DataFrame(data, dtype='float')

k=3  # 假设聚类为3类
# 构建模型
km = KMeans(n_clusters=k) 
km.fit(data.values)

label_pred = km.labels_   # 获取聚类后的样本所属簇对应值
centroids = km.cluster_centers_  # 获取簇心



dd = pd.DataFrame(km.labels_)

dd.to_csv(r'test.csv',header=None)
