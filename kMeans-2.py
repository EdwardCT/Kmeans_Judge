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

# 选取两个特征
data = data[['社保期限','年收入']]

import seaborn as sns
import matplotlib.pyplot as plt

plt.rc('font',family='FangSong')
# sns.scatterplot(x='社保期限',y='年收入',data=data)

# 标准化
from sklearn.preprocessing import StandardScaler
km_df = data
km_df_standardize = StandardScaler().fit_transform(km_df)
km_df_standardize = pd.DataFrame(data=km_df_standardize,columns=list(km_df.columns))
print(km_df_standardize)

# K值选择
from scipy.spatial.distance import cdist
cost = []

K = range(1,15)
for k in K:
    kmeanModel = KMeans(n_clusters=k,random_state=99)
    kmeanModel.fit(km_df_standardize)
    cost.append(kmeanModel.inertia_)
    
# 肘部法则可视化
# plt.xlabel('k')
# plt.ylabel('cost')
# plt.plot(K,cost,'o-')
# plt.show()


# 导入kmeans包
from sklearn.cluster import KMeans
km = KMeans(n_clusters=3,random_state=99)
km.fit(km_df_standardize)
km_label = pd.DataFrame(km.labels_,columns=['新标签'])
km_df = pd.concat([km_df,km_label],axis=1)

sns.scatterplot(x='社保期限',y='年收入',hue='新标签',data=km_df,palette='Set1')


# 聚类分析-对每一项进行进一步分析
final_df = data
