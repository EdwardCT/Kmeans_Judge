# -*- coding: utf-8 -*-
"""
Created on Wed Mar  1 10:37:42 2023

@author: Administrator
"""

# import pandas as pd

# df1 = pd.read_excel(r'C:\Users\Administrator\Desktop\聚睿\普罗米修\普罗米修建模V2\表合并.xlsx',sheet_name='Sheet1')
# df2 = pd.read_excel(r'C:\Users\Administrator\Desktop\聚睿\普罗米修\普罗米修建模V2\表合并.xlsx',sheet_name='Sheet2')

# df = pd.merge(df1,df2,left_on="合同号.1",right_on="会員番号",how='left')



# df.to_excel(r'C:\Users\Administrator\Desktop\聚睿\普罗米修\普罗米修建模V2\初始表添加_案件号.xlsx')


import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import pandas as pd
from  sklearn.cluster import KMeans
import time
import seaborn as sns


data = pd.read_excel(r'C:\Users\Administrator\Desktop\聚睿\普罗米修\普罗米修建模V2\初始表添加_案件号.xlsx',sheet_name="Sheet2")
data1 = data.iloc[:,1:]
plt.rc('font',family='FangSong')
sns.scatterplot(x='判决号评分',y='剩余本金1',data=data1)




from sklearn.cluster import KMeans
km = KMeans(n_clusters=20,random_state=99)
km.fit(data1)
km_label = pd.DataFrame(km.labels_,columns=['新标签'])
km_df = pd.concat([data1,km_label],axis=1)
sns.scatterplot(x='判决号评分',y='剩余本金1',hue='新标签',data=km_df,palette='Set1')
summary_df = km_df.groupby('新标签').mean().round()
summary_df['数目'] = km_df.groupby('新标签').size()
data1 = pd.concat([data1,data.iloc[:,0:1]],axis=1)

summary_df.to_excel(r'new_test1.xlsx')
km_df.to_excel(r'new_test2.xlsx')







# box_newlabel = summary_df.index
# box_number = summary_df['数目']
# summary_df = summary_df.iloc[:,:-1]


# 数据归一化
# summary_df1 = summary_df.astype("int")
# from sklearn.preprocessing import MinMaxScaler
# scaler = MinMaxScaler()
# scaler = scaler.fit(summary_df1.values)
# result = scaler.transform(summary_df1.values)
# summary_df = pd.DataFrame(result)
# summary_df.columns = ['']



# 数值归一化，便于做权重分析
# 数值归一化
# summary_df1 = summary_df.iloc[:,:-1]
# # summary_df1 = summary_df1[summary_df1!="新标签"]
# print(summary_df1)

# summary_df1 = summary_df1.astype("int")
# from sklearn.preprocessing import MinMaxScaler
# scaler = MinMaxScaler()
# scaler = scaler.fit(summary_df1.values)
# result = scaler.transform(summary_df1.values)


# data1 = pd.concat([data1,data[["合同号","客户姓名"]]],axis=1)
# data1.to_excel(r'test1.xlsx')
# summary_df.to_excel(r'test.xlsx')

