# -*- coding: utf-8 -*-
"""
Created on Mon Feb 27 22:03:45 2023

@author: Administrator
"""

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
import seaborn as sns
data = pd.read_excel(r'C:\Users\Administrator\Desktop\聚睿\普罗米修\普罗米修交割表.xlsx')

# data1 = data[["学历","房产类型"]]



# 直接划分
data1 = data[['逾期天数','号码可联状态','未还金额','现在年龄','客户性别','亲友信息评分','婚姻信息评分']]
k=5  # 假设聚类为3类
# 构建模型
km = KMeans(n_clusters=k) 
km.fit(data1)
km_label = pd.DataFrame(km.labels_,columns=['新标签'])
data1 = pd.concat([data1,km_label],axis=1)
summary_df = data1[['逾期天数','号码可联状态','未还金额','现在年龄','客户性别','亲友信息评分','婚姻信息评分','新标签']].groupby('新标签').mean().round()
summary_df['数目'] = data1.groupby('新标签').size()

data1 = pd.concat([data1,data[["合同号","客户姓名"]]],axis=1)

data1.to_excel(r'C:\Users\Administrator\Desktop\聚睿\普罗米修\模型测试\test1.xlsx')
summary_df.to_excel(r'C:\Users\Administrator\Desktop\聚睿\普罗米修\模型测试\test.xlsx')




# # 数值归一化
# summary_df1 = summary_df[summary_df!="数目"]
# # summary_df1 = summary_df1[summary_df1!="新标签"]
# print(summary_df1)

# # summary_df1 = summary_df1.astype("int")
# # from sklearn.preprocessing import MinMaxScaler
# # scaler = MinMaxScaler()
# # scaler = scaler.fit(summary_df1.values)
# # result = scaler.transform(summary_df1.values)
# # # 进行归一化后
# # sumary_df2 = pd.concat([pd.DataFrame(result),summary_df['数目']],axis=1)
# # sumary_df2.columns = ['逾期天数','号码可联状态','未还金额','现在年龄','客户性别','亲友信息评分','婚姻信息评分','新标签','数目'] 
# # sumary_df2.to_excel(r'test.xlsx')



