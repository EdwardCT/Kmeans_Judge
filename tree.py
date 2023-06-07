import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt


data = pd.read_excel(r"C:\Users\Administrator\Desktop\聚睿\普罗米修\模型测试\初始数据-副本-tree.xlsx")
data1 = pd.read_excel(r"C:\Users\Administrator\Desktop\聚睿\普罗米修\模型测试\tree-测试数据.xlsx")


labels = data["学历"].unique().tolist()
data["学历"] = data["学历"].apply(lambda x: labels.index(x))

labels = data1["学历"].unique().tolist()
data1["学历"] = data1["学历"].apply(lambda x: labels.index(x))


X = data.iloc[:,data.columns != "还款情况"]
y = data.iloc[:,data.columns == "还款情况"]


from sklearn.model_selection import train_test_split
Xtrain, Xtest, Ytrain, Ytest = train_test_split(X,y,test_size=0.3)



for i in [Xtrain, Xtest, Ytrain, Ytest]:
    i.index = range(i.shape[0])
    
clf = DecisionTreeClassifier(random_state=25)
clf = clf.fit(Xtrain, Ytrain)
score_ = clf.score(Xtest, Ytest)

value = clf.predict(data1)

print(value)







