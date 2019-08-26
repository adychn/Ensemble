#!/usr/bin/env python
# coding: utf-8
# Adaboost的实现与肿瘤分类 (Ensemble Exercise 3)
# ### 本节课调用adaboost以及基于简单决策树实现adaboost模型
#
# #### 我们将利用adaboost模型对肿瘤类型进行判断与分类
#
# - 数据来源： https://www.kaggle.com/uciml/breast-cancer-wisconsin-data
# - 对比模型： Adaboost模型

# In[1]:
# necessary imports
get_ipython().run_line_magic('matplotlib', 'inline')
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

# In[1]:
np.random.seed(19)
data = pd.read_csv("breastCancer.csv")


# In[3]:
data.head()

# In[4]:
# #### 检查需要预测的目标
data['diagnosis'].value_counts()

# In[5]:
# #### 过滤不需要的信息
data.drop('id',axis=1,inplace=True)
data.drop('Unnamed: 32',axis=1,inplace=True)

# In[6]:
# #### 转换预测的目标   M：1， B：-1
data['diagnosis'] = data['diagnosis'].apply(lambda x: 1 if x == 'M' else -1)

# In[7]:
# #### 观察数据的基本信息
data.describe()

# In[8]:
data.info()

# In[9]:
import seaborn as sns
sns.countplot(data['diagnosis'])

# In[10]:
# #### 利用前6个特征，设定目标变量
features = data.columns[1:7]
target = 'diagnosis'
features
data['radius_mean']data[target] == -1
data[target] == -1
data.loc[data[target] == -1, 'radius_mean']
# In[11]:
# #### 特征探索
#
# - 比较6个特征与恶性肿瘤的关系
i = 0
for feature in features:
    bins = 25
    # 将特征的直方图画出来
    plt.hist(data[feature][data[target] == -1], bins=bins, color='lightblue', label='B-healthy', alpha=1)
    plt.hist(data[feature][data[target] ==  1], bins=bins, color='k', label='M-bad', alpha=0.5)
    plt.xlabel(feature)
    plt.ylabel('Amount of count')
    plt.legend()
    plt.show()

# In[61]:
from sklearn.model_selection import train_test_split
train_data, test_data = train_test_split(data, test_size=0.3)

# In[62]:
trainX, trainY = train_data[data.columns[1:]], train_data[target]
testX, testY = test_data[data.columns[1:]], test_data[target]

# In[63]:
# ### Logistic回归的表现
logistic_model = LogisticRegression()
print("Logistic Regression performance: %f" % (cross_val_score(logistic_model, trainX, trainY, cv=8).mean()))

# In[64]:
# ### 决策树的表现
tree_model = DecisionTreeClassifier()
print("Decision Tree performance: %f" % (cross_val_score(tree_model, trainX, trainY, cv=8).mean()))

# In[65]:
# ### 直接调用adaboost模型的表现
ada_model = AdaBoostClassifier(n_estimators=200)
print("Decision Tree performance: %f" % (cross_val_score(ada_model, trainX, trainY, cv=8).mean()))

# In[66]:
# ### 测试集的表现
logistic_model = LogisticRegression()
logistic_model.fit(trainX, trainY)
print("Logistic Regression test performance: %f" % logistic_model.score(testX, testY))

# In[67]:
tree_model = DecisionTreeClassifier()
tree_model.fit(trainX, trainY)
preY = tree_model.predict(trainX)
trainY.ravel()
preY
(preY != trainY)
preY[0] = -1
ones = np.ones(len(preY)) / preY
ones.dot((preY != trainY))

print("Decision Tree test performance: %f" % tree_model.score(testX, testY))

# In[68]:
ada_model = AdaBoostClassifier(n_estimators=200)
ada_model.fit(trainX, trainY)
print("Adaboost test performance: %f" % ada_model.score(testX, testY))


# In[74]:
# #### Adaboost的实现
from sklearn.base import BaseEstimator
class Adaboost(BaseEstimator):
    def __init__(self, M):
        self.M = M

    def fit(self, X, Y):
        self.models = []
        self.model_weights = []

        N, _ = X.shape
        alpha = np.ones(N) / N

        for m in range(self.M):
            tree = DecisionTreeClassifier(max_depth=2)
            tree.fit(X, Y, sample_weight=alpha)
            prediction = tree.predict(X)

            # 计算加权错误
            weighted_error = alpha.dot(prediction != Y) # a trick, dot with True (1), False (-1).

            # 计算当前模型的权重
            model_weight = 0.5 * (np.log(1 - weighted_error) - np.log(weighted_error))

            # 更新数据的权重
            alpha = alpha * np.exp(-model_weight * prediction * Y)

            # 数据权重normalize
            alpha = alpha / alpha.sum()

            self.models.append(tree)
            self.model_weights.append(model_weight)

    def predict(self, X):
        N, _ = X.shape
        result = np.zeros(N)
        for wt, tree in zip(self.model_weights, self.models):
            result += tree.predict(X) * wt

        return np.sign(result) # -1 or 1, no 0.

    def score(self, X, Y):
        prediction = self.predict(X)
        return np.mean(prediction == Y)


# In[75]:
# ### Adaboost的表现
adamodel = Adaboost(10)
print("Adaboost model performance: %f" % (cross_val_score(adamodel, trainX.as_matrix().astype(np.float64), trainY.as_matrix().astype(np.float64), cv=8).mean()))

# In[77]:
# ### 测试集的表现
trainX.values
adamodel.fit(trainX.values.astype(np.float64), trainY.values.astype(np.float64))
print("Adaboost model test performance: %f" % adamodel.score(testX.values.astype(np.float64), testY.values.astype(np.float64)))
