#!/usr/bin/env python
# coding: utf-8
# 随机森林在房价预测与蘑菇分类的应用 (Ensemble Exercise 2)
# ## 随机森林的直接调用
#
# #### 这一节课我们利用现成的随机森林库函数对蘑菇进行有毒和无毒的简单分类
#
# - 数据来源： https://www.kaggle.com/uciml/mushroom-classification/data
# - 对比模型： 随机森林，决策树，Logistic回归模型

# In[ ]:
get_ipython().run_line_magic('matplotlib', 'inline')
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression

# In[ ]:
np.random.seed(19)
data = pd.read_csv("mushrooms.csv", header=None)

# In[ ]:
data.head()

# In[ ]:
# ### 处理二分类问题的标签
data[0] = data.apply(lambda x: 0 if x[0] == 'e' else 1, axis=1)

# In[ ]:
# ### 处理每列的数据
# 每一列如果有null，用"missing"代替
cols = np.arange(1,23)
for col in cols:
    if np.any(data[col].isnull()):
        data.loc[data[col].isnull(), col] = 'missing'


# In[ ]:
labelEncoders = dict()

# 对每一列进行label encoding
for col in cols:
    encoder = LabelEncoder()
    values = data[col].tolist()
    values.append('missing')  #加入missing这种值
    encoder.fit(values)
    labelEncoders[col] = encoder

# 计算label encoding之后的列数
dimensionality = 0
for col, encoder in labelEncoders.items():
    dimensionality += len(encoder.classes_)

print("dimensionality:  %d" % (dimensionality))

# In[ ]:
# 用于测试数据的变换
def transform(df):
    N, _ = df.shape
    X = np.zeros((N, dimensionality))
    i = 0
    for col, encoder in labelEncoders.items():
        k = len(encoder.classes_)
        X[np.arange(N), encoder.transform(df[col]) + i] = 1 # Indexing Multi-dimensional arrays in pairs
        i += k
    return X

### 0 1 2 3    4 5 6 7 8   dimensionality
# N
# 0   1        1   1
# 1     1            1
# 2 1
# 3       1      1
# 4                    1

# In[ ]:
# 准备数据和标签
X = transform(data)
Y = data[0].values

# In[ ]:
# ### Logistic回归的表现
logistic_model = LogisticRegression()
print("logistic Regression performance: %f" % (cross_val_score(logistic_model, X, Y, cv=8).mean()))

# In[ ]:
# ### 决策树的表现
tree_model = DecisionTreeClassifier()
print("Decision Tree performance: %f" % (cross_val_score(tree_model, X, Y, cv=8).mean()))

# In[ ]:
# ### 随机森林的表现
forest = RandomForestClassifier(5)
print("Random Forest performance: %f" % (cross_val_score(tree_model, X, Y, cv=8).mean()))

# In[ ]:


# In[ ]:
# ### 伪随机森林的实现
from sklearn.base import BaseEstimator
class FakeRandomForest(BaseEstimator):
    def __init__(self, M):
        self.M = M

    def fit(self, X, Y, n_features=None):
        N, D = X.shape # N samples, D dimensional
        if n_features is None:
            # default 採集的特征个数
            n_features = int(np.sqrt(D))

        self.models = []   # Bags
        self.features = [] # Features used in each model

        for m in range(self.M):
            tree = DecisionTreeClassifier(max_depth=4)

            #有放回的随机抽取N个数据
            idx = np.random.choice(N, size=N, replace=True)
            X_current = X[idx]
            Y_current = Y[idx]

            #随机抽取n_features个特征
            features_idx = np.random.choice(D, size=n_features, replace=False)

            #训练当前的决策树模型
            tree.fit(X_current[:, features_idx], Y_current)
            self.features.append(features_idx)
            self.models.append(tree)

    def predict(self, X):
        N = len(X)
        results = np.zeros(N)
        for features_idx, tree in zip(self.features, self.models):
            results += tree.predict(X[:, features_idx])
        return np.round(results / self.M) # greater than 0.5 is 1, less is 0.

    def score(self, X, Y):
        prediction = self.predict(X)
        return np.mean(prediction == Y)

# In[ ]:
# ### Bagging决策树的实现
class BaggedTreeClassifier(BaseEstimator):
    def __init__(self, M):
        self.M = M

    def fit(self, X, Y):
        N = len(X)
        self.models = []
        for m in range(self.M):
            idx = np.random.choice(N, size=N, replace=True)
            Xb = X[idx]
            Yb = Y[idx]

            model = DecisionTreeClassifier(max_depth=2)
            model.fit(Xb, Yb)
            self.models.append(model)

    def predict(self, X):
        # no need to keep a dictionary since we are doing binary classification
        predictions = np.zeros(len(X))
        for model in self.models:
            predictions += model.predict(X)
        return np.round(predictions / self.M)

    def score(self, X, Y):
        P = self.predict(X)
        return np.mean(Y == P)


# In[ ]:
baggedtc = BaggedTreeClassifier(20)
cross_val_score(baggedtc, X, Y, cv=8).mean()


# In[ ]:
fakerf = FakeRandomForest(20)
cross_val_score(fakerf, X, Y, cv=8).mean()


# In[ ]:
# ### 用随机森林做regression
#
# #### 这一节课我们利用现成的随机森林库函数对房价做预测
#
# - 数据来源： https://www.kaggle.com/harlfoxem/housesalesprediction/data
# - 对比模型： 随机森林，线性回归模型
# In[ ]:
house_data = pd.read_csv("kc_house_data.csv")

# In[ ]:
house_data.head()

# In[ ]:
house_data.columns


# In[ ]:
# price is the target
NUMERICAL_COLS = ['price', 'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'sqft_above','sqft_basement',
                 'sqft_living15', 'sqft_lot15']

# In[ ]:
# ### 处理一下每一列的数据
# fit 每一列数据的scaler
scalers = dict()
for col in NUMERICAL_COLS:
    scaler = StandardScaler()
    scaler.fit(house_data[col].values.astype(np.float64).reshape(-1, 1))
    scalers[col] = scaler

# In[ ]:
def transform_numerical(df):
    N, _ = df.shape
    D = len(NUMERICAL_COLS)
    result = np.zeros((N, D))
    i = 0
    for col, scaler in scalers.items():
        result[:, i] = scaler.transform(df[col].values.astype(np.float64).reshape(1, -1))
        i += 1
    return result


# In[ ]:
from sklearn.model_selection import train_test_split
hdata = transform_numerical(house_data)

# In[ ]:
train_data, test_data = train_test_split(hdata, test_size=0.2, random_state=1)

# In[ ]:
trainX, trainY = train_data[:,1:], train_data[:, 0]
testX, testY = test_data[:, 1:], test_data[:, 0]

# In[ ]:
rfregressor = RandomForestRegressor(n_estimators=100)
rfregressor.fit(trainX, trainY)
predictions = rfregressor.predict(testX)


# In[ ]:
# ### 可视化预测的结果
plt.scatter(testY, predictions)
plt.xlabel("target")
plt.ylabel("prediction")
ymin = np.round(min(min(testY), min(predictions)))
ymax = np.ceil(max(max(testY), max(predictions)))
r = range(int(ymin), int(ymax) + 1)
plt.plot(r,r)
plt.show()

# In[ ]:
plt.plot(testY, label='targets')
plt.plot(predictions, label='predictions')
plt.legend()
plt.show()

# In[ ]:
lr = LinearRegression()
print("linear regression performance: %f" % (cross_val_score(lr, trainX, trainY).mean()))

# In[ ]:
print("random forest regressor performance: %f" % (cross_val_score(rfregressor, trainX, trainY).mean()))

# In[ ]:
lr.fit(trainX, trainY)
print("linear regression test score: %f" % (lr.score(testX, testY)))

# In[ ]:
rfregressor.fit(trainX, trainY)
print("random forest regressor test score: %f" % (rfregressor.score(testX, testY)))
