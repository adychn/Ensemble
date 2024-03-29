#!/usr/bin/env python
# coding: utf-8
# ## bias 和variance对比实现
#
# #### 这节课我们在人工点集上利用不同自由度的多项式进行线性回归，对比不同自由度下模型的bias和variance

# In[38]:
get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error as mse

# In[39]:
# 数据集个数
Num_datasets = 50
noise_level = 0.5 # gauss

# 最大的degree
max_degree = 12

# 每个数据集里的数据个数
N = 25

# 用于训练的数据数
trainN = int(25 * 0.9)
np.random.seed(2)

# In[40]:
def make_poly(x, degree):
    """
    input: x  N by 1
    output: N by degree + 1
    """
    N = len(x)
    result = np.empty((N, degree+1))
    for d in range(degree + 1):
        result[:, d] = x ** d
        if d > 1:
            result[:, d] = (result[:, d] - result[:, d].mean()) / result[:,d].std() # z-score, standard score
    return result

def f(X):
    return np.sin(X)


# In[41]:
# #### 需要拟合的目标函数
x_axis = np.linspace(-np.pi, np.pi, 100)
y_axis = f(x_axis)

# 可视化
plt.plot(x_axis, y_axis)

# In[42]:
# 基本训练集
X = np.linspace(-np.pi, np.pi, 25)
np.random.shuffle(X)
f_X = f(X)

# 创建全部的数据
allData = make_poly(X, max_degree)

train_scores = np.zeros((Num_datasets, max_degree))
test_scores = np.zeros((Num_datasets, max_degree))

train_predictions = np.zeros((trainN, Num_datasets, max_degree)) # Every point (20 points) in every dataset in every model
prediction_curves = np.zeros((100, Num_datasets, max_degree))

model = LinearRegression()


# dataset1 $\leftarrow$  $model_1$, $model_2$, ... $model_{10}$
# dataset2 $\leftarrow$  $model_1$, $model_2$,... $model_{10}$
# .
# .
#
# tip: 固定住某一维的坐标理解矩阵

# In[43]:
# #### 训练集是否具有代表性
plt.scatter(X, f_X)

# In[44]:
# #### 每一个数据集上用不同degree的多项式拟合
for k in range(Num_datasets):

    # 每个数据集不失pattern的情况下稍微不一样~
    Y = f_X + np.random.randn(N) * noise_level

    # 20 train data, 5 test data
    trainX, testX = allData[:trainN], allData[trainN:]
    trainY, testY = Y[:trainN], Y[trainN:]

    # 用不同的模型去学习当前数据集
    for d in range(max_degree):

        # 模型学习
        model.fit(trainX[:, :d+2], trainY)

        # 在allData上的预测结果
        all_predictions = model.predict(allData[:, :d+2])

        # 预测并记录一下我们的目标函数
        x_axis_poly = make_poly(x_axis, d + 1)          # true poly x
        axis_predictions = model.predict(x_axis_poly)   # true y
        prediction_curves[:, k, d] = axis_predictions


        train_prediction = all_predictions[:trainN]
        test_prediction = all_predictions[trainN:]

        train_predictions[:, k, d] = train_prediction # 用于计算bias and varaince


        #计算并存储训练集和测试集上的分数
        train_score = mse(train_prediction, trainY)
        test_score = mse(test_prediction, testY)
        train_scores[k, d] = train_score
        test_scores[k, d] = test_score




# #### 画出每一个degree下，50个数据集上预测结果以及预测结果的平均
# In[46]:
for d in range(max_degree): # every model
    for k in range(Num_datasets): # every dataset
        # 给定当前模型，画出它在所有数据集上的表现
        plt.plot(x_axis, prediction_curves[:, k, d], color='green', alpha=0.5)

    # 给定当前模型，画出它在所有数据集上的表现的平均
    plt.plot(x_axis, prediction_curves[:, :, d].mean(axis=1), color='blue', linewidth=2)

    plt.title("curves for degree=%d" %(d+1))
    plt.show()


# #### 计算bias跟variance
#
# - 一种degree对应一种模型
# - 给定一种模型，计算该模型在50个数据集上的预测结果的平均
# - 给定一种模型，计算该模型在50个数据集上预测结果的方差

# In[47]:
# 每一个模型的bias
average_train_prediction = np.zeros((trainN, max_degree))   # 模型的平均表现
squared_bias = np.zeros(max_degree)

trueY_train = f_X[:trainN] # 真值

# $bias^{2} = (average\_performance - true\_value)^2$
for d in range(max_degree):
    for i in range(trainN):
        average_train_prediction[i, d] = train_predictions[i, :, d].mean()
    squared_bias[d] = ((average_train_prediction[:, d] - trueY_train) ** 2).mean()

# In[52]:
# 每一个模型的variance
# - 单看一个训练数据点，计算它在50个数据集上的不同程度
# - 计算全部训练数据点不同程度的平均
variances = np.zeros((trainN, max_degree))
for d in range(max_degree):
    for i in range(trainN):
        difference = train_predictions[i, :, d] - average_train_prediction[i, d]
        variances[i,d] = np.dot(difference, difference) / Num_datasets

variance = variances.mean(axis=0)

# In[17]:
# #### 作图
# - 以degree为横轴，以$bias^2$为纵轴
# - 以degree为横轴，以$variance$为纵轴
# - 以degree为横轴，以测试集分数为纵轴
# - 以degree为横轴，以$bias^2 + variance$为纵轴
degrees = np.arange(max_degree) + 1
best_degree = np.argmin(test_scores.mean(axis=0)) + 1

plt.plot(degrees, squared_bias, label='squared bias')
plt.plot(degrees, variance, label = 'variance')
plt.plot(degrees, test_scores.mean(axis=0), label='test scores')
plt.plot(degrees, squared_bias + variance, label='squared bias + variance')
plt.axvline(x=best_degree, linestyle='--', label='best complexity')
plt.legend()
plt.show()

# In[18]:
# #### 作图
# - 以degree为横轴，以训练集分数为纵轴
# - 以degree为横轴，以测试集分数为纵轴

plt.plot(degrees, train_scores.mean(axis=0), label='train scores')
plt.plot(degrees, test_scores.mean(axis=0), label= 'test scores')
plt.axvline(x=best_degree, linestyle='--', label='best complexity')
plt.legend()
plt.show()
