#!/usr/bin/env python
# coding: utf-8
# Adaboost的完全实现（Ensemble Exercise 4, 大作业）
# necessary imports
# In[ ]:
get_ipython().run_line_magic('matplotlib', 'inline')
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score

# In[ ]:
np.random.seed(19)
data = pd.read_csv("mushrooms.csv")
data.head()
# In[ ]:
# ### 数据预处理
data['class'] = data.apply(lambda row: -1 if row[0] == 'e' else 1, axis=1)

# In[ ]:
def labelencoder(data, columns=['pclass','name_title','embarked', 'sex']):
    for col in columns:
        data[col] = data[col].apply(lambda x: str(x))
        new_cols = [col + '_' + i for i in data[col].unique()]
        data = pd.concat([data, pd.get_dummies(data[col], prefix=col)[new_cols]], axis=1)
        del data[col]
    return data

# In[ ]:
target = 'class'  # y
cols = data.columns.drop(target)

# In[ ]:
data_set = labelencoder(data, columns = cols)

# In[ ]:
from sklearn.model_selection import train_test_split
train_data, test_data = train_test_split(data_set, test_size=0.3)

# In[ ]:
trainX, trainY = train_data[train_data.columns[1:]], pd.DataFrame(train_data[target])
testX, testY = test_data[test_data.columns[1:]], pd.DataFrame(test_data[target])


# In[ ]:
# ### 基于决策树的完全手动实现adaboost
class TreeNode:
    def __init__(self, is_leaf, prediction, split_feature):
        self.is_leaf = is_leaf
        self.prediction = prediction
        self.split_feature = split_feature
        self.left = None
        self.right = None

# In[ ]:
# How to learn my decision tree based on data points' weight?
# To calculate what the predicted output of this node will be, and calculate
# the total error.
def node_weighted_mistakes(targets_in_node, data_weights):
    # 如果全部预测为-1，那么预测错误的数据权重等于weight_positive
    # 如果全部预测为+1，那么预测错误的数据权重等于weight_negative

    # 计算lable 为+1的所有数据的权重和
    weight_positive = sum(data_weights[targets_in_node == 1])

    # 计算lable 为-1的所有数据的权重和
    weight_negative = sum(data_weights[targets_in_node == -1])

    #将加权错误和对应的预测标签一起输出
    if weight_positive < weight_negative:
        return (weight_positive, -1)
    else:
        return (weight_negative, 1)

# In[ ]:
# #### 阶段性测试
# - 测试node_weighted_mistakes函数，对测试样例输出应该是(2.5,-1),表示预测错误的权重为2.5，预测结果为-1
example_targets = np.array([-1, -1, 1, 1, 1])
example_data_weights = np.array([1., 2., .5, 1., 1.])
node_weighted_mistakes(example_targets, example_data_weights)


# In[ ]:
# ### 重新写best_split函数，
# we use min weighted error of a feature to pick the best feature to separte a stump
def best_split_weighted(data, features, target, data_weights):
    # return the best feature
    best_feature = None
    best_error = float("inf")
    num_data_points = float(len(data))

    # Try using each feature to split the tree
    for feature in features:
        # 左分支对应当前特征为0的数据点
        # 右分支对应当前特征为1的数据点
        row_idx_feature_is_0 = data[feature] == 0
        row_idx_feature_is_1 = data[feature] == 1

        # 进入左分支数据点的权重
        left_split = data[row_idx_feature_is_0]
        left_data_weights = data_weights[row_idx_feature_is_0]

        # 进入右分支数据点的权重
        right_split = data[row_idx_feature_is_1]
        right_data_weights =  data_weights[row_idx_feature_is_1]

        # 重点！！
        # 计算左边分支里犯了多少错 (加权结果！！)
        left_misses, left_class = node_weighted_mistakes(left_split[target], left_data_weights)

        # 计算右边分支里犯了多少错 (加权结果！！)
        right_misses, right_class = node_weighted_mistakes(right_split[target], right_data_weights)

        # 计算当前划分之后的分类犯错率
        error = (left_misses + right_misses) * 1.0 / sum(data_weights)

        # 更新应选特征和错误率，注意错误越低说明该特征越好
        if error < best_error:
            best_error = error
            best_feature = feature

    return best_feature

# In[ ]:
# #### 阶段性测试
# - 测试best_split_weighted函数，结果应该是"odor_n" 这个特征
# 根据之前的实现，最佳特征
features = data_set.columns.drop(target)
example_data_weights = np.array(len(train_data) * [2])
best_split_weighted(train_data, features, target, example_data_weights)

# In[ ]:
# 用于创建叶子的函数
def create_leaf(target_values, data_weights):

    leaf = TreeNode(True, None, None)

    # 直接调用node_weighted_mistakes得到叶子节点的预测结果
    _, prediction_class = node_weighted_mistakes(target_values, data_weights)
    leaf.prediction = prediction_class

    # 返回叶子
    return leaf


# In[ ]:
def create_weighted_tree(data, data_weights, features, target, current_depth = 0, max_depth = 10, min_error=1e-15):
    # 拷贝以下可用特征
    remaining_features = features[:]
    target_values = data[target]

    # termination 1
    if node_weighted_mistakes(target_values,data_weights)[0] <= min_error:
        print("Termination 1 reached.")
        return create_leaf(target_values, data_weights)

    # termination 2
    if len(remaining_features) == 0:
        print("Termination 2 reached.")
        return create_leaf(target_values, data_weights)

    # termination 3
    if current_depth >= max_depth:
        print("Termination 3 reached.")
        return create_leaf(target_values, data_weights)

    # 选出最佳当前划分特征, here uses weighted accuracy, can use entropy or gini as long as uses the weight on data points.
    split_feature = best_split_weighted(data, features, target, data_weights)

    # 选出最佳特征后，该特征为0的data分到左边，该特征为1的data分到右边
    left_split = data[data[split_feature] == 0]
    right_split = data[data[split_feature] == 1]

    # 将对应数据的权重也分到左边与右边
    left_data_weights = data_weights[data[split_feature] == 0]
    right_data_weights = data_weights[data[split_feature] == 1]

    # 剔除已经用过的特征
    remaining_features = remaining_features.drop(split_feature)
    print("Split on feature %s. (%s, %s)" % (split_feature, str(len(left_split)), str(len(right_split))))

    # 如果当前数据全部划分到了一边，直接创建叶子节点返回即可
    if len(left_split) == len(data):
        print("Perfect split!")
        return create_leaf(left_split[target], left_data_weights)
    if len(right_split) == len(data):
        print("Perfect split!")
        return create_leaf(right_split[target], right_data_weights)

    # 递归上面的步骤
    left_tree = create_weighted_tree(left_split, left_data_weights, remaining_features, target, current_depth + 1, max_depth, min_error)
    right_tree = create_weighted_tree(right_split, right_data_weights, remaining_features, target, current_depth + 1, max_depth, min_error)

    #生成当前的树节点
    result_node = TreeNode(is_leaf=False, prediction=None, split_feature=split_feature)
    result_node.left = left_tree
    result_node.right = right_tree
    return result_node

# In[ ]:
def count_leaves(tree):
    if tree.is_leaf:
        return 1
    return count_leaves(tree.left) + count_leaves(tree.right)

# In[ ]:
# - 测试create_weighted_tree函数，根据测试样例，输出应该是4
# #### 阶段性测试
example_data_weights = np.array([1.0 for i in range(len(train_data))])
small_data_decision_tree = create_weighted_tree(train_data,example_data_weights, features, target, max_depth=2)
count_leaves(small_data_decision_tree)

# In[ ]:
def predict_single_data(tree, x, annotate = False):
    # 如果已经是叶子节点直接返回叶子节点的预测结果
    if tree.is_leaf:
        if annotate:
            print("leaf node, predicting %s" % tree.prediction)
        return tree.prediction
    else:
        # 查询当前节点用来划分数据集的特征
        split_feature_value = x[tree.split_feature]

        if annotate:
            print("Split on %s = %s" % (tree.split_feature, split_feature_value))
        if split_feature_value == 0:
            #如果数据在该特征上的值为0，交给左子树来预测
            return predict_single_data(tree.left, x, annotate)
        else:
            #如果数据在该特征上的值为0，交给右子树来预测
            return predict_single_data(tree.right, x, annotate)

# In[ ]:
# 测试 test 据样例，输出应该至少是0.95以上
def evaluate_accuracy(tree, data):
    # 将predict函数应用在数据data的每一行
    prediction = data.apply(lambda row: predict_single_data(tree, row), axis=1)
    # 返回正确率
    accuracy = (prediction == data[target]).sum() * 1.0 / len(data)
    return accuracy

evaluate_accuracy(small_data_decision_tree, test_data)

# In[ ]:
from sklearn.base import BaseEstimator
from sklearn.metrics import accuracy_score
class WeightedDecisionTree(BaseEstimator):
    def __init__(self, max_depth, min_error):
        self.max_depth = max_depth
        self.min_error = min_error

    def fit(self, X, Y, data_weights = None):
        data_set = pd.concat([X, Y], axis=1)
        features = X.columns
        target = Y.columns[0]
        self.root_node = create_weighted_tree(data_set, data_weights, features, target, current_depth=0, max_depth=self.max_depth, min_error=self.min_error)

    def predict(self, X):
        prediction = X.apply(lambda row: predict_single_data(self.root_node, row), axis=1)
        return prediction

    def score(self, testX, testY):
        target = testY.columns[0]
        result = self.predict(testX)
        return accuracy_score(testY[target], result)


# ### The equations you need to implement the MyAdaboost class
# $$weighted\_error = \sum_{i=1}^{N} \alpha_{i}I(\hat{y_i} \ne y_i) / \sum_{i}^{N}\alpha_i$$
#
# $$w_t = \frac{1}{2} ln(\frac{1 - weighted\_error}{weighted\_error})$$
#
# $$\alpha_i = \alpha_i * e^{-w_t * f_t(x_i) * y_i}$$
#
# $$\alpha_i = \frac{\alpha_i}{\sum_{j=1}^{N}\alpha_j}$$
# In[ ]:
from sklearn.base import BaseEstimator
class MyAdaboost(BaseEstimator):
    def __init__(self, M):
        self.M = M

    def fit(self, X, Y):
        self.models = []
        self.model_weights = []
        self.target = Y.columns[0]

        N, _ = X.shape
        alpha = np.ones(N) / N    # data weights

        for m in range(self.M):
            tree = WeightedDecisionTree(max_depth=2, min_error=1e-15) # weak learner
            tree.fit(X, Y, data_weights=alpha)
            # fit your current model
            prediction = tree.predict(X)

            # 计算加权错误
            weighted_error = alpha.dot(prediction != Y[self.target]) # [0.33, 0.33, 0.33] dot [0, 0, 1]

            # 计算当前模型的权重
            model_weight = 0.5 * np.log((1 - weighted_error) / weighted_error)

            # 更新数据的权重, label is 1 or -1, don't need if else.
            alpha = alpha * np.exp(-model_weight * Y[self.target] * prediction)

            # 数据权重normalize
            alpha = alpha / alpha.sum()

            self.models.append(tree)
            self.model_weights.append(model_weight)

    def predict(self, X):
        N, _ = X.shape
        result = np.zeros(N)
        for wt, tree in zip(self.model_weights, self.models):
            result += wt * tree.predict(X)

        return np.sign(result)  # > 0 is 1, < 0 is -1

    def score(self, testX, testY):
        result = self.predict(testX)
        return accuracy_score(testY[self.target], result)

# In[ ]:
#### 阶段性测试
# - 根据测试样例，输出应该是接近于1.0的得分
m = MyAdaboost(20)
m.fit(trainX, trainY)
m.score(testX, testY)
