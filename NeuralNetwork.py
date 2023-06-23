## 第一階段-撰寫一個2-n-1的淺層類神經網路
#1. 隱藏層的神經元使用Tangent Sigmoid作為激活函數
#2. 輸出層為線性函數
#3. 類神經網路的輸入為兩個四位數的數值，輸出則為這兩個數的總和
#4. 調整你的隱藏層神經元數量，並畫出一張神經元數量對30次訓練結果之誤差平均的圖。
#> PS1: 撰寫類神經前請先產生你的資料集，建議要有10000筆以上，接著把這個資料集切開成訓練與測試資料集。
#> PS2: 不管是輸入還是輸出都要正規化到0-1之間，而產出結果時，需要把輸出反正規化到原本的空間中。

import numpy as np
import math

# FUN: 初始值
def init():
    global W2, W1, b1, b2, neuron_num, learning_rate
    ## 類神經網路
    # 隱藏層神經元數
    neuron_num = np.random.randint(2, 9)

    ## 權重與偏差
    W1 = np.random.randn(neuron_num, 2)
    W2 = np.random.randn(1, neuron_num)
    b1 = np.random.randn(neuron_num)
    b2 = np.random.randn(1)
    learning_rate = 0.001

# FUN: 正規化
def standardize(numArray):
    # 正規化至0到1之間
    min_val = np.min(numArray)
    range_val = np.max(numArray) - min_val
    numArr_std = (numArray - min_val) / range_val
    return numArr_std, min_val, range_val
# FUN: 反正規化
def unstandardize(numArray, min_val, range_val):
    return numArray * range_val + min_val

## FUN: activation function
# tanh(x) = (e^x - e^(-x)) / (e^x + e^(-x))
# https://www.baeldung.com/cs/sigmoid-vs-tanh-functions
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# FUN: 正向傳播
def forward(X0):
    # 第一層
    Z1 = np.dot(X0, W1.T) + b1
    X1 = sigmoid(Z1)

    # 第二層
    Z2 = np.dot(X1, W2.T) + b2
    X2 = sigmoid(Z2)

    return Z1, X1, Z2, X2

## 反向傳播
# FUN: Tangent Sigmoid 微分
def dsigmoid(x): 
    return (1 - sigmoid(x)) * sigmoid(x)
# FUN: 計算輸出層梯度
def delta_output(Z, Y):
    return (sigmoid(Z) - Y) * dsigmoid(Z)
# FUN: 計算隱藏層梯度
def delta_hidden(Z, D, W):
    return dsigmoid(Z) * np.dot(D, W)
# FUN: 倒傳遞
def backward(Y, Z2, Z1):
    # 計算輸出層梯度
    D2 = delta_output(Z2, Y)

    # 計算隱藏層梯度
    D1 = delta_hidden(Z1, D2, W2)

    return D2, D1

# FUN: 目標函數對權重微分
def deweight(D, X):
    return np.dot(D.T, X)

# FUN: 目標函數對偏差微分
def debias(D):
    return D.sum(axis=0)

# FUN: 參數更新
def update_paras(D2, X1, D1, X0):
    global W2, W1, b1, b2, neuron_num, learning_rate
    # 更新權重
    W2 = W2 - learning_rate * deweight(D2, X1)
    W1 = W1 - learning_rate * deweight(D1, X0)

    # 更新偏差
    b2 = b2 - learning_rate * debias(D2)
    b1 = b1 - learning_rate * debias(D1)

## 學習
def train(X, Y):
    # 正向傳播
    Z1, X1, Z2, X2 = forward(X)

    # 反向傳播
    D2, D1 = backward(Y, Z2, Z1)

    # 參數更新
    update_paras(D2, X1, D1, X)

## 執行
# FUN: 預測
def predict(X):
    # -1為最後
    ans = forward(X)[-1]
    return ans

# FUN: 目標函數
def error_function(Y, X):
    # 1/2*error^2
    a = predict(X)
    return 0.5 * ((Y - a) ** 2).sum()


## DATA
# 數據數量
data_num = 12345
# 輸入值: (四位數, 四位數)陣列
TX = (np.random.rand(data_num, 2)*9998).astype(np.int32)+1
# 輸出值: 四位數 + 四位數 陣列
TY = np.sum(TX, axis=1).reshape(-1, 1)
# 對輸入和輸出進行正規化
TX, min_inputs, range_inputs = standardize(TX)
TY, min_expecteds, range_expecteds = standardize(TY)

# NN
init()

## 批次執行
batch = 100
epoch = 100
for e in range(1, epoch + 1):
    # 隨機打亂訓練資料的索引
    p = np.random.permutation(len(TX))

    # 遍歷每個batch
    for i in range(math.ceil(len(TX) / batch)):
        # 取出當前batch的索引範圍
        indice = p[i * batch:(i + 1) * batch]

        # 根據索引範圍從原始資料中取出對應的batch資料
        X0 = TX[indice]
        Y = TY[indice]

        # 使用當前batch資料進行模型訓練
        train(X0, Y)

    # 輸出訓練誤差
    if e % 1 == 0:
        error = error_function(TY, TX)
        log = f'\
        error = {error} ({e}th epoch),\n \
        '
        #W1:{lib.weights_L1},\n \
        #W2:{lib.weights_L2},\n \
        #b1{lib.bias_L1},\n \
        #b2{lib.bias_L2}\
        print(log)

## 對預測結果進行反正規化
predicted_output = unstandardize(predict(TX), min_expecteds, range_expecteds)
origin_TY = unstandardize(TY, min_expecteds, range_expecteds)
mean_diff = np.mean(np.abs(predicted_output - origin_TY))
print(mean_diff)
