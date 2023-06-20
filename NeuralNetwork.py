## 第一階段-撰寫一個2-n-1的淺層類神經網路
#1. 隱藏層的神經元使用Tangent Sigmoid作為激活函數
#2. 輸出層為線性函數
#3. 類神經網路的輸入為兩個四位數的數值，輸出則為這兩個數的總和
#4. 調整你的隱藏層神經元數量，並畫出一張神經元數量對30次訓練結果之誤差平均的圖。
#> PS1: 撰寫類神經前請先產生你的資料集，建議要有10000筆以上，接著把這個資料集切開成訓練與測試資料集。
#> PS2: 不管是輸入還是輸出都要正規化到0-1之間，而產出結果時，需要把輸出反正規化到原本的空間中。



import numpy as np
import math

# 數據數量
data_num = 12345
# 輸入值: (四位數, 四位數)陣列
inputs = np.random.randint(1000, 9999, size=(data_num, 2))
# 輸出值: 四位數 + 四位數 陣列
expecteds = np.sum(inputs, axis=1)
# FUN: 正規化
def standardize(numArray):
    # 正規化至0到1之間
    numArr_std = (numArray - np.min(numArray)) / (np.max(numArray) - np.min(numArray))
    return numArr_std


## 類神經網路
# 隱藏層神經元數
neuron_num = np.random.randint(2,9)
neuron_num = 3
## 權重與偏差
weights_L1 = np.random.randn(2, neuron_num)
weights_L2 = np.random.randn(neuron_num, 1)
bias = np.random.randn(neuron_num)
# FUN: activation function
# tanh(x) = (e^x - e^(-x)) / (e^x + e^(-x))
# https://www.baeldung.com/cs/sigmoid-vs-tanh-functions
def tangentSigmoid(inputs):
    outputs = (np.exp(inputs) - np.exp(-inputs)) / (np.exp(inputs) + np.exp(-inputs))
    return outputs

# FUN: 正向傳播
def forward(inputs):
     # 第一層
    weighteds_L1 = np.dot(inputs, weights_L1) + bias
    computeds = tangentSigmoid(weighteds_L1)
    
    # 第二層
    weighteds_L2 = np.dot(computeds, weights_L2)
    outputs = tangentSigmoid(weighteds_L2)
    
    return outputs

## 反向傳播
# FUN: Tangent Sigmoid 微分
def dTangentSigmoid(inputs):
    outputs = 1 - np.tanh(inputs)**2
    return outputs
# FUN: 計算輸出層梯度
def delta_output(weighteds, expecteds):
    result = (tangentSigmoid(weighteds) -expecteds) * dTangentSigmoid(weighteds)
    return result
# FUN: 計算隱藏層梯度
def delta_hidden(weighteds, next_weights, next_deltas):
    result = np.dot(next_deltas, next_weights.T) * dTangentSigmoid(weighteds)
    return result
# FUN: 倒傳遞
def backward(inputs, expecteds, weighteds_L1, weighteds_L2):
    # 計算輸出層梯度
    delta_L2 = delta_output(weighteds_L2, expecteds)
    
    # 計算隱藏層梯度
    delta_L1 = delta_hidden(weighteds_L1, weights_L2, delta_L2)
    
    return delta_L2, delta_L1

## 更新
learning_rate = 0.001

# FUN: 目標函數對權重微分
def deweight(delta, inputs):
    return np.dot(delta.T, inputs)

# FUN: 目標函數對偏差微分
def debias(delta):
    return np.mean(delta, axis=0)

# FUN: 參數更新
def update_paras(delta_L2, delta_L1, inputs, weighteds_L1):
    # 更新第二層權重
    dW2 = deweight(delta_L2, weighteds_L1)
    weights_L2 -= learning_rate * dW2

    # 更新第一層權重
    dW1 = deweight(delta_L1, inputs)
    weights_L1 -= learning_rate * dW1

    # 更新偏差
    dB = debias(delta_L2)
    bias -= learning_rate * dB
    
## 學習
num_epochs = 300
def train(inputs, expecteds, weights_L1, weights_L2, bias, num_epochs):
    errors = []
    for epoch in range(num_epochs):
        # 正向傳播
        outputs = forward(inputs)
        
        # 計算誤差
        error = np.mean(np.abs(outputs - expecteds))
        errors.append(error)
        
        # 反向傳播
        delta_L2, delta_L1 = backward(inputs, expecteds, weights_L1, weights_L2)
        
        # 參數更新
        weights_L1, weights_L2, bias = update_paras(delta_L2, delta_L1, inputs, weights_L1)

## 執行
# FUN: 預測
def predict(inputs):
    outputs = forward(inputs)
    return outputs

# FUN: 目標函數
def error_function(inputs, expecteds):
    #1/2*error^2
    return 0.5*((expecteds-predict(inputs)**2).sum())

## 批次執行
batch = 100
for num_epochs in range(1, num_epochs+1):
    # 隨機打亂訓練資料的索引
    p = np.random.permutation(len(inputs))
    
    # 遍歷每個batch
    for i in range(math.ceil(len(inputs)/batch)):
        # 取出當前batch的索引範圍
        indice = p[i*batch:(i+1)*batch]
        
        # 根據索引範圍從原始資料中取出對應的batch資料
        inputs_batch = inputs[indice]
        expecteds_batch = expecteds[indice]
        
        # 使用當前batch資料進行模型訓練
        train(inputs_batch, expecteds_batch)

    # 每1000個epoch輸出一次訓練誤差
    if num_epochs % 1000 == 0:
        log = 'error = {8.4f} ({:5d}th epoch)'
        # 計算訓練資料上的誤差並輸出到控制台
        print(log.format(error_function(inputs, expecteds), num_epochs))

