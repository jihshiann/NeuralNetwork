import numpy as np
import math
import matplotlib.pyplot as plt


# FUN: 初始值
def init(n):
    global W2, W1, b1, b2, neuron_num, learning_rate
    np.random.seed(0)
    ## 類神經網路
    # 隱藏層神經元數
    neuron_num = n

    ## 權重與偏差
    W1 = np.random.randn(neuron_num, 2)
    W2 = np.random.randn(1, neuron_num)
    b1 = np.random.randn(neuron_num)
    b2 = np.random.randn(1)
    learning_rate = 0.001

# FUN: 產生數據
def generateData(stage):
    if stage == 1:
        ## DATA
        # 數據數量
        data_num = 30000
        # 輸入值: (四位數, 四位數)陣列
        X = np.random.randint(1000, 9999+1, size=(data_num, 2))
        # 輸出值: 四位數 + 四位數 陣列
        Y = np.sum(X, axis=1).reshape(-1, 1)

    if stage == 2:
        ## DATA
        # 數據數量
        data_num = 10000

        # 輸入值: ([-0.5, 0.2]或[0.8, 1.5]兩個區塊中)二維陣列
        X = np.random.uniform(-1, 1, size=(data_num, 2))
        X[X >= 0] = 0.7 * X[X >= 0] + 0.8
        X[X < 0] = 0.7 * X[X < 0] + 0.2
        bool_X = np.copy(X)
        bool_X[X >= 0.8] = 1
        bool_X[X < 0.8] = 0

        # 輸出: 1或0
        Y = np.where(bool_X[:, 0] == bool_X[:, 1], 0, 1).reshape(-1, 1)

    # 對輸入和輸出進行正規化
    X, min_inputs, range_inputs = standardize(X)
    Y, min_outputs, range_outputs = standardize(Y)
    # 切成訓練與測試
    train_ratio = 0.7
    train_size = int(train_ratio * len(X))
    train_X = X[:train_size]
    train_Y = Y[:train_size]
    test_X = X[train_size:]
    test_Y = Y[train_size:]
    
    return train_X, train_Y, test_X, test_Y, min_inputs, min_outputs,range_inputs, range_outputs

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
def TangentSigmoid(x):
    return np.tanh(x)

# FUN: 正向傳播
def forward(X0):
    # 第一層
    Z1 = np.dot(X0, W1.T) + b1
    X1 = TangentSigmoid(Z1)

    # 第二層
    Z2 = np.dot(X1, W2.T) + b2
    X2 = TangentSigmoid(Z2)

    return Z1, X1, Z2, X2

## 反向傳播
# FUN: Tangent Sigmoid 微分
def dTangentSigmoid(x):
    return 1 - np.power(np.tanh(x), 2)
# FUN: 計算輸出層梯度
def delta_output(Z, Y):
    return (TangentSigmoid(Z) - Y) * dTangentSigmoid(Z)
# FUN: 計算隱藏層梯度
def delta_hidden(Z, D, W):
    return dTangentSigmoid(Z) * np.dot(D, W)
# FUN: 倒傳遞
def backward(Y, Z2, Z1):
    # 計算輸出層梯度
    D2 = delta_output(Z2, Y)

    # 計算隱藏層梯度
    D1 = delta_hidden(Z1, D2, W2)

    return D2, D1

# FUN: 計算目標函數對權重的梯度
def computeWeight(D, X):
    return np.dot(D.T, X)

# FUN: 計算目標函數對偏差的梯度
def computeBias(D):
    return D.sum(axis=0)

# FUN: 參數更新
def update_paras(D2, X1, D1, X0):
    global W2, W1, b1, b2, neuron_num, learning_rate
    # 更新權重
    W2 = W2 - learning_rate * computeWeight(D2, X1)
    W1 = W1 - learning_rate * computeWeight(D1, X0)

    # 更新偏差
    b2 = b2 - learning_rate * computeBias(D2)
    b1 = b1 - learning_rate * computeBias(D1)

## FUN: 學習
def train(X, Y):
    # 正向傳播
    Z1, X1, Z2, X2 = forward(X)

    # 反向傳播
    D2, D1 = backward(Y, Z2, Z1)

    # 參數更新
    update_paras(D2, X1, D1, X)

# FUN: 預測
def predict(X):
    # -1為最後
    ans = forward(X)[-1]
    return ans

# FUN: 計算誤差
def error_function(Y, X):
    # 1/2*error^2
    a = predict(X)
    return 0.5 * ((Y - a) ** 2).sum()


## 執行
def exec(stage, times, max_n, epoch):
    # 保存每次訓練的誤差
    mean_diffs = []  
    min_n = stage
    epoch_list = []
    errors = np.zeros((max_n+1, epoch))
    train_X, train_Y, test_X, test_Y, min_inputs, min_outputs,range_inputs, range_outputs = generateData(stage)

    # 重複訓練
    for time in range(0, times):

        for n in range(min_n, max_n+1):
            # NN
            init(n)

            ## 批次執行
            batch = 1
            for e in range(0, epoch):
                # 隨機打亂訓練資料的索引
                p = np.random.permutation(len(train_X))

                # 遍歷每個batch
                for i in range(math.ceil(len(train_X) / batch)):
                    # 取出當前batch的索引範圍
                    indice = p[i * batch:(i + 1) * batch]

                    # 根據索引範圍從原始資料中取出對應的batch資料
                    X = train_X[indice]
                    Y = train_Y[indice]

                    # 使用當前batch資料進行模型訓練
                    train(X, Y)

                 
                if stage == 1:
                    # 輸出訓練誤差
                    error = error_function(train_Y, train_X)
                    if e%10==1:
                        log = f'\
                        time = {time}, n = {n}, error = {error} ({e}th epoch),\n \
                        '
                        print(log)
                    ## 預測並反正規化
                    predicted_outputs = unstandardize(predict(test_X), min_outputs, range_outputs)
                    origin_outputs = unstandardize(test_Y, min_outputs, range_outputs)
                    ## 記錄誤差
                    mean_diff = np.mean(np.abs(predicted_outputs - origin_outputs))
                    if len(mean_diffs) < n:
                        mean_diffs.append(mean_diff)
                    else:
                        mean_diffs[n-1] += mean_diff
                    # lose
                    errors[n][e] += error

                if stage == 2:
                    # 預測並分類
                    preds = predict(test_X)
                    classify_preds = np.where(preds <= 0.5, 0, 1)
                    ## 記錄誤差
                    mean_diff = np.mean(np.abs(classify_preds - test_Y))
                    mean_pred = np.mean(classify_preds)
                    mean_ans = np.mean(test_Y)
                    ##print(mean_diff)  
                    # 添加至視覺化列表
                    epoch_list.append(e + 1)
                    current_epoch = len(epoch_list)
                    mean_diffs.append(mean_diff)
                    # 繪製並保存圖片
                    plt.plot(epoch_list, mean_diffs, '-o')
                    plt.xlabel('Epoch')
                    plt.ylabel('Mean Difference')
                    plt.title('Mean Difference vs. Epoch')
                    plt.xlim(1, epoch)
                    plt.ylim(0, 1)
                    # 加入每個點的標籤
                    #for x, y in zip(epoch_list, diffs):
                    #    plt.text(x, y, f'{y:.2f}', ha='center', va='bottom')
                    plt.savefig(f'epoch_diff_{current_epoch}.png')
                    plt.close()

                    # 第二張圖: preds vs. Epoch
                    preds = preds.ravel().tolist()
                    x_values = [epoch_list[current_epoch-1]] * len(preds)
                    plt.figure(figsize=(10.0, 10.0)) #10.0 6.0
                    plt.scatter(x_values, preds, s=3.6, label='pred')#36
                    plt.xlabel('Epoch')
                    plt.ylabel('Pred')
                    plt.title('Pred vs. Epoch')
                    plt.xlim(1, epoch)
                    plt.ylim(0, 1)
                    plt.legend()
                    plt.savefig(f'epoch_value_{current_epoch}.png', dpi=400)#200
                    plt.close()
                
    return mean_diffs, errors