import numpy as np
import math
import matplotlib.pyplot as plt


# FUN: ��l��
def init(n):
    global W2, W1, b1, b2, neuron_num, learning_rate
    ## �����g����
    # ���üh���g����
    neuron_num = n

    ## �v���P���t
    W1 = np.random.randn(neuron_num, 2)
    W2 = np.random.randn(1, neuron_num)
    b1 = np.random.randn(neuron_num)
    b2 = np.random.randn(1)
    learning_rate = 0.001

# FUN: ���ͼƾ�
def generateData(stage):
    if stage == 1:
        ## DATA
        # �ƾڼƶq
        data_num = 30000
        # ��J��: (�|���, �|���)�}�C
        X = np.random.randint(1000, 9999+1, size=(data_num, 2))
        # ��X��: �|��� + �|��� �}�C
        Y = np.sum(X, axis=1).reshape(-1, 1)

    if stage == 2:
        ## DATA
        # �ƾڼƶq
        data_num = 10000

        # ��J��: ([-0.5, 0.2]��[0.8, 1.5]��Ӱ϶���)�G���}�C
        X = np.random.uniform(-1, 1, size=(data_num, 2))
        X[X >= 0] = 0.7 * X[X >= 0] + 0.8
        X[X < 0] = 0.7 * X[X < 0] + 0.2
        bool_X = np.copy(X)
        bool_X[X >= 0.8] = 1
        bool_X[X < 0.8] = 0

        # ��X: 1��0
        Y = np.where(bool_X[:, 0] == bool_X[:, 1], 0, 1).reshape(-1, 1)

    # ���J�M��X�i�楿�W��
    X, min_inputs, range_inputs = standardize(X)
    Y, min_outputs, range_outputs = standardize(Y)
    # �����V�m�P����
    train_ratio = 0.7
    train_size = int(train_ratio * len(X))
    train_X = X[:train_size]
    train_Y = Y[:train_size]
    test_X = X[train_size:]
    test_Y = Y[train_size:]
    
    return train_X, train_Y, test_X, test_Y, min_inputs, min_outputs,range_inputs, range_outputs

# FUN: ���W��
def standardize(numArray):
    # ���W�Ʀ�0��1����
    min_val = np.min(numArray)
    range_val = np.max(numArray) - min_val
    numArr_std = (numArray - min_val) / range_val
    return numArr_std, min_val, range_val
# FUN: �ϥ��W��
def unstandardize(numArray, min_val, range_val):
    return numArray * range_val + min_val

## FUN: activation function
# tanh(x) = (e^x - e^(-x)) / (e^x + e^(-x))
# https://www.baeldung.com/cs/sigmoid-vs-tanh-functions
def TangentSigmoid(x):
    return np.tanh(x)

# FUN: ���V�Ǽ�
def forward(X0):
    # �Ĥ@�h
    Z1 = np.dot(X0, W1.T) + b1
    X1 = TangentSigmoid(Z1)

    # �ĤG�h
    Z2 = np.dot(X1, W2.T) + b2
    X2 = TangentSigmoid(Z2)

    return Z1, X1, Z2, X2

## �ϦV�Ǽ�
# FUN: Tangent Sigmoid �L��
def dTangentSigmoid(x):
    return 1 - np.power(np.tanh(x), 2)
# FUN: �p���X�h���
def delta_output(Z, Y):
    return (TangentSigmoid(Z) - Y) * dTangentSigmoid(Z)
# FUN: �p�����üh���
def delta_hidden(Z, D, W):
    return dTangentSigmoid(Z) * np.dot(D, W)
# FUN: �˶ǻ�
def backward(Y, Z2, Z1):
    # �p���X�h���
    D2 = delta_output(Z2, Y)

    # �p�����üh���
    D1 = delta_hidden(Z1, D2, W2)

    return D2, D1

# FUN: �p��ؼШ�ƹ��v�������
def computeWeight(D, X):
    return np.dot(D.T, X)

# FUN: �p��ؼШ�ƹﰾ�t�����
def computeBias(D):
    return D.sum(axis=0)

# FUN: �ѼƧ�s
def update_paras(D2, X1, D1, X0):
    global W2, W1, b1, b2, neuron_num, learning_rate
    # ��s�v��
    W2 = W2 - learning_rate * computeWeight(D2, X1)
    W1 = W1 - learning_rate * computeWeight(D1, X0)

    # ��s���t
    b2 = b2 - learning_rate * computeBias(D2)
    b1 = b1 - learning_rate * computeBias(D1)

## FUN: �ǲ�
def train(X, Y):
    # ���V�Ǽ�
    Z1, X1, Z2, X2 = forward(X)

    # �ϦV�Ǽ�
    D2, D1 = backward(Y, Z2, Z1)

    # �ѼƧ�s
    update_paras(D2, X1, D1, X)

# FUN: �w��
def predict(X):
    # -1���̫�
    ans = forward(X)[-1]
    return ans

# FUN: �p��~�t
def error_function(Y, X):
    # 1/2*error^2
    a = predict(X)
    return 0.5 * ((Y - a) ** 2).sum()


## ����
def exec(stage, times, max_n, epoch):
    # �O�s�C���V�m���~�t
    diffs = []  
    min_n = stage
    epoch_list = []

    # ���ưV�m
    for time in range(0, times):
        train_X, train_Y, test_X, test_Y, min_inputs, min_outputs,range_inputs, range_outputs = generateData(stage)

        for n in range(min_n, max_n+1):
            # NN
            init(n)

            ## �妸����
            batch = 100
            for e in range(0, epoch):
                # �H�����ðV�m��ƪ�����
                p = np.random.permutation(len(train_X))

                # �M���C��batch
                for i in range(math.ceil(len(train_X) / batch)):
                    # ���X��ebatch�����޽d��
                    indice = p[i * batch:(i + 1) * batch]

                    # �ھگ��޽d��q��l��Ƥ����X������batch���
                    X = train_X[indice]
                    Y = train_Y[indice]

                    # �ϥη�ebatch��ƶi��ҫ��V�m
                    train(X, Y)

                 
                if stage == 1:
                    # ��X�V�m�~�t
                    if e % 10 == 0:
                        error = error_function(train_Y, train_X)
                        log = f'\
                        time = {time}, n = {n}, error = {error} ({e}th epoch),\n \
                        '
                        print(log)
                    ## �w���äϥ��W��
                    predicted_outputs = unstandardize(predict(test_X), min_outputs, range_outputs)
                    origin_outputs = unstandardize(test_Y, min_outputs, range_outputs)
                    ## �O���~�t
                    mean_diff = np.mean(np.abs(predicted_outputs - origin_outputs))
                    if len(diffs) < n:
                        diffs.append(mean_diff)
                    else:
                        diffs[n-1] += mean_diff

                if stage == 2:
                    # �w���ä���
                    preds = predict(test_X)
                    classify_preds = np.where(preds <= 0.5, 0, 1)
                    ## �O���~�t
                    mean_diff = np.mean(np.abs(classify_preds - test_Y))
                    ##print(mean_diff)  
                    # �K�[�ܵ�ı�ƦC��
                    epoch_list.append(e + 1)
                    diffs.append(mean_diff)

                    # ø�s�ëO�s�Ϥ�
                    plt.plot(epoch_list, diffs, '-o')
                    plt.xlabel('Epoch')
                    plt.ylabel('Mean Difference')
                    plt.title('Mean Difference vs. Epoch')
                    plt.xlim(1, epoch)
                    plt.ylim(0, 1)
                    # �[�J�C���I������
                    #for x, y in zip(epoch_list, diffs):
                    #    plt.text(x, y, f'{y:.2f}', ha='center', va='bottom')
                    plt.savefig(f'epoch_{len(epoch_list)}.png')
                    plt.close()
                
    return diffs