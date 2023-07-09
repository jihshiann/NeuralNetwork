## �Ĥ@���q-���g�@��2-n-1���L�h�����g����
#1. ���üh�����g���ϥ�Tangent Sigmoid�@���E�����
#2. ��X�h���u�ʨ��
#3. �����g��������J����ӥ|��ƪ��ƭȡA��X�h���o��Ӽƪ��`�M
#4. �վ�A�����üh���g���ƶq�A�õe�X�@�i���g���ƶq��30���V�m���G���~�t�������ϡC
#> PS1: ���g�����g�e�Х����ͧA����ƶ��A��ĳ�n��10000���H�W�A���ۧ�o�Ӹ�ƶ����}���V�m�P���ո�ƶ��C
#> PS2: ���ެO��J�٬O��X���n���W�ƨ�0-1�����A�Ӳ��X���G�ɡA�ݭn���X�ϥ��W�ƨ�쥻���Ŷ����C

from itertools import repeat
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

# FUN: �ؼШ�ƹ��v���L��
def deweight(D, X):
    return np.dot(D.T, X)

# FUN: �ؼШ�ƹﰾ�t�L��
def debias(D):
    return D.sum(axis=0)

# FUN: �ѼƧ�s
def update_paras(D2, X1, D1, X0):
    global W2, W1, b1, b2, neuron_num, learning_rate
    # ��s�v��
    W2 = W2 - learning_rate * deweight(D2, X1)
    W1 = W1 - learning_rate * deweight(D1, X0)

    # ��s���t
    b2 = b2 - learning_rate * debias(D2)
    b1 = b1 - learning_rate * debias(D1)

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
def exec(times, max_n):
    # �O�s�C���V�m���~�t
    diffs = []  

    # ���ưV�m
    for time in range(0, times):
        ## DATA
        # �ƾڼƶq
        data_num = 30000
        # ��J��: (�|���, �|���)�}�C
        total_X = (np.random.rand(data_num, 2)*9998).astype(np.int32)+1
        # ��X��: �|��� + �|��� �}�C
        total_Y = np.sum(total_X, axis=1).reshape(-1, 1)
        # ���J�M��X�i�楿�W��
        total_X, min_inputs, range_inputs = standardize(total_X)
        total_Y, min_outputs, range_outputs = standardize(total_Y)
        # �����V�m�P����
        train_ratio = 0.7
        train_size = int(train_ratio * len(total_X))
        train_X = total_X[:train_size]
        train_Y = total_Y[:train_size]
        test_X = total_X[train_size:]
        test_Y = total_Y[train_size:]

        for n in range(1, max_n+1):
            # NN
            init(n)

            ## �妸����
            batch = 30
            epoch = 30
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

                 # ��X�V�m�~�t
                if e % 3 == 0:
                    error = error_function(train_Y, train_X)
                    log = f'\
                    time = {time}, n = {n}, error = {error} ({e}th epoch),\n \
                    '
                    print(log)
      
            ## ��w�����G�i��ϥ��W��
            predicted_outputs = unstandardize(predict(test_X), min_outputs, range_outputs)
            origin_outputs = unstandardize(test_Y, min_outputs, range_outputs)
            ## �O���~�t
            mean_diff = np.mean(np.abs(predicted_outputs - origin_outputs))
            if len(diffs) < n:
                diffs.append(mean_diff)
            else:
                diffs[n-1] += mean_diff
    return diffs

times = 30
max_n = 30
diffs = exec(times, max_n)
mean_diffs = [diff / times for diff in diffs]


### ø��
n_list = list(range(1, max_n+1))
plt.plot(n_list, mean_diffs)
plt.xlabel('Number of Neurons')
plt.ylabel('Mean Diff')
plt.title('Number of Neurons vs. Mean Diff')
plt.show()
print('end')



