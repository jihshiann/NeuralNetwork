## �Ĥ@���q-���g�@��2-n-1���L�h�����g����
#1. ���üh�����g���ϥ�Tangent Sigmoid�@���E�����
#2. ��X�h���u�ʨ��
#3. �����g��������J����ӥ|��ƪ��ƭȡA��X�h���o��Ӽƪ��`�M
#4. �վ�A�����üh���g���ƶq�A�õe�X�@�i���g���ƶq��30���V�m���G���~�t�������ϡC
#> PS1: ���g�����g�e�Х����ͧA����ƶ��A��ĳ�n��10000���H�W�A���ۧ�o�Ӹ�ƶ����}���V�m�P���ո�ƶ��C
#> PS2: ���ެO��J�٬O��X���n���W�ƨ�0-1�����A�Ӳ��X���G�ɡA�ݭn���X�ϥ��W�ƨ�쥻���Ŷ����C

import numpy as np
import math

# FUN: ��l��
def init():
    global W2, W1, b1, b2, neuron_num, learning_rate
    ## �����g����
    # ���üh���g����
    neuron_num = np.random.randint(2, 9)

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
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# FUN: ���V�Ǽ�
def forward(X0):
    # �Ĥ@�h
    Z1 = np.dot(X0, W1.T) + b1
    X1 = sigmoid(Z1)

    # �ĤG�h
    Z2 = np.dot(X1, W2.T) + b2
    X2 = sigmoid(Z2)

    return Z1, X1, Z2, X2

## �ϦV�Ǽ�
# FUN: Tangent Sigmoid �L��
def dsigmoid(x): 
    return (1 - sigmoid(x)) * sigmoid(x)
# FUN: �p���X�h���
def delta_output(Z, Y):
    return (sigmoid(Z) - Y) * dsigmoid(Z)
# FUN: �p�����üh���
def delta_hidden(Z, D, W):
    return dsigmoid(Z) * np.dot(D, W)
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

## �ǲ�
def train(X, Y):
    # ���V�Ǽ�
    Z1, X1, Z2, X2 = forward(X)

    # �ϦV�Ǽ�
    D2, D1 = backward(Y, Z2, Z1)

    # �ѼƧ�s
    update_paras(D2, X1, D1, X)

## ����
# FUN: �w��
def predict(X):
    # -1���̫�
    ans = forward(X)[-1]
    return ans

# FUN: �ؼШ��
def error_function(Y, X):
    # 1/2*error^2
    a = predict(X)
    return 0.5 * ((Y - a) ** 2).sum()


## DATA
# �ƾڼƶq
data_num = 12345
# ��J��: (�|���, �|���)�}�C
TX = (np.random.rand(data_num, 2)*9998).astype(np.int32)+1
# ��X��: �|��� + �|��� �}�C
TY = np.sum(TX, axis=1).reshape(-1, 1)
# ���J�M��X�i�楿�W��
TX, min_inputs, range_inputs = standardize(TX)
TY, min_expecteds, range_expecteds = standardize(TY)

# NN
init()

## �妸����
batch = 100
epoch = 100
for e in range(1, epoch + 1):
    # �H�����ðV�m��ƪ�����
    p = np.random.permutation(len(TX))

    # �M���C��batch
    for i in range(math.ceil(len(TX) / batch)):
        # ���X��ebatch�����޽d��
        indice = p[i * batch:(i + 1) * batch]

        # �ھگ��޽d��q��l��Ƥ����X������batch���
        X0 = TX[indice]
        Y = TY[indice]

        # �ϥη�ebatch��ƶi��ҫ��V�m
        train(X0, Y)

    # ��X�V�m�~�t
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

## ��w�����G�i��ϥ��W��
predicted_output = unstandardize(predict(TX), min_expecteds, range_expecteds)
origin_TY = unstandardize(TY, min_expecteds, range_expecteds)
mean_diff = np.mean(np.abs(predicted_output - origin_TY))
print(mean_diff)
