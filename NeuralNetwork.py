## �Ĥ@���q-���g�@��2-n-1���L�h�����g����
#1. ���üh�����g���ϥ�Tangent Sigmoid�@���E�����
#2. ��X�h���u�ʨ��
#3. �����g��������J����ӥ|��ƪ��ƭȡA��X�h���o��Ӽƪ��`�M
#4. �վ�A�����üh���g���ƶq�A�õe�X�@�i���g���ƶq��30���V�m���G���~�t�������ϡC
#> PS1: ���g�����g�e�Х����ͧA����ƶ��A��ĳ�n��10000���H�W�A���ۧ�o�Ӹ�ƶ����}���V�m�P���ո�ƶ��C
#> PS2: ���ެO��J�٬O��X���n���W�ƨ�0-1�����A�Ӳ��X���G�ɡA�ݭn���X�ϥ��W�ƨ�쥻���Ŷ����C



import numpy as np
import math

# �ƾڼƶq
data_num = 12345
# ��J��: (�|���, �|���)�}�C
inputs = np.random.randint(1000, 9999, size=(data_num, 2))
inputs = inputs.reshape(-1, 2)  # �N��J�ƾ��ഫ���G���}�C
# ��X��: �|��� + �|��� �}�C
expecteds = np.sum(inputs, axis=1)

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


## �����g����
# ���üh���g����
neuron_num = np.random.randint(2,9)
## �v���P���t
weights_L1 = np.random.randn(2, neuron_num)
weights_L2 = np.random.randn(neuron_num, 1)
bias_L1 = np.random.randn(neuron_num)
bias_L2 = np.random.randn(1)
# FUN: activation function
# tanh(x) = (e^x - e^(-x)) / (e^x + e^(-x))
# https://www.baeldung.com/cs/sigmoid-vs-tanh-functions
def tangentSigmoid(inputs):
    outputs = (np.exp(inputs) - np.exp(-inputs)) / (np.exp(inputs) + np.exp(-inputs))
    return outputs

# FUN: ���V�Ǽ�
def forward(inputs):
    # �Ĥ@�h
    weighteds_L1 = np.dot(inputs, weights_L1) + bias_L1  
    computeds_L1 = tangentSigmoid(weighteds_L1)

    # �ĤG�h
    weighteds_L2 = np.dot(computeds_L1, weights_L2) + bias_L2 
    computeds_L2 = weighteds_L2

    return weighteds_L1, computeds_L1, weighteds_L2, computeds_L2

## �ϦV�Ǽ�
# FUN: Tangent Sigmoid �L��
def dTangentSigmoid(inputs):
    outputs = 1 - np.tanh(inputs)**2
    return outputs
# FUN: �p���X�h���
def delta_output(weighteds, expecteds):
    result = (tangentSigmoid(weighteds) - expecteds.reshape(-1, 1)) * dTangentSigmoid(weighteds)
    return result
# FUN: �p�����üh���
def delta_hidden(weighteds, next_weights, next_deltas):
    result = np.dot(next_deltas, next_weights.T) * dTangentSigmoid(weighteds)
    return result
# FUN: �˶ǻ�
def backward(expecteds, weighteds_L2, weighteds_L1):
    # �p���X�h���
    delta_L2 = delta_output(weighteds_L2, expecteds)

    # �p�����üh���
    delta_L1 = delta_hidden(weighteds_L1, weights_L2, delta_L2)

    return delta_L2, delta_L1


## ��s
learning_rate = 0.001
# FUN: �ؼШ�ƹ��v���L��
def deweight(delta, inputs):
    return np.dot(inputs.T, delta)

# FUN: �ؼШ�ƹﰾ�t�L��
def debias(delta):
    return np.mean(delta, axis=0)

# FUN: �ѼƧ�s
def update_paras(delta_L2, computeds_L1, delta_L1, inputs):
    global weights_L2, weights_L1, bias_L1, bias_L2  # ��s�����ܶq�C��
    # ��s�ĤG�h�v��
    # �p�� dW2
    delta_L2_repeated = np.repeat(delta_L2, neuron_num, axis=1)
    dW2 = np.sum(delta_L2_repeated * computeds_L1, axis=0)

    # ��s weights_L2
    weights_L2 -= learning_rate * dW2.reshape(weights_L2.shape)

    # ��s�Ĥ@�h�v��
    dW1 = deweight(delta_L1, inputs)
    weights_L1 -= learning_rate * dW1

    # ��s�ĤG�h���t
    dB2 = debias(delta_L2)
    bias_L2 -= learning_rate * dB2

    # ��s�Ĥ@�h���t
    dB1 = debias(delta_L1)
    bias_L1 -= learning_rate * dB1
    
## �ǲ�
def train(inputs, expecteds):
    # ���V�Ǽ�
    weighteds_L1, computeds_L1, weighteds_L2, computeds_L2 = forward(inputs)
        
    # �ϦV�Ǽ�
    delta_L2, delta_L1 = backward(expecteds, weighteds_L2, weighteds_L1)
        
    # �ѼƧ�s
    update_paras(delta_L2, computeds_L1, delta_L1, inputs)

## ����
# FUN: �w��
def predict(inputs):
    weighteds_L1, computeds_L1, weighteds_L2, computeds_L2 = forward(inputs)
    return computeds_L2

# FUN: �ؼШ��
def error_function(inputs, expecteds):
    # 1/2*error^2
    return 0.5 * ((expecteds - predict(inputs)) ** 2).sum()

## �妸����
batch = 100
epoch = 10000
std_inputs, min_inputs, range_inputs = standardize(inputs)
std_expecteds, min_expecteds, range_expecteds = standardize(expecteds)
for e in range(1, epoch+1):
    # �H�����ðV�m��ƪ�����
    p = np.random.permutation(len(std_inputs))
    
    # �M���C��batch
    for i in range(math.ceil(len(std_inputs)/batch)):
        # ���X��ebatch�����޽d��
        indice = p[i*batch:(i+1)*batch]
        
        # �ھگ��޽d��q��l��Ƥ����X������batch���
        inputs_batch = std_inputs[indice]
        expecteds_batch = std_expecteds[indice]
        
        # �ϥη�ebatch��ƶi��ҫ��V�m
        train(inputs_batch, expecteds_batch)

    # �C10��epoch��X�@���V�m�~�t
    if e % 100 == 0:
        log = 'error = {:8.4f} ({:5d}th epoch)'
        # �p��V�m��ƤW���~�t�ÿ�X�챱��x
        print(log.format(error_function(std_inputs, std_expecteds), e))

