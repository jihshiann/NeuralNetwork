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
# ��X��: �|��� + �|��� �}�C
expecteds = np.sum(inputs, axis=1)
# FUN: ���W��
def standardize(numArray):
    # ���W�Ʀ�0��1����
    numArr_std = (numArray - np.min(numArray)) / (np.max(numArray) - np.min(numArray))
    return numArr_std


## �����g����
# ���üh���g����
neuron_num = np.random.randint(2,9)
neuron_num = 3
## �v���P���t
weights_L1 = np.random.randn(2, neuron_num)
weights_L2 = np.random.randn(neuron_num, 1)
bias = np.random.randn(neuron_num)
# FUN: activation function
# tanh(x) = (e^x - e^(-x)) / (e^x + e^(-x))
# https://www.baeldung.com/cs/sigmoid-vs-tanh-functions
def tangentSigmoid(inputs):
    outputs = (np.exp(inputs) - np.exp(-inputs)) / (np.exp(inputs) + np.exp(-inputs))
    return outputs

# FUN: ���V�Ǽ�
def forward(inputs):
     # �Ĥ@�h
    weighteds_L1 = np.dot(inputs, weights_L1) + bias
    computeds = tangentSigmoid(weighteds_L1)
    
    # �ĤG�h
    weighteds_L2 = np.dot(computeds, weights_L2)
    outputs = tangentSigmoid(weighteds_L2)
    
    return outputs

## �ϦV�Ǽ�
# FUN: Tangent Sigmoid �L��
def dTangentSigmoid(inputs):
    outputs = 1 - np.tanh(inputs)**2
    return outputs
# FUN: �p���X�h���
def delta_output(weighteds, expecteds):
    result = (tangentSigmoid(weighteds) -expecteds) * dTangentSigmoid(weighteds)
    return result
# FUN: �p�����üh���
def delta_hidden(weighteds, next_weights, next_deltas):
    result = np.dot(next_deltas, next_weights.T) * dTangentSigmoid(weighteds)
    return result
# FUN: �˶ǻ�
def backward(inputs, expecteds, weighteds_L1, weighteds_L2):
    # �p���X�h���
    delta_L2 = delta_output(weighteds_L2, expecteds)
    
    # �p�����üh���
    delta_L1 = delta_hidden(weighteds_L1, weights_L2, delta_L2)
    
    return delta_L2, delta_L1

## ��s
learning_rate = 0.001

# FUN: �ؼШ�ƹ��v���L��
def deweight(delta, inputs):
    return np.dot(delta.T, inputs)

# FUN: �ؼШ�ƹﰾ�t�L��
def debias(delta):
    return np.mean(delta, axis=0)

# FUN: �ѼƧ�s
def update_paras(delta_L2, delta_L1, inputs, weighteds_L1):
    # ��s�ĤG�h�v��
    dW2 = deweight(delta_L2, weighteds_L1)
    weights_L2 -= learning_rate * dW2

    # ��s�Ĥ@�h�v��
    dW1 = deweight(delta_L1, inputs)
    weights_L1 -= learning_rate * dW1

    # ��s���t
    dB = debias(delta_L2)
    bias -= learning_rate * dB
    
## �ǲ�
num_epochs = 300
def train(inputs, expecteds, weights_L1, weights_L2, bias, num_epochs):
    errors = []
    for epoch in range(num_epochs):
        # ���V�Ǽ�
        outputs = forward(inputs)
        
        # �p��~�t
        error = np.mean(np.abs(outputs - expecteds))
        errors.append(error)
        
        # �ϦV�Ǽ�
        delta_L2, delta_L1 = backward(inputs, expecteds, weights_L1, weights_L2)
        
        # �ѼƧ�s
        weights_L1, weights_L2, bias = update_paras(delta_L2, delta_L1, inputs, weights_L1)

## ����
# FUN: �w��
def predict(inputs):
    outputs = forward(inputs)
    return outputs

# FUN: �ؼШ��
def error_function(inputs, expecteds):
    #1/2*error^2
    return 0.5*((expecteds-predict(inputs)**2).sum())

## �妸����
batch = 100
for num_epochs in range(1, num_epochs+1):
    # �H�����ðV�m��ƪ�����
    p = np.random.permutation(len(inputs))
    
    # �M���C��batch
    for i in range(math.ceil(len(inputs)/batch)):
        # ���X��ebatch�����޽d��
        indice = p[i*batch:(i+1)*batch]
        
        # �ھگ��޽d��q��l��Ƥ����X������batch���
        inputs_batch = inputs[indice]
        expecteds_batch = expecteds[indice]
        
        # �ϥη�ebatch��ƶi��ҫ��V�m
        train(inputs_batch, expecteds_batch)

    # �C1000��epoch��X�@���V�m�~�t
    if num_epochs % 1000 == 0:
        log = 'error = {8.4f} ({:5d}th epoch)'
        # �p��V�m��ƤW���~�t�ÿ�X�챱��x
        print(log.format(error_function(inputs, expecteds), num_epochs))

