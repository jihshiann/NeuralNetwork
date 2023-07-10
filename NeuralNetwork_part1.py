## �Ĥ@���q-���g�@��2-n-1���L�h�����g����
#1. ���üh�����g���ϥ�Tangent Sigmoid�@���E�����
#2. ��X�h���u�ʨ��
#3. �����g��������J����ӥ|��ƪ��ƭȡA��X�h���o��Ӽƪ��`�M
#4. �վ�A�����üh���g���ƶq�A�õe�X�@�i���g���ƶq��30���V�m���G���~�t�������ϡC
#> PS1: ���g�����g�e�Х����ͧA����ƶ��A��ĳ�n��10000���H�W�A���ۧ�o�Ӹ�ƶ����}���V�m�P���ո�ƶ��C
#> PS2: ���ެO��J�٬O��X���n���W�ƨ�0-1�����A�Ӳ��X���G�ɡA�ݭn���X�ϥ��W�ƨ�쥻���Ŷ����C

import Library as NN
import matplotlib.pyplot as plt

## ����
times = 3
max_n = 5
stage = 1
diffs = NN.exec(stage, times, max_n)
mean_diffs = [diff / times for diff in diffs]


## ø��
n_list = list(range(1, max_n+1))
plt.plot(n_list, mean_diffs)
plt.xlabel('Number of Neurons')
plt.ylabel('Mean Diff')
plt.title('Number of Neurons vs. Mean Diff')
plt.show()
print('end')



