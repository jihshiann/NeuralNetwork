## �ĤG���q-�ӧ@�~���q�@�A��ƶ��令XOR����ƶ��A�B�����g�j�p�w�q��2-2-1
#1. �����o�Ӹ�ƶ��������ʧ@�C�b�o�Ӹ�ƶ����A�A�|����10000���������
#2. x���d��i�H���b[-0.5, 0.2]��[0.8, 1.5]��Ӱ϶����Ay���d��i�H���b[-0.5, 0.2]��[0.8, 1.5]��Ӱ϶���
#3. �Yx�Py��Ƹ��b[-0.5, 0.2]���A�N���|�Q�ର0�A�Ϥ��Y���b[0.8, 1.5]���A�N���|�Q�ର1
#4. �C�ӵ�����Ƴ̫��ݩ�������O�h��XOR�޿�M�w�C
#> PS1: �����g����J�Ox�Py���ƭȡA��X�h�O�@�ӼƭȡA�Y���ƭȤp�󵥩�0.5�A�h�Q�����1���A�Ϥ��Y�j��0.5�A�h�Q�����2���C�o�������g����J�P��X���n���������W�ƫ�~��}�l�����g���V�m
#> PS2: �Ч�C���V�m���ѼƵ��G��ı�Ƨe�{�X��(�వ���ʺA��ΡA�p�U�����v���ҥ�)�A�ù��ո�Ū�o�˪���ı�Ƶ��G

import Library as NN
import matplotlib.pyplot as plt

## ����
times = 2
max_n = 2
stage = 2
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



