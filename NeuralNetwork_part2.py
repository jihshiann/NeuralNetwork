## �ĤG���q-�ӧ@�~���q�@�A��ƶ��令XOR����ƶ��A�B�����g�j�p�w�q��2-2-1
#1. �����o�Ӹ�ƶ��������ʧ@�C�b�o�Ӹ�ƶ����A�A�|����10000���������
#2. x���d��i�H���b[-0.5, 0.2]��[0.8, 1.5]��Ӱ϶����Ay���d��i�H���b[-0.5, 0.2]��[0.8, 1.5]��Ӱ϶���
#3. �Yx�Py��Ƹ��b[-0.5, 0.2]���A�N���|�Q�ର0�A�Ϥ��Y���b[0.8, 1.5]���A�N���|�Q�ର1
#4. �C�ӵ�����Ƴ̫��ݩ�������O�h��XOR�޿�M�w�C
#> PS1: �����g����J�Ox�Py���ƭȡA��X�h�O�@�ӼƭȡA�Y���ƭȤp�󵥩�0.5�A�h�Q�����1���A�Ϥ��Y�j��0.5�A�h�Q�����2���C�o�������g����J�P��X���n���������W�ƫ�~��}�l�����g���V�m
#> PS2: �Ч�C���V�m���ѼƵ��G��ı�Ƨe�{�X��(�వ���ʺA��ΡA�p�U�����v���ҥ�)�A�ù��ո�Ū�o�˪���ı�Ƶ��G

import Library as NN
import matplotlib.pyplot as plt
import imageio
import os

## ����
times = 1
max_n = 2
stage = 2
epoch = 100
NN.exec(stage, times, max_n, epoch)

# �N�Ϥ��զX��GIF�ʵe
images = []
for e in range(epoch):
    images.append(imageio.imread(f'epoch_{e+1}.png'))
    #. `fps=50` == `duration=20` (1000 * 1/50).
    imageio.mimsave('dynamic_visualization.gif', images, duration=20)

for e in range(epoch):
    os.remove(f'epoch_{e+1}.png')
print('end')



