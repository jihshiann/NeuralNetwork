## 第一階段-撰寫一個2-n-1的淺層類神經網路
#1. 隱藏層的神經元使用Tangent Sigmoid作為激活函數
#2. 輸出層為線性函數
#3. 類神經網路的輸入為兩個四位數的數值，輸出則為這兩個數的總和
#4. 調整你的隱藏層神經元數量，並畫出一張神經元數量對30次訓練結果之誤差平均的圖。
#> PS1: 撰寫類神經前請先產生你的資料集，建議要有10000筆以上，接著把這個資料集切開成訓練與測試資料集。
#> PS2: 不管是輸入還是輸出都要正規化到0-1之間，而產出結果時，需要把輸出反正規化到原本的空間中。

import Library as NN
import matplotlib.pyplot as plt

## 執行
times = 3
max_n = 5
stage = 1
diffs = NN.exec(stage, times, max_n)
mean_diffs = [diff / times for diff in diffs]


## 繪圖
n_list = list(range(1, max_n+1))
plt.plot(n_list, mean_diffs)
plt.xlabel('Number of Neurons')
plt.ylabel('Mean Diff')
plt.title('Number of Neurons vs. Mean Diff')
plt.show()
print('end')



