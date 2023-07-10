## 第二階段-承作業階段一，資料集改成XOR的資料集，且類神經大小定義為2-2-1
#1. 完成這個資料集的分類動作。在這個資料集中，你會產生10000筆虛擬資料
#2. x的範圍可以落在[-0.5, 0.2]或[0.8, 1.5]兩個區塊中，y的範圍可以落在[-0.5, 0.2]或[0.8, 1.5]兩個區塊中
#3. 若x與y資料落在[-0.5, 0.2]中，代表其會被轉為0，反之若落在[0.8, 1.5]中，代表其會被轉為1
#4. 每個虛擬資料最後屬於哪個類別則由XOR邏輯決定。
#> PS1: 類神經的輸入是x與y的數值，輸出則是一個數值，若此數值小於等於0.5，則被分到第1類，反之若大於0.5，則被分到第2類。這個類神經的輸入與輸出都要先完成正規化後才能開始類神經的訓練
#> PS2: 請把每次訓練的參數結果視覺化呈現出來(能做成動態更佳，如下面的影片所示)，並嘗試解讀這樣的視覺化結果

import Library as NN
import matplotlib.pyplot as plt

## 執行
times = 2
max_n = 2
stage = 2
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



