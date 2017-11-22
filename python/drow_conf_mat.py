import pandas as pd
import numpy as np
import seaborn as sn
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

def print_cmx(y_true,y_pred):
    labels = sorted(list(set(y_true)))
    cmx_data = confusion_matrix(y_true,y_pred,labels=labels)

    df_cmx = pd.DataFrame(cmx_data, index=labels, columns=labels)

    plt.figure(figsize = (10,7))
    # sn.heatmap ヒートマップを可視化
    sn.heatmap(df_cmx, annot=True)
    plt.show()

# 二次元配列を作る　例として10x12大きさの配列に乱数を入れる
np.random.seed(0)
data = np.random.rand(10,12)
sn.heatmap(data,annot = True)
plt.show()

# 今度は自分で配列を作ってそれを表示
data2 = np.array(
        [[1,2,3,4],[5,6,7,8],[9,10,11,12]])

# cmapはカラーマップの指定
# カラーマップは以下のサイト参照
# https://matplotlib.org/examples/color/colormaps_reference.html
# (access: 2017/11/21)
sn.heatmap(data2,annot = True, cmap='Blues')
plt.show()

#---------------------------------------------
# 最後に実際に実験で使う時のように設定してみる
#---------------------------------------------

# 使うデータ
data3 = np.array([\
        [0.8,0.05,0.1,0.025,0.025],\
        [0.01, 0.7,0.04,0.02,0.23],\
        [0.1,0.2,0.6,0.04,0.06],\
        [0.01,0.02,0.03,0.9,0.04],\
        [0,0.03,0.03,0.09,0.85]])
# pandasのdataframeを使って、項目名とかの設定する
df_data3 = pd.DataFrame(data3)
# カラム名、インデックス名をつける
df_data3.columns = ["class1","class2","class3","class4","class5"]
df_data3.index = ["class1_idx","class2","class3","class4","class5"]

# ヒートマップを作る
sn.heatmap(df_data3,annot = True, cmap='Blues')

# ラベル設定
plt.xlabel("class_x")
plt.ylabel("class_y")

# y軸の回転角度を０にする（横書きにする）
plt.yticks(rotation=0)

# 表示
plt.show()
