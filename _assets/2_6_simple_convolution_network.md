```python
import pickle
import numpy as np
from collections import OrderedDict
from common import layers
from common import optimizer
from data.mnist import load_mnist
import matplotlib.pyplot as plt

import sys
sys.path.append('.')


```

効率よく行列計算をするためのデータ変換

im2col <-> col2im の実装


```python
'''
input_data/col : 畳み込み層への入力データ / 変換後データ
filter_h: フィルターの高さ
filter_w: フィルターの横幅
stride: ストライド
pad: パディング
'''
# 画像データを２次元配列に変換
def im2col(input_data, filter_h, filter_w, stride=1, pad=0):
    # N: number, C: channel, H: height, W: width
    N, C, H, W = input_data.shape
    out_h = (H + 2 * pad - filter_h)//stride + 1
    out_w = (W + 2 * pad - filter_w)//stride + 1

    img = np.pad(input_data, [(0,0), (0,0), (pad, pad), (pad, pad)], 'constant')
    col = np.zeros((N, C, filter_h, filter_w, out_h, out_w))

    for y in range(filter_h):
        y_max = y + stride * out_h
        for x in range(filter_w):
            x_max = x + stride * out_w
            col[:, :, y, x, :, :] = img[:, :, y:y_max:stride, x:x_max:stride]
    
    col = col.transpose(0, 4, 5, 1, 2, 3) # (N, C, filter_h, filter_w, out_h, out_w) -> (N, filter_w, out_h, out_w, C, filter_h)    
    
    col = col.reshape(N * out_h * out_w, -1)
    return col

# ２次元配列を画像データに変換
def col2im(col, input_shape, filter_h, filter_w, stride=1, pad=0):
    # N: number, C: channel, H: height, W: width
    N, C, H, W = input_shape
    # 切り捨て除算    
    out_h = (H + 2 * pad - filter_h)//stride + 1
    out_w = (W + 2 * pad - filter_w)//stride + 1
    col = col.reshape(N, out_h, out_w, C, filter_h, filter_w).transpose(0, 3, 4, 5, 1, 2) # (N, filter_h, filter_w, out_h, out_w, C)

    img = np.zeros((N, C, H + 2 * pad + stride - 1, W + 2 * pad + stride - 1))
    for y in range(filter_h):
        y_max = y + stride * out_h
        for x in range(filter_w):
            x_max = x + stride * out_w
            img[:, :, y:y_max:stride, x:x_max:stride] += col[:, :, y, x, :, :]

    return img[:, :, pad:H + pad, pad:W + pad]
```


```python
def print_im2col_col2im(input_data, conv_param):
    filter_h, filter_w, stride, pad = conv_param
    print('====== input_data =======\n', input_data)
    print('==========================')
    col = im2col(input_data, filter_h=filter_h, filter_w=filter_w, stride=stride, pad=pad)
    print('========= im2col ==========\n', col)
    print('=========================')

    img = col2im(col, input_shape=input_data.shape, filter_h=filter_h, filter_w=filter_w, stride=stride, pad=pad)
    print('========= col2im ==========\n', img)
    print('=========================')
    print()

```

im2col の振舞いを理解するため、(わかりやすさのため)全要素1の入力を考える
2x2 のフィルタだと、col2im の結果は
 - stride = 2 にすればオーバーラップがなくなるため、全部 1 になる
 - padding = 1 にすれば全画素が同じだけ使用されて、全部 4 になる
うん。納得。


```python
input_data = np.reshape(np.ones(16), (1, 1, 4, 4)) # number, channel, height, widthを表す

conv_param =  (2,2,1,0)  # filter_h, filter_w, stride, pad
print("filter_h, fitler_w, stride, padding = ", conv_param)
print_im2col_col2im( input_data , conv_param)

conv_param =  (2,2,2,0) # filter_h, filter_w, stride, pad
print("filter_h, fitler_w, stride, padding = ", conv_param)
print_im2col_col2im( input_data , conv_param)


conv_param =  (2,2,1,2) # filter_h, filter_w, stride, pad
print("filter_h, fitler_w, stride, padding = ", conv_param)
print_im2col_col2im( input_data , conv_param)



# input_data = np.random.rand(2, 1, 4, 4)*100//1 # number, channel, height, widthを表す



```

```
    filter_h, fitler_w, stride, padding =  (2, 2, 1, 0)
    ====== input_data =======
     [[[[1. 1. 1. 1.]
       [1. 1. 1. 1.]
       [1. 1. 1. 1.]
       [1. 1. 1. 1.]]]]
    ==========================
    ========= im2col ==========
     [[1. 1. 1. 1.]
     [1. 1. 1. 1.]
     [1. 1. 1. 1.]
     [1. 1. 1. 1.]
     [1. 1. 1. 1.]
     [1. 1. 1. 1.]
     [1. 1. 1. 1.]
     [1. 1. 1. 1.]
     [1. 1. 1. 1.]]
    =========================
    ========= col2im ==========
     [[[[1. 2. 2. 1.]
       [2. 4. 4. 2.]
       [2. 4. 4. 2.]
       [1. 2. 2. 1.]]]]
    =========================
    
    filter_h, fitler_w, stride, padding =  (2, 2, 2, 0)
    ====== input_data =======
     [[[[1. 1. 1. 1.]
       [1. 1. 1. 1.]
       [1. 1. 1. 1.]
       [1. 1. 1. 1.]]]]
    ==========================
    ========= im2col ==========
     [[1. 1. 1. 1.]
     [1. 1. 1. 1.]
     [1. 1. 1. 1.]
     [1. 1. 1. 1.]]
    =========================
    ========= col2im ==========
     [[[[1. 1. 1. 1.]
       [1. 1. 1. 1.]
       [1. 1. 1. 1.]
       [1. 1. 1. 1.]]]]
    =========================
    
    filter_h, fitler_w, stride, padding =  (2, 2, 1, 2)
    ====== input_data =======
     [[[[1. 1. 1. 1.]
       [1. 1. 1. 1.]
       [1. 1. 1. 1.]
       [1. 1. 1. 1.]]]]
    ==========================
    ========= im2col ==========
     [[0. 0. 0. 0.]
     [0. 0. 0. 0.]
     [0. 0. 0. 0.]
     [0. 0. 0. 0.]
     [0. 0. 0. 0.]
     [0. 0. 0. 0.]
     [0. 0. 0. 0.]
     [0. 0. 0. 0.]
     [0. 0. 0. 1.]
     [0. 0. 1. 1.]
     [0. 0. 1. 1.]
     [0. 0. 1. 1.]
     [0. 0. 1. 0.]
     [0. 0. 0. 0.]
     [0. 0. 0. 0.]
     [0. 1. 0. 1.]
     [1. 1. 1. 1.]
     [1. 1. 1. 1.]
     [1. 1. 1. 1.]
     [1. 0. 1. 0.]
     [0. 0. 0. 0.]
     [0. 0. 0. 0.]
     [0. 1. 0. 1.]
     [1. 1. 1. 1.]
     [1. 1. 1. 1.]
     [1. 1. 1. 1.]
     [1. 0. 1. 0.]
     [0. 0. 0. 0.]
     [0. 0. 0. 0.]
     [0. 1. 0. 1.]
     [1. 1. 1. 1.]
     [1. 1. 1. 1.]
     [1. 1. 1. 1.]
     [1. 0. 1. 0.]
     [0. 0. 0. 0.]
     [0. 0. 0. 0.]
     [0. 1. 0. 0.]
     [1. 1. 0. 0.]
     [1. 1. 0. 0.]
     [1. 1. 0. 0.]
     [1. 0. 0. 0.]
     [0. 0. 0. 0.]
     [0. 0. 0. 0.]
     [0. 0. 0. 0.]
     [0. 0. 0. 0.]
     [0. 0. 0. 0.]
     [0. 0. 0. 0.]
     [0. 0. 0. 0.]
     [0. 0. 0. 0.]]
    =========================
    ========= col2im ==========
     [[[[4. 4. 4. 4.]
       [4. 4. 4. 4.]
       [4. 4. 4. 4.]
       [4. 4. 4. 4.]]]]
    =========================
```


そのほかのデータ構造でも試してみる。
 ( im2col までは ) 通し番号の方がわかりやすいので randでなくrangeをつかう。



```python
input_data = np.reshape(range(100), (2, 2, 5, 5)) # number, channel, height, widthを表す

conv_param =  (3,3,2,1)  # filter_h, filter_w, stride, pad
print("filter_h, fitler_w, stride, padding = ", conv_param)
print_im2col_col2im( input_data , conv_param)

```
```
    filter_h, fitler_w, stride, padding =  (3, 3, 2, 1)
    ====== input_data =======
     [[[[ 0  1  2  3  4]
       [ 5  6  7  8  9]
       [10 11 12 13 14]
       [15 16 17 18 19]
       [20 21 22 23 24]]
    
      [[25 26 27 28 29]
       [30 31 32 33 34]
       [35 36 37 38 39]
       [40 41 42 43 44]
       [45 46 47 48 49]]]
    
    
     [[[50 51 52 53 54]
       [55 56 57 58 59]
       [60 61 62 63 64]
       [65 66 67 68 69]
       [70 71 72 73 74]]
    
      [[75 76 77 78 79]
       [80 81 82 83 84]
       [85 86 87 88 89]
       [90 91 92 93 94]
       [95 96 97 98 99]]]]
    ==========================
    ========= im2col ==========
     [[ 0.  0.  0.  0.  0.  1.  0.  5.  6.  0.  0.  0.  0. 25. 26.  0. 30. 31.]
     [ 0.  0.  0.  1.  2.  3.  6.  7.  8.  0.  0.  0. 26. 27. 28. 31. 32. 33.]
     [ 0.  0.  0.  3.  4.  0.  8.  9.  0.  0.  0.  0. 28. 29.  0. 33. 34.  0.]
     [ 0.  5.  6.  0. 10. 11.  0. 15. 16.  0. 30. 31.  0. 35. 36.  0. 40. 41.]
     [ 6.  7.  8. 11. 12. 13. 16. 17. 18. 31. 32. 33. 36. 37. 38. 41. 42. 43.]
     [ 8.  9.  0. 13. 14.  0. 18. 19.  0. 33. 34.  0. 38. 39.  0. 43. 44.  0.]
     [ 0. 15. 16.  0. 20. 21.  0.  0.  0.  0. 40. 41.  0. 45. 46.  0.  0.  0.]
     [16. 17. 18. 21. 22. 23.  0.  0.  0. 41. 42. 43. 46. 47. 48.  0.  0.  0.]
     [18. 19.  0. 23. 24.  0.  0.  0.  0. 43. 44.  0. 48. 49.  0.  0.  0.  0.]
     [ 0.  0.  0.  0. 50. 51.  0. 55. 56.  0.  0.  0.  0. 75. 76.  0. 80. 81.]
     [ 0.  0.  0. 51. 52. 53. 56. 57. 58.  0.  0.  0. 76. 77. 78. 81. 82. 83.]
     [ 0.  0.  0. 53. 54.  0. 58. 59.  0.  0.  0.  0. 78. 79.  0. 83. 84.  0.]
     [ 0. 55. 56.  0. 60. 61.  0. 65. 66.  0. 80. 81.  0. 85. 86.  0. 90. 91.]
     [56. 57. 58. 61. 62. 63. 66. 67. 68. 81. 82. 83. 86. 87. 88. 91. 92. 93.]
     [58. 59.  0. 63. 64.  0. 68. 69.  0. 83. 84.  0. 88. 89.  0. 93. 94.  0.]
     [ 0. 65. 66.  0. 70. 71.  0.  0.  0.  0. 90. 91.  0. 95. 96.  0.  0.  0.]
     [66. 67. 68. 71. 72. 73.  0.  0.  0. 91. 92. 93. 96. 97. 98.  0.  0.  0.]
     [68. 69.  0. 73. 74.  0.  0.  0.  0. 93. 94.  0. 98. 99.  0.  0.  0.  0.]]
    =========================
    ========= col2im ==========
     [[[[  0.   2.   2.   6.   4.]
       [ 10.  24.  14.  32.  18.]
       [ 10.  22.  12.  26.  14.]
       [ 30.  64.  34.  72.  38.]
       [ 20.  42.  22.  46.  24.]]
    
      [[ 25.  52.  27.  56.  29.]
       [ 60. 124.  64. 132.  68.]
       [ 35.  72.  37.  76.  39.]
       [ 80. 164.  84. 172.  88.]
       [ 45.  92.  47.  96.  49.]]]
    
    
     [[[ 50. 102.  52. 106.  54.]
       [110. 224. 114. 232. 118.]
       [ 60. 122.  62. 126.  64.]
       [130. 264. 134. 272. 138.]
       [ 70. 142.  72. 146.  74.]]
    
      [[ 75. 152.  77. 156.  79.]
       [160. 324. 164. 332. 168.]
       [ 85. 172.  87. 176.  89.]
       [180. 364. 184. 372. 188.]
       [ 95. 192.  97. 196.  99.]]]]
    =========================
```


```python
class Convolution:
    # W: フィルター, b: バイアス
    def __init__(self, W, b, stride=1, pad=0):
        self.W = W
        self.b = b
        self.stride = stride
        self.pad = pad
        
        # 中間データ（backward時に使用）
        self.x = None   
        self.col = None
        self.col_W = None
        
        # フィルター・バイアスパラメータの勾配
        self.dW = None
        self.db = None

    def forward(self, x):
        # FN: filter_number, C: channel, FH: filter_height, FW: filter_width
        FN, C, FH, FW = self.W.shape
        N, C, H, W = x.shape
        # 出力値のheight, width
        out_h = 1 + int((H + 2 * self.pad - FH) / self.stride)
        out_w = 1 + int((W + 2 * self.pad - FW) / self.stride)
        
        # xを行列に変換
        col = im2col(x, FH, FW, self.stride, self.pad)
        # フィルターをxに合わせた行列に変換
        col_W = self.W.reshape(FN, -1).T

        out = np.dot(col, col_W) + self.b
        # 計算のために変えた形式を戻す
        out = out.reshape(N, out_h, out_w, -1).transpose(0, 3, 1, 2)

        self.x = x
        self.col = col
        self.col_W = col_W

        return out

    def backward(self, dout):
        FN, C, FH, FW = self.W.shape
        dout = dout.transpose(0, 2, 3, 1).reshape(-1, FN)

        self.db = np.sum(dout, axis=0)
        self.dW = np.dot(self.col.T, dout)
        self.dW = self.dW.transpose(1, 0).reshape(FN, C, FH, FW)

        dcol = np.dot(dout, self.col_W.T)
        # dcolを画像データに変換
        dx = col2im(dcol, self.x.shape, FH, FW, self.stride, self.pad)

        return dx

    
class Pooling:
    def __init__(self, pool_h, pool_w, stride=1, pad=0):
        self.pool_h = pool_h
        self.pool_w = pool_w
        self.stride = stride
        self.pad = pad
        
        self.x = None
        self.arg_max = None

    def forward(self, x):
        N, C, H, W = x.shape
        out_h = int(1 + (H - self.pool_h) / self.stride)
        out_w = int(1 + (W - self.pool_w) / self.stride)
        
        # xを行列に変換
        col = im2col(x, self.pool_h, self.pool_w, self.stride, self.pad)
        # プーリングのサイズに合わせてリサイズ
        col = col.reshape(-1, self.pool_h*self.pool_w)
        
        # 行ごとに最大値を求める
        arg_max = np.argmax(col, axis=1)
        out = np.max(col, axis=1)
        # 整形
        out = out.reshape(N, out_h, out_w, C).transpose(0, 3, 1, 2)

        self.x = x
        self.arg_max = arg_max

        return out

    def backward(self, dout):
        dout = dout.transpose(0, 2, 3, 1)
        
        pool_size = self.pool_h * self.pool_w
        dmax = np.zeros((dout.size, pool_size))
        dmax[np.arange(self.arg_max.size), self.arg_max.flatten()] = dout.flatten()
        dmax = dmax.reshape(dout.shape + (pool_size,)) 
        
        dcol = dmax.reshape(dmax.shape[0] * dmax.shape[1] * dmax.shape[2], -1)
        dx = col2im(dcol, self.x.shape, self.pool_h, self.pool_w, self.stride, self.pad)
        
        return dx

```


```python
# simple CNN の実装・学習例

class SimpleConvNet:
    # conv - relu - pool - affine - relu - affine - softmax
    def __init__(self, input_dim=(1, 28, 28), conv_param={'filter_num':30, 'filter_size':5, 'pad':0, 'stride':1},
                 hidden_size=100, output_size=10, weight_init_std=0.01):
        filter_num = conv_param['filter_num']        
        filter_size = conv_param['filter_size']
        filter_pad = conv_param['pad']
        filter_stride = conv_param['stride']
        input_size = input_dim[1]
        conv_output_size = (input_size - filter_size + 2 * filter_pad) / filter_stride + 1
        pool_output_size = int(filter_num * (conv_output_size / 2) * (conv_output_size / 2))

        # 重みの初期化
        self.params = {}
        self.params['W1'] = weight_init_std * np.random.randn(filter_num, input_dim[0], filter_size, filter_size)
        self.params['b1'] = np.zeros(filter_num)
        self.params['W2'] = weight_init_std * np.random.randn(pool_output_size, hidden_size)
        self.params['b2'] = np.zeros(hidden_size)
        self.params['W3'] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params['b3'] = np.zeros(output_size)

        # レイヤの生成
        self.layers = OrderedDict()
        self.layers['Conv1'] = layers.Convolution(self.params['W1'], self.params['b1'], conv_param['stride'], conv_param['pad'])
        self.layers['Relu1'] = layers.Relu()
        self.layers['Pool1'] = layers.Pooling(pool_h=2, pool_w=2, stride=2)
        self.layers['Affine1'] = layers.Affine(self.params['W2'], self.params['b2'])
        self.layers['Relu2'] = layers.Relu()
        self.layers['Affine2'] = layers.Affine(self.params['W3'], self.params['b3'])

        self.last_layer = layers.SoftmaxWithLoss()

    def predict(self, x):
        for key in self.layers.keys():
            x = self.layers[key].forward(x)
        return x
        
    def loss(self, x, d):
        y = self.predict(x)
        return self.last_layer.forward(y, d)

    def accuracy(self, x, d, batch_size=100):
        if d.ndim != 1 : d = np.argmax(d, axis=1)
        
        acc = 0.0
        
        for i in range(int(x.shape[0] / batch_size)):
            tx = x[i*batch_size:(i+1)*batch_size]
            td = d[i*batch_size:(i+1)*batch_size]
            y = self.predict(tx)
            y = np.argmax(y, axis=1)
            acc += np.sum(y == td) 
        
        return acc / x.shape[0]

    def gradient(self, x, d):
        # forward
        self.loss(x, d)
        
        # backward
        dout = 1
        dout = self.last_layer.backward(dout)
        layers = list(self.layers.values())
        
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        # 設定
        grad = {}
        grad['W1'], grad['b1'] = self.layers['Conv1'].dW, self.layers['Conv1'].db
        grad['W2'], grad['b2'] = self.layers['Affine1'].dW, self.layers['Affine1'].db
        grad['W3'], grad['b3'] = self.layers['Affine2'].dW, self.layers['Affine2'].db

        return grad

```


```python
from common import optimizer

# データの読み込み
(x_train, d_train), (x_test, d_test) = load_mnist(flatten=False)

print("データ読み込み完了")

# 処理に時間のかかる場合はデータを削減 
x_train, d_train = x_train[:5000], d_train[:5000]
x_test, d_test = x_test[:1000], d_test[:1000]


network = SimpleConvNet(input_dim=(1,28,28), conv_param = {'filter_num': 30, 'filter_size': 5, 'pad': 0, 'stride': 1},
                        hidden_size=100, output_size=10, weight_init_std=0.01)

optimizer = optimizer.Adam()

iters_num = 1000
train_size = x_train.shape[0]
batch_size = 100

train_loss_list = []
accuracies_train = []
accuracies_test = []

plot_interval=10



for i in range(iters_num):
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    d_batch = d_train[batch_mask]
    
    grad = network.gradient(x_batch, d_batch)
    optimizer.update(network.params, grad)

    loss = network.loss(x_batch, d_batch)
    train_loss_list.append(loss)

    if (i+1) % plot_interval == 0:
        accr_train = network.accuracy(x_train, d_train)
        accr_test = network.accuracy(x_test, d_test)
        accuracies_train.append(accr_train)
        accuracies_test.append(accr_test)
        
        print('Generation: ' + str(i+1) + '. 正答率(トレーニング) = ' + str(accr_train))
        print('                : ' + str(i+1) + '. 正答率(テスト) = ' + str(accr_test))               

lists = range(0, iters_num, plot_interval)
plt.plot(lists, accuracies_train, label="training set")
plt.plot(lists, accuracies_test,  label="test set")
plt.legend(loc="lower right")
plt.title("accuracy")
plt.xlabel("count")
plt.ylabel("accuracy")
plt.ylim(0, 1.0)
# グラフの表示
plt.show()
```
```
    データ読み込み完了
    Generation: 10. 正答率(トレーニング) = 0.5386
                    : 10. 正答率(テスト) = 0.514
    Generation: 20. 正答率(トレーニング) = 0.5666
                    : 20. 正答率(テスト) = 0.539
    Generation: 30. 正答率(トレーニング) = 0.7254
                    : 30. 正答率(テスト) = 0.713
    Generation: 40. 正答率(トレーニング) = 0.793
                    : 40. 正答率(テスト) = 0.775
    Generation: 50. 正答率(トレーニング) = 0.8136
                    : 50. 正答率(テスト) = 0.783
    Generation: 60. 正答率(トレーニング) = 0.8538

    (中略)

    Generation: 1000. 正答率(トレーニング) = 0.9962
                    : 1000. 正答率(テスト) = 0.96
```

![svg](/assets/2_6_simple_convolution_network_files/2_6_simple_convolution_network_11_1.svg)
    

