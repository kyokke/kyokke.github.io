```python
import sys
sys.path.append('.')

import numpy as np
from collections import OrderedDict
from common import layers
from data.mnist import load_mnist
import matplotlib.pyplot as plt
from multi_layer_net import MultiLayerNet
from common import optimizer

```

パラメータを変えてなんども試すので、関数化しておく。
 - regulation 引数で、正則化 なし、L1,L2 を切り替える
 - 正則化ありのパラメータ更新は勉強のために、オリジナルのコードと同じくベタで書いている (OptimizerはSDG前提のパラメータ更新)
 - network 生成時の 引数を変えることで、dropout true/false, ratio 他も変更可能


```python
def train(network, optimizer_obj, learning_rate, weight_decay_lambda = 0.0, regulation=None,  iters_num = 1000 ):
    (x_train, d_train), (x_test, d_test) = load_mnist(normalize=True)
    # print("データ読み込み完了")

    # 過学習を再現するために、学習データを削減
    x_train = x_train[:300]
    d_train = d_train[:300]

    #iters_num = 1000
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
        loss = 0
        if regulation == None:
            optimizer_obj.update(network.params, grad)
        elif regulation == 'L2':
            weight_decay = 0
            for idx in range(1, network.hidden_layer_num+1):
                grad['W' + str(idx)] = network.layers['Affine' + str(idx)].dW + weight_decay_lambda * network.params['W' + str(idx)]
                grad['b' + str(idx)] = network.layers['Affine' + str(idx)].db
                network.params['W' + str(idx)] -= learning_rate * grad['W' + str(idx)]
                network.params['b' + str(idx)] -= learning_rate * grad['b' + str(idx)]        
                # weight_decay += 0.5 * weight_decay_lambda * np.sqrt(np.sum(network.params['W' + str(idx)] ** 2)) # こうかかれていたが..
                weight_decay += 0.5 * weight_decay_lambda * np.sum(network.params['W' + str(idx)] ** 2) # こっちじゃないか?
            loss = weight_decay 
        elif regulation == 'L1':
            weight_decay = 0
            for idx in range(1, network.hidden_layer_num+1):
                grad['W' + str(idx)] = network.layers['Affine' + str(idx)].dW + weight_decay_lambda * np.sign(network.params['W' + str(idx)])
                grad['b' + str(idx)] = network.layers['Affine' + str(idx)].db
                network.params['W' + str(idx)] -= learning_rate * grad['W' + str(idx)]
                network.params['b' + str(idx)] -= learning_rate * grad['b' + str(idx)]        
                weight_decay += weight_decay_lambda * np.sum(np.abs(network.params['W' + str(idx)]))

            loss = weight_decay            
        
        loss = loss + network.loss(x_batch, d_batch)
        train_loss_list.append(loss)
            
        if (i+1) % plot_interval == 0:
            accr_train = network.accuracy(x_train, d_train)
            accr_test = network.accuracy(x_test, d_test)
            accuracies_train.append(accr_train)
            accuracies_test.append(accr_test)

            # print('Generation: ' + str(i+1) + '. 正答率(トレーニング) = ' + str(accr_train))
            # print('                : ' + str(i+1) + '. 正答率(テスト) = ' + str(accr_test))        

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

基準となる 正則化無し, Dropout無し, SDGの場合は訓練データの正解率100%に対して 検証データの正解率75%程度。かなり結果に差があり過学習が起きていると思われる


```python

# 正則化無し, Dropout 無し
learning_rate = 0.01
network = MultiLayerNet(input_size=784, hidden_size_list=[100, 100, 100, 100, 100, 100], output_size=10)
optimizer_obj = optimizer.SGD(learning_rate=learning_rate)

train(network,optimizer_obj,learning_rate)
```


    
![svg](/assets/2_5_overfiting_files/2_5_overfiting_4_0.svg)
    


L2正則化の重みを変えてみると、0.01では小さすぎ(正則化無しとほぼ変わらない)、1.0 は大きすぎ。
その間で探すと、そこそこの結果っぽくはなるが、これでいいのかはちょっとわからない。



```python
# L2 

lambda_list = [0.01, 0.08, 1.0]
learning_rate = 0.01
regulation = 'L2'
iters_num = 1000

for weight_decay_lambda in lambda_list:
    print(weight_decay_lambda)
    network = MultiLayerNet(input_size=784, hidden_size_list=[100, 100, 100, 100, 100, 100], output_size=10)
    optimizer_obj = optimizer.SGD(learning_rate=learning_rate)
    train(network,optimizer_obj,learning_rate,weight_decay_lambda, regulation,iters_num=iters_num)


```

    0.01



    
![svg](/assets/2_5_overfiting_files/2_5_overfiting_6_1.svg)
    


    0.08



    
![svg](/assets/2_5_overfiting_files/2_5_overfiting_6_3.svg)
    


    1.0



    
![svg](/assets/2_5_overfiting_files/2_5_overfiting_6_5.svg)
    


L1正則化。与えられたコードでは、学習率が0.1 だったが、L2正則化と合わせて 0.01にした。
元のloss関数も、学習に使うデータも変わらないのだし、正則化項とのバランスは weight_decay_lambda 使えばいいので、
そちらのほうが、振舞いなども比較しやすいだろう。

L2正則化と似たような結果が得られる。講義動画でグラフがバタついてたのは L1正則化を使っていたこと「だけ」が原因でなく、その時の学習率が大きすぎて必要以上に0重みが増えたり減ったりしていたからでは? 





```python
# L1
lambda_list = [0.0005, 0.008, 0.05]
learning_rate = 0.01
regulation = 'L1'
iters_num = 1000

for weight_decay_lambda in lambda_list:
    print(weight_decay_lambda)
    network = MultiLayerNet(input_size=784, hidden_size_list=[100, 100, 100, 100, 100, 100], output_size=10)
    optimizer_obj = optimizer.SGD(learning_rate=learning_rate)
    train(network, optimizer_obj, learning_rate, weight_decay_lambda, regulation, iters_num)


```

    0.0005



    
![svg](/assets/2_5_overfiting_files/2_5_overfiting_8_1.svg)
    


    0.008



    
![svg](/assets/2_5_overfiting_files/2_5_overfiting_8_3.svg)
    


    0.05



    
![svg](/assets/2_5_overfiting_files/2_5_overfiting_8_5.svg)
    



```python
class Dropout:
    def __init__(self, dropout_ratio=0.5):
        self.dropout_ratio = dropout_ratio
        self.mask = None

    def forward(self, x, train_flg=True):
        if train_flg:
            self.mask = np.random.rand(*x.shape) > self.dropout_ratio
            return x * self.mask
        else:
            return x * (1.0 - self.dropout_ratio) # ランダム要素をいれない分、学習時に有効なノードの割合をかけて辻褄合わせる?

    def backward(self, dout):
        return dout * self.mask
```

Dropout. SGD で、正則化無。

講義動画では、「うまくいっている」風なことが言われていたが、イタレーションを増やすと、ただ収束がゆっくりになっただけで、訓練データは正解率100% に到達する。
最終的には過学習してしまっているのではないか？という気もするが、たとえば、dropout ratio 0.2 のケースで、early stopping を前提にすると、検証データの正解率がサチリかけたときに、訓練データの正解率がまだ、100%に達しておらず、ワンチャン絶妙なモデルが得られるかもしれない?


```python
# common settings 
use_dropout = True
weight_decay_lambda = 0
regulation = None
learning_rate = 0.01
dropout_ratio_list = [0.15, 0.2, 0.3, 0.4]
iters_num_list    = [3000, 4000, 5000, 6000]

for i in range(len(dropout_ratio_list)):
    dropout_ratio = dropout_ratio_list[i]
    print(dropout_ratio)
    iters_num = iters_num_list[i]
    optimizer_obj = optimizer.SGD(learning_rate=learning_rate)
    network = MultiLayerNet(input_size=784, hidden_size_list=[100, 100, 100, 100, 100, 100], output_size=10,
                            use_dropout = use_dropout, dropout_ratio = dropout_ratio)
    train(network, optimizer_obj, learning_rate, weight_decay_lambda, regulation,iters_num)

```

    0.15



    
![svg](/assets/2_5_overfiting_files/2_5_overfiting_11_1.svg)
    


    0.2



    
![svg](/assets/2_5_overfiting_files/2_5_overfiting_11_3.svg)
    


    0.3



    
![svg](/assets/2_5_overfiting_files/2_5_overfiting_11_5.svg)
    


    0.4



    
![svg](/assets/2_5_overfiting_files/2_5_overfiting_11_7.svg)
    


dropout ratio 0, 0.15, 0.2 で各種 optimizer を試してみる。正則化はとりあえずなし
そもそもdroptout 無しの状態で、SGDと比べて、収束早い & 検証データの成果率が若干高い。
Droptout有りのときもあまり収束がSlowDownしないし、訓練データはすぐに正解率100%になる。これをどう捉えるべきか?



```python
# common settings 
use_dropout = True
weight_decay_lambda = 0
regulation = None
learning_rate = 0.01
dropout_ratio_list = [0, 0.15, 0.2]
iters_num_list    = [1000, 3000, 4000]

# -------------------------------------------------------
print("Momentum+Droptout")
for i in range(len(dropout_ratio_list)):
    dropout_ratio = dropout_ratio_list[i]
    print(dropout_ratio)
    iters_num = iters_num_list[i]
    optimizer_obj = optimizer.Momentum(learning_rate=learning_rate,momentum=0.9)
    network = MultiLayerNet(input_size=784, hidden_size_list=[100, 100, 100, 100, 100, 100], output_size=10,
                            use_dropout = use_dropout, dropout_ratio = dropout_ratio)
    train(network, optimizer_obj, learning_rate, weight_decay_lambda, regulation,iters_num)
# -------------------------------------------------------
print("AdaGrad+Droptout")
for i in range(len(dropout_ratio_list)):
    dropout_ratio = dropout_ratio_list[i]
    print(dropout_ratio)
    iters_num = iters_num_list[i]
    optimizer_obj = optimizer.AdaGrad(learning_rate=learning_rate)
    network = MultiLayerNet(input_size=784, hidden_size_list=[100, 100, 100, 100, 100, 100], output_size=10,
                            use_dropout = use_dropout, dropout_ratio = dropout_ratio)
    train(network, optimizer_obj, learning_rate, weight_decay_lambda, regulation,iters_num)
# -------------------------------------------------------
print("Adam+Droptout")
for i in range(len(dropout_ratio_list)):
    dropout_ratio = dropout_ratio_list[i]
    print(dropout_ratio)
    iters_num = iters_num_list[i]
    optimizer_obj = optimizer.Adam()
    network = MultiLayerNet(input_size=784, hidden_size_list=[100, 100, 100, 100, 100, 100], output_size=10,
                            use_dropout = use_dropout, dropout_ratio = dropout_ratio)
    train(network, optimizer_obj, learning_rate, weight_decay_lambda, regulation,iters_num)

```

    Momentum+Droptout
    0



    
![svg](/assets/2_5_overfiting_files/2_5_overfiting_13_1.svg)
    


    0.15



    
![svg](/assets/2_5_overfiting_files/2_5_overfiting_13_3.svg)
    


    0.2



    
![svg](/assets/2_5_overfiting_files/2_5_overfiting_13_5.svg)
    


    AdaGrad+Droptout
    0



    
![svg](/assets/2_5_overfiting_files/2_5_overfiting_13_7.svg)
    


    0.15



    
![svg](/assets/2_5_overfiting_files/2_5_overfiting_13_9.svg)
    


    0.2



    
![svg](/assets/2_5_overfiting_files/2_5_overfiting_13_11.svg)
    


    Adam+Droptout
    0



    
![svg](/assets/2_5_overfiting_files/2_5_overfiting_13_13.svg)
    


    0.15



    
![svg](/assets/2_5_overfiting_files/2_5_overfiting_13_15.svg)
    


    0.2



    
![svg](/assets/2_5_overfiting_files/2_5_overfiting_13_17.svg)
    


Dropout (ratio 0.1) + L2 正則化。
最初に実装したtrain関数は、正則化を行う時のパラメータ更新のコードがSGD前提になってしまっている。
各種Optimizer 全部ためすのは面倒だなぁと思っていたが、multi_layer_net.py のコードを見ると、
実はnetwork生成時に、weight decay lambda の数値を与えると、コスト関数に 重み係数の二乗和が追加されるので、L2正則化っぽいことが試せる. 

(ちなみに、デフォルトのコードでは、Droptouのコード例で、このweight_decay_lambda が network 生成時に与えられているので、何も考えずに頭から実行した場合は、実は、L1正則化単体の実験で使われていた、weight_decay_lambda = 0.005 が使われることになるのだが、コードを準備した方の意図通りだろうか?)





```python
# common settings 
use_dropout = True
weight_decay_lambda = 0.05
regulation = None
learning_rate = 0.01
dropout_ratio = 0.08
iters_num    = 2000

#------------------------------------------------------------
print("SGD+Droptout+L2")
optimizer_obj = optimizer.SGD(learning_rate=learning_rate)
network = MultiLayerNet(input_size=784, hidden_size_list=[100, 100, 100, 100, 100, 100], output_size=10,
                        weight_decay_lambda=weight_decay_lambda,use_dropout = use_dropout, dropout_ratio = dropout_ratio)
train(network, optimizer_obj, learning_rate, weight_decay_lambda, regulation,iters_num)

# -------------------------------------------------------
print("Momentum+Droptout+L2")
optimizer_obj = optimizer.Momentum(learning_rate=learning_rate, momentum=0.9)
network = MultiLayerNet(input_size=784, hidden_size_list=[100, 100, 100, 100, 100, 100], output_size=10,
                        weight_decay_lambda=weight_decay_lambda,  use_dropout = use_dropout, dropout_ratio = dropout_ratio)
train(network, optimizer_obj, learning_rate, weight_decay_lambda, regulation,iters_num)

# -------------------------------------------------------
print("AdaGrad+Droptout+L2")
optimizer_obj = optimizer.AdaGrad(learning_rate=learning_rate)
network = MultiLayerNet(input_size=784, hidden_size_list=[100, 100, 100, 100, 100, 100], output_size=10,
                        weight_decay_lambda=weight_decay_lambda, use_dropout = use_dropout, dropout_ratio = dropout_ratio)
train(network, optimizer_obj, learning_rate, weight_decay_lambda, regulation,iters_num)
# -------------------------------------------------------
print("Adam+Droptout+L2")
optimizer_obj = optimizer.Adam()
network = MultiLayerNet(input_size=784, hidden_size_list=[100, 100, 100, 100, 100, 100], output_size=10,
                        weight_decay_lambda=weight_decay_lambda, use_dropout = use_dropout, dropout_ratio = dropout_ratio)
train(network, optimizer_obj, learning_rate, weight_decay_lambda, regulation,iters_num)
```

    SGD+Droptout+L2



    
![svg](/assets/2_5_overfiting_files/2_5_overfiting_15_1.svg)
    


    Momentum+Droptout+L2



    
![svg](/assets/2_5_overfiting_files/2_5_overfiting_15_3.svg)
    


    AdaGrad+Droptout+L2



    
![svg](/assets/2_5_overfiting_files/2_5_overfiting_15_5.svg)
    


    Adam+Droptout+L2



    
![svg](/assets/2_5_overfiting_files/2_5_overfiting_15_7.svg)
    


SGDで Dropout + L1 正則化。反復回数だけ増やして実行。
1000回まではよさげにみえたが、その後は、訓練データは微増に対して、検証データは微減。このケースの場合は 1000回ちょっと手前くらいで止めておこうということになるのかな?



```python
from common import optimizer
(x_train, d_train), (x_test, d_test) = load_mnist(normalize=True)

print("データ読み込み完了")

# 過学習を再現するために、学習データを削減
x_train = x_train[:300]
d_train = d_train[:300]

# ドロップアウト設定 ======================================
use_dropout = True
dropout_ratio = 0.08
# ====================================================

network = MultiLayerNet(input_size=784, hidden_size_list=[100, 100, 100, 100, 100, 100], output_size=10,
                        use_dropout = use_dropout, dropout_ratio = dropout_ratio)

iters_num = 3000
train_size = x_train.shape[0]
batch_size = 100
learning_rate=0.01

train_loss_list = []
accuracies_train = []
accuracies_test = []
hidden_layer_num = network.hidden_layer_num

plot_interval=10

# 正則化強度設定 ======================================
weight_decay_lambda=0.004
# =================================================

for i in range(iters_num):
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    d_batch = d_train[batch_mask]

    grad = network.gradient(x_batch, d_batch)
    weight_decay = 0
    
    for idx in range(1, hidden_layer_num+1):
        grad['W' + str(idx)] = network.layers['Affine' + str(idx)].dW + weight_decay_lambda * np.sign(network.params['W' + str(idx)])
        grad['b' + str(idx)] = network.layers['Affine' + str(idx)].db
        network.params['W' + str(idx)] -= learning_rate * grad['W' + str(idx)]
        network.params['b' + str(idx)] -= learning_rate * grad['b' + str(idx)]        
        weight_decay += weight_decay_lambda * np.sum(np.abs(network.params['W' + str(idx)]))

    loss = network.loss(x_batch, d_batch) + weight_decay
    train_loss_list.append(loss)        
        
    if (i+1) % plot_interval == 0:
        accr_train = network.accuracy(x_train, d_train)
        accr_test = network.accuracy(x_test, d_test)
        accuracies_train.append(accr_train)
        accuracies_test.append(accr_test)
        
        # print('Generation: ' + str(i+1) + '. 正答率(トレーニング) = ' + str(accr_train))
        # print('                : ' + str(i+1) + '. 正答率(テスト) = ' + str(accr_test))               
        
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

    データ読み込み完了



    
![svg](/assets/2_5_overfiting_files/2_5_overfiting_17_1.svg)
    



```python

```
