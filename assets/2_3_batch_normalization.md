```python
import sys
sys.path.append('.')
```


```python
## バッチ正則化 layer の定義
import numpy as np
from collections import OrderedDict
from common import layers
from data.mnist import load_mnist
import matplotlib.pyplot as plt
from multi_layer_net import MultiLayerNet
from common import optimizer

# バッチ正則化 layer
class BatchNormalization:
    '''
    gamma: スケール係数
    beta: オフセット
    momentum: 慣性
    running_mean: テスト時に使用する平均
    running_var: テスト時に使用する分散
    '''
    def __init__(self, gamma, beta, momentum=0.9, running_mean=None, running_var=None):
        self.gamma = gamma
        self.beta = beta
        self.momentum = momentum
        self.input_shape = None

        self.running_mean = running_mean
        self.running_var = running_var  
        
        # backward時に使用する中間データ
        self.batch_size = None
        self.xc = None
        self.std = None
        self.dgamma = None
        self.dbeta = None

    def forward(self, x, train_flg=True):
        if self.running_mean is None:
            N, D = x.shape
            self.running_mean = np.zeros(D)
            self.running_var = np.zeros(D)
        #### batch normalization (バッチ正規化に該当する箇所)　ここから-----------------                
        if train_flg:
            mu = x.mean(axis=0) # 平均
            xc = x - mu # xをセンタリング
            var = np.mean(xc**2, axis=0) # 分散
            std = np.sqrt(var + 10e-7) # スケーリング
            xn = xc / std
            
            self.batch_size = x.shape[0]
            self.xc = xc
            self.xn = xn
            self.std = std
            self.running_mean = self.momentum * self.running_mean + (1-self.momentum) * mu # 平均値の加重平均
            self.running_var = self.momentum * self.running_var + (1-self.momentum) * var #分散値の加重平均
        else:
            xc = x - self.running_mean
            xn = xc / ((np.sqrt(self.running_var + 10e-7)))
            
        out = self.gamma * xn + self.beta 
        
        return out

    def backward(self, dout):
        dbeta = dout.sum(axis=0)
        dgamma = np.sum(self.xn * dout, axis=0)
        dxn = self.gamma * dout
        dxc = dxn / self.std
        dstd = -np.sum((dxn * self.xc) / (self.std * self.std), axis=0)
        dvar = 0.5 * dstd / self.std
        dxc += (2.0 / self.batch_size) * self.xc * dvar
        dmu = np.sum(dxc, axis=0)
        dx = dxc - dmu / self.batch_size
        
        self.dgamma = dgamma
        self.dbeta = dbeta

        return dx    

```


```python
def learn_and_plot(network, iters_num=1000):
    (x_train, d_train), (x_test, d_test) = load_mnist(normalize=True)

    #iters_num = 1000
    train_size = x_train.shape[0]
    batch_size = 100
    learning_rate=0.01

    train_loss_list = []
    accuracies_train = []
    accuracies_test = []

    plot_interval=10


    for i in range(iters_num):
        batch_mask = np.random.choice(train_size, batch_size)
        x_batch = x_train[batch_mask]
        d_batch = d_train[batch_mask]

        grad = network.gradient(x_batch, d_batch)
        for key in ('W1', 'W2', 'W3', 'b1', 'b2', 'b3'):
            network.params[key] -= learning_rate * grad[key]

            loss = network.loss(x_batch, d_batch)
            train_loss_list.append(loss)        
            
        if (i + 1) % plot_interval == 0:
            accr_test = network.accuracy(x_test, d_test)
            accuracies_test.append(accr_test)        
            accr_train = network.accuracy(x_batch, d_batch)
            accuracies_train.append(accr_train)
            
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


```python
## 元々あった状態
learn_and_plot(MultiLayerNet(input_size=784, hidden_size_list=[40, 20], output_size=10,
                        activation='sigmoid', weight_init_std='Xavier', use_batchnorm=True))



```

    データ読み込み完了



    
![svg](/assets/2_3_batch_normalization_files/2_3_batch_normalization_3_1.svg)
    



```python
## gauss 初期化にしたり、層の数を増やしたりして勾配消失問題が起きやすい状況をつくってみる
print("gauss 初期化で収束しにくくする")
learn_and_plot(MultiLayerNet(input_size=784, hidden_size_list=[40, 20], output_size=10,
                        activation='sigmoid', weight_init_std=0.01, use_batchnorm=True))
print("さらに層を増やして勾配消失問題も起きやすくする")
learn_and_plot(MultiLayerNet(input_size=784, hidden_size_list=[40, 30 ,20], output_size=10,
                        activation='sigmoid', weight_init_std=0.01, use_batchnorm=True))
```

    gauss 初期化で収束しにくくする



    
![svg](/assets/2_3_batch_normalization_files/2_3_batch_normalization_4_1.svg)
    


    さらに層を増やして勾配消失問題も起きやすくする



    
![svg](/assets/2_3_batch_normalization_files/2_3_batch_normalization_4_3.svg)
    



```python
print("He,relu,bn")
learn_and_plot(MultiLayerNet(input_size=784, hidden_size_list=[40, 30 ,20], output_size=10,
                        activation='relu', weight_init_std='He', use_batchnorm=True))

print("xavier,sigmoid,bn")
learn_and_plot(MultiLayerNet(input_size=784, hidden_size_list=[40, 30 ,20], output_size=10,
                        activation='sigmoid', weight_init_std='Xavier', use_batchnorm=True))
```

    He,relu,bn



    
![svg](/assets/2_3_batch_normalization_files/2_3_batch_normalization_5_1.svg)
    


    xavier,sigmoid,bn



    
![svg](/assets/2_3_batch_normalization_files/2_3_batch_normalization_5_3.svg)
    

