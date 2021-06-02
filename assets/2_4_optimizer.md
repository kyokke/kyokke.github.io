```python
import sys
sys.path.append('.')

import sys, os
import numpy as np
from collections import OrderedDict
from common import layers
from data.mnist import load_mnist
import matplotlib.pyplot as plt
from multi_layer_net import MultiLayerNet



```


```python
def learn_and_plot(network, optimizer="sgd", iters_num=1000, learning_rate=0.01):
    # データの読み込み
    (x_train, d_train), (x_test, d_test) = load_mnist(normalize=True, one_hot_label=True)


    train_size = x_train.shape[0]
    batch_size = 100
    # momentum 
    momentum = 0.9
    # adagrad
    theta = 1e-4
    # rmsprop
    decay_rate = 0.99
    # adam
    beta1 = 0.9
    beta2 = 0.999

    train_loss_list = []
    accuracies_train = []
    accuracies_test = []

    plot_interval=10

    for i in range(iters_num):
        batch_mask = np.random.choice(train_size, batch_size)
        x_batch = x_train[batch_mask]
        d_batch = d_train[batch_mask]

        # 勾配
        grad = network.gradient(x_batch, d_batch)
        
        if i == 0:
            if optimizer == "momentum":
                v = {} 
            elif optimizer == "adagrad" or optimizer == "rmsprop" :
                h = {}
            elif optimizer == "adam":
                m = {}
                v = {} 
        

        for key in ('W1', 'W2', 'W3', 'b1', 'b2', 'b3'):
            if optimizer == "sgd":
                network.params[key] -= learning_rate * grad[key]
            elif optimizer == "momentum":
                if i == 0:
                    v[key] = np.zeros_like(network.params[key]) 
                v[key] = momentum * v[key] - learning_rate * grad[key]
                network.params[key] += v[key]
            elif optimizer == "adagrad":
                if i == 0:
                    h[key] = np.full_like(network.params[key],theta)
                h[key] = h[key] + np.square(grad[key]) 
                network.params[key] -= learning_rate / np.sqrt(h[key]) * grad[key]
            elif optimizer == "rmsprop":
                if i == 0:
                    h[key] = np.zeros_like(network.params[key])
                h[key] *= decay_rate
                h[key] += (1-decay_rate) * np.square(grad[key]) 
                network.params[key] -= learning_rate * grad[key] / (np.sqrt(h[key]) + 1e-7) 
            elif optimizer == "adam":
                if i == 0:
                    m[key] = np.zeros_like(network.params[key])
                    v[key] = np.zeros_like(network.params[key])

                m[key] += (1-beta1) * (grad[key] - m[key])
                v[key] += (1-beta2) * (grad[key]**2 - v[key])

                network.params[key] -= learning_rate * m[key] / (np.sqrt(v[key]) + 1e-7) 

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
# default 状態
iters_num = 1000
activation = "sigmoid"
initiazlier = 0.01
use_batchnorm = False

optimizer_list = ["sgd", "momentum", "adagrad", "adagrad", "rmsprop", "adam"]

for optimizer in optimizer_list:
    print(optimizer)
    network = MultiLayerNet(input_size=784, hidden_size_list=[40, 20], output_size=10, activation=activation, weight_init_std=initiazlier,
                        use_batchnorm=use_batchnorm)
    learn_and_plot(network,optimizer, iters_num)

```

    sgd



    
![svg](/assets/2_4_optimizer_files/2_4_optimizer_2_1.svg)
    


    momentum



    
![svg](/assets/2_4_optimizer_files/2_4_optimizer_2_3.svg)
    


    adagrad



    
![svg](/assets/2_4_optimizer_files/2_4_optimizer_2_5.svg)
    


    rmsprop



    
![svg](/assets/2_4_optimizer_files/2_4_optimizer_2_7.svg)
    


    adam



    
![svg](/assets/2_4_optimizer_files/2_4_optimizer_2_9.svg)
    



```python
# 学習率を増やしてみる 0.01 -> 0.1 
iters_num = 1000
activation = "sigmoid"
learning_rate = 0.1
initiazlier = 0.01
use_batchnorm = False

optimizer_list = ["sgd", "momentum", "adagrad", "adagrad", "rmsprop", "adam"]

for optimizer in optimizer_list:
    print(optimizer)
    network = MultiLayerNet(input_size=784, hidden_size_list=[40, 20], output_size=10, activation=activation, weight_init_std=initiazlier,
                        use_batchnorm=use_batchnorm)
    learn_and_plot(network,optimizer, iters_num,learning_rate)

```

    sgd



    
![svg](/assets/2_4_optimizer_files/2_4_optimizer_3_1.svg)
    


    momentum



    
![svg](/assets/2_4_optimizer_files/2_4_optimizer_3_3.svg)
    


    adagrad



    
![svg](/assets/2_4_optimizer_files/2_4_optimizer_3_5.svg)
    


    rmsprop



    
![svg](/assets/2_4_optimizer_files/2_4_optimizer_3_7.svg)
    


    adam



    
![svg](/assets/2_4_optimizer_files/2_4_optimizer_3_9.svg)
    



```python
# デフォルト状態から 活性化関数を
iters_num = 1000
activation = "relu"
initiazlier = 0.01
use_batchnorm = False

optimizer_list = ["sgd", "momentum", "adagrad", "adagrad", "rmsprop", "adam"]

for optimizer in optimizer_list:
    print(optimizer)
    network = MultiLayerNet(input_size=784, hidden_size_list=[40, 20], output_size=10, activation=activation, weight_init_std=initiazlier,
                        use_batchnorm=use_batchnorm)
    learn_and_plot(network,optimizer, iters_num)

```

    sgd



    
![svg](/assets/2_4_optimizer_files/2_4_optimizer_4_1.svg)
    


    momentum



    
![svg](/assets/2_4_optimizer_files/2_4_optimizer_4_3.svg)
    


    adagrad



    
![svg](/assets/2_4_optimizer_files/2_4_optimizer_4_5.svg)
    


    adagrad



    
![svg](/assets/2_4_optimizer_files/2_4_optimizer_4_7.svg)
    


    rmsprop



    
![svg](/assets/2_4_optimizer_files/2_4_optimizer_4_9.svg)
    


    adam



    
![svg](/assets/2_4_optimizer_files/2_4_optimizer_4_11.svg)
    



```python
# さらに初期化を He へ
iters_num = 1000
activation = "relu"
initiazlier = "He"
use_batchnorm = False

optimizer_list = ["sgd", "momentum", "adagrad", "adagrad", "rmsprop", "adam"]

for optimizer in optimizer_list:
    print(optimizer)
    network = MultiLayerNet(input_size=784, hidden_size_list=[40, 20], output_size=10, activation=activation, weight_init_std=initiazlier,
                        use_batchnorm=use_batchnorm)
    learn_and_plot(network,optimizer, iters_num)

```

    sgd



    
![svg](/assets/2_4_optimizer_files/2_4_optimizer_5_1.svg)
    


    momentum



    
![svg](/assets/2_4_optimizer_files/2_4_optimizer_5_3.svg)
    


    adagrad



    
![svg](/assets/2_4_optimizer_files/2_4_optimizer_5_5.svg)
    


    adagrad



    
![svg](/assets/2_4_optimizer_files/2_4_optimizer_5_7.svg)
    


    rmsprop



    
![svg](/assets/2_4_optimizer_files/2_4_optimizer_5_9.svg)
    


    adam



    
![svg](/assets/2_4_optimizer_files/2_4_optimizer_5_11.svg)
    



```python
# すでにひとつ前の状態でどの手法でもある程度収束しているので、
# BNいれても変化が見つけづらそう? -> 活性化関数・初期化手法を戻して BN入れて見る
iters_num = 1000
activation = "sigmoid"
initiazlier = 0.01
use_batchnorm = True

optimizer_list = ["sgd", "momentum", "adagrad", "adagrad", "rmsprop", "adam"]

for optimizer in optimizer_list:
    print(optimizer)
    network = MultiLayerNet(input_size=784, hidden_size_list=[40, 20], output_size=10, activation=activation, weight_init_std=initiazlier,
                        use_batchnorm=use_batchnorm)
    learn_and_plot(network,optimizer, iters_num)

```

    sgd



    
![svg](/assets/2_4_optimizer_files/2_4_optimizer_6_1.svg)
    


    momentum



    
![svg](/assets/2_4_optimizer_files/2_4_optimizer_6_3.svg)
    


    adagrad



    
![svg](/assets/2_4_optimizer_files/2_4_optimizer_6_5.svg)
    


    adagrad



    
![svg](/assets/2_4_optimizer_files/2_4_optimizer_6_7.svg)
    


    rmsprop



    
![svg](/assets/2_4_optimizer_files/2_4_optimizer_6_9.svg)
    


    adam



    
![svg](/assets/2_4_optimizer_files/2_4_optimizer_6_11.svg)
    



```python

```
