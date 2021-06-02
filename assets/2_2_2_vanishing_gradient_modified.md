```python
import sys
sys.path.append('/.')
```


```python
# MLP クラス
import numpy as np
from common import layers
from collections import OrderedDict
from common import functions
from data.mnist import load_mnist
import matplotlib.pyplot as plt


class MultiLayerNet:
    '''
    input_size: 入力層のノード数
    hidden_size_list: 隠れ層のノード数のリスト
    output_size: 出力層のノード数
    activation: 活性化関数
    weight_init_std: 重みの初期化方法
    '''
    def __init__(self, input_size, hidden_size_list, output_size, activation='relu', weight_init_std='relu'):
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size_list = hidden_size_list
        self.hidden_layer_num = len(hidden_size_list)
        self.params = {}

        # 重みの初期化
        self.__init_weight(weight_init_std)

        # レイヤの生成, sigmoidとreluのみ扱う
        activation_layer = {'sigmoid': layers.Sigmoid, 'relu': layers.Relu}
        self.layers = OrderedDict() # 追加した順番に格納
        for idx in range(1, self.hidden_layer_num+1):
            self.layers['Affine' + str(idx)] = layers.Affine(self.params['W' + str(idx)], self.params['b' + str(idx)])
            self.layers['Activation_function' + str(idx)] = activation_layer[activation]()

        idx = self.hidden_layer_num + 1
        self.layers['Affine' + str(idx)] = layers.Affine(self.params['W' + str(idx)], self.params['b' + str(idx)])

        self.last_layer = layers.SoftmaxWithLoss()

    def __init_weight(self, weight_init_std):
        all_size_list = [self.input_size] + self.hidden_size_list + [self.output_size]
        for idx in range(1, len(all_size_list)):
            scale = weight_init_std
            if str(weight_init_std).lower() in ('relu', 'he'):
                scale = np.sqrt(2.0 / all_size_list[idx - 1])
            elif str(weight_init_std).lower() in ('sigmoid', 'xavier'):
                scale = np.sqrt(1.0 / all_size_list[idx - 1])

            self.params['W' + str(idx)] = scale * np.random.randn(all_size_list[idx-1], all_size_list[idx])
            self.params['b' + str(idx)] = np.zeros(all_size_list[idx])

    def predict(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)

        return x

    def loss(self, x, d):
        y = self.predict(x)

        weight_decay = 0
        for idx in range(1, self.hidden_layer_num + 2):
            W = self.params['W' + str(idx)]

        return self.last_layer.forward(y, d) + weight_decay

    def accuracy(self, x, d):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        if d.ndim != 1 : d = np.argmax(d, axis=1)

        accuracy = np.sum(y == d) / float(x.shape[0])
        return accuracy

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
        for idx in range(1, self.hidden_layer_num+2):
            grad['W' + str(idx)] = self.layers['Affine' + str(idx)].dW
            grad['b' + str(idx)] = self.layers['Affine' + str(idx)].db

        return grad

```


```python
def learn_and_plot(network, iters_num=2000):

    # データの読み込み
    (x_train, d_train), (x_test, d_test) = load_mnist(normalize=True, one_hot_label=True)

    #iters_num = 2000
    train_size = x_train.shape[0]
    batch_size = 100
    learning_rate = 0.1

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

# 活性化関数 と 初期化関数の組み合わせ
hidden_size_list= [40,20]

print("----------------- Sigmoid")
print("initializer: gauss")
learn_and_plot(MultiLayerNet(input_size=784, hidden_size_list=hidden_size_list, output_size=10, activation='sigmoid', weight_init_std=0.01))

print("initializer: Xavier")
learn_and_plot(MultiLayerNet(input_size=784, hidden_size_list=hidden_size_list, output_size=10, activation='sigmoid', weight_init_std='Xavier'))

print("initializer: He") # try
learn_and_plot(MultiLayerNet(input_size=784, hidden_size_list=hidden_size_list, output_size=10, activation='sigmoid', weight_init_std='He'))

print("----------------- Relu")

print("initializer: gauss")
learn_and_plot(MultiLayerNet(input_size=784, hidden_size_list=hidden_size_list, output_size=10, activation='relu', weight_init_std=0.01))

print("initializer: Xavier") # try 
learn_and_plot(MultiLayerNet(input_size=784, hidden_size_list=hidden_size_list, output_size=10, activation='relu', weight_init_std='Xavier'))

print("initializer: He")
learn_and_plot(MultiLayerNet(input_size=784, hidden_size_list=hidden_size_list, output_size=10, activation='relu', weight_init_std='He'))


```

    ----------------- Sigmoid
    initializer: gauss



    
![svg](/assets/2_2_2_vanishing_gradient_modified_files/2_2_2_vanishing_gradient_modified_3_1.svg)
    


    initializer: Xavier



    
![svg](/assets/2_2_2_vanishing_gradient_modified_files/2_2_2_vanishing_gradient_modified_3_3.svg)
    


    initializer: He



    
![svg](/assets/2_2_2_vanishing_gradient_modified_files/2_2_2_vanishing_gradient_modified_3_5.svg)
    


    ----------------- Relu
    initializer: gauss



    
![svg](/assets/2_2_2_vanishing_gradient_modified_files/2_2_2_vanishing_gradient_modified_3_7.svg)
    


    initializer: Xavier



    
![svg](/assets/2_2_2_vanishing_gradient_modified_files/2_2_2_vanishing_gradient_modified_3_9.svg)
    


    initializer: He



    
![svg](/assets/2_2_2_vanishing_gradient_modified_files/2_2_2_vanishing_gradient_modified_3_11.svg)
    



```python

# 活性化関数 と 初期化関数の組み合わせ
hidden_size_list= [40,30,20]

print("----------------- Sigmoid")
print("initializer: gauss")
learn_and_plot(MultiLayerNet(input_size=784, hidden_size_list=hidden_size_list, output_size=10, activation='sigmoid', weight_init_std=0.01),iters_num=4000)

print("initializer: Xavier")
learn_and_plot(MultiLayerNet(input_size=784, hidden_size_list=hidden_size_list, output_size=10, activation='sigmoid', weight_init_std='Xavier'),iters_num=4000)

print("initializer: He") # try
learn_and_plot(MultiLayerNet(input_size=784, hidden_size_list=hidden_size_list, output_size=10, activation='sigmoid', weight_init_std='He'),iters_num=4000)

print("----------------- Relu")

print("initializer: gauss")
learn_and_plot(MultiLayerNet(input_size=784, hidden_size_list=hidden_size_list, output_size=10, activation='relu', weight_init_std=0.01),iters_num=4000)

print("initializer: Xavier") # try 
learn_and_plot(MultiLayerNet(input_size=784, hidden_size_list=hidden_size_list, output_size=10, activation='relu', weight_init_std='Xavier'),iters_num=4000)

print("initializer: He")
learn_and_plot(MultiLayerNet(input_size=784, hidden_size_list=hidden_size_list, output_size=10, activation='relu', weight_init_std='He'),iters_num=4000)


```

    ----------------- Sigmoid
    initializer: gauss



    
![svg](/assets/2_2_2_vanishing_gradient_modified_files/2_2_2_vanishing_gradient_modified_4_1.svg)
    


    initializer: Xavier



    
![svg](/assets/2_2_2_vanishing_gradient_modified_files/2_2_2_vanishing_gradient_modified_4_3.svg)
    


    initializer: He



    
![svg](/assets/2_2_2_vanishing_gradient_modified_files/2_2_2_vanishing_gradient_modified_4_5.svg)
    


    ----------------- Relu
    initializer: gauss



    
![svg](/assets/2_2_2_vanishing_gradient_modified_files/2_2_2_vanishing_gradient_modified_4_7.svg)
    


    initializer: Xavier



    
![svg](/assets/2_2_2_vanishing_gradient_modified_files/2_2_2_vanishing_gradient_modified_4_9.svg)
    


    initializer: He



    
![svg](/assets/2_2_2_vanishing_gradient_modified_files/2_2_2_vanishing_gradient_modified_4_11.svg)
    

