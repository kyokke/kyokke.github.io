#### 実装演習 simple RNN (バイナリ加算)


```python
import sys
sys.path.append('./data')
```


```python
import numpy as np
from common import functions
import matplotlib.pyplot as plt

def d_tanh(x):
    return 1-np.tanh(x)**2 

def learn_simple_rnn(hidden_layer_size=16, learning_rate=0.1, weight_init_std=1, initializer='Normal', activation='sigmoid', iters_num=10000):
    # データを用意
    # 2進数の桁数
    binary_dim = 8
    # 最大値 + 1
    largest_number = pow(2, binary_dim)
    # largest_numberまで2進数を用意
    binary = np.unpackbits(np.array([range(largest_number)],dtype=np.uint8).T,axis=1)

    input_layer_size = 2
    output_layer_size = 1

    plot_interval = 100

    if activation == 'sigmoid':
        h_activation = functions.sigmoid
        h_activation_d = functions.d_sigmoid
    elif activation == 'tanh':
        h_activation = np.tanh
        h_activation_d = d_tanh
    elif activation == 'relu':
        h_activation = functions.relu
        h_activation_d = functions.d_relu
    else: 
        print('activation is not valid')

    if initializer == 'Normal':
        # ウェイト初期化 (バイアスは簡単のため省略)
        W_in = weight_init_std * np.random.randn(input_layer_size, hidden_layer_size)
        W_out = weight_init_std * np.random.randn(hidden_layer_size, output_layer_size)
        W = weight_init_std * np.random.randn(hidden_layer_size, hidden_layer_size)
    elif initializer == 'Xavier':
        # Xavier
        W_in =  np.random.randn(input_layer_size, hidden_layer_size)/np.sqrt(input_layer_size)
        W_out = np.random.randn(hidden_layer_size, output_layer_size)/np.sqrt(hidden_layer_size)
        W = np.random.randn(hidden_layer_size, hidden_layer_size)/np.sqrt(hidden_layer_size)
    elif initializer == 'He':
        # He
        W_in = np.random.randn(input_layer_size, hidden_layer_size)/np.sqrt(input_layer_size)*np.sqrt(2)
        W_out = np.random.randn(hidden_layer_size, output_layer_size)/np.sqrt(hidden_layer_size)*np.sqrt(2)
        W = np.random.randn(hidden_layer_size, hidden_layer_size)/np.sqrt(hidden_layer_size)*np.sqrt(2)
    else:
        print("initializer is not valid!")


    # 勾配
    W_in_grad = np.zeros_like(W_in)
    W_out_grad = np.zeros_like(W_out)
    W_grad = np.zeros_like(W)

    u = np.zeros((hidden_layer_size, binary_dim + 1))
    z = np.zeros((hidden_layer_size, binary_dim + 1))
    y = np.zeros((output_layer_size, binary_dim))

    delta_out = np.zeros((output_layer_size, binary_dim))
    delta = np.zeros((hidden_layer_size, binary_dim + 1))

    all_losses = []

    for i in range(iters_num):
        
        # A, B初期化 (a + b = d)
        a_int = np.random.randint(largest_number/2)
        a_bin = binary[a_int] # binary encoding
        b_int = np.random.randint(largest_number/2)
        b_bin = binary[b_int] # binary encoding
        
        # 正解データ
        d_int = a_int + b_int
        d_bin = binary[d_int]
        
        # 出力バイナリ
        out_bin = np.zeros_like(d_bin)
        
        # 時系列全体の誤差
        all_loss = 0    
        
        # 順伝播 
        for t in range(binary_dim):
            # 入力値
            X = np.array([a_bin[ - t - 1], b_bin[ - t - 1]]).reshape(1, -1) # (1, i)
            # 時刻tにおける正解データ
            dd = np.array([d_bin[binary_dim - t - 1]]) # (1,)
            
            u[:,t+1] = np.dot(X, W_in) + np.dot(z[:,t].reshape(1, -1), W) # (1,i) (i,hidden) + (1, hidden) (hidden,hidden)  = (hidden)
            z[:,t+1] = h_activation(u[:,t+1]) # = (hidden, )

            y[:,t] = functions.sigmoid(np.dot(z[:,t+1].reshape(1, -1), W_out)) # (1,hidden) * (hidden,0ut) = (1,out)


            #誤差
            loss = functions.mean_squared_error(dd, y[:,t])
            
            delta_out[:,t] = functions.d_mean_squared_error(dd, y[:,t]) * functions.d_sigmoid(y[:,t]) # (o,) \otimes (o,) = (o,)
            
            all_loss += loss

            out_bin[binary_dim - t - 1] = np.round(y[:,t])
        
        ## 誤差逆伝播
        for t in range(binary_dim)[::-1]:
            X = np.array([a_bin[-t-1],b_bin[-t-1]]).reshape(1, -1) # (i,i)
            delta[:,t] = (np.dot(delta[:,t+1].T, W.T) + np.dot(delta_out[:,t].T, W_out.T)) * h_activation_d(u[:,t+1]) # (1,hidden) (hidden,hidden) + (1,o)(o,hidden) = (hidden,)

            # 勾配更新
            W_out_grad += np.dot(z[:,t+1].reshape(-1,1), delta_out[:,t].reshape(-1,1)) # (hidden,1) (o,1) = (hidden, o) 
            W_grad += np.dot(z[:,t].reshape(-1,1), delta[:,t].reshape(1,-1)) # (hidden, 1) (1, hidden) = ( hidden,hidden )
            W_in_grad += np.dot(X.T, delta[:,t].reshape(1,-1)) # 
        
        # 勾配適用
        W_in -= learning_rate * W_in_grad
        W_out -= learning_rate * W_out_grad
        W -= learning_rate * W_grad
        
        W_in_grad *= 0
        W_out_grad *= 0
        W_grad *= 0
        

        if(i % plot_interval == 0):
            all_losses.append(all_loss)
            # print("iters:" + str(i))
            # print("Loss:" + str(all_loss))
            # print("Pred:" + str(out_bin))
            # print("True:" + str(d_bin))
            out_int = 0
            for index,x in enumerate(reversed(out_bin)):
                out_int += x * pow(2, index)
            # print(str(a_int) + " + " + str(b_int) + " = " + str(out_int))
            # print("------------")

    lists = range(0, iters_num, plot_interval)
    plt.plot(lists, all_losses, label="loss")
    plt.show()
```

隠れ層の数を変えてみる。8層で必要十分? 


```python

learn_simple_rnn(hidden_layer_size = 2,  iters_num=20000)
learn_simple_rnn(hidden_layer_size = 4,  iters_num=20000)
learn_simple_rnn(hidden_layer_size = 8,  iters_num=20000)
learn_simple_rnn(hidden_layer_size = 16, iters_num=20000)
learn_simple_rnn(hidden_layer_size = 32, iters_num=20000)
learn_simple_rnn(hidden_layer_size = 64, iters_num=20000)

```


    
![svg](/assets/3_1_simple_RNN_files/3_1_simple_RNN_4_0.svg)
    



    
![svg](/assets/3_1_simple_RNN_files/3_1_simple_RNN_4_1.svg)
    



    
![svg](/assets/3_1_simple_RNN_files/3_1_simple_RNN_4_2.svg)
    



    
![svg](/assets/3_1_simple_RNN_files/3_1_simple_RNN_4_3.svg)
    



    
![svg](/assets/3_1_simple_RNN_files/3_1_simple_RNN_4_4.svg)
    



    
![svg](/assets/3_1_simple_RNN_files/3_1_simple_RNN_4_5.svg)
    


初期化によってもかなり変わる


```python

learn_simple_rnn(hidden_layer_size = 8,  weight_init_std=0.1, iters_num=10000)
learn_simple_rnn(hidden_layer_size = 8,  weight_init_std=0.5, iters_num=10000)
learn_simple_rnn(hidden_layer_size = 8,  weight_init_std=1, iters_num=10000)
learn_simple_rnn(hidden_layer_size = 8,  weight_init_std=2, iters_num=10000)
learn_simple_rnn(hidden_layer_size = 8,  weight_init_std=5, iters_num=10000)

```


    
![svg](/assets/3_1_simple_RNN_files/3_1_simple_RNN_6_0.svg)
    



    
![svg](/assets/3_1_simple_RNN_files/3_1_simple_RNN_6_1.svg)
    



    
![svg](/assets/3_1_simple_RNN_files/3_1_simple_RNN_6_2.svg)
    



    
![svg](/assets/3_1_simple_RNN_files/3_1_simple_RNN_6_3.svg)
    



    
![svg](/assets/3_1_simple_RNN_files/3_1_simple_RNN_6_4.svg)
    


learning rate は 0.1付近~0.5 の間くらい?? 0.01 のときは単に収束が遅いだけではなさそうにもみえる。


```python

learn_simple_rnn(hidden_layer_size = 8,  weight_init_std=1, learning_rate=0.01, iters_num=50000)
learn_simple_rnn(hidden_layer_size = 8,  weight_init_std=1, learning_rate=0.1, iters_num=20000)
learn_simple_rnn(hidden_layer_size = 8,  weight_init_std=1, learning_rate=0.5, iters_num=10000)
learn_simple_rnn(hidden_layer_size = 8,  weight_init_std=1, learning_rate=1.0, iters_num=10000)
learn_simple_rnn(hidden_layer_size = 8,  weight_init_std=1, learning_rate=2.0, iters_num=10000)

```


    
![svg](/assets/3_1_simple_RNN_files/3_1_simple_RNN_8_0.svg)
    



    
![svg](/assets/3_1_simple_RNN_files/3_1_simple_RNN_8_1.svg)
    



    
![svg](/assets/3_1_simple_RNN_files/3_1_simple_RNN_8_2.svg)
    



    
![svg](/assets/3_1_simple_RNN_files/3_1_simple_RNN_8_3.svg)
    



    
![svg](/assets/3_1_simple_RNN_files/3_1_simple_RNN_8_4.svg)
    




activation は sigmoid なので Xavier がよいはずだが、あまり効果なさそう?? むしろ Normalが一番安定しているように見える.


```python
learn_simple_rnn(weight_init_std=1, initializer='Normal', learning_rate=0.2, iters_num=10000)
learn_simple_rnn(initializer='Xavier', learning_rate=0.2, iters_num=10000)
learn_simple_rnn(initializer='He', learning_rate=0.2, iters_num=10000)

```


    
![svg](/assets/3_1_simple_RNN_files/3_1_simple_RNN_11_0.svg)
    



    
![svg](/assets/3_1_simple_RNN_files/3_1_simple_RNN_11_1.svg)
    



    
![svg](/assets/3_1_simple_RNN_files/3_1_simple_RNN_11_2.svg)
    


活性化関数をtanhにしてみると、Normal より Xavier 収束が早くなっていそう。



```python
learn_simple_rnn(activation='tanh',    initializer='Normal', learning_rate=0.075, iters_num=10000)
learn_simple_rnn(activation='tanh',    initializer='Xavier', learning_rate=0.075, iters_num=10000)

```


    
![svg](/assets/3_1_simple_RNN_files/3_1_simple_RNN_13_0.svg)
    



    
![svg](/assets/3_1_simple_RNN_files/3_1_simple_RNN_13_1.svg)
    


活性化関数を relu にかえると。よいときと悪い時とで差が.. 
初期化方法が Normal だと ときどき基本的に収束はむずかしそう。
かなりの頻度で、NaN や overflow (これは勾配爆発が原因だろう) が起きる。


```python
learn_simple_rnn(activation='relu',    initializer='Normal', learning_rate=0.1, iters_num=20000)



```


    
![svg](/assets/3_1_simple_RNN_files/3_1_simple_RNN_15_0.svg)
    



```python
learn_simple_rnn(activation='relu',    initializer='Normal', learning_rate=0.1, iters_num=20000)



```

```
    ---------------------------------------------------------------------------

    ValueError                                Traceback (most recent call last)

    ValueError: cannot convert float NaN to integer

    
    The above exception was the direct cause of the following exception:


    ValueError                                Traceback (most recent call last)

    <ipython-input-67-6a829eb4e065> in <module>
    ----> 1 learn_simple_rnn(activation='relu',    initializer='Normal', learning_rate=0.1, iters_num=20000)
          2 
          3 


    <ipython-input-43-3497d874a704> in learn_simple_rnn(hidden_layer_size, learning_rate, weight_init_std, initializer, activation, iters_num)
        103             all_loss += loss
        104 
    --> 105             out_bin[binary_dim - t - 1] = np.round(y[:,t])
        106 
        107         ## 誤差逆伝播


    ValueError: setting an array element with a sequence.
```

He を つかうと 収束していそうなケースと、失敗してそうなのが半々くらい。
傾向としては、NaN や overflow が観測される数が著しく減った。


```python
learn_simple_rnn(activation='relu',    initializer='He', learning_rate=0.1, iters_num=20000)
```


    
![svg](/assets/3_1_simple_RNN_files/3_1_simple_RNN_18_0.svg)
    



```python
learn_simple_rnn(activation='relu',    initializer='He', learning_rate=0.1, iters_num=20000)
```


    
![svg](/assets/3_1_simple_RNN_files/3_1_simple_RNN_19_0.svg)
    

