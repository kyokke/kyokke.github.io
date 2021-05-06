### ハンズオン


```python
# Local実行のため、Google ドライブのマウントは行わない
# from google.colab import drive
# drive.mount('/content/drive')
```


```python
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

%matplotlib inline

#seaborn設定
sns.set()
#背景変更
sns.set_style("darkgrid", {'grid.linestyle': '--'})
#大きさ(スケール変更)
sns.set_context("paper")
```


```python
n=100
def true_func(x):
    z = 1-48*x+218*x**2-315*x**3+145*x**4
    return z 

def linear_func(x):
    z = x
    return z 
```


```python
# 真の関数からノイズを伴うデータを生成

# 真の関数からデータ生成
data = np.random.rand(n).astype(np.float32)
data = np.sort(data)
target = true_func(data)

# 　ノイズを加える
noise = 0.5 * np.random.randn(n) 
target = target  + noise

# ノイズ付きデータを描画
plt.scatter(data, target)
plt.title('NonLinear Regression')
# plt.legend(loc=2)
```




    Text(0.5, 1.0, 'NonLinear Regression')




    
![svg](/assets/skl_nonlinear_regression_files/skl_nonlinear_regression_4_1.svg)
    


線形回帰では十分にフィットできない(未学習の状態)。


```python
from sklearn.linear_model import LinearRegression

clf = LinearRegression()
data = data.reshape(-1,1)
target = target.reshape(-1,1)
clf.fit(data, target)

p_lin = clf.predict(data)

plt.scatter(data, target, label='data')
plt.plot(data, p_lin, color='darkorange', marker='', linestyle='-', linewidth=1, markersize=6, label='linear regression')
plt.legend(loc=2)
print(clf.score(data, target))
```

    0.40814627626298416



    
![svg](/assets/skl_nonlinear_regression_files/skl_nonlinear_regression_6_1.svg)
    


#### 多項式回帰
回帰モデルの形は、
$$ f(x) = \sum_{n=0}^{N-1} w_n x^n $$

入力 $x$ から $\phi_n(x) = x^n, n=0,1,2,\ldots$ を計算しこれを $n$ 次元ベクトルの説明変数だと思って、重回帰を行えばよい。
動画講義のスライド同様の結果が得られている



```python
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
deg = [1,2,3,4,5,6,7,8,9,10]

plt.figure(figsize=(8,5))
plt.scatter(data, target, label='data')
plt.title("polinomial regression")
for d in deg:
    regr = Pipeline([
        ('poly', PolynomialFeatures(degree=d)),
        ('linear', LinearRegression())
    ])
    regr.fit(data, target)
    # make predictions
    p_poly = regr.predict(data)
    # plot regression result
    plt.plot(data, p_poly, label='degree %d' % (d))
plt.legend(loc="lower right")

```




    <matplotlib.legend.Legend at 0x7f749c2b7f40>




    
![svg](/assets/skl_nonlinear_regression_files/skl_nonlinear_regression_8_1.svg)
    



#### ガウシアン基底 + 正則化項

次はRBFカーネル(ガウシアン基底)を使う。(RBFカーネルにも色々あるらしいが、scikit-learnの関数を追っかけていくとガウシアンだと分かる)

つまり fit 時に与えたデータ $ x_0, x_1,\ldots, x_{N-1}$ に対して、
$$
f(x) =\sum_{i=0}^{N-1} w_i \exp\bigl(-\gamma (x-x_i)^2\bigr) 
$$
で回帰を行うということ。

回帰の方式は、Ridge回帰ということで、評価関数は
$$ J(\mathrm{w}) = \sum_i \bigl( y - f(x_i) \bigr)^2 + \alpha \sum_i w_i^2  $$
となる。


まずは、sklearn の KernelRidge を使った簡単な実装で、
- $\gamma=1$ (デフォルト値)
- $alpha=0.0002$
の場合。


```python
from sklearn.kernel_ridge import KernelRidge

clf = KernelRidge(alpha=0.0002, kernel='rbf')
clf.fit(data, target)
p_kridge = clf.predict(data)
```

rbf_kernel() 関数によって、$ \mathrm{X} \rightarrow \Phi(\mathrm{X})$ の変換ができるので、多項式回帰と同様に
線形回帰との組み合わせで実装することもできる。
以下は、

 - $\alpha=30$ 
 - $\gamma=50$

の場合。はじめのモデルと比べてペナルティを強くかけている分、係数が0に集まっていることが分かる



```python
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.linear_model import Ridge

# 学習
kx = rbf_kernel(X=data, Y=data, gamma=50)
clf2 = Ridge(alpha=30)
clf2.fit(kx, target)
p_ridge = clf2.predict(kx)

# plot　
plt.scatter(data, target,label='data')
for i in range(len(kx)):
    if i == 0:
        plt.plot(data, kx[i], color='black', linestyle='-', linewidth=1, markersize=3, label='rbf kernels for each data', alpha=0.2)
    else:
        plt.plot(data, kx[i], color='black', linestyle='-', linewidth=1, markersize=3, alpha=0.2)

plt.plot(data, p_ridge, color='green', linestyle='-', linewidth=1, markersize=3,label='ridge regression #2 (alpha=30,gamma=50)')
plt.plot(data, p_kridge, color='orange', linestyle='-', linewidth=1, markersize=3,label='ridge regression #1 (alpha=0.0002,gamma=1)')
plt.legend()
plt.show()
# score
print("ridge regression #1 : %f" % (clf.score(data,target)))
print("ridge regression #2 : %f" % (clf2.score(kx, target)))

# weight coeff histogram

plt.hist([clf.dual_coef_[:,0], clf2.coef_[0,:]],label=['ridge regression 1','ridge regression 2'], bins=10)
plt.title("coef. histgram")
plt.legend()

```


    
![svg](/assets/skl_nonlinear_regression_files/skl_nonlinear_regression_12_0.svg)
    


    ridge regression #1 : 0.857598
    ridge regression #2 : 0.836881





    <matplotlib.legend.Legend at 0x7f7497a3d190>




    
![svg](/assets/skl_nonlinear_regression_files/skl_nonlinear_regression_12_3.svg)
    


つぎに Lasso 回帰の効果を確認する。ハンズオンに最初から入力されていた $alpha=10000$ はペナルティが強すぎてほとんど、誤差が減らない。
トライアル&エラーで $alpha=0.002$ がそれなりに fit することを見つけた。

実際に学習された係数を確認してみると、100のうち 4つの重みのみが非ゼロとなった。スパース推定ができていそう。


```python
#Lasso
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.linear_model import Lasso

kx = rbf_kernel(X=data, Y=data, gamma=5)

lasso_clf = Lasso(alpha=0.002, max_iter=50000)
lasso_clf.fit(kx, target)
p_lasso = lasso_clf.predict(kx)

lasso_clf2 = Lasso(alpha=10000, max_iter=50000)
lasso_clf2.fit(kx, target)
p_lasso2 = lasso_clf2.predict(kx)

# plot
plt.scatter(data, target)
plt.plot(data, p_lasso, label="alpha=0.002")
plt.plot(data, p_lasso2,label="alpha=10000")
plt.title("Lasso Regression")
plt.legend()
plt.show()

# score
print("alpha=0.002 :%f" % (lasso_clf.score(kx, target)))
print("alpha=10000 :%f" % (lasso_clf2.score(kx, target)))

print("weight for the model with alpha=0.002")
print(lasso_clf.coef_)
```


    
![svg](/assets/skl_nonlinear_regression_files/skl_nonlinear_regression_14_0.svg)
    

```
    alpha=0.002 :0.848822
    alpha=10000 :-0.000000
    weight for the model with alpha=0.002
    [-0.         -0.         -0.         -0.         -0.         -0.
     -0.         -0.         -0.         -0.         -0.         -2.4409611
     -7.19585    -1.6620084  -4.9211154  -0.         -0.         -0.
     -0.         -0.         -0.         -0.         -0.         -0.
     -0.         -0.         -0.         -0.         -0.         -0.
     -0.         -0.         -0.         -0.         -0.         -0.
     -0.         -0.         -0.         -0.         -0.         -0.
     -0.         -0.         -0.         -0.         -0.          0.
      0.          0.          0.          0.          0.          0.
      0.          0.          0.          0.          0.          0.
      0.          0.          0.          0.          0.         -0.
     -0.         -0.         -0.         -0.         -0.         -0.
     -0.         -0.         -0.         -0.         -0.         -0.
     -0.         -0.         -0.         -0.         -0.         -0.
     -0.         -0.         -0.         -0.         -0.         -0.
     -0.         -0.         -0.         -0.         -0.         -7.282311
     -6.780508   -0.03432941 -0.         -0.        ]
```

#### サポートベクトル回帰(SVR)

名前だけ見ると、これは後に学習する サポートベクトルマシンを回帰問題に応用したもの？
[調べてみると](https://datachemeng.com/supportvectorregression/)

- モデル $f(x)$ は基底関数法と同じ
- L2ノルムをペナルティにもつ
- 誤差関数が二乗誤差和でなく $\max(0, |f(x) - y|-\epsilon)$  

らしい。ハンズオンに元々入力されていた $gamma=0.1$ ではうまくフィットしなかったため、
$gamma=1$で学習。 


```python
from sklearn import model_selection, preprocessing, linear_model, svm

# SVR-rbf
clf_svr = svm.SVR(kernel='rbf', C=1e3, gamma=1, epsilon=0.1)
clf_svr.fit(data, target[:,0])
y_rbf = clf_svr.fit(data, target[:,0]).predict(data)

clf_svr2 = svm.SVR(kernel='rbf', C=1e3, gamma=0.1, epsilon=0.1)
clf_svr2.fit(data, target[:,0])
y_rbf2 = clf_svr2.fit(data, target[:,0]).predict(data)


# plot
plt.scatter(data, target, color='darkorange', label='data')
plt.title("Support Vector Regression (RBF)")
plt.plot(data, y_rbf, label='gamma=1')
plt.plot(data, y_rbf2, label='gamma=0.1')

plt.legend()
plt.show()
```


    
![svg](/assets/skl_nonlinear_regression_files/skl_nonlinear_regression_16_0.svg)
    


#### 多層パーセプトロンによる回帰

Kerasによる回帰の例も示されている。が、これはきっと Stage3移行で改めて勉強するので、実行して結果を確認する程度にする


```python
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(data, target, test_size=0.1, random_state=0)
```


```python
!mkdir -p ./out/checkpoints 
!mkdir -p ./out/tensorBoard
```


```python
from keras.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint

cb_cp = ModelCheckpoint('./out/checkpoints/weights.{epoch:02d}-{val_loss:.2f}.hdf5', verbose=1, save_weights_only=True)
cb_tf  = TensorBoard(log_dir='./out/tensorBoard', histogram_freq=0)
```


```python
def relu_reg_model():
    model = Sequential()
    model.add(Dense(10, input_dim=1, activation='relu'))
    model.add(Dense(1000, activation='relu'))
    model.add(Dense(1000, activation='relu'))
    model.add(Dense(1000, activation='relu'))
    model.add(Dense(1000, activation='relu'))
    model.add(Dense(1000, activation='relu'))
    model.add(Dense(1000, activation='relu'))
    model.add(Dense(1000, activation='relu'))
    model.add(Dense(1000, activation='linear'))
#     model.add(Dense(100, activation='relu'))
#     model.add(Dense(100, activation='relu'))
#     model.add(Dense(100, activation='relu'))
#     model.add(Dense(100, activation='relu'))
    model.add(Dense(1))

    model.compile(loss='mean_squared_error', optimizer='adam')
    return model
```


```python
from keras.models import Sequential
from keras.layers import Input, Dense, Dropout, BatchNormalization
from keras.wrappers.scikit_learn import KerasRegressor

# use data split and fit to run the model
estimator = KerasRegressor(build_fn=relu_reg_model, epochs=100, batch_size=5, verbose=1)

history = estimator.fit(x_train, y_train, callbacks=[cb_cp, cb_tf], validation_data=(x_test, y_test))
```
出力は抜粋(本当は1エポック毎出力されているのだが)

```
    Epoch 1/100
    18/18 [==============================] - 1s 25ms/step - loss: 1.6244 - val_loss: 1.5469

    ...

    Epoch 00009: saving model to ./out/checkpoints/weights.09-1.04.hdf5
    Epoch 10/100
    18/18 [==============================] - 0s 11ms/step - loss: 1.0322 - val_loss: 0.8556
    
    ...
    
    Epoch 00049: saving model to ./out/checkpoints/weights.49-0.47.hdf5
    Epoch 50/100
    18/18 [==============================] - 0s 11ms/step - loss: 0.3095 - val_loss: 0.4469

    ...

    Epoch 00099: saving model to ./out/checkpoints/weights.99-0.49.hdf5
    Epoch 100/100
    18/18 [==============================] - 0s 12ms/step - loss: 0.2995 - val_loss: 0.3197
    
    Epoch 00100: saving model to ./out/checkpoints/weights.100-0.32.hdf5
```


```python
y_pred = estimator.predict(x_train)
```

    18/18 [==============================] - 0s 3ms/step



```python
plt.title('NonLiner Regressions via DL by ReLU')
plt.plot(data, target, 'o')
plt.plot(data, true_func(data), '.')
plt.plot(x_train, y_pred, "o", label='predicted: deep learning')
#plt.legend(loc=2)
```




    [<matplotlib.lines.Line2D at 0x7f746052cf70>]




    
![svg](/assets/skl_nonlinear_regression_files/skl_nonlinear_regression_24_1.svg)
    



```python
plt.figure(figsize=(8,5))
plt.title('compare all regression models')
x=np.linspace(0,1,num=100)
plt.plot(x, true_func(x),label="true function")

plt.plot(x_train, y_pred, "o", label='DL')
plt.plot(data, y_rbf ,label="SVR(RBF)")
plt.plot(data, p_lasso ,label="Lasso(RBF)")
plt.plot(data, p_kridge ,label="Ridge(RBF)")
plt.legend(loc=2)
plt.show()

#plt.plot(data, clf ,label="Polinomial")


#plt.legend(loc=2)
```


    
![svg](/assets/skl_nonlinear_regression_files/skl_nonlinear_regression_25_0.svg)
    



```python

```
