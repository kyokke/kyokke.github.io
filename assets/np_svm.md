###  ハンズオン


```python
%matplotlib inline
import numpy as np
import matplotlib.pyplot as plt
```

#### 線形分離可能なケース


```python
def fit(X_train, t, n_iter=500,  eta1=0.01, eta2=0.001, C=0):
    K = X_train.dot(X_train.T) 
    H = np.outer(t, t) * K
    n_samples = len(t)

    a = np.ones(n_samples) # 初期値
    for _ in range(n_iter):
        grad = 1 - H.dot(a)

        # 基本の更新式
        a += eta1 * grad
        # 制約部分
        a -= eta2 * a.dot(t) * t
        # 未定定数 a が全部0以上というのは凸なので、
        # 無理やり射影してもまあ悪いことにはならない
        if C == 0:
            a = np.where(a > 0, a, 0) 
        elif C>0: # ソフトマージンのときはここだけ違う
            a = np.clip(a, 0, C)


    # w,b を求める
    index = a > 1e-6 # epsilon
    support_vectors = X_train[index]
    support_vector_t = t[index]
    support_vector_a = a[index]

    term2 = K[index][:, index].dot(support_vector_a * support_vector_t)
    b = (support_vector_t - term2).mean()
    
    w = np.zeros(X_train.shape[1]) 
    for a, sv_t, sv in zip(support_vector_a, support_vector_t, support_vectors):
        w += a * sv_t * sv

    return w, b, support_vectors

def predict(X,w,b):
    y_project = np.zeros(len(X))
    for i in range(len(X)):
         y_project[i] = w.dot(X[i]) + b
    return np.sign(y_project), y_project
```


```python
# (2,2) と (-2,-2) の周辺にランダムに点を発生させる
def gen_data():
    x0 = np.random.normal(size=50).reshape(-1, 2) - 2.
    x1 = np.random.normal(size=50).reshape(-1, 2) + 2.
    X_train = np.concatenate([x0, x1])
    ys_train = np.concatenate([np.zeros(25), np.ones(25)]).astype(np.int)
    return X_train, ys_train
```


```python
# -------- データ生成
X_train, ys_train = gen_data()
# plt.scatter(X_train[:, 0], X_train[:, 1], c=ys_train)
t = np.where(ys_train == 1.0, 1.0, -1.0)

# --------- 学習
w,b,support_vectors = fit(X_train,t)


# --------- 学習結果の確認
# 領域可視化のために mesh 上の各点を入力データとして生成 
xx0, xx1 = np.meshgrid(np.linspace(-5, 5, 100), np.linspace(-5, 5, 100))
xx = np.array([xx0, xx1]).reshape(2, -1).T
X_test = xx

y_pred, y_project = predict(X_test,w,b)

# 訓練データを可視化
plt.scatter(X_train[:, 0], X_train[:, 1], c=ys_train)

# サポートベクトルを可視化
plt.scatter(support_vectors[:, 0], support_vectors[:, 1],
                    s=100, facecolors='none', edgecolors='k')

# 領域を可視化
plt.contourf(xx0, xx1, y_pred.reshape(100, 100), alpha=0.2, levels=np.linspace(0, 1, 3))
# マージンと決定境界を可視化
plt.contour(xx0, xx1, y_project.reshape(100, 100), colors='k',
                     levels=[-1, 0, 1], alpha=0.5, linestyles=['--', '-', '--'])

# マージンと決定境界を可視化
plt.quiver(0, 0, 0.1, 0.35, width=0.01, scale=1, color='pink')


```




    <matplotlib.quiver.Quiver at 0x7f1ce7904070>




    
![SVG](/assets/np_svm_files/np_svm_5_1.svg)
    


#### 線形分離不可能なケース

元のデータ空間では線形分離は出来ないケースに対して、特徴空間上で線形分離することを考える。
今回はカーネルとしてRBFカーネル（ガウシアンカーネル）を利用する。


```python
def fit_k(K, X_train, t, n_iter=500,  eta1=0.01, eta2=0.001):
    n_samples = len(t)
    eta1 = 0.01
    eta2 = 0.001
    n_iter = 5000

    H = np.outer(t, t) * K
    # a の学習
    a = np.ones(n_samples)
    for _ in range(n_iter):
        grad = 1 - H.dot(a)
        a += eta1 * grad
        a -= eta2 * a.dot(t) * t
        a = np.where(a > 0, a, 0)

    # support vector の情報取得
    index = a > 1e-6
    support_vectors = X_train[index]
    support_vector_t = t[index]
    support_vector_a = a[index]

    # b の計算
    term2 = K[index][:, index].dot(support_vector_a * support_vector_t)
    b = (support_vector_t - term2).mean()

    return a, b, support_vectors, support_vector_t, support_vector_a


def predict_k(X, b, support_vectors, support_vector_t, support_vector_a):
    y_project = np.ones(len(X)) * b
    for i in range(len(X)):
        for a, sv_t, sv in zip(support_vector_a, support_vector_t, support_vectors):
            y_project[i] += a * sv_t * rbf(X[i], sv)
    y_pred = np.sign(y_project)
    return y_pred,  y_project
```


```python
# 半径の異なる円周上に点をばらつかせ, 内側を1, 外側を0 に分類
factor = .2
n_samples = 50
linspace = np.linspace(0, 2 * np.pi, n_samples // 2 + 1)[:-1]
outer_circ_x = np.cos(linspace)
outer_circ_y = np.sin(linspace)
inner_circ_x = outer_circ_x * factor
inner_circ_y = outer_circ_y * factor

X = np.vstack((np.append(outer_circ_x, inner_circ_x),
               np.append(outer_circ_y, inner_circ_y))).T
y = np.hstack([np.zeros(n_samples // 2, dtype=np.intp),
               np.ones(n_samples // 2, dtype=np.intp)])
X += np.random.normal(scale=0.15, size=X.shape)
x_train = X
y_train = y

# RBFカーネルの計算
def rbf(u, v):
        sigma = 0.8
        return np.exp(-0.5 * ((u - v)**2).sum() / sigma**2)

X_train = x_train
t = np.where(y_train == 1.0, 1.0, -1.0)

K = np.zeros((n_samples, n_samples))
for i in range(n_samples):
    for j in range(n_samples):
        K[i, j] = rbf(X_train[i], X_train[j])

# 学習
a,b,support_vectors,support_vector_t, support_vector_a = fit_k(K, X_train, t)

# 予測
xx0, xx1 = np.meshgrid(np.linspace(-1.5, 1.5, 100), np.linspace(-1.5, 1.5, 100))
xx = np.array([xx0, xx1]).reshape(2, -1).T
X_test = xx

y_pred, y_project = predict_k(X_test, b, support_vectors, support_vector_t, support_vector_a)

# 訓練データを可視化
plt.scatter(x_train[:, 0], x_train[:, 1], c=y_train)
# サポートベクトルを可視化
plt.scatter(support_vectors[:, 0], support_vectors[:, 1],
                    s=100, facecolors='none', edgecolors='k')
# 領域を可視化
plt.contourf(xx0, xx1, y_pred.reshape(100, 100), alpha=0.2, levels=np.linspace(0, 1, 3))
# マージンと決定境界を可視化
plt.contour(xx0, xx1, y_project.reshape(100, 100), colors='k',
                     levels=[-1, 0, 1], alpha=0.5, linestyles=['--', '-', '--'])
                     
```




    <matplotlib.contour.QuadContourSet at 0x7f1cec0fb0a0>




    
![SVG](/assets/np_svm_files/np_svm_9_1.svg)
    


### 重なりのあるケースでの ソフトマージンSVM

要点まとめに書いた通り、最終的にパラメータ更新における、$a$の不等式制約が少し違うだけ。
関数をまとめて、Cの値によって場合分け実装した (オリジナルの実装だと epsilon が異なっていたがまぁ、共通でもいいだろう。)

```python
if C == 0: # 通常のSVM
    a = np.where(a > 0, a, 0) 
elif C>0: # ソフトマージンのとき
    a = np.clip(a, 0, C)
```


```python
# 通常のSVMで使用したデータよりも集合が近い (1,1)と(-1,-1)の中心で散らばりをもたせたデータ
x0 = np.random.normal(size=50).reshape(-1, 2) - 1.
x1 = np.random.normal(size=50).reshape(-1, 2) + 1.
x_train = np.concatenate([x0, x1])
y_train = np.concatenate([np.zeros(25), np.ones(25)]).astype(np.int)

plt.scatter(x_train[:, 0], x_train[:, 1], c=y_train)

# 学習
X_train = x_train
t = np.where(y_train == 1.0, 1.0, -1.0)
w, b, support_vectors  = fit(X_train, t, n_iter=1000, C=1)

# 領域可視化のため、mesh上の点を生成
xx0, xx1 = np.meshgrid(np.linspace(-4, 4, 100), np.linspace(-4, 4, 100))
xx = np.array([xx0, xx1]).reshape(2, -1).T
X_test = xx 
y_pred, y_project = predict(X_test,w,b)

# 訓練データを可視化
plt.scatter(x_train[:, 0], x_train[:, 1], c=y_train)

# サポートベクトルを可視化
plt.scatter(support_vectors[:, 0], support_vectors[:, 1],
                    s=100, facecolors='none', edgecolors='k')
# 領域を可視化
plt.contourf(xx0, xx1, y_pred.reshape(100, 100), alpha=0.2, levels=np.linspace(0, 1, 3))

# マージンと決定境界を可視化
plt.contour(xx0, xx1, y_project.reshape(100, 100), colors='k',
                     levels=[-1, 0, 1], alpha=0.5, linestyles=['--', '-', '--'])

```




    <matplotlib.contour.QuadContourSet at 0x7f1ce7e997c0>




    
![SVG](/assets/np_svm_files/np_svm_11_1.svg)
    

