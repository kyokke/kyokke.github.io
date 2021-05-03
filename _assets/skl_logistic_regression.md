### ハンズオン

#### データ読み込み & 前処理


```python
# from google.colab import drive
# drive.mount('/content/drive')
```


```python
#from モジュール名 import クラス名（もしくは関数名や変数名）
import pandas as pd
from pandas import DataFrame
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#matplotlibをinlineで表示するためのおまじない (plt.show()しなくていい)
%matplotlib inline
```


```python
# titanic data csvファイルの読み込み
titanic_df = pd.read_csv('../data/titanic_train.csv')

# ファイルの先頭部を表示し、データセットを確認する
titanic_df.head(5)
```



~~~
<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PassengerId</th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Name</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Ticket</th>
      <th>Fare</th>
      <th>Cabin</th>
      <th>Embarked</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0</td>
      <td>3</td>
      <td>Braund, Mr. Owen Harris</td>
      <td>male</td>
      <td>22.0</td>
      <td>1</td>
      <td>0</td>
      <td>A/5 21171</td>
      <td>7.2500</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>Cumings, Mrs. John Bradley (Florence Briggs Th...</td>
      <td>female</td>
      <td>38.0</td>
      <td>1</td>
      <td>0</td>
      <td>PC 17599</td>
      <td>71.2833</td>
      <td>C85</td>
      <td>C</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>1</td>
      <td>3</td>
      <td>Heikkinen, Miss. Laina</td>
      <td>female</td>
      <td>26.0</td>
      <td>0</td>
      <td>0</td>
      <td>STON/O2. 3101282</td>
      <td>7.9250</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>1</td>
      <td>1</td>
      <td>Futrelle, Mrs. Jacques Heath (Lily May Peel)</td>
      <td>female</td>
      <td>35.0</td>
      <td>1</td>
      <td>0</td>
      <td>113803</td>
      <td>53.1000</td>
      <td>C123</td>
      <td>S</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>0</td>
      <td>3</td>
      <td>Allen, Mr. William Henry</td>
      <td>male</td>
      <td>35.0</td>
      <td>0</td>
      <td>0</td>
      <td>373450</td>
      <td>8.0500</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
  </tbody>
</table>
</div>
~~~



```python
#予測に不要と考えるカラムをドロップ (本当はここの情報もしっかり使うべきだと思っています)
titanic_df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1, inplace=True)

#一部カラムをドロップしたデータを表示
# titanic_df.head()

#nullを含んでいる行を表示
titanic_df[titanic_df.isnull().any(1)].head(10)
```



~~~
<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Fare</th>
      <th>Embarked</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>5</th>
      <td>0</td>
      <td>3</td>
      <td>male</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
      <td>8.4583</td>
      <td>Q</td>
    </tr>
    <tr>
      <th>17</th>
      <td>1</td>
      <td>2</td>
      <td>male</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
      <td>13.0000</td>
      <td>S</td>
    </tr>
    <tr>
      <th>19</th>
      <td>1</td>
      <td>3</td>
      <td>female</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
      <td>7.2250</td>
      <td>C</td>
    </tr>
    <tr>
      <th>26</th>
      <td>0</td>
      <td>3</td>
      <td>male</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
      <td>7.2250</td>
      <td>C</td>
    </tr>
    <tr>
      <th>28</th>
      <td>1</td>
      <td>3</td>
      <td>female</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
      <td>7.8792</td>
      <td>Q</td>
    </tr>
    <tr>
      <th>29</th>
      <td>0</td>
      <td>3</td>
      <td>male</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
      <td>7.8958</td>
      <td>S</td>
    </tr>
    <tr>
      <th>31</th>
      <td>1</td>
      <td>1</td>
      <td>female</td>
      <td>NaN</td>
      <td>1</td>
      <td>0</td>
      <td>146.5208</td>
      <td>C</td>
    </tr>
    <tr>
      <th>32</th>
      <td>1</td>
      <td>3</td>
      <td>female</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
      <td>7.7500</td>
      <td>Q</td>
    </tr>
    <tr>
      <th>36</th>
      <td>1</td>
      <td>3</td>
      <td>male</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
      <td>7.2292</td>
      <td>C</td>
    </tr>
    <tr>
      <th>42</th>
      <td>0</td>
      <td>3</td>
      <td>male</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
      <td>7.8958</td>
      <td>C</td>
    </tr>
  </tbody>
</table>
</div>
~~~



```python
#Ageカラムのnullを中央値で補完
titanic_df['AgeFill'] = titanic_df['Age'].fillna(titanic_df['Age'].mean())

#再度nullを含んでいる行を表示 (Ageのnullは補完されている)
titanic_df[titanic_df.isnull().any(1)]

#titanic_df.dtypes
```



~~~
<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Fare</th>
      <th>Embarked</th>
      <th>AgeFill</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>5</th>
      <td>0</td>
      <td>3</td>
      <td>male</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
      <td>8.4583</td>
      <td>Q</td>
      <td>29.699118</td>
    </tr>
    <tr>
      <th>17</th>
      <td>1</td>
      <td>2</td>
      <td>male</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
      <td>13.0000</td>
      <td>S</td>
      <td>29.699118</td>
    </tr>
    <tr>
      <th>19</th>
      <td>1</td>
      <td>3</td>
      <td>female</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
      <td>7.2250</td>
      <td>C</td>
      <td>29.699118</td>
    </tr>
    <tr>
      <th>26</th>
      <td>0</td>
      <td>3</td>
      <td>male</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
      <td>7.2250</td>
      <td>C</td>
      <td>29.699118</td>
    </tr>
    <tr>
      <th>28</th>
      <td>1</td>
      <td>3</td>
      <td>female</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
      <td>7.8792</td>
      <td>Q</td>
      <td>29.699118</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>859</th>
      <td>0</td>
      <td>3</td>
      <td>male</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
      <td>7.2292</td>
      <td>C</td>
      <td>29.699118</td>
    </tr>
    <tr>
      <th>863</th>
      <td>0</td>
      <td>3</td>
      <td>female</td>
      <td>NaN</td>
      <td>8</td>
      <td>2</td>
      <td>69.5500</td>
      <td>S</td>
      <td>29.699118</td>
    </tr>
    <tr>
      <th>868</th>
      <td>0</td>
      <td>3</td>
      <td>male</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
      <td>9.5000</td>
      <td>S</td>
      <td>29.699118</td>
    </tr>
    <tr>
      <th>878</th>
      <td>0</td>
      <td>3</td>
      <td>male</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
      <td>7.8958</td>
      <td>S</td>
      <td>29.699118</td>
    </tr>
    <tr>
      <th>888</th>
      <td>0</td>
      <td>3</td>
      <td>female</td>
      <td>NaN</td>
      <td>1</td>
      <td>2</td>
      <td>23.4500</td>
      <td>S</td>
      <td>29.699118</td>
    </tr>
  </tbody>
</table>
<p>179 rows × 9 columns</p>
</div>
~~~


#### 実装(チケット価格から生死を判別)


```python
#運賃だけのリストを作成
data1 = titanic_df.loc[:, ["Fare"]].values

#生死フラグのみのリストを作成
label1 =  titanic_df.loc[:,["Survived"]].values


```


```python
from sklearn.linear_model import LogisticRegression

# 学習
model=LogisticRegression()
model.fit(data1, label1[:,0])

# 学習結果の出力
print("モデルパラメータ")
print (model.intercept_)
print (model.coef_)

# 試しにいくつか予測 : 運賃 61USD, 62USD の間で生死の確率が逆転する
print("運賃 61USD")
print( model.predict([[61]]))
print( model.predict_proba([[61]]))

print("運賃 62USD")
print( model.predict([[62]]))
print( model.predict_proba([[62]]))
```
```
    モデルパラメータ
    [-0.94131796]
    [[0.01519666]]
    運賃 61USD
    [0]
    [[0.50358033 0.49641967]]
    運賃 62USD
    [1]
    [[0.49978123 0.50021877]]
```


```python
## 可視化 (おまけ程度なので飛ばします、だそう)
X_test_value = model.decision_function(data1) 

# # 決定関数値（絶対値が大きいほど識別境界から離れている）
# X_test_value = model.decision_function(X_test) 
# # 決定関数値をシグモイド関数で確率に変換
# X_test_prob = normal_sigmoid(X_test_value) 

w_0 = model.intercept_[0]
w_1 = model.coef_[0,0]

def sigmoid(x):
    return 1 / (1+np.exp(-(w_1*x+w_0)))

x_range = np.linspace(-1, 500, 3000)

plt.figure(figsize=(9,5))

plt.plot(data1, model.predict_proba(data1)[:,0], 'o', label="predicted prob. (survive)")
plt.plot(data1, model.predict_proba(data1)[:,1], 'o', label="predicted prob. (not survive)")
plt.plot(x_range, sigmoid(x_range), '--', label="sigmoid function with param")
plt.legend()
plt.show()
```

![svg](/assets/skl_logistic_regression_files/skl_logistic_regression_10_1.svg)
    


#### 実装(2変数から生死を判別)

ただ変数を増やすだけでなく、特徴量エンジニアリングを経験するのがポイント


```python
#AgeFillの欠損値を埋めたので
#titanic_df = titanic_df.drop(['Age'], axis=1)

# 'Gender' の male,female を それぞれ 0,1 に置き換え
titanic_df['Gender'] = titanic_df['Sex'].map({'female': 0, 'male': 1}).astype(int)
titanic_df.head(3)

```



~~~
<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Fare</th>
      <th>Embarked</th>
      <th>AgeFill</th>
      <th>Gender</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>3</td>
      <td>male</td>
      <td>22.0</td>
      <td>1</td>
      <td>0</td>
      <td>7.2500</td>
      <td>S</td>
      <td>22.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>1</td>
      <td>female</td>
      <td>38.0</td>
      <td>1</td>
      <td>0</td>
      <td>71.2833</td>
      <td>C</td>
      <td>38.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>3</td>
      <td>female</td>
      <td>26.0</td>
      <td>0</td>
      <td>0</td>
      <td>7.9250</td>
      <td>S</td>
      <td>26.0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>
~~~



```python
# Pclass と Gender を足し合わせた新たな特徴量 'Pclass_Gender' をつくｒ
titanic_df['Pclass_Gender'] = titanic_df['Pclass'] + titanic_df['Gender']
titanic_df.head()
```



~~~
<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Fare</th>
      <th>Embarked</th>
      <th>AgeFill</th>
      <th>Gender</th>
      <th>Pclass_Gender</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>3</td>
      <td>male</td>
      <td>22.0</td>
      <td>1</td>
      <td>0</td>
      <td>7.2500</td>
      <td>S</td>
      <td>22.0</td>
      <td>1</td>
      <td>4</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>1</td>
      <td>female</td>
      <td>38.0</td>
      <td>1</td>
      <td>0</td>
      <td>71.2833</td>
      <td>C</td>
      <td>38.0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>3</td>
      <td>female</td>
      <td>26.0</td>
      <td>0</td>
      <td>0</td>
      <td>7.9250</td>
      <td>S</td>
      <td>26.0</td>
      <td>0</td>
      <td>3</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>1</td>
      <td>female</td>
      <td>35.0</td>
      <td>1</td>
      <td>0</td>
      <td>53.1000</td>
      <td>S</td>
      <td>35.0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>3</td>
      <td>male</td>
      <td>35.0</td>
      <td>0</td>
      <td>0</td>
      <td>8.0500</td>
      <td>S</td>
      <td>35.0</td>
      <td>1</td>
      <td>4</td>
    </tr>
  </tbody>
</table>
</div>
~~~



```python
# 不要になった特徴量を drop
titanic_df = titanic_df.drop(['Pclass', 'Sex', 'Gender','Age'], axis=1)
titanic_df.head()
```



~~~
<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Survived</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Fare</th>
      <th>Embarked</th>
      <th>AgeFill</th>
      <th>Pclass_Gender</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>7.2500</td>
      <td>S</td>
      <td>22.0</td>
      <td>4</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>71.2833</td>
      <td>C</td>
      <td>38.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>7.9250</td>
      <td>S</td>
      <td>26.0</td>
      <td>3</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>53.1000</td>
      <td>S</td>
      <td>35.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>8.0500</td>
      <td>S</td>
      <td>35.0</td>
      <td>4</td>
    </tr>
  </tbody>
</table>
</div>
~~~



```python
#　使用する2変数だけのリストを作成
data2 = titanic_df.loc[:, ["AgeFill", "Pclass_Gender"]].values

#生死フラグのみのリストを作成
label2 =  titanic_df.loc[:,["Survived"]].values
```


```python
# skleanr 実装

# 学習
model2 = LogisticRegression()
model2.fit(data2, label2[:,0])

# 予測　
print("10歳でPclass_Gender=1")
print(model2.predict([[10,1]]))
print(model2.predict_proba([[10,1]]))

# 予測
print("10歳でPclass_Gender=4")
print(model2.predict([[10,4]]))
print(model2.predict_proba([[10,4]]))

```

    10歳でPclass_Gender=1
    [1]
    [[0.03754749 0.96245251]]
    10歳でPclass_Gender=4
    [0]
    [[0.78415473 0.21584527]]



```python
# numpy 実装
def add_one(x):
    return np.concatenate([np.ones(len(x))[:, None], x], axis=1)

# overflow,underflow 対策 のため素直な実装はしない
def sigmoid(x):
    return (np.tanh(x/2) + 1)/2

# うまく収束しているかどうか確認するための loss計算
def loss(X,y,w):
    yhat = sigmoid(np.dot(X,w))
    # これだと、yhat = 0,1の時にエラーがでる。
    # temp = y * np.log(yhat) + (1-y) * np.log(1-yhat)
    temp = np.log( (1-y)-(-1)**y * yhat)
    return -sum(temp)/len(temp)

# バッチの共役勾配法
def sgd(X_train, y_train, max_iter, eta):
    # w = np.random.rand(X_train.shape[1])
    w = np.zeros(X_train.shape[1])
    for i in range(max_iter):
        if i % 100000 == 0:
            print(loss(X_train, y_train, w))
        w_prev = np.copy(w)
        yhat = sigmoid(np.dot(X_train, w))
        w -= eta * np.dot(X_train.T, (yhat - y_train))
        #if np.allclose(w, w_prev):
        #    return w
    return w

# ヒューリスティックにパラメータチューニング
max_iter=2000000 
eta = 0.000001
w = sgd(add_one(data2), label2[:,0], max_iter, eta)
```

```
    0.6931471805599373
    0.5079276216927482
    0.4905915541820542
    0.4862090047552061
    0.48489864641172786
    0.48447258191832004
    0.48432764482843516
    0.48427706879224364
    0.48425915804405467
    0.48425275998902756
    0.4842504626951552
    0.4842496352899579
    0.4842493367393749
    0.4842492288953244
    0.48424918991352967
    0.4842491758173902
    0.48424917071889134
    0.484249168874524
    0.4842491682072763
    0.4842491679658649
```

```python
print("numpy実装で得られたパラメータ")
print(w)

print("sklearn実装で得られたパラメータ")
print(model2.intercept_)
print(model2.coef_)

```
```
    numpy実装で得られたパラメータ
    [ 5.27828007 -0.04669006 -1.52804692]
    sklearn実装で得られたパラメータ
    [5.2174269]
    [[-0.04622413 -1.51130754]]
```


```python
titanic_df.head(3)
```



~~~
<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Survived</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Fare</th>
      <th>Embarked</th>
      <th>AgeFill</th>
      <th>Pclass_Gender</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>7.2500</td>
      <td>S</td>
      <td>22.0</td>
      <td>4</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>71.2833</td>
      <td>C</td>
      <td>38.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>7.9250</td>
      <td>S</td>
      <td>26.0</td>
      <td>3</td>
    </tr>
  </tbody>
</table>
</div>
~~~


特徴量空間上にサンプルと、決定境界をプロットしてみよう。

ロジスティック回帰の場合、確率0.5が境界。

\begin{eqnarray}
\frac{1}{1+e^{\boldsymbol{w}^T \boldsymbol{x} + w_0}} &> \frac{1}{2} \\
e^{\boldsymbol{w}^T \boldsymbol{x} + w_0} &< 1 \\
\boldsymbol{w}^T \boldsymbol{x} + w_0 &< 0
\end{eqnarray}
が決定境界になる。



```python
from matplotlib.colors import ListedColormap

np.random.seed = 0
h = 0.02
xmin, xmax = -5, 85
ymin, ymax = 0.5, 4.5
index_survived = titanic_df[titanic_df["Survived"]==0].index
index_notsurvived = titanic_df[titanic_df["Survived"]==1].index

xx, yy = np.meshgrid(np.arange(xmin, xmax, h), np.arange(ymin, ymax, h))
Z = model2.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
Z = Z.reshape(xx.shape)

fig, ax = plt.subplots()
levels = np.linspace(0, 1.0)
cm = plt.cm.RdBu
cm_bright = ListedColormap(['#FF0000', '#0000FF'])
#contour = ax.contourf(xx, yy, Z, cmap=cm, levels=levels, alpha=0.5)

sc = ax.scatter(titanic_df.loc[index_survived, 'AgeFill'],
                titanic_df.loc[index_survived, 'Pclass_Gender']+(np.random.rand(len(index_survived))-0.5)*0.1,
                color='r', label='Not Survived', alpha=0.3)
sc = ax.scatter(titanic_df.loc[index_notsurvived, 'AgeFill'],
                titanic_df.loc[index_notsurvived, 'Pclass_Gender']+(np.random.rand(len(index_notsurvived))-0.5)*0.1,
                color='b', label='Survived', alpha=0.3)

ax.set_xlabel('AgeFill')
ax.set_ylabel('Pclass_Gender')
ax.set_xlim(xmin, xmax)
ax.set_ylim(ymin, ymax)
#fig.colorbar(contour)

# 決定境界
x1 = xmin
x2 = xmax
y1 = -1*(model2.intercept_[0]+model2.coef_[0][0]*xmin)/model2.coef_[0][1]
y2 = -1*(model2.intercept_[0]+model2.coef_[0][0]*xmax)/model2.coef_[0][1]
ax.plot([x1, x2] ,[y1, y2], 'g--', label="decision boundary(sklearn)")

y1 = -1*(w[0]+w[1]*xmin)/w[2]
y2 = -1*(w[0]+w[1]*xmax)/w[2]
ax.plot([x1, x2] ,[y1, y2], 'y--', label="decision boundary(numpy)")

# ax.legend(bbox_to_anchor=(1.4, 1.03))
ax.legend(bbox_to_anchor=(1.0, 1.0))
```

![svg](/assets/skl_logistic_regression_files/skl_logistic_regression_21_1.svg)
    


#### モデル評価

ここでは sklearn, seaborn などを使ったモデル評価の方法をいくつか使ってみる。
可視化はおまけ、ということなので、何をやってるか簡単に理解するにとどめておく。

 - confusion matrix や、Recall, Precision などはビデオ講義で学んだ
 - seaborn の [point plot](https://seaborn.pydata.org/generated/seaborn.pointplot.html?highlight=pointplot) では $x$ で指定したクラス分類それぞれに付いて、$y$ の点推定・信頼区間をプロットする. 
   - 今回 estimator には何も指定していないので 母平均の推定を行っている
 - seaborn の [lmplot](https://seaborn.pydata.org/generated/seaborn.lmplot.html) は、回帰+その結果の可視化をしてくれるもの。
   - 今回は logistic に True を指定しているため、ロジスティック回帰を行うが、線形回帰や多項式回帰なども行えるようだ。
   - hue,col の指定は そのままでは回帰に使えない, カテゴリ変数を扱うためのもの? グラフを分けたり、色を分けたりする. (この機能を使うことをさして faceted と言っているもよう)
   - 薄い範囲は信頼区間(デフォルト95%)を示すようだ。
   




```python
# データの分割 (本来は同じデータセットを分割しなければいけない。(簡易的に別々に分割している)
from sklearn.model_selection import train_test_split
traindata1, testdata1, trainlabel1, testlabel1 = train_test_split(data1, label1, test_size=0.2)
traindata2, testdata2, trainlabel2, testlabel2 = train_test_split(data2, label2, test_size=0.2)

# 学習
eval_model1=LogisticRegression()
predictor_eval1=eval_model1.fit(traindata1, trainlabel1[:,0]).predict(testdata1)

eval_model2=LogisticRegression()
predictor_eval2=eval_model2.fit(traindata2, trainlabel2[:,0]).predict(testdata2)

# 評価
print("")
print("score:")
print("(model 1, train) = %f" % (eval_model1.score(traindata1, trainlabel1)))
print("(model 1, test)  = %f" % (eval_model1.score(testdata1,testlabel1)))
print("(model 2, train)  = %f" % (eval_model2.score(traindata2, trainlabel2)))
print("(model 1, test)  = %f" % (eval_model2.score(testdata2,testlabel2)))

from sklearn import metrics
print("")
print("metrics:")
print("model1")
print(metrics.classification_report(testlabel1, predictor_eval1))
print("model2")
print(metrics.classification_report(testlabel2, predictor_eval2))
```

```
    score:
    (model 1, train) = 0.653090
    (model 1, test)  = 0.715084
    (model 2, train)  = 0.761236
    (model 1, test)  = 0.821229
    
    metrics:
    model1
                  precision    recall  f1-score   support
    
               0       0.70      0.97      0.81       113
               1       0.86      0.27      0.41        66
    
        accuracy                           0.72       179
       macro avg       0.78      0.62      0.61       179
    weighted avg       0.76      0.72      0.67       179
    
    model2
                  precision    recall  f1-score   support
    
               0       0.82      0.91      0.86       108
               1       0.83      0.69      0.75        71
    
        accuracy                           0.82       179
       macro avg       0.82      0.80      0.81       179
    weighted avg       0.82      0.82      0.82       179
```    



```python
def plot_confusion_matrix(cm,title):
    fig = plt.figure(figsize = (7,7))
    plt.title(title)
    sns.heatmap(
        cm,
        vmin=None,
        vmax=None,
        cmap="Blues",
        center=None,
        robust=False,
        annot=True, fmt='.2g',
        annot_kws=None,
        linewidths=0,
        linecolor='white',
        cbar=True,
        cbar_kws=None,
        cbar_ax=None,
        square=True, ax=None, 
        #xticklabels=columns,
        #yticklabels=columns,
        mask=None)
    return 


from sklearn.metrics import confusion_matrix
confusion_matrix1=confusion_matrix(testlabel1, predictor_eval1)
confusion_matrix2=confusion_matrix(testlabel2, predictor_eval2)


plot_confusion_matrix(confusion_matrix1, "model1")
plot_confusion_matrix(confusion_matrix2, "model2")


```


    
![svg](/assets/skl_logistic_regression_files/skl_logistic_regression_24_0.svg)
    



    
![svg](/assets/skl_logistic_regression_files/skl_logistic_regression_24_1.svg)
    



```python
#Paired categorical plots

import seaborn as sns
sns.set(style="whitegrid")

# Load the example Titanic dataset
titanic = sns.load_dataset("titanic")

# Set up a grid to plot survival probability against several variables
g = sns.PairGrid(titanic, y_vars="survived",
                 x_vars=["class", "sex", "who", "alone"],
                 size=5, aspect=.5)

# Draw a seaborn pointplot onto each Axes
g.map(sns.pointplot, color=sns.xkcd_rgb["plum"])
g.set(ylim=(0, 1))
sns.despine(fig=g.fig, left=True)

plt.show()
```
実際には warning が表示されていたが、割愛。ローカルPC内のパスが含まれていたため
    
![svg](/assets/skl_logistic_regression_files/skl_logistic_regression_25_1.svg)
    



```python
#Faceted logistic regression

import seaborn as sns
sns.set(style="darkgrid")

# Load the example titanic dataset
df = sns.load_dataset("titanic")

# Make a custom palette with gendered colors
pal = dict(male="#6495ED", female="#F08080")

# Show the survival proability as a function of age and sex
g = sns.lmplot(x="age", y="survived", col="sex", hue="sex", data=df,
               palette=pal, y_jitter=.02, logistic=True)
g.set(xlim=(0, 80), ylim=(-.05, 1.05))
plt.show()
```


    
![svg](/assets/skl_logistic_regression_files/skl_logistic_regression_26_0.svg)
    

