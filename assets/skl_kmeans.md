### ハンズオン


```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import confusion_matrix

from sklearn import cluster, preprocessing, datasets
from sklearn.cluster import KMeans




```


```python
#https://datahexa.com/kmeans-clustering-with-wine-dataset
wine = datasets.load_wine()
print(wine["DESCR"])
```
```
    .. _wine_dataset:
    
    Wine recognition dataset
    ------------------------
    
    **Data Set Characteristics:**
    
        :Number of Instances: 178 (50 in each of three classes)
        :Number of Attributes: 13 numeric, predictive attributes and the class
        :Attribute Information:
     		- Alcohol
     		- Malic acid
     		- Ash
    		- Alcalinity of ash  
     		- Magnesium
    		- Total phenols
     		- Flavanoids
     		- Nonflavanoid phenols
     		- Proanthocyanins
    		- Color intensity
     		- Hue
     		- OD280/OD315 of diluted wines
     		- Proline
    
        - class:
                - class_0
                - class_1
                - class_2
    		
        :Summary Statistics:
        
        ============================= ==== ===== ======= =====
                                       Min   Max   Mean     SD
        ============================= ==== ===== ======= =====
        Alcohol:                      11.0  14.8    13.0   0.8
        Malic Acid:                   0.74  5.80    2.34  1.12
        Ash:                          1.36  3.23    2.36  0.27
        Alcalinity of Ash:            10.6  30.0    19.5   3.3
        Magnesium:                    70.0 162.0    99.7  14.3
        Total Phenols:                0.98  3.88    2.29  0.63
        Flavanoids:                   0.34  5.08    2.03  1.00
        Nonflavanoid Phenols:         0.13  0.66    0.36  0.12
        Proanthocyanins:              0.41  3.58    1.59  0.57
        Colour Intensity:              1.3  13.0     5.1   2.3
        Hue:                          0.48  1.71    0.96  0.23
        OD280/OD315 of diluted wines: 1.27  4.00    2.61  0.71
        Proline:                       278  1680     746   315
        ============================= ==== ===== ======= =====
    
        :Missing Attribute Values: None
        :Class Distribution: class_0 (59), class_1 (71), class_2 (48)
        :Creator: R.A. Fisher
        :Donor: Michael Marshall (MARSHALL%PLU@io.arc.nasa.gov)
        :Date: July, 1988
    
    This is a copy of UCI ML Wine recognition datasets.
    https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data
    
    The data is the results of a chemical analysis of wines grown in the same
    region in Italy by three different cultivators. There are thirteen different
    measurements taken for different constituents found in the three types of
    wine.
    
    Original Owners: 
    
    Forina, M. et al, PARVUS - 
    An Extendible Package for Data Exploration, Classification and Correlation. 
    Institute of Pharmaceutical and Food Analysis and Technologies,
    Via Brigata Salerno, 16147 Genoa, Italy.
    
    Citation:
    
    Lichman, M. (2013). UCI Machine Learning Repository
    [https://archive.ics.uci.edu/ml]. Irvine, CA: University of California,
    School of Information and Computer Science. 
    
    .. topic:: References
    
      (1) S. Aeberhard, D. Coomans and O. de Vel, 
      Comparison of Classifiers in High Dimensional Settings, 
      Tech. Rep. no. 92-02, (1992), Dept. of Computer Science and Dept. of  
      Mathematics and Statistics, James Cook University of North Queensland. 
      (Also submitted to Technometrics). 
    
      The data was used with many others for comparing various 
      classifiers. The classes are separable, though only RDA 
      has achieved 100% correct classification. 
      (RDA : 100%, QDA 99.4%, LDA 98.9%, 1NN 96.1% (z-transformed data)) 
      (All results using the leave-one-out technique) 
    
      (2) S. Aeberhard, D. Coomans and O. de Vel, 
      "THE CLASSIFICATION PERFORMANCE OF RDA" 
      Tech. Rep. no. 92-01, (1992), Dept. of Computer Science and Dept. of 
      Mathematics and Statistics, James Cook University of North Queensland. 
      (Also submitted to Journal of Chemometrics).
```    



```python
df = pd.DataFrame(wine.data, columns=wine.feature_names)
df.head()

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
      <th>alcohol</th>
      <th>malic_acid</th>
      <th>ash</th>
      <th>alcalinity_of_ash</th>
      <th>magnesium</th>
      <th>total_phenols</th>
      <th>flavanoids</th>
      <th>nonflavanoid_phenols</th>
      <th>proanthocyanins</th>
      <th>color_intensity</th>
      <th>hue</th>
      <th>od280/od315_of_diluted_wines</th>
      <th>proline</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>14.23</td>
      <td>1.71</td>
      <td>2.43</td>
      <td>15.6</td>
      <td>127.0</td>
      <td>2.80</td>
      <td>3.06</td>
      <td>0.28</td>
      <td>2.29</td>
      <td>5.64</td>
      <td>1.04</td>
      <td>3.92</td>
      <td>1065.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>13.20</td>
      <td>1.78</td>
      <td>2.14</td>
      <td>11.2</td>
      <td>100.0</td>
      <td>2.65</td>
      <td>2.76</td>
      <td>0.26</td>
      <td>1.28</td>
      <td>4.38</td>
      <td>1.05</td>
      <td>3.40</td>
      <td>1050.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>13.16</td>
      <td>2.36</td>
      <td>2.67</td>
      <td>18.6</td>
      <td>101.0</td>
      <td>2.80</td>
      <td>3.24</td>
      <td>0.30</td>
      <td>2.81</td>
      <td>5.68</td>
      <td>1.03</td>
      <td>3.17</td>
      <td>1185.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>14.37</td>
      <td>1.95</td>
      <td>2.50</td>
      <td>16.8</td>
      <td>113.0</td>
      <td>3.85</td>
      <td>3.49</td>
      <td>0.24</td>
      <td>2.18</td>
      <td>7.80</td>
      <td>0.86</td>
      <td>3.45</td>
      <td>1480.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>13.24</td>
      <td>2.59</td>
      <td>2.87</td>
      <td>21.0</td>
      <td>118.0</td>
      <td>2.80</td>
      <td>2.69</td>
      <td>0.39</td>
      <td>1.82</td>
      <td>4.32</td>
      <td>1.04</td>
      <td>2.93</td>
      <td>735.0</td>
    </tr>
  </tbody>
</table>
</div>
~~~



```python
df.describe()
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
      <th>alcohol</th>
      <th>malic_acid</th>
      <th>ash</th>
      <th>alcalinity_of_ash</th>
      <th>magnesium</th>
      <th>total_phenols</th>
      <th>flavanoids</th>
      <th>nonflavanoid_phenols</th>
      <th>proanthocyanins</th>
      <th>color_intensity</th>
      <th>hue</th>
      <th>od280/od315_of_diluted_wines</th>
      <th>proline</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>178.000000</td>
      <td>178.000000</td>
      <td>178.000000</td>
      <td>178.000000</td>
      <td>178.000000</td>
      <td>178.000000</td>
      <td>178.000000</td>
      <td>178.000000</td>
      <td>178.000000</td>
      <td>178.000000</td>
      <td>178.000000</td>
      <td>178.000000</td>
      <td>178.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>13.000618</td>
      <td>2.336348</td>
      <td>2.366517</td>
      <td>19.494944</td>
      <td>99.741573</td>
      <td>2.295112</td>
      <td>2.029270</td>
      <td>0.361854</td>
      <td>1.590899</td>
      <td>5.058090</td>
      <td>0.957449</td>
      <td>2.611685</td>
      <td>746.893258</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.811827</td>
      <td>1.117146</td>
      <td>0.274344</td>
      <td>3.339564</td>
      <td>14.282484</td>
      <td>0.625851</td>
      <td>0.998859</td>
      <td>0.124453</td>
      <td>0.572359</td>
      <td>2.318286</td>
      <td>0.228572</td>
      <td>0.709990</td>
      <td>314.907474</td>
    </tr>
    <tr>
      <th>min</th>
      <td>11.030000</td>
      <td>0.740000</td>
      <td>1.360000</td>
      <td>10.600000</td>
      <td>70.000000</td>
      <td>0.980000</td>
      <td>0.340000</td>
      <td>0.130000</td>
      <td>0.410000</td>
      <td>1.280000</td>
      <td>0.480000</td>
      <td>1.270000</td>
      <td>278.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>12.362500</td>
      <td>1.602500</td>
      <td>2.210000</td>
      <td>17.200000</td>
      <td>88.000000</td>
      <td>1.742500</td>
      <td>1.205000</td>
      <td>0.270000</td>
      <td>1.250000</td>
      <td>3.220000</td>
      <td>0.782500</td>
      <td>1.937500</td>
      <td>500.500000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>13.050000</td>
      <td>1.865000</td>
      <td>2.360000</td>
      <td>19.500000</td>
      <td>98.000000</td>
      <td>2.355000</td>
      <td>2.135000</td>
      <td>0.340000</td>
      <td>1.555000</td>
      <td>4.690000</td>
      <td>0.965000</td>
      <td>2.780000</td>
      <td>673.500000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>13.677500</td>
      <td>3.082500</td>
      <td>2.557500</td>
      <td>21.500000</td>
      <td>107.000000</td>
      <td>2.800000</td>
      <td>2.875000</td>
      <td>0.437500</td>
      <td>1.950000</td>
      <td>6.200000</td>
      <td>1.120000</td>
      <td>3.170000</td>
      <td>985.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>14.830000</td>
      <td>5.800000</td>
      <td>3.230000</td>
      <td>30.000000</td>
      <td>162.000000</td>
      <td>3.880000</td>
      <td>5.080000</td>
      <td>0.660000</td>
      <td>3.580000</td>
      <td>13.000000</td>
      <td>1.710000</td>
      <td>4.000000</td>
      <td>1680.000000</td>
    </tr>
  </tbody>
</table>
</div>
~~~



```python
df.isnull().any()
```



```
    alcohol                         False
    malic_acid                      False
    ash                             False
    alcalinity_of_ash               False
    magnesium                       False
    total_phenols                   False
    flavanoids                      False
    nonflavanoid_phenols            False
    proanthocyanins                 False
    color_intensity                 False
    hue                             False
    od280/od315_of_diluted_wines    False
    proline                         False
    dtype: bool
```



```python
X=wine.data
y=wine.target

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3, random_state=55)
```


```python
def species_label(theta):
    if theta == 0:
        return wine.target_names[0]
    if theta == 1:
        return wine.target_names[1]
    if theta == 2:
        return wine.target_names[2]

print(wine.target_names)
y_label = [species_label(theta) for theta in wine.target]
df['species']=y_label
```

    ['class_0' 'class_1' 'class_2']



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
```

#### k-nn で分類

k-nn は学習のプロセスは無いが、識別のために事前にクラスが判明しているサンプルが必要。上で分割したデータを使って試してみる。
つまり、X_train, y_train の情報を用いて、X_test の各サンプルのクラスを判別する。




```python
from scipy import stats

def distance(x1, x2):
    return np.sum((x1 - x2)**2, axis=1)

def knc_predict(n_neighbors, x_train, y_train, X_test):
    y_pred = np.empty(len(X_test), dtype=y_train.dtype)
    for i, x in enumerate(X_test):
        distances = distance(x, X_train)
        nearest_index = distances.argsort()[:n_neighbors]
        mode, _ = stats.mode(y_train[nearest_index])
        y_pred[i] = mode
    return y_pred

# X_train, y_train のデータをつかって y_test の識別を行う
n_neighbors = 3
y_pred = knc_predict(n_neighbors, X_train, y_train, X_test)

print(metrics.classification_report(y_test, y_pred))
plot_confusion_matrix(confusion_matrix(y_test,y_pred),"knn")

```

```
                  precision    recall  f1-score   support
    
               0       0.87      1.00      0.93        13
               1       0.74      0.74      0.74        23
               2       0.69      0.61      0.65        18
    
        accuracy                           0.76        54
       macro avg       0.76      0.78      0.77        54
    weighted avg       0.75      0.76      0.75        54
```    



    
![svg](/assets/skl_kmeans_files/skl_kmeans_10_1.svg)
    


#### k-means クラスタリング

k-means クラスタリングは、教師なしのクラスタリング手法として知られるので、まずはデータ全体に対してクラスタリングを試してみた。

numpy 実装、sklearn 実装ともに 同様のクラスタリング結果が得られたようだ (k-means により割り当てられるクラスタの番号が、正解データの class 番号と一致する保証は全くないが、対角成分が最大になるように何度か実行した)



```python
# skelearn で　k-means
model = KMeans(n_clusters=3)
labels = model.fit_predict(X) 

df['species_sklearn'] = labels
pd.crosstab(df['species_sklearn'], df['species'])

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
      <th>species</th>
      <th>class_0</th>
      <th>class_1</th>
      <th>class_2</th>
    </tr>
    <tr>
      <th>species_sklearn</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>46</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>50</td>
      <td>19</td>
    </tr>
    <tr>
      <th>2</th>
      <td>13</td>
      <td>20</td>
      <td>29</td>
    </tr>
  </tbody>
</table>
</div>
~~~



```python
# numpy 実装の k-means
# distance 関数は k-nn で定義したものがそのまま使い回せる
#  毎イタレーションで 配列を確保して速度的には効率が悪いが目をつぶる

def predict(x, centers, n_clusters):
    D = np.zeros((len(x), n_clusters))
    for i, x in enumerate(x):
        D[i] = distance(x, centers)
        cluster_index = np.argmin(D, axis=1)
    return cluster_index

def fit(x, cluster_index, n_clusters):
    centers = np.zeros( (n_clusters,x.shape[1]) )
    # 3) 各クラスタの平均ベクトル（中心）を計算する
    for k in range(n_clusters):
        index_k = cluster_index == k
        centers[k] = np.mean(x[index_k], axis=0)
    return centers

n_clusters = 3
iter_max = 100

# 1) 各クラスタ中心の初期値を設定する ここでは ランダム
centers = X[np.random.choice(len(X), n_clusters, replace=False)]
for _ in range(iter_max):
    prev_centers = np.copy(centers)

    # 2) 各データ点に対して、各クラスタ中心との距離を計算し、最も距離が近いクラスタを割り当てる
    cluster_index = predict(X, centers, n_clusters)

    # 3) 各クラスタの平均ベクトル（中心）を計算する
    centers = fit(X, cluster_index, n_clusters)

    # 4) 収束するまで2, 3の処理を繰り返す    
    if np.allclose(prev_centers, centers):
        break

# 
df['species_np'] = predict(X,centers,n_clusters)
pd.crosstab(df['species_np'], df['species'])

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
      <th>species</th>
      <th>class_0</th>
      <th>class_1</th>
      <th>class_2</th>
    </tr>
    <tr>
      <th>species_np</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>46</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>50</td>
      <td>19</td>
    </tr>
    <tr>
      <th>2</th>
      <td>13</td>
      <td>20</td>
      <td>29</td>
    </tr>
  </tbody>
</table>
</div>
~~~



```python
# クラスタリング結果の比較
pd.crosstab(df['species_np'], df['species_sklearn'])

# 各クラスのセントロイド間の ユークリッド距離
print(distance(model.cluster_centers_, centers))
```

    [2.20190800e-28 1.29254120e-26 2.32245581e-28]


ちょっとナンセンスな気もするが、あえて教師あり学習することも可能なのでやってみる。


```python
centers_train = fit(X_train, y_train, n_clusters)
y_pred_kmeans = predict(X_test, centers_train, n_clusters)

print(metrics.classification_report(y_test, y_pred_kmeans))
plot_confusion_matrix(confusion_matrix(y_test,y_pred_kmeans),"k-means")

```

```
                  precision    recall  f1-score   support
    
               0       0.86      0.92      0.89        13
               1       0.74      0.87      0.80        23
               2       0.77      0.56      0.65        18
    
        accuracy                           0.78        54
       macro avg       0.79      0.78      0.78        54
    weighted avg       0.78      0.78      0.77        54
```



    
![svg](/assets/skl_kmeans_files/skl_kmeans_16_1.svg)
    
