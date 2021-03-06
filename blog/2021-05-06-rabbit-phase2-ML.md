@def title = "ラビットチャレンジ: Phase.2 機械学習"
@def author = "kyokke" 
@def tags = [ "Deep-Learning", "Rabbit-Challenge" ]


# ラビットチャレンジ: Phase.2 機械学習 

本ページはラビットチャレンジの、
Phase.2 機械学習のレポート提出を兼ねた受講記録です。
提出指示に対して、下記の方針でまとめました

1. 動画講義の要点まとめ
   - 自分がその手法を思い出して実装するのに必要な最低限の情報 (モデル定義・評価関数・解放の着想があれば十分かなぁ.. )
   - 講義で口頭補足されたトピックなどあれば。

2. 実装演習(ハンズオン実行)
   - 基本的には scikit-learn のノートの内容に準じる(より実践的?っぽいので)
     - 対応する numpy実装を見て、sikit-learn側に応用できそうなことがあればやってみる (コアアルゴリズム部分を numpy 実装に置き換えてみるとか) 
     - その他パラメータのチューニングなど。その他、何か追加でやりたくなったら、適当に追加。
   - ノートブックを jupyter nbconvert で markdown に変換しその抜粋を掲載
     - markdown /コード/出力 のそれぞれに関して、blog上での表示を考慮してあとから微調整を行う
     - 手法に関する一般的な説明・数式展開などは、要点のまとめ側に移動する場合も　

## 目次
\toc

## 各論前の講義内容　

 - プロローグ ( なぜ 非ディープラーニングの機械学習の勉強をするのか? )
   - 数学的な背景を元にした機械学習の基礎を抑えていないエンジニアは、フレームワークを使って機械学習モデルを組める「程度」の人材にしかならない

   - 機械学習のモデリングプロセスをしっかりと抑えることが重要
     1. 問題設定
        - 最終的な使われ方をイメージしよう
        - 必ずしも機械学習を使う必要は無く、ルールベースで解けるならそっちの方が楽
          - 技術的負債になりやすい
            - 自分が開発した技術を運用チームに移管するとして、移管先に専門知識を持った人間がいなければSLA(Service Level Agreement)を担保できない
          - テストしにくい・考慮すべきことが多い
          - データ集めるの大変
     2. データ選定
        - GIGO(garbage in garbage out) = ゴミを突っ込んだらゴミが出てくる
          - ex. 集めたデータに(意図しない)バイアスがかかっているなど。
     3. データの前処理
        - 開発時間の殆どは、このプロセスに費やされるといっても過言でない
        - 実務経験がモノをいうところ。練習のチャンスがない人は、kaggleなどやるといい
     4. モデルの選定
        - ディープラーニング(で扱われるモデル)も、機械学習モデルの一部に過ぎず、このプロセスの具体的な作業が異なるだけ
     5. モデルパラメータの決定
     6. モデルの評価

 - ルールベースモデルと機械学習モデル
   - 技術者として、機械学習とは?と聞かれたら何らかの形で答えられるようになっておくこと
     - 講師の体感では、機械学習ってなに? と聞かれるよりも、人工知能って何? と聞かれることの方が多いらしい 

## 線形回帰モデル 

### 動画講義メモ(要点のまとめ)

 - 回帰問題 = 入力から出力を予測する問題。線形回帰は、回帰式が線形。
   - 線形回帰モデルの形 (線形とは?) 
     - ざっくり説明 = 入出力が比例関係である (直線・平面・超平面)
     - $ y = w_0 + \sum_{i=1}^{n-1} w_i x_i $
       - $ = \sum_{i=0}^{n-1} w_i x_i $　
       - $ = \bm{w}^T \bm{x} $  ただし $ \bm{w}^T = (w_0, w_1, ... , w_{n-1})$, $ \bm{x}^T = (1, x_1, ... , x_{n-1})$ 
     - 記法としては ベクトルが便利、訳わからなったら sigma 記法にする
       - このベクトル/行列と 要素ごとの記法の行き来ができることがポイント
       - この後の記述とあわせて係数は $w$ にした
    - 出力について
      - 連続値とスライドに書かれているが、離散値でもいい
        - 例: 諸条件から、競馬の順位を予想する
          - cf. "vapnik の原理" には反する
            - 順位 = 大小関係 がわかるだけでいいのに、もっと難しい回帰問題を解くべきではないという話。
            - [論文](https://www.ism.ac.jp/editsec/toukei/pdf/58-2-141.pdf)
      - スカラーと書いてあるが、多次元のベクトルにしてもいい(マルチタスク学習など)
      - データ分割・学習
        - 未知のデータに対する予測精度をあげたいので、テスト用のデータと、学習用のデータは分ける
        - 最尤法による解とMSEの最小化は等価と書いてあったが、それは誤差を確率変数としたとき、これが正規分布に従う場合では? (他の分布なら、別の解でてくる場合もあるよね?) 
    - 誤差について
      - 必ずしも偶発的な誤差だけではなく、モデル化誤差もある
        - y = 来店者数、x = 気温 で予測ができる? いや、曜日にも影響する。このとき、曜日の項の影響は誤差になる
    - 未知数の数と方程式の数について
      - 今、重みw は m+1 次元なので、基本的には、 m+1 以上の方程式がないと厳しい
        - データの方が少ないケースを扱う手法もあるが、それは advanced 
        - DLの場合パラメータがたくさんあるので、データが必要
    - パラメータ推定方法 : 最小二乗法
      - $ \mathrm{MSE_{train}} = \frac{1}{n} \sum_{i=1}^{n_{train}} (\hat{y_i}^{(train)}-y_i^{(train)})^2 = \sum \epsilon^2 :=  J(w) $
        - MSE = mean square error
      - これをパラメータについて微分して、勾配が0になる点を求める
        - $ \bm{\hat{w}}= (X^T X) ^{-1} X^T \bm{y} $  : (train) は省略
          - $\bm{y}$ に 一般化逆行列をかけた形
        - 必ずしも誤差関数として二乗誤差は最適ではないことに注意
          - 基本的にハズレ値の存在にかなり弱くなる
          - Huber loss, Tukey loss とか　を使うとハズレ値に強くなる
       - よく使う ベクトルの微分の形
         - $ \frac{\partial}{\partial \bm{w}} \bm{w} ^T \bm{x} = x$
         - $ \frac{\partial}{\partial \bm{w}} \bm{w}^T A \bm{w} = (A+A^T) \bm{x} = 2A\bm{x}$ (Aが対称行列のとき)
         - 参考図書: Matrix Cook Book 

 - ハンズオンに関するコメント
   - 全部の説明変数を使う必要はないし、使うべきではないことがある
    - 12番目の "アフリカ系アメリカ人居住者の割合" など
   - 現実のデータセットを扱う上では、よくデータをみる必要がある
     - 得られたデータの中に使用すべきでないもの(ハズレ値など) があるかもしれない
     - 要約統計量などを適宜活用
     - pandas で 12行目まで観察 ( 頭の数行みて、怪しい所があったから。最初の5行 CHAS全部0じゃね?)
   - エイヤで学習すると、マイナスの価格が出てしまった件
     - こういう現象が出た時点でおかしい、と気付かなければいけない
     - 何がダメだったんだろう -> 1部屋のケースなどがデータに入ってないんだろうな
       - 外挿問題にDLは基本的には弱い 5~10部屋の範囲のデータから学習したとき、11部屋, 2部屋の時の予測は上手く行かない
   - scikit-learn や tensorflow で動かすのは小学生でもできる。なんとなく動かすんじゃなくて、数式に対する理解を持つこと
   - 多重回帰の場合も試しに実行してみてね
     - 学習されたモデルを評価する
      - 部屋を増やしてみる(価格増えるはず)
      - 犯罪率増やしてみる(価格下がるはず)

\textinput{skl_regression.md}


## 非線形回帰 

### 動画講義メモ(要点のまとめ)

- モデルの形
  - 線形回帰 $ y = \sum w_i x_i $ で、$ x_i $ の代わりに 非線形関数 $\phi_i (\bm{x})$ を使う。
    - $\phi_0 = 1, w_0 =1 $
    - $ w $ については線形 (linear-in-parameter) のまま
    - -> 二乗誤差に対する解としてパラメータは求められるってことね
    - 旧動画では、これを基底関数法と呼んでいた
- 未学習(underfitting)・過学習(overfitting)について
  - 多項式近似の例:  4次以上はフィッティング具合ほとんどかわらない
    - 左のグラフ(1,2次ぐらい) -> 未学習
    - 真ん中(4次ぐらい) -> 望ましい状態
    - 右のグラフ(もっと次数をあげた) -> 過学習
  - 訓練誤差とテスト誤差を比較することで、判断ができる
    - cf. 2018年の DL 論文で、過学習可と思いきや、ずっと学習続けてると、またロスが下がり始める現象が報告されていた(double descent)
- 過学習の対策
  1. 学習データを増やす
  2. 不要な基底関数を削除して表現力を抑止 
    - 情報量基準などを使う
    - 愚直にやると、組合せ爆発を起こすので大変
  3. 正則化法を利用して表現力を抑止
    - モデルの複雑さに伴って、その値が大きくなるようなペナルティを関数に課す
      - 今回の場合は重みパラメータの大きさ
        - 4次まで誤差を十分小さくできており、7次以上の項による誤差低減効果はほぼない。この状態で、正則化を行えば、7次以上の重みパラネータが小さく抑えられるはず。
      - $ min. \  MSE + \lambda R(\bm{w}) $ を解けば良い。( $R(\bm{w})$ が正則化項)
        - KKT条件によって、$ min. \ \mathrm{MSE} \ s.t. \ R(\bm{w}) \le r $ を一つの目的関数の最小化問題に変換
          - 今回の場合、必ずしも上記不等式制約を満たす必要はない(なるべく小さくすればいいだけ)なので上記の議論で問題ない
          - $r$ は $\bm{w}$ によらないので目的関数からは削除されてる 
     - 正則化項の例
       - Ridge推定量　- 正則化項 にL2ノルムを使う (パラメータを0に近づける)
       - Lasso推定量  - 正則化項 にL1ノルムを使う (スパース推定) 
     - 正則化項にかかるパラメータは hyper parameter 　
 - モデル選択
    - 評価するときには、学習に使ってないデータ(検証データ)に対する評価値を見る
    - ホールドアウト法
      - 学習に使うデータと、検証に使うデータを固定する
        - 大量にデータが手元にあるときはこれをやることがおおい
      - 注意点 : 
        - 検証用データに "はずれ値"が入ってしまうと、はずれ値にフィットするモデルが選ばれる、ということが起きる
    - 交差検証法 (クロスバリデーション) 
      - データを学習用と評価用に分割するのだが、その分割の仕方をローテーションする。例えば、5分割したうちの1つが検証、残りが学習、みたいな
      - 1つのモデルに対して、 5回の評価が行われる。精度の平均値 CV値で評価する
      - Q&A 精度ってどうやって計算するの? -> 基本的には 学習時のloss をそのまま使う
      - 基本的には交差検証法の方の値を報告する
 - ハイパーパラメータの調整をどうしよう? という話　
   - グリッドサーチは 実装の練習はしてもいいと思うけど、実践的にはあまり使わない。
   - ベイズ最適化(BO)が最近だとよく使われる? 
     - PFN の optunaなど

\textinput{skl_nonlinear_regression.md}

## ロジスティック回帰
### 動画講義メモ(要点のまとめ)

- 参考: google AI blog の記事
  - DNNじゃなくてあえてlogistic 回帰使いました
  - [using ML to Predict Parking Difficulty (2017年 2月)](https://ai.googleblog.com/2017/02/using-machine-learning-to-predict.html)

 - 分類問題に対するアプローチ
    1. 識別的アプローチ
      - $ p(C_k | x ) $ を直接モデル化する方法
        - logistic 回帰は、これ。
      - 識別関数をつかう方法
        - $f(x) >  0$ なら　$C = 1$,  $f(x) < 0$ なら　$C = 0$ みたいな
        - SVMはこれ
    2. 生成的アプローチ
      - $ p(C_k) $ と $ P(x|C_k)$ を model化して、その後 Bayesの定理を用いて、$ p(C_k | x ) $ を求める
        - ハズレ値の検出をもとめたり、生成モデルをつくれたりする
      - ロジスティック回帰での使い方
        - 一般に実数全体をとり得る入力の線形結合に sigmoid 関数をかけて値域を0~1にすることで確率とみなせるようにしている
        - 確率が0.5以上ならば、1, 未満なら、0と予測
      - なんでわざわざ確率にする? -> 判断の保留などが可能
        - 識別関数だと、識別結果をそのまま使用するしかない
 - 最尤推定: ベルヌーイ分布のパラメータ推定
    - データ-> ベールヌーイ分布の計算
      - 1回の試行 $ P(y) = p^y (1-p)^{1-y} $
      - n回の試行で、$y_1,..,y_n$ が同時に起こる確率 $P(y_1,..,y_n;p)=  \prod_{i=0}^{i=n-1} p^{y_i} (1-p)^{1-y_i} $
    - パラメータである p を データ $y_1,..,y_n$ から推定したい。
      - pを色々変えて、$y_1,..,y_n$ が得られる確率が最大になるような $p$ を求める->最尤推定
    - ロジスティク回帰モデルの最尤推定
      - $P(Y=y_1 | \bm{x}_1 ) =p_1^{y_1} (1-p_1)^{1-y_1} $
        - 確率はロジスティック回帰 $ p_1 = \sigma(\bm{w}^T \bm{x}_1)$ で計算できるので、尤度が最大になる $ \bm{w} $ を探す問題
  - 更新式の導出: ひたすら微分。もう何度もやったのでここでは省略. 
    - 対数尤度を使う理由
      - 微分の計算が簡単ってのは、正直どうでもいい (それならそう書かないでくれ)
      - 尤度 : 確率を何度も掛け算する -> 桁落ちの可能性
    - 微分のchane ルールを理解して使いこなすことが重要
 - 確率的勾配法
   - Iteration によってデータを入れ替えることで、パラメータ更新の勾配方向をある程度不規則にして、初期値依存制を低減する効果を狙う
   - ただし、ロジスティック回帰なら単峰制なので局所最適に陥る心配はないのだが
   - [p.70 の参考文献](http://ibis.t.u-tokyo.ac.jp/suzuki/lecture/2018/kyoto/Kyoto_02.pdf)
     - 層を深くすること近似できる関数の自由度がどう増えていくか? とか興味深いがそもそも、どんな意図で貼られているんだろう?
 - モデルの評価
   - 混合行列でマトリックスを使って整理.
     - 行 = モデルの予測結果の Positive/Negative
     - 列 = 検証データの Positive/Negative 
     - 対角成分が 正解(True), 非対角成分が不正解(False)となる
   - FalsePotsitve と FalseNegativeとを分けて評価する
     - Link: http://ibisforest.org/index.php?F%E5%80%A4
     - 検証データ中の Posi/Nega のバランスが取れてないとき何をみてるかわからなくなる
       - 間違い方によって何が嫌かが違う
     - 再現率(Recall): Positiveデータに対する正答率 = TP/(TP+FN)
       - 過剰検出を許容して、ヌケモレのなさを重視する評価
     - 適合率(Precision): モデルがPositiveと予測したものに対する正答率 = TP/(TP+FP)
       - 見逃しを許容して、慎重な検出を評価する
     - F値 : Recall と Precision の調和平均
       - どっちにも振れない微妙なタスクにつかう?
     - 現場でよくやるのは、上記3つ全部並べておくこと 

\textinput{skl_logistic_regression.md}


## 主成分分析 (PCA) 

### 動画講義メモ(要点のまとめ)

 - 次元圧縮(変量の個数を減らす)のに使われる方法
    - 線形変換によって、$N$ 個の $M$ 次元のデータを $J$次元にしたい
      - データ行列から、データを引いた $\bar{X} := X - \bar{x}$ として、 $\bm{s}_j = \bar{X} \bm{a}_j$ が 各データの $j$ 次元めの値となる。
    - 変換前後のデータの散らばり具合を維持できれば、データの情報量が減らないと考える
      - $\bm{a}_j j=1,2,\ldots, J$ は、元のデータの広がりが大きい方向。
      - $\bm{a}_j$ の見つけ方
        - 目的関数: 変換後の分散最大化
          - $\argmax_{a} = \bm{a}_j^T Var(\bar{X}) \bm{a}_j $
        - 制約条件: ノルム一定の制約で解の任意制を取り除く
          - $\bm{a}_j^T \bm{a}_j =1$
        - ラグランジュ未定乗数法をつかって解く -> 解: $Var(\bar{X}) \bm{a}_j = \lambda \bm{a}_j $
          - 取り出すべき軸の方向は、固有ベクトルの方向
          - 射影先の分散は固有値と一致
      - 寄与率 : 圧縮前後の情報のロスを定量化したもの
        - 第 $k$ 成分の寄与率は、 $c_k = \frac{\lambda_k}{\sum_{m=1}^{M} \lambda_m}$
        - 第 1 ~ $k$ 主成分までの累積寄与率は $ r_k = \frac{\sum_{j=1}^{J}\lambda_j}{\sum_{m=1}^{M} \lambda_m}$


\textinput{skl_pca.md}

## k-nn/means 法 (レポート提出課題の "アルゴリズム" の単元に該当?)

### 動画講義メモ(要点のまとめ)

 - k近傍法  
   - 分類問題の機械学習手法
   - 最近傍のデータをk個撮ってきて、それらがもっとも多く所属するクラスに識別
     - ということは、2クラス分類なら、k は奇数にするということか
   - k はハイパパラメータ
     - k を大きくすると決定境界がなめらかになる(多すぎると過学習気味)
 - k-平均法(k-means) 
   - 教師なし学習, クラスタリングを行う手法
   - アルゴリズム
      1. 各クラスタについて、中心の初期値を設定
      2. 各データ点に対して、各クラスタ中心との距離を計算し、最も距離が近いクラスタを割り当てる
      3. 各クラスたの平均ベクトルを計算しあらたな中心とする
      4. 収束するまで2,3の処理を繰り返す
   - 初期値重要
     - 初期値を変えるとクラスタリング結果も変わりうる
       - 初期値が離れていたほうがいい?
       - cf. k-means++ (試験でも出たらしい)
     - クラスタ数 k を変えてももちろん結果が変わる

\textinput{skl_kmeans.md}


## サポートベクターマシン(SVM)

### 動画講義メモ(要点のまとめ)

 - 2クラス分類を解くための教師あり学習の手法
   - 基本の手法
     - 線形判別を前提に境界面(識別関数)を直接的に求める
       - $ t = \mathrm{sign} ( \bm{w}^T \bm{x} + b)$ の取る値 +1,-1 が分類クラスに対応
     - 問題設定: マージン(境界面と境界面に最も近い点の距離)が最大になるケース汎化性能が高い、という仮定で境界面を決める
       - ということは、最終的に最近接点(これをサポートベクターと呼ぶ) のみが学習に必要ということになる
       - $\bm{w}, b$ について "最小化"すべき目的関数: $\frac{1}{2} ||\bm{w} ||^{2}$
         - マージン最大の時、境界面から各クラスの最近接点までの距離は等しくなる (どちらかに偏ると最大でなくなる)
         - 最近接点に対しては、$t_i (\bm{w}^T \bm{x} + b) = 1$ が成立
           - 右辺1 とすることで、$\bm{w}, b$ のスケールの任意性が取り除かれる
         - 上記2点から マージンは、$\frac{1}{|| \bm{w} ||}$ と分かる (境界面の法線ベクトル $\bm{w}$ への射影を使う)
         - 逆数にすれば最小化問題。問題が解きやすいように 2乗した上で 1/2 しておく 
       - 制約条件:  $ t_{i}(\bm{w}^T\bm{x}_{i} + b) \ge 1 \quad (i=1, 2, \cdots, n)$
         - この制約条件は、問題の定義から自明
     - 主問題 <-> 双対問題 (結果だけ)
       - 上記の問題設定は解きにくいので問題を解きやすい形に変換する。ここでは結果だけしめすと..
         - $\bm{a}$ について"最大化" すべき目的関数 : $\tilde{L}(\bm{a}) = \sum_{i=1}^{n} a_{i} - \frac{1}{2} \sum_{i=1}^{n} \sum_{j=1}^{n} a_{i} a_{j} t_{i} t_{j} \bm{x}_i^T \bm{x}_i = \bm{a}^{\mathrm{T}} \bm{1} - \frac{1}{2} \bm{a}^{\mathrm{T}} H \bm{a} $
         - 制約条件: $\bm{a}^T \bm{t} = 0, \quad a_i \ge 0 \ (i=1,2,\ldots n) $
         - $\bm{a}$ と $\bm{w}, b$ の関係: 
           - $ \bm{w} = \sum_{i=1}^{n} a_{i} t_{i} \bm{x}_{i}$
           - 境界面に最も近い点 $s$ に対して、$ b =\frac{1}{t_i} (t_s - \bm{w}^T \bm{x_s})$
         - 主問題と双対問題の変換は、ラグランジュ, KKT条件 等を用いた議論が必要だが、"要点のまとめ"としては割愛
       - 数値解法 (ざっくりの理解)
         - 基本的には勾配法
         - 双対問題の目的関数の勾配つかって更新 (最大化)
           - $\bm{a} \leftarrow \bm{a} + \eta_{1} (\bm{1} - H \bm{a})$
         - 等式制約条件の左辺=0になるよう、こっちはこっちで勾配使って更新 (最小化)
           - $\bm{a} \leftarrow \bm{a} - \eta_{2} (\bm{a}^{\mathrm{T}} \bm{t}) \bm{t}$
         - 更新の途中で、$a_i \ge 0$ を満たさない場合は、無理やり0に射影する。
           - 非負制約は凸な集合なので、そんな悪いことは怒らないはず..
   - カーネル関数を使って判別関数を曲面にする
     - 線形判別できない場合も、$\phi(\bm{x})$ と予め変換した別の特徴量空間に飛ばせばうまくいく?
       - パット思いつくのは 単位円が境界となっているような場合? $r = \phi(\bm{x}) = x_1^2 + x_2^2$ と原点からの距離に変換してあげれば、$ r = 1 $ というめちゃくちゃシンプルな直線で判別できる
       - $\bm{\tilde{x}}_i = \phi(\bm{x}_i)$ と変換してあげれば、定式化は基本のSVMと同じになるはず     
     - 式を展開していくと、必ず $\phi(\bm{x}_i)^T \phi(\bm{x}_j)$ とまとまった形で現れる。これを 2サンプル間の類似度を計算する"カーネル関数" として使う。
       - カーネルのサイズは、$\phi$ の次元数によらず、サンプル数で演算量が抑えられる
       - $\phi(x)$ の設計・計算経由することなく、直接カーネル関数を扱える
         - RBFカーネルに対応する $phi$ は無限次元! 
   - ソフトマージンSVM
     - 境界面できれいに2クラス分類できないときに、はみ出てしまうサンプルがあることを許容する
     - はみ出し具合: $\xi_i = \mathrm{max} ( 1 - t_i (\bm{w}^T \bm{x}_i + b), 0 )$ として これをペナルティ項として元の目的関数に加える
     - 元の問題設定:
       - $\text{min}_{\bm{w}, b} \qquad \frac{1}{2} || \bm{w} ||^{2} + C \sum_{i=1}^{n} \xi_{i}$
         - ペナルティ項の重み係数 C はハイパパラメータ
       - $\text{subject to} \qquad  t_{i}(\bm{w}^T \bm{x}_{i} + b) \ge 1 - \xi_{i} \quad (i=1, 2, \cdots, n)$
     - 双対問題
       - 基本は通常のSVMと同じ。ただし、
       - $a$ の非負制約が $0 \le \bm{a}_i \le C \quad (i=1,2,\ldots,n) $ になる


\textinput{np_svm.md}


### 双対問題の導出 (通常のSVMのケースについて)


 - 自分もSVMよく覚えてなかったので、ざっくり思い出したい
 - ハンズオンの説明でも双対問題への変換の説明はすっ飛ばされているように感じる
 - SVMのケースに特化して、納得した気になるだけならそこまで難しくなさそう

ということで、厳密性 < 直感的理解 重視で、自分なりの納得の仕方を記す。

ラビットチャレンジの同期生で、早々と Stage.2 を終えた方はたくさんいるようだが、双対問題がどうやって導かれたかわからん、という人が一定数いるようなので、参考になれば幸い。(厳密に理解したい人は教科書をちゃんと読むのがいいと思います)


最小化問題と不等式制約の２つを分けて扱うのは大変なので、
「不等式制約のことは一旦わすれて、最小化問題を解いていたら、勝手に不等式制約もみたされていた」という状況を作れると嬉しい。
「目的関数を小さくしつつ、こういう性質を満たして欲しい」みたいな解を探す方法は色々あるが、(ラビットチャレンジ勢的に) 最初に思いつくのは ペナルティ項の追加(正則化)ですよね。不等式制約が満たされないと大きな値をとるような関数$f(\bm{w},b) $を足してやればよさそう。

$$ J(\bm{w},b) = \frac{1}{2} ||\bm{w} ||^{2} + f(\bm{w},b)$$ 

ただし、今まで扱ってきた正則化では、「なるべくノルムが小さくなるように」でヨカッタのだが、不等式制約は絶対に満たさなければいけないので、下記注意が必要。

- 不等式制約が満たされているときに $f(\bm{w},b)$ が何らかの値を持ってしまうとすると、
この新たな目的関数の勾配は、元の目的関数 $\frac{1}{2} ||\bm{w} ||^2$ をもっとも減少させる勾配とは少しズレた方向を向くことになって問題だ。
つまり、不等式制約を満たす時、$f=0$でなければならない。
- また 制約を満たさないペナルティに対して、制約を犯したときの第一項目が減少量が大きいければ、不等式制約を満たす保証ができない。
つまり、不等式制約を満たさないとき、$f=+\infty$ でないとならない。


このような性質をもった関数はあるだろうか? ぱっと考え着くのは、微分不可能・不連続点のある関数だが、これを扱うのは数学的に難しそう。
こういうちょっと特殊な関数の振る舞いを、再現するときは、新しい変数・次元を増やしてやるとうまくいくことがある。(カーネルSVMもそうだよね)
ここでは、新しい"非負の" 変数を導入して、それに対する max をとる関数として定義してみよう。

$ f(\bm{w},b) = \mathrm{max}_{\bm{a}}  \sum_i a_i \bigl(-(t_i (\bm{w}^T \bm{x}_i + b) -1)\bigr) $

- 不等式制約を満たすとき つまり $(t_i (\bm{w}^T \bm{x}_i + b) -1) \ge 0 $ の時、 $a_i$ と掛け算する値が負なので、最大化するには $a_i=0$　で $f=0$  ※ これKKT条件 等式制約に一致する
- 不等式制約を満たさないとき つまり $(t_i (\bm{w}^T \bm{x}_i + b) -1) < 0 $ の時　$a_i$ と掛け算する値が生なので、最大化するには $a_i \rightarrow \infty$ で $f=\infty$ 

うん、よさそうだ。ということで解くべき問題は、

$$ \mathrm{min}_{\bm{w},b}  \biggl( \frac{1}{2} ||\bm{w} ||^{2} +　\mathrm{max}_{\bm{a}>0}  \sum_i a_i \bigl(-(t_i (\bm{w}^T \bm{x}_i + b) -1)\bigr) \biggr) \qquad a_i \ge 0 \quad (i=1,2,\ldots) $$ 

ということになるが、$\bm{w}, b, \bm{a}$ は本来独立に動かせる変数で、第一項目に $\bm{a}$ は含まれないので、

$$ \mathrm{min}_{\bm{w},b} \mathrm{max}_{\bm{a}>0} 　 \biggl( \frac{1}{2} ||\bm{w} ||^{2} + \sum_i a_i \bigl(-(t_i (\bm{w}^T \bm{x}_i + b) -1)\bigr) \biggr) \qquad a_i \ge 0 \quad (i=1,2,\ldots) := \mathrm{min}_{\bm{w},b} \mathrm{max}_{\bm{a}>0} \tilde{L}(\bm{w},b,\bm{a}) $$ 

と書き換えてもよい。

ここで、内側で $\mathrm{max}_{\bm{a}>0}$ の操作をすることは、より小さな最適値を見つけるための可能性を狭めることにつながるので、一般には
$$
\mathrm{min}_{\bm{w},b} \mathrm{max}_{\bm{a}>0} \tilde{L}(\bm{w},b,\bm{a}) \ge  \mathrm{max}_{\bm{a}>0} \mathrm{min}_{\bm{w},b}  \tilde{L}(\bm{w},b,\bm{a})
$$

の関係が成り立ち、この等号が成立するかどうかは自明ではない。(cf. 弱双対定理) 

しかしながら、今、各パラメータによる$\tilde{L}$ の勾配 = 0 から得られる式は、基本的に各パラメータに対しての1次ないし0次の式。また制約式も線形だ。最大化、最小化を行う順番を変える、というのは、勾配=0の式を使って、そのパラメータを消去する順番が違うということに相当するわけだが、問題がずっと線形のままならば、どういう順番でパラメータを消去していったところで、直感的には最適値が変わることはなさそうである。

ということで、問題は、
$$
\mathrm{max}_{\bm{a}>0} \mathrm{min}_{\bm{w},b}  \tilde{L}(\bm{w},b,\bm{a})
$$
と書き換えられる事になり、ハンズオンの議論につながる。


## 取り組み概要 & 感想

### 取り組みの記録

- 3/31, 4/1: まずはざっと新しい動画を視聴(1.5倍速)しながらメモをとり・微分計算。基本的に動画を途中でストップすることはなかった。約200分で完了。まぁ、問題なさそうだなぁ。という感触だけ得る。

- 4/1~28 : しばらく塩漬け状態.. (GTC21の聴講などをしていたので)

- 4/29 : ハンズオンの実行環境の構築。Collaborately はセル実行のレスポンスが遅いのでローカルに環境を整える。win10上に pyenv + poetry でを仮想環境を作る。30分程度。

- 4/30-5/1 : 線形回帰 旧動画の方をみながらハンズオン。新旧動画をみながら残したメモを適当に要点まとめとしてblog記事にする。その後 ハンズオンのノートブックファイルを markdown 化し、課題提出用blogの記事ty中にinclude。ハンズオンが 45分, 要点まとめや記事の整形が 1h. 

- 5/1 : 非線形回帰。旧動画をみながら、新動画をみたときのメモと見比べていくつかコメント追加し、要点まとめ。その後、ハンズオンを実行。scikit-learn で RBFカーネルといったときには何をさすのか? SVRってなんだ? など調べていたらすこし時間をくった。合計 3h。

- 5/2~5/5 : 非線形回帰と同様の取り組み方で、残りの単元を1日1つずつ仕上げる。各日3h程度なので、合計12hくらい 

- 5/6 : ステージテスト。所要時間は 30 分程度で、結果は14点/15点。ハヤトチリしました。。スタートテストと違って制限時間が無いのでリラックスして受けられた。


### 感想ほか　

講義動画について。旧動画と新動画でけっこう趣が違う。時間に余裕があれば、両方見てもよいのではなかろうか。
 - 新動画
   - 「基本きっちり押さえれば、後の細かい所は頑張れるでしょ」的な解説。講師の男性はもともとは研究寄りの人な印象を受ける
   - 手計算してるところをみせてくれる (ロジスティック回帰の勾配計算とかやったことない人はやりましょう)
   - 実戦的な注意点などは、参考情報が旧動画より多め
   - その他資料に載っていない背景説明などもある。結果的に時間の都合で資料の説明を一部すっとばすことがある
   - 「お前らコレ難しいからわかってねーだろ」的な発言がチラホラ。人によってはイラッとするかも?
  
 - 旧動画のいいところ
   - 基本的に資料に沿った説明をたんたんとしてくれる感じで、 講師の女性はどちらかというとソフトウェア開発寄の人な印象
   - ハンズオンの説明は こちらの方が丁寧


ハンズオンは基本的に、資料/動画 で学習した手法を使ったデータ分析で、自分で編集しなくてもそのまま実行できてしまう。
コードから学ぶことに慣れている人は、買ってに勘所を見つけて得るべきものを得るが、
人によっては、「へーこんな結果になるんだぁ」で終わってしまうんでなかろうか? それだとちょっともったいない。
例えば、大事なところを数カ所穴埋めにして、考えるきっかけにしてもらう、などやっても良い気がした。


テストについての感想。(他の受講生もあとから受ける可能性があるため具体的な内容には触れない)
15問中13問はビデオ講義のどこかででやった内容そのままだった。
単純なミスや早とちりこそあれ、これらの問題に自信を持って回答できないのであれば、しっかり復習をした方がよさそうだ。
残り2問は、ラビットチャレンジで初めて勉強を始めた人にとっては、未知の手法が出題されたように感じるかもしれない。
しかし、この2問のうち先に出題される問題が、その未知の手法自体を問題文で説明してくれているので、問題文の意味をしっかり捉えれば正解は導ける。
また、それができれば、芋づる式に残り1問も計算問題として解けるので、難しすぎるとか、講義内容とのミスマッチなどを感じることは個人的にはなかった。

## 計画の見直し (2021/05/06 時点)

4/18に終わらせるはずだったステージ2が半月遅れてしまった。
実際に学習に費やしている時間はそこまで長くないが、取り組む時間を確保するのが難しい。

これまで ステージテストはスムーズに合格しているので、各ステージうまく理解ができていると思いたい。
この調子ですすめれば、修了テストにかかる時間は少し縮めてもよいだろう。スケジュールを下記の様に修正する。

 - ~2021/2/15  : スタートテスト (2021/02/07完了)
 - ~2021/3/30  : ステージ1      (2021/03/30完了)
 - ~2021/5/6   : ステージ2      (2021/05/06完了)
 - ~2021/5/30  : ステージ3 
 - ~2021/6/27  : ステージ4 
 - ~2021/7/4   : 復習 -> 修了テスト 
 - ~2021/7/15  : Eもぎライト -> 今後の計画具体化 
 - ~2021/7/30  : シラバスの未習箇所の学習 
 - ~2021/8/26  : 全体の復習
 - 2021/8/27,28: E資格 受験 

だんだん 8月に万全を期しての受験が厳しくなってきた気もするが、、淡々と進めていこう。