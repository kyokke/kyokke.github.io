@def title = "ラビットチャレンジ: Stage.4 深層学習 Day3,4"
@def author = "kyokke" 
@def tags = [ "Deep-Learning", "Rabbit-Challenge" ]


# ラビットチャレンジ: Stage.4 深層学習 Day3, 4  

本ページはラビットチャレンジの、
Stage.4 "深層学習 Day3,4" のレポート提出を兼ねた受講記録です。
提出指示を満たすように、下記の方針でまとめました。

1. 動画講義の要点まとめ (Day3,4)
   - 自分が講義の流れを思い出せるようなメモを残す。通常であれば要点として記載すべき図・数式などがあっても、それが自分にとって既知であれば、言葉の説明ですませることもある
2. 実装演習 (Day.3 のみ)
   - 各Sectionで取り上げられた .ipynb or .py ファイルの内容を実行した結果を記載
   - ただし、初学者向けにやや冗長な内容がある場合、抜粋することもある
3. 確認テスト　(Day.3 のみ)
   - 確認テストの解答に相当する内容は、個別の節をもうけず、要点まとめに含めたり、コードに対するコメントとして記載する
     - 確認テストは重要事項だから、出題されているのであって、まとめ/演習と内容がかぶるはず。
     - 事務局がレポートチェックをする時のために、(確認テスト:1-2) のようなタグを付す。１つ目の数字が section番号、２つ目の数字が 「何番目のテストか?」
4. 1~3 をまとめる上で思うことがあれば、考察やコメントも適宜残しておく。


## 目次
\toc

## Day 3 

### 0: CNN の復習 

動画 day 3-1 0:00 ~ 18:00 

 - 内容は割愛
 - (確認テスト:0-1) 入力 5x5 フィルタ 3x3 パディング 1 ストライド 2 のとき出力の画像は 3x3 
   - 5 + 1*2 - (3-1) = 5 に対して、開始点が 1, 3, 5 と動く
  
### Section.1 : 再帰型 ニューラルネットワークの概念

動画 day 3-1 18:00 - 2:56:30 くらい 

 - RNN (Recurrent Neural Network ) の全体像
   - 時系列データを対象とする
     - データの並び順に意味があり、観測間隔が一定, かつ 相互に統計的な依存関係が認められるデータ
       - 音声や、テキストデータなど
         - テキストデータは、単語ひとつひとつを各時刻の観測データとみなす
   - 構造
     - ![rnn](/assets/rnn.jpg)
       - (確認テスト:1-1) RNNのネットワークには、入力層->中間層, 中間層->出力層の他に、前の時刻の中間層->現時刻の中間層の重みがある。
     - ![rnn-math](/assets/rnn-math.jpg)
       - (確認テスト:1-3) $y_1$ の計算式 
          - $y_1 = g(W_{out} z_1+c)$ 
          - $z_1 = f(W z_0 + W_{in} x_1 + b) $
   - example 
     - Simple Network は バイナリ加算(2進数同士の足し算) をRNNでやる toy problem
       - 2進数の各桁を各時刻のサンプルと考えて、下から上に順々に処理する
         - 一桁下からの繰り上がり処理が、中間層の再帰に相当
     - 時系列データの頭からお尻までを全部処理したら 1エポック
       - 
   - 演習チャレンジ
     - ~~~
<figure style="text-align:center;">
<img src="/assets/day3-enshu1.jpg" style="padding:0;width:100%;" alt="#1"/>
<figcaption></figcaption>
</figure>
~~~
     - 回答は(2) 
       - 理由は、"行列W のサイズ (embed_size x 2*embed_size)と矛盾しないのは (2)しかないから" 。
       - 講義ではどういう演算が２つの単語の特徴をうまく統合でるかという論点で解説されているが、そのような解説のアプローチをとるならば 背景知識の説明は構文木でなく embeddingの話などすべきだと思う。
       - 和や積で情報がうまく扱えるか否かは特徴量ベクトルの性質によるので、自然言語処理のこと知らない人がこの話を聞いてもさっぱりで、結局天下りで回答を与えられた印象しか残らない。

 - BPTT (Back Propagation Through time)
   - 誤差逆伝播の復習
     -  (確認テスト:1-2) $z=t^2, t=x+y$ のとき $dz/dx = 2t * 1 = 2(x+y)$
   - RNNのパラメータ最適化で用いる 誤差逆伝播の一種
     - ポイントは、時刻t+1 の誤差が 時刻 t の誤差へ逆伝播するところ
     - 以降、自分がわかりやすいよう、導出過程は自分がわかりやすいよう修正する。
       - MLPはくさるほど計算したが、RNNは昔いっかいやっただけなので、一通り計算しなおしてみる
       - 最初は連鎖律の関係を理解することに注力するため(各ステップの微分でいちいちで次元を気にするのが面倒)に、スカラー前提で計算する 
       - パラメータの更新式をまとめるときにサイズのことを気にした表記にしてみる
   - BPTTの全体像 
     - 評価関数は各時刻$t$でのエラーの和 : $ E = \sum_t E^t $ と考える
     - $ E^t = L(y^t, d^t) $ を $z^t$ について展開　
       - $ = L(g(W_{(out)} z^t + c), d^t) $
       - $ = L(g(W_{(out)} f(W_{(in)}x^t + W z^{t-1} +b ) + c), d^t) $
       - $ = L(g(W_{(out)} f(W_{(in)}x^t + W f(W_{(in)}x^{t-1} + W z^{t-2} + b) +b ) + c), d^t) $
       - ... (以下いくらでも $z^t$ を展開していける)
     - ここからわかること
       - RNNは無限時間の過去情報を持っている (IIRフィルタといっしょ)
       - $W_{(in)}, W$ が 複数箇所に出てくる -> それぞれのW の場所まで、誤差逆伝播する必要がある
   - BPTT の数学的記述(=パラメータ更新式の導出)
     - パラメータ $X$ についてのロスの微分は、$\frac{\partial E}{\partial X} = \sum_t \frac{\partial E^t}{\partial X}$ と表せるので、各時刻の微分を考えればよい。
     - $\frac{\partial E^t}{\partial W_{(out)}} = \frac{\partial E^t}{\partial y^t} \frac{\partial y^t }{\partial v^t} \frac{\partial v^t}{\partial W_{(out)}} =  \frac{\partial E^t}{\partial y^t} g'(v^t ) z_t := \delta^{out,t} z^t $
       - $\delta^{out,t} := \frac{\partial E }{\partial v^t} = \frac{\partial E^t }{\partial v^t} $ と置く。
     - $\frac{\partial E^t}{\partial W } = \delta^{out,t} \frac{\partial v^t}{\partial z^t} \frac{\partial z^t}{\partial W} = \delta^{out,t} W_{(out)} \frac{\partial z^t}{\partial W}$  
       - $ t=1 $ のとき (具体的な計算)
         - $ \frac{\partial z^1}{\partial W } = f'(u^1) \frac{\partial }{\partial W} ( W_{(in)} x_1 + W z^{0} + b) = f'(u^1) z^0 $
         - $\frac{\partial E^1}{\partial W } =\frac{\partial E^1}{\partial z^1 } \frac{\partial z^1}{\partial W } =  \frac{\partial E^1}{\partial z^1 } f'(u^1) z^0 = \delta^t z^0 $
           - $ \delta^t := \frac{\partial E^t}{\partial u^t } =  \frac{\partial E^t}{\partial z^t } f'(u^t) $ と置いた 
       - $ t=2 $ のとき (具体的な計算 -> 解釈)
         - $ \frac{\partial z^2}{\partial W } =  f'(u^2) \frac{\partial }{\partial W} ( W_{(in)} x_2 + W z^{1} + b) $
           - $ = f'(u^2) ( z^{1} + W \frac{\partial z^{1}}{\partial W}) =  f'(u^2) z^{1} + f'(u^2) W \frac{\partial z^{1}}{\partial W})$  
           - 二項現れたのは、高校でも学習する関数の積の微分を適用したに過ぎないのだが、この一項目をあえて、$z^t$ 自体が $W$ の関数だと思わない場合の微分とみなし、 $ \frac{\partial z^{2,+}}{\partial W }$ で書くと、
           - $ \frac{\partial z^2}{\partial W } = \frac{\partial z^{2,+}}{\partial W } + \frac{\partial z^{2}}{\partial z^1 } \frac{\partial z^{1}}{\partial W} $　= (直近の逆伝播経路に起因する項) + (時間をさかのぼった別経路起因の項) と解釈でき、誤差逆伝播の文脈における誤差計算は、パラメータまで伝播する「全ての」経路の総和を取る必要があるということと矛盾しない. 
         - $ \frac{\partial E^2}{\partial W }  = \frac{\partial E^2}{\partial z^2 } \frac{\partial z^2}{\partial W } $
           - $ = \frac{\partial E^2}{\partial z^2 }  ( f'(u^2) z^{1} + f'(u^2) W \frac{\partial z^{1}}{\partial W}) $
           - $ = \delta^2 z^1 + \delta^2 W (  f'(u^1) z^0 )  =  \delta^2 z^1 + \delta^1 z^0 ) $ 
       - $ t =3 $ のとき (一般化のための足がかり)
         - $ \frac{\partial z^3}{\partial W } = \frac{\partial z^{3,+}}{\partial W } + \frac{\partial z^{3}}{\partial z^2 } ( \frac{\partial z^{2,+}}{\partial W } + \frac{\partial z^{2}}{\partial z^1} (\frac{\partial z^{1,+}}{\partial W } ) )  $
           - $ = \frac{\partial z^{3,+}}{\partial W } + \frac{\partial z^{3}}{\partial z^2 } \frac{\partial z^{2,+}}{\partial W } + \frac{\partial z^{3}}{\partial z^2 } \frac{\partial z^{2}}{\partial z^1} \frac{\partial z^{1,+}}{\partial W }  $
           - $ = \frac{\partial z^{3}}{\partial z^3 } \frac{\partial z^{3,+}}{\partial W } + \frac{\partial z^{3}}{\partial z^2 } \frac{\partial z^{2,+}}{\partial W } + \frac{\partial z^{3}}{\partial z^2 } \frac{\partial z^{2}}{\partial z^1 }  \frac{\partial z^{1,+}}{\partial W }  $
           - $ = \sum_{k=1}^{3} ( \prod_{m=k+1}^{3}\frac{\partial z^{m}}{\partial z^{m-1} } ) \frac{\partial z^{k,+}}{\partial W } $
         - $ \frac{\partial E^3}{\partial W } = \frac{\partial E^3}{\partial z^3}  \sum_{k=1}^{3} ( \prod_{m=k+1}^{3}\frac{\partial z^{m}}{\partial z^{m-1} } ) \frac{\partial z^{k,+}}{\partial W } $ 
           - $ = \sum_{k=1}^{3} \frac{\partial E^3}{\partial z^3}  ( \prod_{m=k+1}^{3}\frac{\partial z^{m}}{\partial z^{m-1} } ) \frac{\partial z^{k,+}}{\partial W } $
           - $ = \sum_{k=1}^{3} \frac{\partial E^3}{\partial z^3}  ( \prod_{m=k+1}^{3} f'( u^m) W ) f'(u^k) z^{k-1}  $
           - $ = \sum_{k=1}^{3} \frac{\partial E^3}{\partial z^3} f'(u^3) ( \prod_{m=k}^{2} W f'(u^m) ) z^{k-1}  $
           - $ = \sum_{k=1}^{3} \delta^{3} ( \prod_{m=k}^{2} W f'(u^m)  ) z^{k-1} $
           - $ = \sum_{k=1}^{3} \delta^{k} z^{k-1} $
             - $ \because \delta^{t-1} =  \frac{\partial E}{\partial u^{t}} \frac{\partial u^t}{\partial u^{t-1}} $
             - $ = \delta^t ( \frac{\partial u^t}{\partial z^{t-1}} \frac{\partial z^{t-1}}{\partial u^{t-1}}) = \delta^t (W f'(u^{t-1}))$ 
       - よって、一般の $ t $ のとき
         - $ \frac{\partial z^t}{\partial W} = \sum_{k=1}^{t} ( \prod_{m=k+1}^t \frac{\partial z^{m}}{\partial z^{m-1}})  \frac{\partial z^{k,+}}{\partial W}$ 
           - $ = \sum_{k=1}^{t} f'(u^t) ( \prod_{m=k}^t W f'(u^m) ) z^{k-1} $ 
             - $ \because \frac{\partial z^{k,+}}{\partial W} = f'(u^k) z^{k-1} $
             - $ \because  \frac{\partial z^{m}}{\partial z^{m-1}} = \frac{\partial z^{m}}{\partial u^{m}}\frac{\partial u^{m}}{\partial z^{m-1}} = f'(u^{m}) W $ 
         - $ \frac{\partial E^t}{\partial W}  = \sum_{k=1}^t \delta^k z^{k-1} $
     - $\frac{\partial E^t}{\partial W_{(in)}} $ も同様に考えればよく, 
       - $ \frac{\partial z^t}{\partial W_{(in)}} = \sum_{k=1}^{t} ( \prod_{m=k+1}^t \frac{\partial z^{m}}{\partial z^{m-1}})  \frac{\partial z^{k,+}}{\partial W_{(in)}}$
         - $ = \sum_{k=1}^{t} f'(u^t ) ( \prod_{m=k}^t W f'(u^m)) x^k $
       - $ \frac{\partial E^t}{\partial W_{(in)}} = \sum_{k=1}^t \delta^k x^{k} $ 
   - 更新式の導出 ( 実装面, 多次元考慮)  
     - 全時刻の情報をバッファせずにパラメータが更新できる. 
       - $ \frac{\partial E}{\partial X} = \sum_{m=1}^{t} \frac{\partial E^m}{\partial X} $ のため、(実際にはパラメータ自体は時間変化しないのだが) 着目する時間ステップを1つずつ進めながら、$ X^{t+1} = X^t - \epsilon \sum_t \frac{\partial E^t}{\partial X}$ と、誤差項を累積加算するように パラメータ更新が行える
     - 途中で遡る時間を打ち切る
       - $ \delta^{t-1} = \delta^t W f'(u^{t-1}) W $ から $\delta^{t-p}$ の $p$ が大きくなるほど勾配は小さくなっていくため、長時間遡る経路起因の誤差項は無視できる
       - そのため、これまで $ \sum_{k=1}^t \delta^k x^k $ などと書いていた箇所は、所定の時間長 $T_t$ だけ遡ることにして、$ \sum_{k=1}^T \delta^{t-k} x^{t-k} $ と打ち切る
       - 逆に重み $W$ によっては勾配爆発の可能性もある。そのための打ち切りということもできる( 勾配消失については打ち切りはよい近似だが、こちらは完全な対症療法) 
     - 入力 $x$ のサイズを $(n_i,)$, 隠れ層 $z$ のサイズを $(n_h,)$, 出力 $y$ のサイズを $(n_o,)$ とする
       - $W_{(in)}$のサイズは$(n_h, n_i)$,  $W$ のサイズは $(n_h, n_h)$, $W_{(out)}$ のサイズは $(n_o, n_h)$ 
       - $\delta^{out,t} = \frac{\partial E}{\partial v}$ スカラー$E$ をサイズ $n_o$ のベクトルで微分するので サイズ $(n_o,)$ 
       - $\delta^{t} = \frac{\partial E}{\partial u}$ は サイズ $(n_h,)$ 
       - よって $\delta^t$ や $\delta^{out,t}$ と積を取っている部分は更新式における行列のサイズとの関係から、転置をとったベクトル積となる
         - つまり、講義で講師の先生がいってた発言「$[\cdot]^T$の記号は転置でなく、時間を遡って計算する、の意味だ」は間違いだと思われる  
     - 以上を考慮して、各パラメータの更新式は下記の通りとなる
       - $ t $ を 1ステップずつ進めながら
         - $ \delta^{out, t} = L'(y^t) g'(v^t) $
         - $ W_({out})^{t+1} = W_({out})^{t} - \epsilon \delta^{out, t} [z^{t}]^T $ 
         - $ c^{t+1} = c^{t} - \epsilon \delta^{out, t} $
         - さらに 入れ子で $ t $ から 遡りつつ
           - $ \delta^t = \delta^{out,t} W_{(out)},  \delta^{t-p-1} = \delta^{t-p} W f(u^{t-p}) $         
           - $W_{(in)}^{t+1} = W_{(in)}^{t} - \epsilon \sum_{k=0}^{T_t} \delta^{t-k} [x^{t-k}]^T$ 
           - $W^{t+1} = W^{t} - \epsilon \sum_{k=0}^{T_t} \delta^{t-k} [z^{t-k-1}]^T $
           - $ b^{t+1} = b^{t} - \epsilon \sum_{k=0}^{T_t} \delta^{t-k} $
   - コード演習問題
     - ~~~
<figure style="text-align:center;">
<img src="/assets/day3-enshu2.jpg" style="padding:0;width:100%;" alt="#1"/>
<figcaption></figcaption>
</figure>
~~~
     - 回答:2 
       - 問題で埋める部分は、BPTTで時間を遡る $\delta_{t-1}$ を $\delta_{t}$ から求めるところ.
       - 現在、1時刻前の中間層出力にかける重みが $U$ なので、$\frac{dh_{t-1}}{dh_{t}} = U$となる.  

\textinput{3_1_simple_RNN.md}


### Section.2 : LSTM

### Section.3 : GRU 

### Section.4 : 双方向RNN

### Section.5 : Seq2Seq

### Section.6 : Word2vec

### Section.7 : Attention Mechanism 


## Day.4 

### Section.1 : 強化学習

### Section.2 : AlphaGo

### Section.3 : 軽量化・高速化技術

### Section.4 : 応用モデル

### Section.5 : Transformer

### Section.6 : 物体検知・セグメンテーション

 - 本Section のスコープ
   - 物体検知の学習を進めたり、論文を読む上で大前提となる共通知識を取り扱う
 - 広義の物体認識タスク (どれも入力は画像で、ラベルを出力する)
   - "得ようとしている情報が 大まか"な順に並べる
   - 分類 : Classification
     - 出力 = "画像に対して" 単一または複数のクラスラベル
       - 画像内の対象が含まれるか否か? だけが問題
   - 物体検知 : Object Detection
     - 出力 = Bounding Box (bbox/BB) 
       - 矩形領域でざっくり位置も検出, インスタンスは区別しない
   - 意味領域分割 : Semantic Segmentation
     - 出力 = "各ピクセルに対して" 単一のクラスラベル
       - ピクセル単位での検出だが、インスタンスの区別はしない
   - 個体領域分割 : Instance Segmentation
     - 出力 = "各ピクセルに対して" 単一のクラスラベル
      - semantic segmentation した上で、さらに 対象が複数あるばあい個体毎の領域も分割

 - 物体検知の基礎
    - どこに、何が、どの程度のconfidence (確率と厳密にイコールではないのでこう呼ぶ) で?
    - データセット
      - why dataset? -> 目的に応じて適切なデータセットを選択できるようにしたい
        - 例えば...
        - クラス数
          - 必ずしも多ければいいというわけではなく、データセットの質にも注意
            - cf. ImageNet はクラス数多いが批判もある
              - 同じものに違うクラス名が振られている (laptop/notebook)
              - 人が気づかないような米粒のようアリにまでラベルが.. 
        - Box/画像 (1画像に含まれるBoxの平均数) 
          - 目的に応じたBox/画像 (1画像あたりのBox数平均)の選択をする
            - 大 -> 物体どうしの重なりやサイズ違いなどを扱う必要がある、実践的
            - 小 -> 日常的に撮影される写真などとは異なるが、対象がアイコン的な写っている画像を扱う場合にはよい。
      - 代表的データセット (どれも物体検出コンペティションで用いられたもの)
        - VOC12 
          - PASCAL VOC Object Detection Challenge 2012で用いられたもの
            - 2012年、主要貢献者がなくなり 本コンペは終了
          - クラス: 20, Box数/画像:  2.4   
          - Train+Val : 11,540
          - 画像サイズ 470x380
          - 個々の物体にラベルが与えられている
        - ILSVRC17 
          - ILSVRC(ImageNet Scale Visual Recognition Challenge) Object Detection Challenge 
            - 2017年にコンペ終了 (後継: Open Images Challenge)
            - ImageNet (21841 クラス/1400万枚以上)のサブセット
          - クラス 200,　 Box数/画像:  1.1 
          - Train+Val :  476,668 
          - 画像サイズ 500x400
          - 個々の物体にはラベルが"ない"
        - MS(Microsoft) COCO18
          - クラス : 80,  Box/画像: 7.3
          - Train+Val : 123,287  
          - 画像サイズ 640x480
          - 個々の物体にラベルが与えられている
        - OICOD18
          - OIC(Open Images Challenge) Object Detection Challenge 
            - Open Images V4 (6000クラス以上/900万枚以上のサブセット)
          - クラス : 500,  Box/画像: 7.0 
          - Train+Val : 1,743,042
          - 画像サイズ 一様ではない
          - 個々の物体にラベルが与えられている
    - 評価指標
      - (復習) Confusion Matrix 
        - Positive/Negative は予測の結果、True/Falseは予測の正否
          - 例: TN(TrueNegative) = 予測結果はnegative でありその正否は正しい (両方Negative) 
          - Precision = TP / (TP+FP) , Positive と予測したもののうち 実際にPositive だったもの (過剰検出のすくなさ)
          - Recall = TP / (TP+FN), 真のラベルがPositive のもののうち、Positiveと予測したもの (取りこぼしのすくなさ)
        - Confidence のしきい値変化に対する振舞い
          - クラス分類: 1画像に対して必ず1ラベルが出力される -> confusion matrix の各マスに入る数字の合計は、しきい値を変えても 変わらない
          - 物体検出: 1画像に対して、その閾値を超える(一般には複数の)BBが出てくる -> しきい値を変えることによって、出力されるBB数の合計が変わる -> Confusion Matrix に入る数字の合計は変わる
      - IoU : Intersection over Union
        - 定義: (Area ov Overlap) / (Area of Union) 
          - ~~~
<figure style="text-align:center;">
<img src="/assets/IoU.jpg" style="padding:0;width:100%;" alt="#1"/>
<figcaption></figcaption>
</figure>
~~~
          - 共通部分 / (Ground Truth BB) だと Predicted BB を画像全体にすれば スコアが上がってしまう
          - 共通部分 / (Predicted BB) だとPredicted BB を小さくして、Ground Truth 内に完全に包含されれば スコアが上がってしまう 
        - Confidence とともに、位置推定の精度の良さを表す指標としてしきい値を設定
      - Precition/Recall の計算方法
        - 入力画像1枚のケース
          - 基本的には、confidence が所定のしきい値(例 0.5)以上かつ、IoUが所定のしき値(例 0.5)以上のものを TPとする
            - ただし、画像に含まれる対象のBBの数 < Positive な BB数 の場合、confidence が高いものをTPとし、過剰な分は FP とする
          - TP, FP がきまったので、Precision そのまま計算できる
          - 検出できなかったクラスがあれば、それを FNとして、Recall を計算 (つまり分母は 全Ground Truth の数となる)
        - 複数画像 (クラス単位) の計算例
          - TP, FP の決め方は、入力画像1枚のケースと同じ
          - TP, FP は画像をまたいでカウントし計算
      - AP (Average Precision)
        - IoUを固定した状態で、confidence を変化させ、"クラスラベルごとに" Precision, Recall を計算
        - Precision を Recall の関数 で表したとして,,
          - $ AP = \int_{0}^1 P(R) dR $ : Precision-Recall curve の下側面積
            - 実際には、confidence は離散値なので、実際には和文として計算する
      - mAP (mean Average Precision)
        - 全クラスでのAPの平均値 $ mAP = \frac{1}{C}\sum_C AP_c $
        - 論文などで、"AP" と書かれていても、実は mAPのこともある
        - 「どのデータセットでの結果なのか?」も気をつける
      - 参考 $AP_{COCO} $
        - IoUを 0.5から 0.95 まで 0.05刻みで変えながら、mAPを計算して、算術平均
        - IoU が x の場合の mAP を $mAP_{x}$ とすると,, $ \frac{1}{10}\sum_x mAP_x$ 
      - FPS (Frame per Second) 
        - 応用上の陽性で、検出速度も評価されることがある
          - FPSは単位時間あたりの処理可能な枚数だが、Inference Time (1枚の画像を処理するための時間) で評価されている場合もあり
    - 物体検知の大枠
      - 2012年: AlexNet の登場を皮切りに 時代はSIFT特徴量をベースとした手法から、(Deep)CNNへ
        - [AlexNet の論文](https://papers.nips.cc/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf) 
      - 2013年~ 
        - 画像系の代表的なネットワーク
          - VGGNet, GoogleNet, ResNet,  Inception-ResNet, DenseNet, MobileNet, AmoebaNet など.. 
        - 物体検知のフレームワーク(抜粋)
          - 2段階検出
            - 候補領域の検出と、クラス推定を別々に行う
            - 相対的には精度が高く、計算量が大きい傾向
            - 例: RCNN, SPPNet, 
          - 1段階検出 RCNN, YOLO, SSD, ...
            - 候補領域の検出とクラス推定を同時に行う
            - 相対的には精度は低いが、計算量が少ない傾向
            - 例: SSD, YOLO,..
    - 1段階検出器の例 : SSD (Single Shot Detector)
      - 超概要
        1. BB を適当な位置に適当な大きさで、用意する (Default Box)
        2. Default BOX を変形して、Predicted BB とし、conf. も出力
      - SSDのネットワークアーキテクチャの特徴
        - ベースネットワーク (VGG16) [論文](https://arxiv.org/abs/1409.1556)
          - 合計 16層 : Convolution 層が 13, 全結合層 3 層 (pooling 層や, ReLU単体は数えてない)
        - SSDの徳用
          - 入力サイズに応じて SSD300, SSD512 など
          - VGG16 の FC2層がConvolution層に、最後のFCは削除
          - VGG16 の 10層目以降、複数の層の出力から、最終出力がつくられている (マルチスケール特徴マップ)
            - より解像度の高い特徴マップで小さい物体を検出している
      - 出力 
        - k * (クラス数 + 4) * m * n 
          - k: Default box の数をコントロールするハイパパラメータ
          - +4: Default Box の中心座標の変更量 $\Delta x, \Delta y$, サイズの変更量 $\Delta w, \Delta h$ 
          - m * n : 特徴マップのサイズ
          - つまり、default box 数は k * m * n 
          - 青,赤でk=4だったり、6だったりするのは演算量などの都合
          - ~~~
<figure style="text-align:center;">
<img src="/assets/ssd_network_output.jpg" style="padding:0;width:100%;" alt="#1"/>
<figcaption></figcaption>
</figure>
~~~
      - その他の工夫 
        - 1つの対象に対して、複数のBBがでてきてしまう
          - Non-Maximum Supression : IoU がある程度高いもののなかから、最も confidence の高いものを残して他のBBは削除する
        - 背景(Negative)と判断されるクラスに属するBBがたくさん出てしまう
          - Positve : Negative の比が 偏りすぎないように 背景に属するBBを減らす
        - 以上は、SSDの中で初出の方法では無い
        - Data Augmentation や、Default Box のアスペクト比のなど
      - 損失関数
        - confidenct と location に関する誤差の和 (数式は割愛)
  - Semantic Segmentation の基礎
    - ざっくりわかる Semantic Segmentation の肝
      - convolution , pooling を繰り返すことで解像度が落ちていく 
      - 対して、出力は、"ピクセルごとの" ラベル
      - 元の解像度に戻す Up-Sampling をいかに行うか? が Semantic Segmentation を理解するポイントになる
    - 無邪気すぎる疑問 : そもそも Pooling しなければいいのでは?
      - 例えば 犬/猫 判定。ごくごく一部の少領域だけでは不可能 -> 正しく認識するためには、受容野(kernel)にある程度の大きさが必要
        - 受容野を広げるための代表的な手段
          - 深い Conv 層 
          - プーリング や ストライド
          - Dilated Convolution (後述)
      - Conv層の多層化は、演算量、メモリの制約もあるので、プーリング・ストライドの併用は実用的には必須
    - ネットワークの形 
      - VGG16 をベースに最後 FCをConv層に置き換え (Fully Convolutional Network)
    - アップサンプリング方法
      - Deconvolution / Transposed convolution 
        - 計算例: input 3x3 -> output 5x5 ( kernel size = 3, padding = 1, stride = 1) 
          - 特徴マップのピクセル間隔を stride 分だけあける
          - 特徴マップの周りに ( kernel size - 1 ) - padding だけ余白を作る
          - 畳み込み演算を行う
        - 厳密には、畳込みの逆演算ではない
      - 輪郭情報の補完
        - 高いレイヤーのpooling層出力は、低解像度になっていて輪郭情報が失われる -> 低レイヤーPooling層の出力を要素ごとに足すことで、輪郭情報を取り戻す
          - U-Net における Skip-connection では、チャネル方向にデータを結合のためやっていることは異なるが、着想, 対処したい問題は似ている
      - UnPooling 
        - MaxPooling を行うとき、どこに最大値があったかを記録(Switch Variables)しておく
        - UnPooling では、Switch Variables の箇所に値をいれ、他をゼロで埋める
        - ~~~
<figure style="text-align:center;">
<img src="/assets/unpooling.jpg" style="padding:0;width:100%;" alt="#1"/>
<figcaption></figcaption>
</figure>
~~~
    - Dilated Convolution
      - pooling を用いずに受容野のサイズを広げる他の手段。とびとびフィルタ。
      - 例: kernel 3x3, dilation rate=2 の場合、一つおきに畳み込むことで 5x5 の範囲をカバーすることができる

## 取り組み概要 & 感想

### 取り組みの記録

今回は、M4MT()

実装演習にじっくり時間をかける感じでもなかったので、
Section単位で 動画みながらまとめ->実装演習 と進めていった。

- 5/7 : Day1 Section.1 までの動画、コードをざっと眺めて、レポートの要件について事務局問い合わせ。(1h)
  - レポート提出方法の"実装演習"というのが何を指しているのか不明確であったため。
  - 事務局回答を踏まえて、本ページ 冒頭に書いたようなまとめ方をすることにした。
　- Stage.1,2 動画は基本座学で、コードの内容や動画で取り上げられることはほぼなかったが、このStage.3 では動画講義中にもコードを見て動かすシーンが多発するので、エディタ上で動画を見ながらレポートをまとめつつ、コードも実行しつつのスタイルですすめる形を取ることにした.
- 5/13: プロローグ を動画みながらざっくりまとめ  (0.5h)
- 5/14: Day1 Section 0, 1, 2 を動画みながらざっくりまとめ (1.5h)
- 5/15: Day1 Section 3 (1h)
- 5/16: Day1 Section 4 (1.5h)
- 5/17: Day1 Section 5 (1h)
- 5/20: Day1 のこり (1h くらい)
- 5/22: Day2 Section 1 (1.75h)
- 5/23: Day2 Section 2 (2.5h) 
  - 演習でのハイパパラメータを色々試していたら時間を食った
- 5/25: Day2 Section 3 (3h)
  - 同じく、ハイパパラメータを色々試していたら時間がかかってしまった。特に Dropoutを入れると収束するまでに 必要なiterationが増えるのが大きかった。
- 5/31: Day2 Section 4,5 (2h)
- 6/1 : ステージテスト (2h)

### 感想ほか　

今回のステージの動画に出てきた講師の方は Stage1,2の方とは違う方。(スタートテスト前のPython講座などをやっていた方)
講師として実績はある方なのだと思うが、私とは、指導・解説の仕方のフィーリングが合わず、意図を汲み取りかねる箇所などがあった。
同じような感想を持った方は、資料の数式を正として要旨を理解し、講師の先生の言い回しや独自の説明の内容についてはあまり細かいところまで気にしすぎないのがよいかもしれない。

例えば.. 

 - 説明で使う言葉に若干の違和感を感じることがあった
   - 一例: 重み付き和を　まぜ合わせる, シャッフルと表現。 
     - "カードを交ぜる" / "醤油と塩を混ぜる" という２つの表現は、まぜたあと個々が区別できるかどうかが違う。シャッフルは前者に該当。絶対間違えだと主張するつもりはないが、なぜあえて、シャッフルという表現をつかったのかな?と1分立ち止まってしまった。

 - 図の説明が独特
   - 例: 学習率の違いによって収束のしかたが異なることを手書きの図を使って説明したとき、loss 関数のグラフの傾きが異なる点でも、同じようにパラメータが変化していくような矢印を書いている。おそらく、初学者にとっては混乱を招く図の書き方だろう

 - 細かい点の見間違いがちらほら
   - 例: 確認テストの細かい意味を取り違えて、おそらく想定した正解とは異なる解答を示すことがある(確認テストの5-2 など)

 - 前後関係を無視した説明, 用語の使い方
   - 例: ミニバッチ勾配降下法の説明をしたあとで、確認テストでその利点を説明させる流れなのに、分散学習的な実装を想定した説明をする。箇所箇所で言っていることは正しいのだが、資料作成者が想定した流れではないと思うし、初学者はミニバッチ勾配降下法の理解に不安を持つと思う。分散学習の話するならするで、「これはまたミニバッチ勾配降下法の基礎の学習とは別の段階の話ですが、分散学習というのがあって、、」みたいな前置きをするのがいいのかな、と思う。


実装演習に関しては、やはり Stage.2 と同じで基本そのままで動いてしまうので、"勉強の仕方が上手い人" でないとさーっと通り過ぎてしまってコードから多くを学ぶことができないのではないのかな? と思った。

テストについて。他の受講生の方でちらほら 合格に苦労したという方がいたので、ケアレスミスの内容にゆっくり解いた。
全問正解だったが、必要以上に時間を書けすぎてしまったかな。。同じペースで本番やったら、問題解ききれない。

内容については 詳細は触れないが、Stage.3 にかぎらずこれまでの講義のどこかでやった内容からの出題で、特に未知の情報は無い。
出題文の日本語の意味がちょっとわかりにくいな、という出題はいくらかあったが、4択問題なので正解はまぁ導けるかな、、とは思う。
「誤ってる/正しいものを選べ」形式の設問でいくつか迷ったものがあったので、それは復習をしておこう。

## 計画 (前回から変更無し)

Stage.2 終了時に見直したスケジュールはほぼキープできた。
ペースは上がってきているが、ステージ3より4の方がボリュームありそうなので、
前倒しにはせず、今の所変更無しとしておく。
前評判ではステージ4のテストはめっぽう難しいらしいので、クリア前に計画見直す可能性もあるかも?

 - ~2021/2/15  : スタートテスト (2021/02/07完了, 10h)
 - ~2021/3/30  : ステージ1      (2021/03/30完了, 8h)
 - ~2021/5/6   : ステージ2      (2021/05/06完了, 21h)
 - ~2021/5/30  : ステージ3      (2021/06/02完了, 17.75h) 
 - ~2021/6/27  : ステージ4 
 - ~2021/7/4   : 復習 -> 修了テスト 
 - ~2021/7/15  : Eもぎライト -> 今後の計画具体化 
 - ~2021/7/30  : シラバスの未習箇所の学習 
 - ~2021/8/26  : 全体の復習
 - 2021/8/27,28: E資格 受験 
