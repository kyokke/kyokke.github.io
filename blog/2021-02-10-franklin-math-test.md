@def title = "Franklin.jl で数式を使ってみるテスト"
@def author = "kyokke" 
@def tags = [ "julialang", "Franklin.jl" ]


# Franklin.jl で数式を使ってみるテスト

このブログは Juliaで実装された、
静的サイトジェネレーター "Franklin.jl" を使って作成されています。

今後数式を扱うことがあると思うので、前もって数式を使うことができるか試してみた。
この記事はそのための投稿です。

## テスト入力

TeX基本を伴う下記の段落

```
$ x = a $ から $ x = b $ までの関数 $f(x)$ の積分は

$$
\int^{b}_{a} f(x) dx = \lim_{n \to \infty} \sum^{n-1}_{i=0} f(x_{i}) \Delta x
$$

と置き換えて考えることができる．
```

が以下に表示されているはず。

---

$ x = a $ から $ x = b $ までの関数 $f(x)$ の積分は

$$
\int^{b}_{a} f(x) dx = \lim_{n \to \infty} \sum^{n-1}_{i=0} f(x_{i}) \Delta x
$$

と置き換えて考えることができる．

---


## ローカル(Windows)で起動した場合の結果

下記のようにローカル環境では正しく数式が表示されているようだ。数式番号もつくのがいい。

![スクリーンショット](/assets/2021-02-10_181534.jpg)

さてさて、github pages の方はどうなるだろうか?



