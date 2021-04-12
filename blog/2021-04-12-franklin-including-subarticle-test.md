@def title = "Franklin.jl で 別記事をインクルード"
@def author = "kyokke" 
@def tags = [ "Franklin.jl", "julialang" ]


# Franklin.jl で 別記事をインクルードするテスト

Franklin.jl の記事の markdown に別の markdown のファイルを include したいことがあるため、いくつかのケースについてテストしてみる。


## インクルードされる記事を assets フォルダに置く場合

基本的には、Franklin とは関係ないところで書かれた記事を持ってきて、
そのまま include したい時に使う。

_assets/2021-04-12-test-subarticle-in-assets.md に置いたファイルをインクルードしてみる。`\textinput`のパス指定の基準は 生成された assets フォルダのようだ.
```
\textinput{2021-04-12-test-subarticle-in-assets.md}
```
\textinput{2021-04-12-test-subarticle-in-assets.md}

--- サブ記事 in _assets ここまで

## インクルードされる記事を assets フォルダに置きたくないケース

例えば、Franklinで公開している複数の記事を一つにまとめた記事を作りたいことがあるかもしれない。

deploy される環境における各ファイルの相対位置がわかっていれば、assets に置かなくても参照できるかもしれない。

ローカルマシンでの位置関係は下記の通り。
```
/__site/ # 生成されたサイトフォルダ
        - assets/  # ここが \textinput の指定パス基準
        - blog/
              - 2021-01-07-rabbit-challenge-application/index.html
/_assets
/_css
/_layout
/_libs
/blog/
     - 2021-01-07-rabbit-challenge-application.md # 読み込みたい記事
(以下略)
```

生成された __site フォルダ以下の html を直接取り込めればよいのだが、それには js や php などの力が必要でちょっと面倒だ。

そこで、めちゃくちゃ行儀が悪いのだが、__site フォルダの外、ソースを直接参照してみよう.

`\textinput` コマンドのパスしていの基準は __site/assets なので、
```
\textinput{../../blog/2021-01-07-rabbit-challenge-application.md}
```
とすれば良さそうだ。下に結果が得られるはず。(ローカルでは成功しているが、github pages ではどうだろうか)
\textinput{../../blog/2021-01-07-rabbit-challenge-application.md}
