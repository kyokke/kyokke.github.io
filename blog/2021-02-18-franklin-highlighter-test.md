@def title = "Franklin.jl で code highlight を使ってみるテスト"
@def author = "kyokke" 
@def tags = [ "julialang", "Franklin.jl" ]


# Franklin.jl で code highlight を使ってみるテスト

このブログは Juliaで実装された、
静的サイトジェネレーター "Franklin.jl" を使って作成されています。

先日数式のテスト投稿を行いましたが、ソースコードの表示も行うことが増えると思うので、code highlight 機能のテストを行います。

## 依存モジュールのインストール on Local Windows 

```julia
using Pkg
Pkg.add("NodeJS")
using NodeJS
run(`$(npm_cmd()) install highlight.js`)
```

