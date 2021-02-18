@def title = "Franklin.jl で code highlight を使ってみるテスト"
@def author = "kyokke" 
@def tags = [ "julialang", "Franklin.jl" ]


# Franklin.jl で code highlight を使ってみるテスト

このブログは Juliaで実装された、
静的サイトジェネレーター "Franklin.jl" を使って作成されています。

先日数式のテスト投稿を行いましたが、ソースコードの表示も行うことが増えると思うので、code highlight 機能のテストを行います。

## 依存モジュールのインストール on Local Windows 

[ドキュメント](https://franklinjl.org/index.html#installing_optional_extras) を見てそのとおりにやるのみ。

```julia
using Pkg
Pkg.add("NodeJS")
using NodeJS
run(`$(npm_cmd()) install highlight.js`)
```

で Local Server では動いていそう。

## 依存モジュールのインストール on Github Actions の Ubuntu

.github/workflows/deploy.yml の中で、

```yml
    - run: julia -e '
            using Pkg; 
            Pkg.add(["NodeJS"]);
            Pkg.add(Pkg.PackageSpec(name="Franklin", rev="master"));
            using NodeJS; run(`$(npm_cmd()) install highlight.js`);
            using Franklin;
            Pkg.activate("."); Pkg.instantiate();
            optimize()'
```

というように、NodeJS モジュールと highlight.js のインストールを行うコードがあることを確認 (最初にこのサイトを開いたときにすでに設定を行っていた)

これで問題なく動いていそう。
