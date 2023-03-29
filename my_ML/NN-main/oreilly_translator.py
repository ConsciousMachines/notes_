# NOTE: removed all the alt= tags because they give me errors with their latex and such. 
# NOTE: empty paragraphs ruin the flow in my system, remove <p><br/></p>
# NOTE: reading from file doesn't work, only when i paste in here. lol.

#f = open(r'C:\Users\pwnag\Desktop\OEBPS\ch29.xhtml', 'r', encoding = 'utf8')
#text = f.read()
#f.close()

text = u"""

<?xml version='1.0' encoding='utf-8'?>
<html xmlns:epub="http://www.idpf.org/2007/ops" xmlns:ops="http://www.idpf.org/2007/ops" xmlns="http://www.w3.org/1999/xhtml" xml:lang="ja">

<head>
  <link href="oreilly.css" rel="stylesheet" type="text/css"/>
  <meta content="Re:VIEW" name="generator"/>
  <title>高階微分以外の用途</title>
</head>

<body>

  <h1 id="h36"><span class="chapno">ステップ36</span><br/>高階微分以外の用途</h1>

  <p>これまで私たちは、DeZeroを使って高階微分を求めてきました。そのために行ったことは、本質的には1つです。それは、逆伝播で行う計算に関しても「つながり」を作るようにしたことです。ここで重要な点は、逆伝播の計算グラフ化こそが、DeZeroの新しい機能だということです。高階微分は、その1つの応用例にすぎません。ここでは、新しくなったDeZeroの高階微分以外の用途について見ていきます。</p>

  <div class="note">

    <table class="note">

      <tr>
        <td class="center top" rowspan="2">
          <img alt="[注記]" class="noteicon" src="images/note.png"/>
        </td>
      </tr>

      <tr>
        <td>
          <p>新しいDeZeroは、逆伝播で行った計算に対して、さらに逆伝播を行うことができます。その機能は、double backpropagation<!-- IDX:double backpropagation -->と呼ばれます（以降、double backprop<!-- IDX:double backprop -->と表記）。double backpropは、現代のディープラーニングのフレームワークのほとんどがサポートしています。</p>
        </td>
      </tr>

    </table>

  </div>

  <h2 id="h36-1"><span class="secno">36.1　</span>double backpropの用途</h2>

  <p>それでは、double backpropの高階微分以外の用途を見ていきましょう。まずは、次の問題を考えてみましょう。</p>

  <p><br/></p>

  <hr/>

  <p class="noindent">問：次の2つの式が与えられたとき、<span class="equation mathimage"><img alt="x=2.0" class="math_gen_b317ab2ecacb1992f0c798afef83d9f2ad1dd1073534c3d13020be615a8f7a0b" src="images/_review_math/_gen_b317ab2ecacb1992f0c798afef83d9f2ad1dd1073534c3d13020be615a8f7a0b.png"/></span>における<span class="equation mathimage"><img alt="\frac{\partial z}{\partial x}" class="math_gen_41bcfd1068dd83aee9c5a2df487c285c42564823268057eaf00e7511c8a0e870" src="images/_review_math/_gen_41bcfd1068dd83aee9c5a2df487c285c42564823268057eaf00e7511c8a0e870.png"/></span>（<span class="equation mathimage"><img alt="x" class="math_gen_2d711642b726b04401627ca9fbac32f5c8530fb1903cc4db02258717921a4881" src="images/_review_math/_gen_2d711642b726b04401627ca9fbac32f5c8530fb1903cc4db02258717921a4881.png"/></span>に関する<span class="equation mathimage"><img alt="z" class="math_gen_594e519ae499312b29433b7dd8a97ff068defcba9755b6d5d00e84c524d67b06" src="images/_review_math/_gen_594e519ae499312b29433b7dd8a97ff068defcba9755b6d5d00e84c524d67b06.png"/></span>の微分）を求めよ。</p>

  <div class="caption-equation" id="eq36-1">

    <table>

      <tr>
        <td>
          <div class="equation">

            <img alt="y = x^2" class="math_gen_ab4fa3a40eaf7510dab9a6daf7a456e1c709a41af740a16103f4c08769793179" src="images/_review_math/_gen_ab4fa3a40eaf7510dab9a6daf7a456e1c709a41af740a16103f4c08769793179.png"/>

          </div>
        </td>
        <td class="mathno">(36.1)</td>
      </tr>

    </table>

  </div>

  <div class="caption-equation" id="eq36-2">

    <table>

      <tr>
        <td>
          <div class="equation">

            <img alt="z = \biggl( {\frac{\partial y}{\partial x}} \biggr)^3 + y" class="math_gen_53984fd06c471fec2a388194b88ec1d63b5e096ebbaaa264582177b1f76a3f6d" src="images/_review_math/_gen_53984fd06c471fec2a388194b88ec1d63b5e096ebbaaa264582177b1f76a3f6d.png"/>

          </div>
        </td>
        <td class="mathno">(36.2)</td>
      </tr>

    </table>

  </div>

  <hr/>

  <p><br/></p>

  <p>これは、これまでにも見てきた微分を求める問題です。これまでと異なるのは、<span class="eqref"><a href="./ch36.xhtml#eq36-2">式(36.2)</a></span>の中に微分が含まれる点です。つまりは、微分が含まれた式に対して、さらに微分を求める必要があります。この問題も、double backpropによって計算できます。その説明の前に、まずは手計算で<span class="equation mathimage"><img alt="\frac{\partial z}{\partial x}" class="math_gen_41bcfd1068dd83aee9c5a2df487c285c42564823268057eaf00e7511c8a0e870" src="images/_review_math/_gen_41bcfd1068dd83aee9c5a2df487c285c42564823268057eaf00e7511c8a0e870.png"/></span>を求めてみます。それには、次のように式を展開します。</p>

  <div class="caption-equation">

    <table>

      <tr>
        <td>
          <div class="equation">

            <img alt="\begin{aligned} \frac{\partial y}{\partial x} &amp; = 2x \\ z &amp; = \biggl( {\frac{\partial y}{\partial x}} \biggr)^3 + y = 8x^3 + x^2 \\ \frac{\partial z}{\partial x} &amp; = 24x^2 + 2x \\ \end{aligned}" class="math_gen_251ea52a4f5c04da0502b894478a76d75ebd4e5874b33b21012a4285ebcf1d7a" src="images/_review_math/_gen_251ea52a4f5c04da0502b894478a76d75ebd4e5874b33b21012a4285ebcf1d7a.png"/>

          </div>
        </td>
      </tr>

    </table>

  </div>

  <p>上記のように式を展開したら、<span class="equation mathimage"><img alt="24x^2 + 2x" class="math_gen_97f6c5b88627d3c050d80bad2318c43cbc84c195e8488aa21da560789eeb89ce" src="images/_review_math/_gen_97f6c5b88627d3c050d80bad2318c43cbc84c195e8488aa21da560789eeb89ce.png"/></span>に<span class="equation mathimage"><img alt="x=2.0" class="math_gen_b317ab2ecacb1992f0c798afef83d9f2ad1dd1073534c3d13020be615a8f7a0b" src="images/_review_math/_gen_b317ab2ecacb1992f0c798afef83d9f2ad1dd1073534c3d13020be615a8f7a0b.png"/></span>を代入すれば、<span class="equation mathimage"><img alt="100.0" class="math_gen_43b87f618caab482ebe4976c92bcd6ad308b48055f1c27b4c574f3e31d7683e0" src="images/_review_math/_gen_43b87f618caab482ebe4976c92bcd6ad308b48055f1c27b4c574f3e31d7683e0.png"/></span>という答えが得られます。</p>

  <div class="caution">

    <table class="note">

      <tr>
        <td class="center top" rowspan="2">
          <img alt="[警告]" class="warningicon" src="images/warning.png"/>
        </td>
      </tr>

      <tr>
        <td>
          <p>上の数式の<span class="equation mathimage"><img alt="\frac{\partial y}{\partial x}" class="math_gen_32a158aa60198bf3b846228c8016d11c347ab950799c8e046aa5d273e09ad834" src="images/_review_math/_gen_32a158aa60198bf3b846228c8016d11c347ab950799c8e046aa5d273e09ad834.png"/></span>は値ではなく、<span class="equation mathimage"><img alt="x" class="math_gen_2d711642b726b04401627ca9fbac32f5c8530fb1903cc4db02258717921a4881" src="images/_review_math/_gen_2d711642b726b04401627ca9fbac32f5c8530fb1903cc4db02258717921a4881.png"/></span>の式です。もしここで、<span class="equation mathimage"><img alt="x=2.0" class="math_gen_b317ab2ecacb1992f0c798afef83d9f2ad1dd1073534c3d13020be615a8f7a0b" src="images/_review_math/_gen_b317ab2ecacb1992f0c798afef83d9f2ad1dd1073534c3d13020be615a8f7a0b.png"/></span>における<span class="equation mathimage"><img alt="\frac{\partial y}{\partial x}" class="math_gen_32a158aa60198bf3b846228c8016d11c347ab950799c8e046aa5d273e09ad834" src="images/_review_math/_gen_32a158aa60198bf3b846228c8016d11c347ab950799c8e046aa5d273e09ad834.png"/></span>の値を求めて、それを<span class="equation mathimage"><img alt="z = \left(\frac{\partial y}{\partial x}\right)^3 + y" class="math_gen_973744b93f61f585cbb35ab99a03671e32e7a05b2058814ce4e41c854e97ddd6" src="images/_review_math/_gen_973744b93f61f585cbb35ab99a03671e32e7a05b2058814ce4e41c854e97ddd6.png"/></span>に代入した場合、正しい結果は得られません。</p>
        </td>
      </tr>

    </table>

  </div>

  <p>以上の点を踏まえて、DeZeroを使って上の問題を解いてみます。コードは次のようになります。</p>

  <div class="emlist-code">

    <p class="caption"><span class="bg">steps/step36.py</span></p>

    <pre class="emlist language-python highlight"><code class="kn">import</code> <code class="nn">numpy</code> <code class="kn">as</code> <code class="nn">np</code>
<code class="kn">from</code> <code class="nn">dezero</code> <code class="kn">import</code> <code class="n">Variable</code>

<code class="n">x</code> <code class="o">=</code> <code class="n">Variable</code><code class="p">(</code><code class="n">np</code><code class="o">.</code><code class="n">array</code><code class="p">(</code><code class="mf">2.0</code><code class="p">))</code>
<code class="n">y</code> <code class="o">=</code> <code class="n">x</code> <code class="o">**</code> <code class="mi">2</code>
<code class="n">y</code><code class="o">.</code><code class="n">backward</code><code class="p">(</code><code class="n">create_graph</code><code class="o">=</code><code class="bp">True</code><code class="p">)</code>
<code class="n">gx</code> <code class="o">=</code> <code class="n">x</code><code class="o">.</code><code class="n">grad</code>
<code class="n">x</code><code class="o">.</code><code class="n">cleargrad</code><code class="p">()</code>

<code class="n">z</code> <code class="o">=</code> <code class="n">gx</code> <code class="o">**</code> <code class="mi">3</code> <code class="o">+</code> <code class="n">y</code>
<code class="n">z</code><code class="o">.</code><code class="n">backward</code><code class="p">()</code>
<code class="k">print</code><code class="p">(</code><code class="n">x</code><code class="o">.</code><code class="n">grad</code><code class="p">)</code>
</pre>

  </div>

  <div class="cmd-code">

    <p class="caption">実行結果</p>

    <pre class="cmd">variable(100.)
</pre>

  </div>

  <p>このコードで重要な点は、<code class="inline-code tt">y.backward(create_graph=True)</code>です。そこでは微分を求めるために逆伝播を行います。それによって、新たに計算グラフが作られます（この場合、<code class="inline-code tt">2*x</code>という計算グラフが、ユーザの見えないところで作られます）。そして、その逆伝播によって作られた計算グラフを使って新しい計算を行い、さらに逆伝播を行います。そうすることで正しい微分が求められます。</p>

  <div class="note">

    <table class="note">

      <tr>
        <td class="center top" rowspan="2">
          <img alt="[注記]" class="noteicon" src="images/note.png"/>
        </td>
      </tr>

      <tr>
        <td>
          <p>上のコードの<code class="inline-code tt">gx = x.grad</code>は、単なる変数（値）ではなく、計算グラフ（式）です。そのため、<code class="inline-code tt">x.grad</code>の計算グラフに対して、さらに逆伝播を行うことができます。</p>
        </td>
      </tr>

    </table>

  </div>

  <p>以上のような問題――微分の式を求め、その式を使って計算を行い、さらに微分を求める問題――が、double backpropを使って解くことができます。このような用途は、ディープラーニングの研究でも見られます。続いて、その例をいくつか紹介します。</p>

  <h2 id="h36-2"><span class="secno">36.2　</span>ディープラーニングの研究での使用例</h2>

  <p>ディープラーニングに関連した用途でdouble backpropを使う研究はいくつもあります。たとえば、WGAN-GP<!-- IDX:WGAN-GP --><a href="bib.xhtml#bib-wgan">[21]</a>という論文では、<span class="imgref"><a href="./ch36.xhtml#id_ch3_2F3-35">図36-1</a></span>で表される数式を最適化します。</p>

  <div class="image" id="id_ch3_2F3-35">

    <img alt="WGAN-GPの最適化する関数（数式は文献&lt;a href=&quot;bib.xhtml#bib-wgan&quot;&gt;[21]&lt;/a&gt;より引用）" src="images/html/ch3/3-35.png"/>

    <p class="caption">
図36-1 WGAN-GPの最適化する関数（数式は文献<a href="bib.xhtml#bib-wgan">[21]</a>より引用）
</p>

  </div>

  <p><span class="imgref"><a href="./ch36.xhtml#id_ch3_2F3-35">図36-1</a></span>で注目したい点は、最適化を行う式に勾配が入っている点です（勾配とは、テンソルの各要素に関しての微分です）。その勾配は、1回目の逆伝播で求めることができます。そして、その勾配を使って関数<span class="equation mathimage"><img alt="L" class="math_gen_72dfcfb0c470ac255cde83fb8fe38de8a128188e03ea5ba5b2a93adbea1062fa" src="images/_review_math/_gen_72dfcfb0c470ac255cde83fb8fe38de8a128188e03ea5ba5b2a93adbea1062fa.png"/></span>を計算し、関数<span class="equation mathimage"><img alt="L" class="math_gen_72dfcfb0c470ac255cde83fb8fe38de8a128188e03ea5ba5b2a93adbea1062fa" src="images/_review_math/_gen_72dfcfb0c470ac255cde83fb8fe38de8a128188e03ea5ba5b2a93adbea1062fa.png"/></span>を最適化するために、2回目の逆伝播を行います。</p>

  <p>このように、最新の研究においてもdouble backpropは使われます。WGAN-GPの他にも、MAML<!-- IDX:MAML --><a href="bib.xhtml#bib-maml">[22]</a>やTRPO<!-- IDX:TRPO --><a href="bib.xhtml#bib-trpo">[23]</a>など、有名な研究でdouble backpropの機能が実際に使われています。</p>

  <div class="note">

    <table class="note">

      <tr>
        <td class="center top" rowspan="2">
          <img alt="[注記]" class="noteicon" src="images/note.png"/>
        </td>
      </tr>

      <tr>
        <td>
          <p>TRPOでは、ヘッセ行列とベクトルの積を求める際に、double backpropを使います。double backpropを使えば、その計算を効率良く行うことができます。ヘッセ行列とベクトルの積については、次の<a class="columnref" href="#column-1">「コラム：ニュートン法とdouble backpropの補足」</a>で説明します。</p>
        </td>
      </tr>

    </table>

  </div>

  <div class="tada">
★ ★ ★ ★ ★ ★ ★ ★
</div>

  <p>以上で第3ステージは終了です。このステージでは、DeZeroの逆伝播を作り変えることで、double backpropが可能になりました。それによって、高階微分を求められ、ニュートン法を実装できました。次のステップからは、今のDeZeroを、ニューラルネットワークに向けて整備していきます。</p>

  <div class="column">

    <h2 id="id_column_3Anewton"><a id="column-1"></a>コラム：ニュートン法とdouble backpropの補足</h2>

    <p>本コラムでは、第3ステージの補足を行います。まずは入力がベクトルの場合のニュートン法について説明します。続いてニュートン法に代わる別の手法を紹介します。そして最後に、double backpropの実用的な用途を紹介します。なお、本コラムはやや高度な内容を扱い、数式を多用します。難しく感じる場合は、飛ばして先に進んでください（本コラムの内容は、これ以降の内容に深く関連しません）。</p>

    <h4 id="h36-2-0-1">多変数関数のニュートン法</h4>

    <p>私たちは第3ステージでニュートン法を実装しました。そこでは、<span class="equation mathimage"><img alt="y = x^4 - 2x^2" class="math_gen_e24d7633b9b7b0924a3a6dddbd0fad8108685d28fbbf293576dacc82a45b3900" src="images/_review_math/_gen_e24d7633b9b7b0924a3a6dddbd0fad8108685d28fbbf293576dacc82a45b3900.png"/></span>という数式に対し、ニュートン法を使って最小値を求めました。見てのとおり、この式の入力変数は<span class="equation mathimage"><img alt="x" class="math_gen_2d711642b726b04401627ca9fbac32f5c8530fb1903cc4db02258717921a4881" src="images/_review_math/_gen_2d711642b726b04401627ca9fbac32f5c8530fb1903cc4db02258717921a4881.png"/></span>だけです。そのため、私たちが行ったことは、正確には「入力変数が1変数（スカラ）の場合のニュートン法を実装した」ということになります。</p>

    <p>それでは、入力が多次元配列の場合のニュートン法について見ていきましょう。ここでは、入力変数をベクトル<span class="equation mathimage"><img alt="\mathbf{x}" class="math_gen_e23b7458c4e2634af1a37ca43d5c2d9039d375c394b01f4d35b93137eb90898c" src="images/_review_math/_gen_e23b7458c4e2634af1a37ca43d5c2d9039d375c394b01f4d35b93137eb90898c.png"/></span>として、関数<span class="equation mathimage"><img alt="y=f(\mathbf{x})" class="math_gen_bc6e8e173caae8d40c6df1259ddb72d8efede6f5d21eb0e25f49cdd3cc6b9fcc" src="images/_review_math/_gen_bc6e8e173caae8d40c6df1259ddb72d8efede6f5d21eb0e25f49cdd3cc6b9fcc.png"/></span>の場合を考えます。このとき、<span class="equation mathimage"><img alt="\mathbf x" class="math_gen_2e6d3d996f44d01d85ba85c49b04a9698c99fc911ad2aad94c63d9b1d7b8f33e" src="images/_review_math/_gen_2e6d3d996f44d01d85ba85c49b04a9698c99fc911ad2aad94c63d9b1d7b8f33e.png"/></span>はベクトルで、<span class="equation mathimage"><img alt="\mathbf{x} = (x_1, x_2, \cdots,x_n)" class="math_gen_1b34592e65d072d6b5f5c1d369a15811958cd639960ee6f92cf88357d42248aa" src="images/_review_math/_gen_1b34592e65d072d6b5f5c1d369a15811958cd639960ee6f92cf88357d42248aa.png"/></span>と<span class="equation mathimage"><img alt="n" class="math_gen_1b16b1df538ba12dc3f97edbb85caa7050d46c148134290feba80f8236c83db9" src="images/_review_math/_gen_1b16b1df538ba12dc3f97edbb85caa7050d46c148134290feba80f8236c83db9.png"/></span>個の要素を持つものとします。</p>

    <div class="caution">

      <table class="note">

        <tr>
          <td class="center top" rowspan="2">
            <img alt="[警告]" class="warningicon" src="images/warning.png"/>
          </td>
        </tr>

        <tr>
          <td>
            <p>本書の数式表記では、変数がスカラでない場合、<span class="equation mathimage"><img alt="\mathbf{x}" class="math_gen_e23b7458c4e2634af1a37ca43d5c2d9039d375c394b01f4d35b93137eb90898c" src="images/_review_math/_gen_e23b7458c4e2634af1a37ca43d5c2d9039d375c394b01f4d35b93137eb90898c.png"/></span>のように太字で表すことにします。変数がスカラの場合は、<span class="equation mathimage"><img alt="x" class="math_gen_2d711642b726b04401627ca9fbac32f5c8530fb1903cc4db02258717921a4881" src="images/_review_math/_gen_2d711642b726b04401627ca9fbac32f5c8530fb1903cc4db02258717921a4881.png"/></span>のように通常の太さで表記します。</p>
          </td>
        </tr>

      </table>

    </div>

    <p>それでは、<span class="equation mathimage"><img alt="y=f(\mathbf{x})" class="math_gen_bc6e8e173caae8d40c6df1259ddb72d8efede6f5d21eb0e25f49cdd3cc6b9fcc" src="images/_review_math/_gen_bc6e8e173caae8d40c6df1259ddb72d8efede6f5d21eb0e25f49cdd3cc6b9fcc.png"/></span>に対するニュートン法を次に示します。</p>

    <div class="caption-equation" id="eqc-1">

      <table>

        <tr>
          <td>
            <div class="equation">

              <img alt="\mathbf{x} \leftarrow \mathbf{x} - [\nabla ^2 f(\mathbf{x})]^{-1} \nabla  f (\mathbf{x})" class="math_gen_1ddae773c932d6975268d5410b13d517cee898d99adf5496c20323c9ad511cab" src="images/_review_math/_gen_1ddae773c932d6975268d5410b13d517cee898d99adf5496c20323c9ad511cab.png"/>

            </div>
          </td>
          <td class="mathno">(C.1)</td>
        </tr>

      </table>

    </div>

    <p>まずは記号の説明から始めます。<span class="eqref"><a href="./ch36.xhtml#eqc-1">式(C.1)</a></span>の<span class="equation mathimage"><img alt="\nabla f(\mathbf{x})" class="math_gen_aa92700d5bb7c7294a62c2bb3deb78d20cf909cda3c5ad27a360f040c19f4d54" src="images/_review_math/_gen_aa92700d5bb7c7294a62c2bb3deb78d20cf909cda3c5ad27a360f040c19f4d54.png"/></span>は勾配（gradient）を表します。勾配は<span class="equation mathimage"><img alt="\mathbf{x}" class="math_gen_e23b7458c4e2634af1a37ca43d5c2d9039d375c394b01f4d35b93137eb90898c" src="images/_review_math/_gen_e23b7458c4e2634af1a37ca43d5c2d9039d375c394b01f4d35b93137eb90898c.png"/></span>の各要素に関しての微分です。実際に、その要素を書くと次のようになります。</p>

    <div class="caption-equation" id="eqc-2">

      <table>

        <tr>
          <td>
            <div class="equation">

              <img alt="\nabla f(\mathbf{x}) =  \begin{pmatrix} \frac{\partial f}{\partial x_1} \\ \frac{\partial f}{\partial x_2} \\ \vdots \\ \frac{\partial f}{\partial x_n} \\ \end{pmatrix}" class="math_gen_8c365f38208fa639732e76ba7461c3bd13806060e450a304344b40cc84e08b00" src="images/_review_math/_gen_8c365f38208fa639732e76ba7461c3bd13806060e450a304344b40cc84e08b00.png"/>

            </div>
          </td>
          <td class="mathno">(C.2)</td>
        </tr>

      </table>

    </div>

    <p>また、<span class="equation mathimage"><img alt="\nabla ^2 f(\mathbf{x})" class="math_gen_0f30a9c3d37d0b94d1b1b9592cc3dc86109f077f707625e74efcf710a0bd82e9" src="images/_review_math/_gen_0f30a9c3d37d0b94d1b1b9592cc3dc86109f077f707625e74efcf710a0bd82e9.png"/></span>はヘッセ行列<!-- IDX:ヘッセ行列 -->（Hessian matrix<!-- IDX:Hessian matrix -->）です。ヘッセ行列は次の式で表されます。</p>

    <div class="caption-equation" id="eqc-3">

      <table>

        <tr>
          <td>
            <div class="equation">

              <img alt="\nabla ^2 f(\mathbf{x}) =  \begin{pmatrix} \frac{\partial^2 f}{\partial x_1^2} &amp; \frac{\partial^2 f}{\partial x_1 \partial x_2} &amp; \cdots &amp; \frac{\partial^2 f}{\partial x_1 \partial x_n}\\ \frac{\partial^2 f}{\partial x_2 \partial x_1} &amp; \frac{\partial^2 f}{\partial x_2^2} &amp; \cdots &amp; \frac{\partial^2 f}{\partial x_2 \partial x_n}\\ \vdots &amp;  \vdots  &amp; \ddots &amp; \vdots \\ \frac{\partial^2 f}{\partial x_n \partial x_1} &amp; \frac{\partial^2 f}{\partial x_n \partial x_2} &amp; \cdots &amp; \frac{\partial^2 f}{\partial x_n^2} \\ \end{pmatrix}" class="math_gen_85b5101d84b9743fe07495a818e50c89073dc93b806759346c85d4cd75fca200" src="images/_review_math/_gen_85b5101d84b9743fe07495a818e50c89073dc93b806759346c85d4cd75fca200.png"/>

            </div>
          </td>
          <td class="mathno">(C.3)</td>
        </tr>

      </table>

    </div>

    <p><span class="eqref"><a href="./ch36.xhtml#eqc-3">式(C.3)</a></span>のとおり、ヘッセ行列は、<span class="equation mathimage"><img alt="\mathbf{x}" class="math_gen_e23b7458c4e2634af1a37ca43d5c2d9039d375c394b01f4d35b93137eb90898c" src="images/_review_math/_gen_e23b7458c4e2634af1a37ca43d5c2d9039d375c394b01f4d35b93137eb90898c.png"/></span>の2つの要素に関する微分となります。2つの要素の組み合わせになるため、行列の形で定義されます。</p>

    <div class="note">

      <table class="note">

        <tr>
          <td class="center top" rowspan="2">
            <img alt="[注記]" class="noteicon" src="images/note.png"/>
          </td>
        </tr>

        <tr>
          <td>
            <p>勾配<span class="equation mathimage"><img alt="\nabla f(\mathbf{x})" class="math_gen_aa92700d5bb7c7294a62c2bb3deb78d20cf909cda3c5ad27a360f040c19f4d54" src="images/_review_math/_gen_aa92700d5bb7c7294a62c2bb3deb78d20cf909cda3c5ad27a360f040c19f4d54.png"/></span>は、<span class="equation mathimage"><img alt="\frac{\partial f}{\partial \mathbf{x}}" class="math_gen_580cb1613012983261d2ab49ca84f508980006db53ef2403408286e899d8a156" src="images/_review_math/_gen_580cb1613012983261d2ab49ca84f508980006db53ef2403408286e899d8a156.png"/></span>と表記することもできます。また、ヘッセ行列<span class="equation mathimage"><img alt="\nabla ^2 f(\mathbf{x})" class="math_gen_0f30a9c3d37d0b94d1b1b9592cc3dc86109f077f707625e74efcf710a0bd82e9" src="images/_review_math/_gen_0f30a9c3d37d0b94d1b1b9592cc3dc86109f077f707625e74efcf710a0bd82e9.png"/></span>は、<span class="equation mathimage"><img alt="\frac{\partial ^2 f}{\partial \mathbf{x}\partial \mathbf{x}^\mathrm{T}}" class="math_gen_183ac3bdcb361fb455399ce5fc4e3e489e09c9f68c17c944086a2899ea7c0350" src="images/_review_math/_gen_183ac3bdcb361fb455399ce5fc4e3e489e09c9f68c17c944086a2899ea7c0350.png"/></span>と表記することもできます。</p>
          </td>
        </tr>

      </table>

    </div>

    <p><span class="eqref"><a href="./ch36.xhtml#eqc-1">式(C.1)</a></span>では、勾配とヘッセ行列を使って、<span class="equation mathimage"><img alt="\mathbf{x}" class="math_gen_e23b7458c4e2634af1a37ca43d5c2d9039d375c394b01f4d35b93137eb90898c" src="images/_review_math/_gen_e23b7458c4e2634af1a37ca43d5c2d9039d375c394b01f4d35b93137eb90898c.png"/></span>を更新しています（<span class="eqref"><a href="./ch36.xhtml#eqc-1">式(C.1)</a></span>には、<span class="equation mathimage"><img alt="[\nabla ^2 f(\mathbf{x})]^{-1}" class="math_gen_ff64057206ca16f566dc7c45f2c56dd75c66c077fb49973dcb1f423e85110ca5" src="images/_review_math/_gen_ff64057206ca16f566dc7c45f2c56dd75c66c077fb49973dcb1f423e85110ca5.png"/></span>とありますが、これはヘッセ行列<span class="equation mathimage"><img alt="\nabla ^2 f(\mathbf{x})" class="math_gen_0f30a9c3d37d0b94d1b1b9592cc3dc86109f077f707625e74efcf710a0bd82e9" src="images/_review_math/_gen_0f30a9c3d37d0b94d1b1b9592cc3dc86109f077f707625e74efcf710a0bd82e9.png"/></span>の逆行列を表します）。このとき、<span class="equation mathimage"><img alt="\mathbf{x}" class="math_gen_e23b7458c4e2634af1a37ca43d5c2d9039d375c394b01f4d35b93137eb90898c" src="images/_review_math/_gen_e23b7458c4e2634af1a37ca43d5c2d9039d375c394b01f4d35b93137eb90898c.png"/></span>を勾配方向へと更新し、その進む距離を、ヘッセ行列の逆行列を使って調整します。ヘッセ行列という2階微分の情報の利用により、よりアグレッシブに進むことができ、早く目的地に辿り着くことが期待できます。しかし残念ながら、機械学習――特にニューラルネットワーク――において、ニュートン法は滅多に使われません。</p>

    <h4 id="h36-2-0-2">ニュートン法の問題点</h4>

    <p>機械学習などの問題においては、ニュートン法には大きな問題があります。それは、パラメータの数が多くなると、ニュートン法のヘッセ行列――正確には、ヘッセ行列の逆行列――の計算量が大きくなりすぎるという問題です。具体的には、パラメータの数が<span class="equation mathimage"><img alt="n" class="math_gen_1b16b1df538ba12dc3f97edbb85caa7050d46c148134290feba80f8236c83db9" src="images/_review_math/_gen_1b16b1df538ba12dc3f97edbb85caa7050d46c148134290feba80f8236c83db9.png"/></span>個の場合、<span class="equation mathimage"><img alt="n^2" class="math_gen_12dcb271d25c745a1e7dba54909f819492e795a52d2f1754c5d6d6f51bca57c6" src="images/_review_math/_gen_12dcb271d25c745a1e7dba54909f819492e795a52d2f1754c5d6d6f51bca57c6.png"/></span>のオーダのメモリスペースが必要になります。また、<span class="equation mathimage"><img alt="n \times n" class="math_gen_914b998e86b35790d59614955d38bcccf0bd6dc1b0b02477bd9ff742fe409472" src="images/_review_math/_gen_914b998e86b35790d59614955d38bcccf0bd6dc1b0b02477bd9ff742fe409472.png"/></span>の逆行列の計算には、<span class="equation mathimage"><img alt="n^3" class="math_gen_0969e62c250901c5280db9ebffe5e347413aeedbe8be07fa6012ebe1add4a392" src="images/_review_math/_gen_0969e62c250901c5280db9ebffe5e347413aeedbe8be07fa6012ebe1add4a392.png"/></span>のオーダの計算量が必要になります。</p>

    <div class="note">

      <table class="note">

        <tr>
          <td class="center top" rowspan="2">
            <img alt="[注記]" class="noteicon" src="images/note.png"/>
          </td>
        </tr>

        <tr>
          <td>
            <p>ニューラルネットワークの場合、そのパラメータの数が100万を超えることは普通にありえます。もし100万個のパラメータをニュートン法によって更新するとすれば、100万×100万のサイズのヘッセ行列が必要になります。しかし、そのような巨大なサイズを収めるメモリは現実的ではありません。</p>
          </td>
        </tr>

      </table>

    </div>

    <p>ニュートン法は現実的な解決法でない場合が多いことから、それに代わる別のアプローチも提案されています。その代表例が<!-- IDX:準ニュートン法 --><b>準ニュートン法</b>です。準ニュートン法は、ニュートン法における「ヘッセ行列の逆行列」を近似して使用する手法の総称です（準ニュートン法という具体的な手法が存在するわけではありません）。</p>

    <p>準ニュートン法は、これまでいくつかの手法が提案されています。その中でも有名なのがL-BFGS<!-- IDX:L-BFGS -->という手法です。L-BFGSは、勾配だけからヘッセ行列を近似します。それによって計算コストとメモリスペースを節約します。PyTorchでは、L-BFGS<a href="bib.xhtml#bib-pytorch2">[20]</a>は実装されており、気軽に試すことができます。ただしディープラーニングの分野では、今のところ勾配だけを使った最適化――SGD、Momentum、Adamなど――が主流です。L-BFGSなどの準ニュートン法が使われる例は多くはありません。</p>

    <h4 id="h36-2-0-3">double backpropの用途：ヘッセ行列とベクトルの積</h4>

    <p>最後に、double backpropについて補足します。double backpropの使用用途に、「ヘッセ行列とベクトルの積<!-- IDX:ヘッセ行列とベクトルの積 -->（Hessian-vector product<!-- IDX:Hessian-vector product -->）」の計算があります。ヘッセ行列を求める計算コストは、先ほど述べたとおり、要素数が大きくなると膨大になります。しかし、ヘッセ行列とベクトルの積の「結果」だけが必要であれば、double backpropを使い、効率良く求めることができます。</p>

    <p>たとえば、<span class="equation mathimage"><img alt="y=f(\mathbf{x})" class="math_gen_bc6e8e173caae8d40c6df1259ddb72d8efede6f5d21eb0e25f49cdd3cc6b9fcc" src="images/_review_math/_gen_bc6e8e173caae8d40c6df1259ddb72d8efede6f5d21eb0e25f49cdd3cc6b9fcc.png"/></span>と<span class="equation mathimage"><img alt="\mathbf{v}" class="math_gen_888480f5c3b2d1d0d6e51edbac94c26aee4c5ff3dbab9b1eb7b6633d214734e5" src="images/_review_math/_gen_888480f5c3b2d1d0d6e51edbac94c26aee4c5ff3dbab9b1eb7b6633d214734e5.png"/></span>を考え、ヘッセ行列を<span class="equation mathimage"><img alt="\nabla ^2 f(\mathbf{x})" class="math_gen_0f30a9c3d37d0b94d1b1b9592cc3dc86109f077f707625e74efcf710a0bd82e9" src="images/_review_math/_gen_0f30a9c3d37d0b94d1b1b9592cc3dc86109f077f707625e74efcf710a0bd82e9.png"/></span>で表します。このとき、<span class="equation mathimage"><img alt="\nabla ^2 f(\mathbf{x}) \mathbf{v}" class="math_gen_0e19357647e2ad431fd35510ad0af268b49b40fcd8b95d689073f40b2d8c711d" src="images/_review_math/_gen_0e19357647e2ad431fd35510ad0af268b49b40fcd8b95d689073f40b2d8c711d.png"/></span>――ヘッセ行列<span class="equation mathimage"><img alt="\nabla ^2 f(\mathbf{x})" class="math_gen_0f30a9c3d37d0b94d1b1b9592cc3dc86109f077f707625e74efcf710a0bd82e9" src="images/_review_math/_gen_0f30a9c3d37d0b94d1b1b9592cc3dc86109f077f707625e74efcf710a0bd82e9.png"/></span>とベクトル<span class="equation mathimage"><img alt="\mathbf{v}" class="math_gen_888480f5c3b2d1d0d6e51edbac94c26aee4c5ff3dbab9b1eb7b6633d214734e5" src="images/_review_math/_gen_888480f5c3b2d1d0d6e51edbac94c26aee4c5ff3dbab9b1eb7b6633d214734e5.png"/></span>の積――を求めたいとします。それには、次の式変換を行います。</p>

    <div class="caption-equation" id="eqc-4">

      <table>

        <tr>
          <td>
            <div class="equation">

              <img alt="\nabla ^2 f(\mathbf{x}) \mathbf{v} = \nabla (\mathbf{v} ^ {\mathrm{T}} \nabla f(\mathbf{x}))" class="math_gen_498cf1290851ec69e97af274ee19656e0abbea1188ffa1fdb2417833db8c5328" src="images/_review_math/_gen_498cf1290851ec69e97af274ee19656e0abbea1188ffa1fdb2417833db8c5328.png"/>

            </div>
          </td>
          <td class="mathno">(C.4)</td>
        </tr>

      </table>

    </div>

    <p>この変換が成り立つことは、右辺と左辺の要素を書き下せば分かります。実際に、要素数が2のベクトルの場合に限定して式を展開すると、次のようになります。</p>

    <div class="caption-equation">

      <table>

        <tr>
          <td>
            <div class="equation">

              <img alt="\begin{aligned} \nabla ^2 f(\mathbf{x}) \mathbf{v} &amp;= \begin{pmatrix} \frac{\partial^2 f}{\partial x_1^2} &amp; \frac{\partial^2 f}{\partial x_1 \partial x_2}\\ \frac{\partial^2 f}{\partial x_2 \partial x_1} &amp; \frac{\partial^2 f}{\partial x_2^2}\\ \end{pmatrix} \begin{pmatrix}v_1 \\ v_2\end{pmatrix} \\ &amp;= \begin{pmatrix} \frac{\partial^2 f}{\partial x_1^2}v_1 +  \frac{\partial^2 f}{\partial x_1 \partial x_2}v_2\\  \frac{\partial^2 f}{\partial x_2 \partial x_1}v_1 + \frac{\partial^2 f}{\partial x_2^2}v_2 \end{pmatrix} \\[3pt] % \end{aligned} % //} %  % //texequation{ % \begin{aligned} \nabla ({\mathbf{v}} ^ \mathrm{T} \nabla f(\mathbf{x}))  &amp;= \nabla (\begin{pmatrix}v_1 &amp; v_2\end{pmatrix}\begin{pmatrix}\frac{\partial f}{\partial x_1} \\ \frac{\partial f}{\partial x_2}\end{pmatrix}) \\ &amp;= \nabla (\frac{\partial f}{\partial x_1} v_1+  \frac{\partial f}{\partial x_2}v_2) \\ &amp;=  \begin{pmatrix} \frac{\partial^2 f}{\partial x_1^2}v_1 +  \frac{\partial^2 f}{\partial x_1 \partial x_2}v_2\\  \frac{\partial^2 f}{\partial x_2 \partial x_1}v_1 + \frac{\partial^2 f}{\partial x_2^2}v_2 \end{pmatrix} \end{aligned}" class="math_gen_9b0c1e0ea5f2b684b30b580f7b5862ba388977c6bb279ca1395554c3d60062d3" src="images/_review_math/_gen_9b0c1e0ea5f2b684b30b580f7b5862ba388977c6bb279ca1395554c3d60062d3.png"/>

            </div>
          </td>
        </tr>

      </table>

    </div>

    <p>ここでは要素数が2のベクトルの場合に限定しましたが、これは要素数が<span class="equation mathimage"><img alt="n" class="math_gen_1b16b1df538ba12dc3f97edbb85caa7050d46c148134290feba80f8236c83db9" src="images/_review_math/_gen_1b16b1df538ba12dc3f97edbb85caa7050d46c148134290feba80f8236c83db9.png"/></span>の場合へと容易に拡張できます。これより、<span class="eqref"><a href="./ch36.xhtml#eqc-4">式(C.4)</a></span>が成り立つことが分かります。</p>

    <p>それでは改めて、<span class="eqref"><a href="./ch36.xhtml#eqc-4">式(C.4)</a></span>を見てみましょう。<span class="eqref"><a href="./ch36.xhtml#eqc-4">式(C.4)</a></span>の右辺が意味することは、ベクトル<span class="equation mathimage"><img alt="\mathbf{v}" class="math_gen_888480f5c3b2d1d0d6e51edbac94c26aee4c5ff3dbab9b1eb7b6633d214734e5" src="images/_review_math/_gen_888480f5c3b2d1d0d6e51edbac94c26aee4c5ff3dbab9b1eb7b6633d214734e5.png"/></span>と勾配<span class="equation mathimage"><img alt="\nabla f(\mathbf{x})" class="math_gen_aa92700d5bb7c7294a62c2bb3deb78d20cf909cda3c5ad27a360f040c19f4d54" src="images/_review_math/_gen_aa92700d5bb7c7294a62c2bb3deb78d20cf909cda3c5ad27a360f040c19f4d54.png"/></span>の積――ベクトルの内積――を先に求めて、その結果に対してさらに勾配を求めるということです。これによって、ヘッセ行列を作らなくて済むので計算効率が良くなります。</p>

    <p>それでは、DeZeroを使ってヘッセ行列とベクトルの積を求めてみましょう。要素数が2のベクトルを使った計算例を示します（ここでは、行列の積を求める<code class="inline-code tt">F.matmul</code>関数を先取りして利用します）。</p>

    <div class="emlist-code">

      <pre class="emlist language-python highlight"><code class="kn">import</code> <code class="nn">numpy</code> <code class="kn">as</code> <code class="nn">np</code>
<code class="kn">from</code> <code class="nn">dezero</code> <code class="kn">import</code> <code class="n">Variable</code>
<code class="kn">import</code> <code class="nn">dezero.functions</code> <code class="kn">as</code> <code class="nn">F</code>

<code class="n">x</code> <code class="o">=</code> <code class="n">Variable</code><code class="p">(</code><code class="n">np</code><code class="o">.</code><code class="n">array</code><code class="p">([</code><code class="mf">1.0</code><code class="p">,</code> <code class="mf">2.0</code><code class="p">]))</code>
<code class="n">v</code> <code class="o">=</code> <code class="n">Variable</code><code class="p">(</code><code class="n">np</code><code class="o">.</code><code class="n">array</code><code class="p">([</code><code class="mf">4.0</code><code class="p">,</code> <code class="mf">5.0</code><code class="p">]))</code>

<code class="k">def</code> <code class="nf">f</code><code class="p">(</code><code class="n">x</code><code class="p">):</code>
    <code class="n">t</code> <code class="o">=</code> <code class="n">x</code> <code class="o">**</code> <code class="mi">2</code>
    <code class="n">y</code> <code class="o">=</code> <code class="n">F</code><code class="o">.</code><code class="n">sum</code><code class="p">(</code><code class="n">t</code><code class="p">)</code>
    <code class="k">return</code> <code class="n">y</code>

<code class="n">y</code> <code class="o">=</code> <code class="n">f</code><code class="p">(</code><code class="n">x</code><code class="p">)</code>
<code class="n">y</code><code class="o">.</code><code class="n">backward</code><code class="p">(</code><code class="n">create_graph</code><code class="o">=</code><code class="bp">True</code><code class="p">)</code>

<code class="n">gx</code> <code class="o">=</code> <code class="n">x</code><code class="o">.</code><code class="n">grad</code>
<code class="n">x</code><code class="o">.</code><code class="n">cleargrad</code><code class="p">()</code>

<code class="n">z</code> <code class="o">=</code> <code class="n">F</code><code class="o">.</code><code class="n">matmul</code><code class="p">(</code><code class="n">v</code><code class="p">,</code> <code class="n">gx</code><code class="p">)</code>
<code class="n">z</code><code class="o">.</code><code class="n">backward</code><code class="p">()</code>
<code class="k">print</code><code class="p">(</code><code class="n">x</code><code class="o">.</code><code class="n">grad</code><code class="p">)</code>
</pre>

    </div>

    <div class="cmd-code">

      <p class="caption">実行結果</p>

      <pre class="cmd">variable([ 8. 10.])
</pre>

    </div>

    <p>上のコードを数式で表すと、<span class="equation mathimage"><img alt="\nabla (\mathbf{v} ^ \mathrm{T} \nabla f(\mathbf{x}))" class="math_gen_e92f2b5a48771d017f9ea4066934b9dd11aec91442892c955c1c8798e434f8a1" src="images/_review_math/_gen_e92f2b5a48771d017f9ea4066934b9dd11aec91442892c955c1c8798e434f8a1.png"/></span>に対応します。<span class="equation mathimage"><img alt="\mathbf{v} ^ \mathrm{T} \nabla f(\mathbf{x})" class="math_gen_2b8e068781e209869797895ce13915421899d740df2923d9584e9b3e4e74cf87" src="images/_review_math/_gen_2b8e068781e209869797895ce13915421899d740df2923d9584e9b3e4e74cf87.png"/></span>の計算が、<code class="inline-code tt">z = F.matmul(v, gx)</code>に対応します。そして、<code class="inline-code tt">z.backward()</code>で、<code class="inline-code tt">z</code>に対してさらに勾配を求めます。これにより、ヘッセ行列とベクトルの積が求められます。ちなみに、上の出力は正しい結果です。以上で本コラムは終了です。</p>

  </div>

</body>

</html>

""".replace('<p><br/></p>', '')


tran = u"""


Step 36

Applications other than higher-level differentiation

So far, we have been using DeZero to search for higher-level differentiation. What we've done for that is essentially one. It also made a "connection" for calculations performed by reverse propagation. The key here is that reverse propagation computational graphing is a new feature of DeZero. Higher-order differentiation is just one application. Here, we will look at the new DeZero uses other than higher-order differentiation.

The new DeZero can further reverse propagation of calculations made by reverse propagation. Its functionality is called double backpropagation (hereinafter referred to as double backprop). double backprop supports most modern deep learning frameworks.

36.1 What to use double backprop

Let's take a look at some uses other than higher-level differentiation for double backprop. First, consider the following problem:

This is a differential problem that we've seen so far. What's different is that the expression (see 36.2) contains derivatives. In other words, you need to find more derivatives for expressions that contain differentiation. This problem can also be calculated by double backprop. Before explaining that, let's first calculate by hand. To do this, expand the expression as follows:

Once you've expanded the expression as described above, you can assign the to.

The formula above is an expression of, not a value. If you find the value of here, and assign it to , you will not get the correct result.

With the above in mind, let's solve the above problem using DeZero. The code looks like this:

Results of execution

An important point in this code is y.backward(create_graph=True). There, reverse propagation is performed to determine the differentiation. This creates a new calculation graph (in this case, a calculated graph of 2*x is created out of the user's view). Then, a new calculation is performed using the calculation graph created by the reverse propagation, and further reverse propagation is performed. Doing so requires the correct differentiation.

The gx = x.grad in the code above is not just a variable (value), but a computational graph (expression). Therefore, further reverse propagation can be performed on x.grad calculation graphs.

These problems -- the problem of determining an expression for a differentiation, calculating using that expression, and further determining the differentiation -- can be solved using double backprop. These applications can also be found in deep learning research. Here are some examples:

36.2 Use cases in deep learning research

There are a number of studies that use double backprop for deep learning-related applications. For example, a paper called WGAN-GP[21] optimizes the formula represented in Figure 36-1.

Fig. 36-1 Functions for optimizing WGAN-GP (equation quoted from literature [21])

What we want to focus on in Figure 36-1 is that the optimization expression has a slope (the slope is a derivative of each element of the tensor). Its slope can be determined by the first reverse propagation. Then, in order to calculate the function using that gradient and optimize the function, we do a second reverse propagation.

In this way, double backprop is also used in the latest research. In addition to WGAN-GP, double backprop features have actually been used in well-known studies such as MAML[22] and TRPO[23].

TRPO uses double backprop to find the product of hesse matrices and vectors. Double backprop can be used to perform that calculation efficiently. The product of hesse matrices and vectors is described in the following column: Newton Method and Double Backprop Supplement.

This concludes the third stage. At this stage, double backprop is now possible by recreating deZero reverse propagation. As a result, higher-level differentiation was required, and the Newton method was implemented. From the next step, we will develop the current DeZero for neural networks.

In this column, I will supplement the third stage. First, let's talk about the Newton method when the input is a vector. Next, I will introduce another alternative to the Newton method. And finally, I will show you the practical uses of double backprop. This column deals with a slightly more advanced content and uses a lot of formulas. If you find it difficult, skip it and move on (the content of this column is not deeply related to the rest of the table).

We implemented the Newton Method in the third stage. There, we used the Newton method to find the minimum value for the formula. As you can see, the only input variable for this expression. So what we did is to be precise: "We implemented the Newton method when the input variable is one variable (scalar)."

Let's look at the Newton method when the input is a multidimensional array. In this case, consider the case of a function, using the input variable as a vector. At this time, assume that it is a vector and has a piece of elements.

In the formula notation in this book, if the variable is not scalar, it should be represented in bold, as shown. If the variable is scalar, it is represented by normal thickness as shown.

Here's the Newton Act for :

Let's start with a description of the symbols. Equation (C.1) represents gradient. The slope is a derivative of each element of . In fact, when you write that element:

is also the Hessian matrix. The hesse matrix is represented by the following formula:

As equation (C.3), the hesse matrix is a differentiation of the two elements of . It is defined in the form of a matrix because it is a combination of two elements.

The slope can also be written as . Hesse matrices can also be written as .

Equation (C.1) uses gradients and hesse matrices to update (equation (C.1) says, but this represents the inverse matrix of the Hesse matrix). This is to update the slope direction and adjust its forward distance using the inverse matrix of the Hesse matrix. By using the information of the second floor differentiation called hesse procession, you can proceed more aggressively and expect to reach your destination faster. Unfortunately, newton methods are rarely used in machine learning, especially neural networks.

In problems such as machine learning, the Newton method has a big problem. The problem is that as the number of parameters increases, the calculation amount of the Hesse matrix of the Newton method -- to be precise, the inverse matrix of the Hesse matrix -- becomes too large. Specifically, if the number of parameters is several, the order's memory space is required. Also, calculating the inverse matrix of requires the amount of the order calculation.

In the case of neural networks, it is usually possible that the number of its parameters exceeds one million. If one million parameters were to be updated by the Newton method, a hesse matrix × a size of one million would be required. However, memory that fits such a huge size is impractical.

Since the Newton method is often not a practical solution, an alternative approach has been proposed. A typical example is the Quasi-Newton Method. The quasi-Newton method is a generic term for the use of the "inverse matrix of hesse matrices" in the Newton method (there is no specific method called quasi-Newton method).

Several techniques have been proposed for quasi-Newton methods. Among them, the L-BFGS method is famous. L-BFGS approximates the Hesse matrix from the slope only. This saves calculation costs and memory space. In PyTorch, L-BFGS[20] is implemented and you can easily try it out. However, in the field of deep learning, optimization using only gradients -- SGD, Momentum, Adam, etc.- is the mainstream for now. There are not many examples where quasi-Newton methods such as L-BFGS are used.

Finally, I'll supplement with double backprop. A use of double backprop is to calculate the Hessian-vector product. As I mentioned earlier, the computational cost of determining a hesse matrix becomes enormous as the number of elements increases. However, if you only need the "result" of the hesse matrix and the product of the vector, you can use double backprop to find it efficiently.

For example, consider the Hesse matrix as . Suppose you want to get a hesse matrix and a product of vectors. To do this, do the following expression conversion:

You can see that this transformation holds by writing down the elements on the right and left sides. In fact, if you expand the expression only for vectors with 2 elements:

We've limited it to vectors with 2 elements, which can be easily extended to the case where the number of elements is . From this, it can be seen that the expression (C.4) holds.

Let's take a look at equation (C.4) again. The right side of equation (C.4) means that the product of the vector and gradient -- the dot product of the vector -- is first, and the gradient is further determined for the result. This makes calculation efficiency more efficient because you don't have to create a hesse matrix.

Now let's use DeZero to get the product of the Hesse matrix and vector. Here is an example of a calculation using a vector with an element number of 2 (here we will use the F.matmul function for the product of the matrix in advance).

Results of execution

The code above is represented by a formula to correspond to . corresponds to z = F.matmul(v, gx). Then, in z.backward(), we get more gradients for z. This requires the product of the hesse matrix and vector. By the way, the above output is the correct result. This concludes this column.

"""
tran = [i for i in tran.split('\n') if len(i) > 4]

# https://ebooks.stackexchange.com/questions/6449/calibre-ignoring-font-settings

def find_next_sequence(_text, _i, _seq = '</p>'):
    while _i != len(_text):                   # for each index
        if _text[_i] == _seq[0]:              # potentially match
            counter = -1                      # counter to iterate _text's chars against _seq chars
            match = True                      # we will check all the chars. if one doesn't match, this sets to False
            for _c in _seq:                   # iterate over all chars we need to check in _seq
                counter += 1                  # index of _text's char to compare with 
                if _text[_i + counter] != _c: # if they don't match, set condition to False
                    match = False  
                    break
            if match:                         # if they match return the original index
                return _i
        _i += 1                               # otherwise check the next one 
    raise Exception(f'did not find sequence {_seq}')





# preprocess: remove alt tags 
# get the indices of where they start: something like alt="f(x)"
alt_indices = []
_i = 0
while _i != len(text):
    _start = find_next_sequence(text, _i, 'alt="') + len('alt="') # first char after alt 
    _end = find_next_sequence(text, _start, '"')  # loc of closing "
    alt_indices.append((_start, _end, 'alt'))
    _i = _end 
    _i += 1
# now we want all the in-between chunks that go up to the start of an alt, and begin agaain after they end
betweens = []
_chunk_start = 0
for i in alt_indices:
    _alt_start, _alt_end, _typ = i                    
    _chunk_end = _alt_start - 5
    betweens.append((_chunk_start, _chunk_end, 'keep'))  
    _chunk_start = _alt_end + 1                              
betweens.append((_chunk_start, len(text), 'keep'))       
# make new text using these subsets 
cleaned = ''
for i in betweens:
    cleaned += text[i[0]:i[1]]
text = cleaned



# part 1. get the chapter's title sections (2 of them)
# _start is always the index of the first char of interest
# _end is always the closing '<' 
_sequence = '<span class="chapno">'
_idx = find_next_sequence(text, 0, _sequence)
name_1_start = _idx + len(_sequence)
name_1_end = find_next_sequence(text, name_1_start, '<')
_sequence = '</span><br/>'
_idx = find_next_sequence(text, name_1_end, _sequence)
name_2_start = _idx + len(_sequence)
name_2_end = find_next_sequence(text, name_2_start, '<')
text[name_1_start:name_1_end]
text[name_2_start:name_2_end]
name_indices = [(name_1_start, name_1_end, 'name'), (name_2_start, name_2_end, 'name')]


# part 2: get indices of paragraphs
paragraph_indices = []
_i = 0
while _i != len(text):
    _start = find_next_sequence(text, _i, '<p>') + 3 # loc of left arrow
    _end = find_next_sequence(text, _start, '</p>')  # loc of right arrow
    # check if there is math in this paragraph
    para_text = text[_start:_end]
    if para_text.find('equation mathimage') > -1:
        paragraph_indices.append((_start, _end, 'para_math'))
    else:
        paragraph_indices.append((_start, _end, 'para'))
    _i = _end 
    _i += 1


# part 3: get indices of figure captions
caption_indices = []
_i = 0 
while _i != len(text):
    _start = find_next_sequence(text, _i, '<p class="caption">') + len('<p class="caption">')
    _end = find_next_sequence(text, _start, '</p>')
    _i = _end 
    _i += 1
    # you can check if the next character isn't '<' so that we aren't in a code box
    if text[_start] == '<':  
        continue 
    caption_indices.append((_start , _end, 'cap'))


# part 4: get indices of sections
section_indices = []
_i = 0
while _i != len(text):
    _start = find_next_sequence(text, _i, '<span class="secno">')
    _mid = find_next_sequence(text, _start, '</span>') + len('</span>')
    _end = find_next_sequence(text, _mid, '</h2>')
    section_indices.append((_mid , _end, 'sec'))
    _i = _end 
    _i += 1


# part 5: get all the chunks that we will keep unchanged 
betweens = []
pre_soy = name_indices + paragraph_indices + caption_indices + section_indices
pre_soy.sort(key=lambda x: x[0])
_chunk_start = 0
for i in pre_soy:
    _para_start, _para_end, _typ = i                      # locations where paragraphs start
    _chunk_end = _para_start                              # keep-chunk ends where paragraph to be replaced starts
    betweens.append((_chunk_start, _chunk_end, 'keep'))   # add this
    _chunk_start = _para_end                              # next keep-chunk starts where paragraph ends 
betweens.append((_chunk_start, len(text), 'keep'))        # add the last one which starts after the last paragraph, till the end




# part 6: sort all
soy = name_indices + paragraph_indices + caption_indices + section_indices + betweens
soy.sort(key=lambda x: x[0])


# part 7: replace the parts
set([i[2] for i in soy])
output = ''
counter = 0 # we iterate over longer list so we need to keep track of which element we are using from shorter list
for i in range(len(soy)):
    _start, _end, _typ = soy[i]

    if _typ == 'name': 
        new_para = tran[counter]

    elif _typ == 'para_math':
        new_para = tran[counter]

        para = text[_start:_end]
        __end = 0
        for j in range(para.count('equation mathimage')):
            __start = find_next_sequence(para, __end, 'equation mathimage') + len('equation mathimage') + 2
            __end = find_next_sequence(para, __start, '</span>')
            math_part = '<span class="equation mathimage">' + para[__start:__end] + '</span>'
            new_para += math_part

    elif _typ == 'para': 
        new_para = tran[counter]

    elif _typ == 'sec': 
        new_para = ' '.join(tran[counter].split(' ')[1:])

    elif _typ == 'cap': 
        new_para = '\n' + tran[counter] + '\n'

    elif _typ == 'keep':
        new_para = text[_start:_end]
        counter -= 1 # same as only adding 1 in all the other cases 

    counter += 1
    output += new_para


print('\n' * 20)
print(output)
assert len(pre_soy) == len(tran), 'translation and parts different lengths!'
