# Spatial Transformer Network
[paper](https://arxiv.org/abs/1506.02025)
[code](https://github.com/vicsesi/PyTorch-STN)
[tutorial](https://pytorch.org/tutorials/intermediate/spatial_transformer_tutorial.html)

## Abstract
畳み込みニューラルネットワークは、非常に強力なモデルクラスであるが、計算量やパラメータ効率の面から、入力データに対する空間的不変性が欠落しているといえる。
本研究では、学習可能な新しいモジュールであるSpatial Transformer を導入し、ネットワーク内でのデータの空間操作を明示的に可能にする。 
この微分可能なモジュールは、既存の畳み込みアーキテクチャに挿入することができ、ニューラルネットワークに、余分な学習監督や最適化プロセスの変更なしに、特徴マップそのものを条件として、特徴マップを能動的に空間変換する能力を与える。 
Spatial Transformerを使用することで、並進、スケール、回転、およびより一般的なワーピングに対する不変性を学習するモデルが得られることを示す。

## 1. Introduction
近年、高速でスケーラブルなend-to-endの学習フレームワークである畳み込みニューラルネットワーク(CNN)[21]の採用により、コンピュータ・ビジョンの状況は劇的に変化し、前進している。  
最近の発明ではないが、分類[19, 28, 35]、localization[31, 37]、semantic segmentaion[24]、行動認識[12, 32]などのタスクで、CNNベースのモデルが最先端の結果を達成している。 
画像を推論できるシステムの望ましい特性は、物体のポーズや部分の変形をテクスチャや形状から切り離すことである。
CNNに局所的なmax-pooling層を導入することで、ネットワークが特徴の位置に対してある程度空間的に不変であることを可能にし、この特性を満たすのに役立っている。 
しかし、一般的にマックス・プーリングの空間的サポートは小さいため(例えば $2\times2$ pixel)、この空間的不変性はmax-poolingと畳み込みの深い階層にわたってのみ実現され、CNNの中間的な特徴マップ(畳み込み層の活性化)は入力データの大きな変換に対して実際には不変ではない[6, 22]。 
このCNNの限界は、データの空間的配置の変化に対処するための、限定された、あらかじめ定義されたプーリング機構しか持たないことに起因する。 
本研究では、空間変換機能を提供するために、標準的なニューラルネットワークアーキテクチャに組み込むことができる、Spatial Transformerモジュールを導入する。 
Spatial Transformerの動作は、個々のデータサンプルに条件付けされ、問題のタスクに適した動作が訓練中に(追加の監視なしで)学習される。
受容野が固定的かつ局所的であるプーリングレイヤーとは異なり、Spatial Transformerモジュールは、各入力サンプルに対して適切な変換を生成することにより、画像(または特徴マップ)を能動的に空間変換することができる動的なメカニズムである。 
変換は(非局所的に)特徴マップ全体に対して実行され、スケーリング、クロッピング、回転、および非剛体変形を含むことができる。 
これにより、Spatial Transformerを含むネットワークは、最も関連性の高い(注目すべき)画像の領域を選択するだけでなく、それらの領域を正規の期待されるポーズに変換し、次のレイヤーでの認識を単純化することができる。 
注目すべきは、空間変換器は標準的なバックプロパゲーションで学習することができ、それらが注入されたモデルのend-to-endの学習が可能であることである。

Spatial TransformerをCNNに組み込むことで、例えば次のような様々なタスクに役立てることができる：
(i)画像分類：CNNが、特定の数字を含むかどうかに応じて画像の多方向分類を行うように学習されているとする；
ここで、桁の位置と大きさはサンプルごとに大きく異なる(そしてクラスとは無相関である)可能性がある；
適切な領域を切り出し、スケール正規化するSpatial Transformerは、その後の分類タスクを単純化し、優れた分類性能につながる；
(ii)共定位：同じ(しかし未知の)クラスの異なるインスタンスを含む画像セットが与えられると、空間変換器を使用して、各画像内でそれらを定位させることができる；
(iii) 空間的注意：Spatial Transformerは、[14, 39]のような注意メカニズムを必要とするタスクに使うことができるが、より柔軟性が高く、強化学習を使わずに純粋にバックプロパゲーションで学習することができる。
アテンションを使用する主な利点は、変換された(つまりアテンションされた)解像度の低い入力を、解像度の高い生の入力に優先して使用することができ、計算効率が向上することである。

本稿の残りの部分は以下のように構成されている： 
Sect. 2で我々の研究に関連するいくつかの研究について述べ、Sect. 3で空間変換器の定式化と実装を紹介し、最後にSect.4.追加の実験を示し、実装の詳細は付録Aに示す。

![Figure1](images/Figure1.png)
(b) The localisation network of the spatial transformer predicts a transformation to apply to the input image. 
(c) The output of the spatial transformer, after applying the transformation.
(d) The classification prediction produced by the subsequent fully-connected network on the output of the spatial transformer. 
The spatial transformer network (a CNN including a spatial transformer module) is trained end-to-end with only class labels – no knowledge of the groundtruth transformations is given to the system.

Figure 1：歪んだMNISTの数字を分類するために訓練された全結合ネットワークの第1層としてSpatial Transformerを使用した結果。 
(a) Spatial Transformerネットワークへの入力は、ランダムな平行移動、スケール、回転、乱雑さで、ゆがまされたMNISTの数字の画像である。 
(b) Spatial Transformerのローカリゼーションネットワークは、入力画像に適用する変換を予測する。 
(c) 変換を適用した後のSpatial Transformerの出力。 
(d) 空間変換器の出力に対して後続の完全連結ネットワークが生成した分類予測。 
空間変換器ネットワーク（空間変換器モジュールを含むCNN）は、クラスラベルのみを用いてエンドツーエンドで学習される。

## 2 Related Work
In this section we discuss the prior work related to the paper, covering the central ideas of modelling transformations with neural networks [15, 16, 36], learning and analysing transformation-invariant representations [4, 6, 10, 20, 22, 33], as well as attention and detection mechanisms for feature selection [1, 7, 11, 14, 27, 29].
Early work by Hinton [15] looked at assigning canonical frames of reference to object parts, a theme which recurred in [16] where 2D affine transformations were modeled to create a generative model composed of transformed parts. 
The targets of the generative training scheme are the transformed input images, with the transformations between input images and targets given as an additional input to the network. 
The result is a generative model which can learn to generate transformed images of objects by composing parts. 
The notion of a composition of transformed parts is taken further by Tieleman [36], where learnt parts are explicitly affine-transformed, with the transform predicted by the network. 
Such generative capsule models are able to learn discriminative features for classification from transformation supervision.
The invariance and equivariance of CNN representations to input image transformations are studied in [22] by estimating the linear relationships between representations of the original and transformed images. 
Cohen & Welling [6] analyse this behaviour in relation to symmetry groups, which is also exploited in the architecture proposed by Gens & Domingos [10], resulting in feature maps that are more invariant to symmetry groups. 
Other attempts to design transformation invariant representa- tions are scattering networks [4], and CNNs that construct filter banks of transformed filters [20, 33].  
Stollenga et al. [34] use a policy based on a network’s activations to gate the responses of the network’s filters for a subsequent forward pass of the same image and so can allow attention to specific features. 
In this work, we aim to achieve invariant representations by manipulating the data rather than the feature extractors, something that was done for clustering in [9].
Neural networks with selective attention manipulate the data by taking crops, and so are able to learn translation invariance. 
Work such as [1, 29] are trained with reinforcement learning to avoid the  need for a differentiable attention mechanism, while [14] use a differentiable attention mechansim by utilising Gaussian kernels in a generative model. 
The work by Girshick et al. [11] uses a region proposal algorithm as a form of attention, and [7] show that it is possible to regress salient regions with a CNN. 
The framework we present in this paper can be seen as a generalisation of differentiable attention to any spatial transformation.

![Figure2](images/Figure2.png)
Figure 2：Spatial Transformer モジュールのアーキテクチャ。 
入力特徴マップ$U$は、変換パラメータ $\theta$ を回帰するlocalizationネットワークに渡される。
$V$ 上の規則的な空間グリッド $G$ はサンプリンググリッド $\Tau_\theta(G)$ に変換され、3.3で説明するように $U$ に適用され、ワープされた出力特徴マップVを生成する。 
localizationネットワークとサンプリングメカニズムの組み合わせは、Spatial Transformerを定義する。

## 3 Spatial Transformers
このセクションでは、Spatial Transformer の定式化について述べる。 
これは微分可能なモジュールであり、1回のフォワードパスで特徴マップに空間変換を適用する。
マルチチャンネル入力の場合、同じワーピングが各チャンネルに適用される。 
簡単のため、このセクションでは、変換器ごとに単一の変換と単一の出力を考えるが、実験で示したように、複数の変換に一般化することができる。
Spatial Transformerの仕組みは、Figure 2に示すように3つの部分に分かれている。
計算の順序としては、まずlocalisation ネットワーク(Section 3.1)が入力特徴マップを受け取り、いくつかの隠れ層を通して、特徴マップに適用すべき空間変換のパラメータを出力する。
この特徴マップが入力に対する変形の制約を与える。
推定された変換パラメータはサンプリンググリッドを生成するために使用される。
サンプリンググリッドとは、変換された出力を生成するために入力マップがサンプリングされるべき点の集合である。 
これはSection 3.2にて説明するgrid generatorによって行われる。
最後に、特徴マップとサンプリンググリッドがサンプラーへの入力として取り込まれ、グリッドポイントで入力からサンプリングされた出力マップが生成される(Section 3.3節)。 
これら3つのコンポーネントの組み合わせは空間変換器を形成し、以下の節でより詳細に説明される。

### 3.1 Localisation Network
localisation ネットワークは、幅 $W$ 、高さ$H$ 、チャンネル $C$ を持つ入力特徴マップ $U\in \mathcal{R}^{H\times W\times C}$ を受け取り $\theta$ を出力する。
$\theta$ は特徴マップに適用される変換 $\Tau_\theta$ のパラメータである：$\theta = f_{loc}(U)$。 
$\theta$ のサイズは、パラメータ化される変換タイプによって変化し、例えばアフィン変換の場合、$\theta$ は式(10)のように6次元である。 
localisation ネットワーク関数 $f_{loc}()$ は、全結合ネットワークや畳み込みネットワークなど、どのような形式をとることもできるが、変換パラメータ $\theta$ を生成するための最終回帰層を含まなければならない。

### 3.2 Parameterised Sampling Grid
入力特徴マップのワーピングを実行するために、各出力ピクセルは、入力特徴マップの特定の位置を中心とするサンプリングカーネルを適用することによって計算される(これについては次のセクションで詳しく説明する)。 
*pixel* とは、一般的な特徴マップの要素のことであり、必ずしも画像のことではない。 
一般に、出力ピクセルは、ピクセル $G_i = (x^t_i,y^t_i)$ 群の規則的なグリッド $G = {G_i}$ 上に位置するように定義され、出力特徴マップ $V\in \mathcal{R}^{H'\times W'\times C}$ を形成する。
ここで、$H'$ と $W'$ はグリッドの高さと幅であり、$C$ はチャンネル数であり、入力と出力で同じである。
説明を明確にするため、とりあえず $\Tau_\theta$ が2次元アフィン変換 $A_\theta$ であると仮定する。 
他の変換については後述する。このアフィンの場合、点ごとの変換は


$$
\left(
\begin{aligned}
    x^s_i\\
    y^s_i
\end{aligned}
\right)
= \Tau_\theta(G_i) = A_\theta
\left(
\begin{aligned}
    x^t_i\\
    y^t_i\\
    1
\end{aligned}
\right) =
\left[
\begin{aligned}
    \theta_{11} \theta_{12} \theta_{13}\\
    \theta_{21} \theta_{22} \theta_{23}
\end{aligned}
\right]
\left(
\begin{aligned}
    x^t_i\\
    y^t_i\\
    1
\end{aligned}
\right) \tag{1}
$$

ここで、$(x^t_i, y^t_i)$ は出力特徴マップの規則的なグリッドのターゲット座標、$(x^s_i, y^s_i)$ はサンプル点を定義する入力特徴マップのソース座標、$A_\theta$ はアフィン変換行列である。
出力の空間的境界内にあるときは $-1\le x^t_i, y^t_i\le 1$、入力の空間的境界内にあるときは $-1\le x^s_i, y^s_i \le 1$ となるように、高さと幅を正規化した座標を使用する( $y$ 座標についても同様)。
ソース／ターゲットの変形およびサンプリングは、グラフィックスで使用される標準的なテクスチャマッピングと座標に相当する[8]。
式 (10)で定義される変換は、入力特徴マップにトリミング、平行移動、回転、スケール、スキューを適用することを可能にし、localisationネットワークによって生成される6つのパラメータ( $A_\theta$ の6要素)のみを必要とする。
これは、変換が縮小である場合(すなわち、左の $2\times 2$ 部分行列の行列式が $1$ より小さい大きさを持つ場合)、写像された正方格子は、$x^s_i, y^s_i$ の範囲より小さい面積の平行四辺形に位置することになるため、トリミングを可能にする。 
この変換がグリッドに与える影響を、同一変換と比較してFigure 3に示す。

![Figure3](images/Figure3.png)
Figure 3：出力 $V$ を生成する画像 $U$ にパラメータ化されたサンプリンググリッドを適用した2つの例。 (a) サンプリンググリッドは正格子 $G = T_I(G)$ であり、$I$ は同一変換パラメータである。 
(b) サンプリンググリッドは、アフィン変換 $T_\theta(G)$ で正規グリッドをワーピングした結果である。


変換 $T_\theta$ のクラスは、注意のために使用されるような、より制約されたものであってもよい。

$$
A_\theta =
\left[
    \begin{aligned}
    s\ 0\ t_x \\
    0\ s\ t_y 
    \end{aligned}
\right] \tag{2}
$$

$s, t_x, t_y$ を変化させることで、トリミング、平行移動、等方的なスケーリングが可能になる。
変換 $T_\theta$ は、8つのパラメータを持つ平面射影変換、ピースワイズアフィン、薄板スプラインなど、より一般的なものにもできる。

実際、この変換は、パラメータに関して微分可能であれば、どのようなパラメータ化された形でもよい。
- これにより、サンプル点 $T_\theta(G_i)$ からlocalisationネットワーク出力 $\theta$ まで、勾配がバックプロパゲートされる。
変換が構造化された低次元の方法でパラメータ化されていれば、ローカリゼーション・ネットワークに割り当てられるタスクの複雑さが軽減される。
例えば、構造化された微分可能な変換の一般的なクラスは、注目変換、アフィン変換、射影変換、薄板スプライン変換のスーパーセットであり、$\Tau_\theta＝M_\theta B$ であり、ここで $B$ はターゲットグリッド表現(例えば(10)では、$B$は同次座標の正格子$G$)であり、$M_\theta$は $\theta$ でパラメータ化された行列である。
この場合、あるサンプルに対する $\theta$ の予測方法を学習するだけでなく、目の前の課題に対する $B$ を学習することも可能である。

### 3.3 Differentiable Image Sampling
入力特徴マップの空間変換を行うために、サンプラーは入力特徴マップ $U$ とともにサンプリング点の集合 $T_\theta(G)$ を取り、サンプリングされた出力特徴マップ $V$ を生成しなければならない。
$T_\theta(G)$ の各座標 $(x^s_i, y^s_i)$ は、出力 $V$ の特定のピクセルでの値を得るためにサンプリングカーネルが適用される入力の空間位置を定義する。 これは次のように書くことができる。

$$
V^c_i = \sum_n^H \sum_m^W U^c_{nm} k(x^s_i - m; \Phi_x) k(y^s_i - n; \Phi_y) \  \forall i \in [1... H'W']\ \  \forall c \in [1...C] \tag{3}
$$

ここで $\Phi_x$ と $\Phi_y$ は、画像補間(バイリニアなど)を定義する一般的なサンプリング・カーネル $k()$ のパラメータ、$U^c{nm}$ は入力のチャンネル $c$ の位置 $(n,m)$ における値、$V^c_i$ はチャンネル $c$ の位置 $(x^t_i, y^t_i)$ におけるピクセル $i$ の出力値である。 
サンプリングは，入力の各チャンネルに対して同じように行われるので，どのチャンネルも同じように変換されることに注意する(これにより，チャンネル間の空間的な一貫性が保たれる)。
理論的には，$x^s_i$ と $y^s_i$ に対して(副)勾配を定義できる限り，どのようなサンプリングカーネルを用いても構わない。
例えば、整数サンプリング・カーネルを使うと、式(3)は次のようになる。

$$
V^c_i = \sum_n^H \sum_m^W U^c_{nm} \delta([x^s_i + 0.5] - m) \delta([y^s_i + 0.5] - n)  \tag{4}
$$

ここで $[x + 0.5]$ は $x$ を最も近い整数に丸めることであり、$\delta()$ はクロネッカーデルタ関数である。
このサンプリングカーネルは、$(x^s_i, y^s_i)$ に最も近いピクセルの値を出力位置 $(x^t_i, y^s_i)$ にコピーするだけに等しい(nearest neighbor)。
あるいは、バイリニアサンプリングカーネルを使うこともでき、次のようになる

$$
V^c_i = \sum_n^H \sum_m^W U^c_{nm} \max(0, 1 - |x^s_i - m|) \max(0, 1 - |y^s_i - n|)  \tag{5}
$$

このサンプリングメカニズムによる損失のバックプロパゲーションを可能にするために、 $U$ と $G$ に関する勾配を定義することができる。
バイリニアサンプリング 式(5)の偏導関数は次のようになる。

$$
\begin{align}
\frac{\partial V^c_i}{\partial U^c_{nm}} = \sum^H_n \sum^W_m \max(0, 1 - |x^s_i - m|)\max(0, 1 - |y^s_i - n|) \tag{6} \\
\frac{\partial V^c_i}{\partial x^s_i} = \sum^H_n \sum^W_m U^c_{nm} \max(0, 1 - |y^s_i - n|) 
\left\{
    \begin{aligned}
    0\ & if\ |m-x^s_i| \ge 1\\
    1\ & if\ m \ge x^s_i\\
    -1\ & if\ m < x^s_i
    \end{aligned}
\right.
\tag{7}
\end{align}
$$

式(7)に関しては $\frac{\partial V^c_i}{\partial x^s_i}$ も同様である。
本式により(副)微分可能なサンプリング機構を与え、損失勾配を入力特徴マップ(6)だけでなく、サンプリンググリッド座標(7)に戻すことができる。
従ってたとえば式(10)のように$\frac{\partial x^s_i}{\partial \theat}$　and $\frac{\partial x^s_i}{\partial \theat}$は簡単に微分できるので、変換パラメータ $\theta$ とlocalisationネットワークに戻すことができる。 
サンプリング関数の不連続性のため、部分勾配を使用する必要がある。 
このサンプリングメカニズムはGPU上で非常に効率的に実装することができ、入力の全ての位置の和を見るのではなく、代わりに各出力画素のカーネルサポート領域を見るだけである。

## 3.4 Spatial Transformer Networks
loalisationネットワーク、グリッド・ジェネレーター、サンプラーを組み合わせると、Spatial Transformerになる(Figure 2)。 
これは自己完結型のモジュールであり、CNNアーキテクチャに任意の時点、任意の数で落とし込むことができ、空間変換ネットワークを生成する。 
このモジュールは計算速度が非常に速く、学習速度を損なわないので、なにも気にせずに使っても時間のオーバーヘッドはほとんど発生せず、変形処理の出力に適用できる後続のダウンサンプリングによって、注意深いモデルでも高速化することができる。
CNN内に空間変換器を配置することで、ネットワークは学習中に特徴マップを能動的に変換する方法を学習し、ネットワークの全体的なコスト関数を最小化することができる。
各トレーニングサンプルをどのように変換するかという知識は、トレーニング中に圧縮され、localisationネットワークの重み(およびSpatial Transformerの前の層の重み)にキャッシュされる。
いくつかのタスクでは、localisationネットワークの出力 $\theta$ をネットワークの残りの部分にフィードフォワードすることも有用である。
出力次元 $H'$ と $W'$ を入力次元 $H$ と $W$ と異なるように定義できるので、特徴マップをダウンサンプリングまたはオーバーサンプリングするためにSpatial Transformerを使用することも可能である。
しかし、固定された小さな空間サポートを持つサンプリングカーネル(バイリニアカーネルなど)では、空間変換器を使用したダウンサンプリングはエイリアシング効果を引き起こす可能性がある。
最後に、CNNでは複数の空間変換器を持つことが可能である。 
複数の空間変換器をネットワークの深さ方向に配置することで、より抽象的な表現に変換することができ、またローカライゼーションネットワークに、予測される変換パラメータのベースとなる、潜在的により有益な表現を与えることができる。 
また、複数の空間変換器を並列に使用することもできます。
これは、特徴マップに複数のオブジェクトや個別に注目すべき部分がある場合に便利です。 
純粋なフィードフォワードネットワークにおけるこのアーキテクチャの限界は、並列空間変換器の数によって、ネットワークがモデル化できるオブジェクトの数が制限されることである。

## 4 Experiments
このセクションでは、多くの教師あり学習タスクにおけるSpatial Transformerネットワークの利用を探求する。 
Section 4.1では、まずMNIST手書きデータセットの歪んだバージョンに対する実験から始める。
4.1では、まずMNIST手書きデータセットの歪んだバージョンに対する実験から始め、入力画像を能動的に変換することによって分類性能を向上させる空間変換器の能力を示す。 
4.2では、Spatial Transformerをテストする。
4.2では、実世界の困難なデータセットであるStreet View House Numbers [25]を用いて、数字認識のためのSpatial Transformerネットワークをテストし、CNNの畳み込みスタックに埋め込まれた複数のSpatial Transformerを用いた最先端の結果を示す。 
最後に、Section 4.3で空間変換器の利用について調べる。
4.3では、CUB-200-2011鳥類データセット[38]において、オブジェクトのパーツを発見し、それらに注目するように学習することで、きめ細かな分類のための複数の並列空間変換器の利用を調査し、最先端の性能を示す。 
MNISTの追加と共局在化の更なる実験は付録Aにある。

### 4.1 Distorted MNIST
In this section we use the MNIST handwriting dataset as a testbed for exploring the range of transformations to which a network can learn invariance to by using a spatial transformer.
We begin with experiments where we train different neural network models to classify MNIST data that has been distorted in various ways: 
rotation (R), rotation, scale and translation (RTS), projective transformation (P), and elastic warping (E) – note that elastic warping is destructive and can not be inverted in some cases. 
The full details of the distortions used to generate this data are given in Appendix A. 
We train baseline fully-connected (FCN) and convolutional (CNN) neural networks, as well as networks with spatial transformers acting on the input before the classification network (ST-FCN and ST-CNN). 
The spatial transformer networks all use bilinear sampling, but variants use different transformation functions: an affine transformation (Aff), projective transformation (Proj), and a 16-point thin plate spline transformation (TPS) [2]. 
The CNN models include two max-pooling layers. 
All networks have approximately the same number of parameters, are trained with identical optimisation schemes (backpropagation, SGD, scheduled learning rate decrease, with a multinomial cross entropy loss), and all with three weight layers in the classification network.
The results of these experiments are shown in Table 1 (left). Looking at any particular type of distortion of the data, it is clear that a spatial transformer enabled network outperforms its counterpart base network. 
For the case of rotation, translation, and scale distortion (RTS), the ST-CNN achieves 0.5% and 0.6% depending on the class of transform used for Tθ, whereas a CNN, with two max- pooling layers to provide spatial invariance, achieves 0.8% error. 
This is in fact the same error that the ST-FCN achieves, which is without a single convolution or max-pooling layer in its network, showing that using a spatial transformer is an alternative way to achieve spatial invariance. 
ST-CNN models consistently perform better than ST-FCN models due to max-pooling layers in ST-CNN providing even more spatial invariance, and convolutional layers better modelling local structure. 
We also test our models in a noisy environment, on 60 ×60 images with translated MNIST digits and background clutter (see Fig. 1 third row for an example): an FCN gets 13.2% error, a CNN gets 3.5% error, while an ST-FCN gets 2.0% error and an ST-CNN gets 1.7% error.
Looking at the results between different classes of transformation, the thin plate spline transformation (TPS) is the most powerful, being able to reduce error on elastically deformed digits by reshaping the input into a prototype instance of the digit, reducing the complexity of the task for the classification network, and does not over fit on simpler data e.g. R. 
Interestingly, the transformation of inputs for all ST models leads to a “standard” upright posed digit – this is the mean pose found in the training data. In Table 1 (right), we show the transformations performed for some test cases where a CNN is unable to correctly classify the digit, but a spatial transformer network can. 
Further test examples are visualised in an animation here https://goo.gl/qdEhUu

![Table1](images/Table1.png)
Table 1: 
左：異なる歪んだMNISTデータセットに対する異なるモデルの誤差の割合。
我々がテストした異なる歪んだMNISTデータセットは以下の通りである。
TC: 平行移動と乱雑化
R: 回転 
RTS: 回転, 平行移動, 拡大縮小
P: 射影変形 
E: 弾性変形
各実験に使用されたモデルはすべて、同じ数のパラメータを持ち、すべての実験において同じ基本構造を持つ。
右：空間変換ネットワークは桁を正しく分類するが、CNNは失敗するテスト画像の例。
(a) ネットワークへの入力
(b) Spatial Transformerの推論した変形。 $\Tau_\theta(G)$ のグリッドで可視化している。
(c) Spatial Transformerの出力
E と RTS の例では  thin plate spline の spatial transformer (ST-CNN TPS)を使用している
一方 R の例では affine変換の spatial transformers (ST-CNN Aff) に角度を空建てて使用している。
これらの実験の動画によるアニメーションは https://goo.gl/qdEhUu を見てください

### 4.2 Street View House Numbers
We now test our spatial transformer networks on a challenging real-world dataset, Street View House Numbers (SVHN) [25]. 
This dataset contains around 200k real world images of house numbers, with the task to recognise the sequence of numbers in each image. 
There are between 1 and 5 digits in each image, with a large variability in scale and spatial arrangement.
We follow the experimental setup as in [1, 13], where the data is preprocessed by taking 64 ×64 crops around each digit sequence. We also use an additional more loosely 128 ×128 cropped dataset as in [1]. 
We train a baseline character sequence CNN model with 11 hidden layers leading to five independent softmax classifiers, each one predicting the digit at a particular position in the sequence.  
This is the character sequence model used in [19], where each classifier includes a null-character output to model variable length sequences. This model matches the results obtained in [13].
We extend this baseline CNN to include a spatial transformer immediately following the input (ST-CNN Single), where the localisation network is a four-layer CNN. We also define another extension where before each of the first four convolutional layers of the baseline CNN, we insert a spatial transformer (ST-CNN Multi), where the localisation networks are all two layer fully connected networks with 32 units per layer. 
In the ST-CNN Multi model, the spatial transformer before the first convolutional layer acts on the input image as with the previous experiments, however the subsequent spatial transformers deeper in the network act on the convolutional feature maps, predicting a
transformation from them and transforming these feature maps (this is visualised in Table 2 (right) (a)). 
This allows deeper spatial transformers to predict a transformation based on richer features rather than the raw image. 
All networks are trained from scratch with SGD and dropout [17], with randomly initialised weights, except for the regression layers of spatial transformers which are initialised to predict the identity transform. 
Affine transformations and bilinear sampling kernels are used for all spatial transformer networks in these experiments.

![Table2](images/Table2.png)
Table 2: Left: The sequence error for SVHN multi-digit recognition on crops of 64 ×64 pixels (64px), and inflated crops of 128 ×128 (128px) which include more background. 
* The best reported result from [1] uses model averaging and Monte Carlo averaging, whereas the results from other models are from a single forward pass of a single model. 
Right: 
(a) The schematic of the ST-CNN Multi model. The transformations applied by each spatial transformer (ST) is applied to the convolutional feature map produced by the previous layer. 
(b) The result of multiplying out the affine transformations predicted by the four spatial transformers in ST-CNN Multi, visualised on the input image.

### 4.3 Fine-Grained Classification
In this section, we use a spatial transformer network with multiple transformers in parallel to perform fine-grained bird classification. 
We evaluate our models on the CUB-200-2011 birds dataset [38], containing 6k training images and 5.8k test images, covering 200 species of birds. 
The birds appear at a range of scales and orientations, are not tightly cropped, and require detailed texture and shape analysis to distinguish. In our experiments, we only use image class labels for training.
We consider a strong baseline CNN model – an Inception architecture with batch normalisation [18] pre-trained on ImageNet [26] and fine-tuned on CUB – which by itself achieves the state-of-the-art accuracy of 82.3% (previous best result is 81.0% [30]). We then train a spatial transformer network, ST-CNN, which contains 2 or 4 parallel spatial transformers, parameterised for attention and acting on the input image. Discriminative image parts, captured by the transformers, are passed to the part description sub-nets (each of which is also initialised by Inception). 
The resulting part representations are concatenated and classified with a single softmax layer. The whole architecture is trained on image class labels end-to-end with backpropagation (full details in Appendix A).  
The results are shown in Table 3 (left). The ST-CNN achieves an accuracy of 84.1%, outperforming the baseline by 1.8%. 
It should be noted that there is a small (22/5794) overlap between the ImageNet training set and CUB-200-2011 test set1 – removing these images from the test set results in 84.0% accuracy with the same ST-CNN. 
In the visualisations of the transforms predicted by 2×ST-CNN (Table 3 (right)) one can see interesting behaviour has been learnt: one spatial transformer(red) has learnt to become a head detector, while the other (green) fixates on the central part of the body of a bird. The resulting output from the spatial transformers for the classification network is a somewhat pose-normalised representation of a bird. 
While previous work such as [3] explicitly define parts of the bird, training separate detectors for these parts with supplied keypoint training data, the ST-CNN is able to discover and learn part detectors in a data-driven manner without any additional supervision. 
In addition, the use of spatial transformers allows us to use 448px resolution input images without any impact in performance, as the output of the transformed 448px images are downsampled to 224px before being processed.

![Table3](images/Table3.png)
Table 3: Left: The accuracy on CUB-200-2011 bird classification dataset. Spatial transformer networks with two spatial transformers (2×ST-CNN) and four spatial transformers (4×ST-CNN) in parallel achieve higher accuracy. 
448px resolution images can be used with the ST-CNN without an increase in computational cost due to downsampling to 224px after the transformers. 
Right: The transformation predicted by the spatial transformers of 2×ST-CNN (top row) and 4×ST-CNN (bottom row) on the input image. Notably for the 2×ST-CNN, one of the transformers (shown in red) learns to detect heads, while the other (shown in green) detects the body, and similarly for the 4×ST-CNN.

## 5 Conclusion
In this paper we introduced a new self-contained module for neural networks – the spatial transformer. 
This module can be dropped into a network and perform explicit spatial transformations of features, opening up new ways for neural networks to model data, and is learnt in an end-to- end fashion, without making any changes to the loss function. 
While CNNs provide an incredibly strong baseline, we see gains in accuracy using spatial transformers across multiple tasks, resulting in state-of-the-art performance. 
Furthermore, the regressed transformation parameters from the spatial transformer are available as an output and could be used for subsequent tasks. 
While we only explore feed-forward networks in this work, early experiments show spatial transformers to be powerful in recurrent models, and useful for tasks requiring the disentangling of object reference frames, as well as easily extendable to 3D transformations (see Appendix A.3).

## A Appendix
In this section we present the results of two further experiments – that of MNIST addition showing spatial transformers acting on multiple objects in Sect. A.1, and co-localisation in Sect. A.2 showing the application to semi-supervised scenarios. 
In addition, we give an example of the extension to 3D in Sect. A.3. We also expand upon the details of the experiments from Sect. 4.1 in Sect. A.4, Sect. 4.2 in Sect. A.5, and Sect. 4.3 in Sect. A.6.

### A.1 MNIST Addition
In this section we demonstrate another use case for multiple spatial transformers in parallel: 
to model multiple objects. We define an MNIST addition task, where the network must output the sum of the two digits given in the input. 
Each digit is presented in a separate 42 ×42 input channel (giving 2-channel inputs), but each digit is transformed independently, with random rotation, scale, and translation (RTS).
We train fully connected (FCN), convolutional (CNN) and single spatial transformer fully connected (ST-FCN) networks, as well as spatial transformer fully connected networks with two parallel spatial transformers (2×ST-FCN) acting on the input image, each one taking both channels as input and transforming both channels. 
The two 2-channel outputs of the two spatial transformers are concatenated into a 4-channel feature map for the subsequent FCN. As in Sect. 4.1, all networks have the same number of parameters, and are all trained with SGD to minimise the multinomial cross entropy loss for 19 classes (the possible addition results 0-18).
The results are given in Table 4 (left). Due to the complexity of this task, the FCN reaches a minimum error of 47.7%, however a CNN with max-pooling layers is far more accurate with 14.7% error. 
Adding a single spatial transformer improves the capability of an FCN by focussing on a single region of the input containing both digits, reaching 18.5% error. 
However, by using two spatial transformers, each transformer can learn to focus on transforming the digit in a single channel (though receiving both channels as input), visualised in Table 4 (right). 
The transformers co-adapt, producing stable representations of the two digits in two of the four output channels of the spatial transformers. 
This allows the 2×ST-FCN model to achieve 5.8% error, far exceeding that of other models.

![Table4](images/Table4.png)
Table 4: Left: The percentage error for the two digit MNIST addition task, where each digit is transformed independently in separate channels, trained by supplying only the label of the sum of the two digits. 
The use of two spatial transformers in parallel, 2×ST-FCN, allows the fully-connected neural network to become invariant to the transformations of each digit, giving the lowest error. 
All the models used for each column have approximately the same number of parameters. 
Right: A test example showing the learnt behaviour of each spatial transformer (using a thin plate spline (TPS) transformation). 
The 2-channel input (the blue bar denotes separation between channels) is fed to two independent spatial transformers, ST1 and ST2, each of which operate on both channels. 
The outputs of ST1 and ST2 and concatenated and used as a 4-channel input to a fully connected network (FCN) which predicts the addition of the two original digits. 
During training, the two spatial transformers co-adapt to focus on a single channel each.

![Table5](images/Table5.png)
Table 5: Left: The percent of correctly co-localised digits for different MNIST digit classes, for just translated digits (T), and for translated digits with clutter added (TC). 
Right: The optimisation architecture. We use a hinge loss to enforce the distance between the two outputs of the spatial transformer (ST) to be less than the distance to a random crop, hoping to encourage the spatial transformer to localise the common objects.