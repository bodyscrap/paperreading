# Exploring Simple Siamese Representation Learning
[paper](https://arxiv.org/abs/2011.10566)
[code](https://github.com/facebookresearch/simsiam)

## Abstract
siamese networkは、教師なし視覚表現学習のための最近の様々なモデルにおいて一般的な構造となっている。
これらのモデルは、解の崩壊を回避するためのある条件に従って、1つの画像の2つの拡張間の類似度を最大化する。
本論文では、単純なsiamese networkが、以下のいずれを用いなくても、意味のある表現を学習できるという驚くべき実証結果を報告する： (i)負のサンプルペア、(ii)大きなバッチ、(iii)momentum encoder
我々の実験によれば、損失と構造には破綻解が存在するが、破綻を防ぐためにはstop-gradient(勾配停止)演算が重要な役割を果たす。
我々はstop-gradientの重要性に関する仮説を提供し、さらにそれを検証する概念実証実験を示す。
提案手法 "SimSiam"は、ImageNetや下流タスクにおいて競争力のある結果を達成した。
このシンプルなベースラインが、教師なし表現学習におけるシャムアーキテクチャの役割を再考する動機付けとなることを期待している。
コードは公開予定。

## 1. Introduction
近年、教師なし／自己教師ありの表現学習は着実に進歩しており、複数の視覚タスクで有望な結果が得られている（例えば[2, 17, 8, 15, 7]）。 
様々な当初の動機にもかかわらず、これらの方法は一般的にある種のsiamese network[4]を含んでいる。 
siamese networkは、2つ以上の入力に適用される重み共有ニューラル・ネットワークである。 
siamese networkは、実体を比較する(「対照」を含むが、これに限定されない)ための自然なツールである。 
最近の手法は、入力を1つの画像の2つの拡張として定義し、異なる条件下で類似度を最大化する。
siamese networkの望ましくない些細な解は、すべての出力が定数に"崩壊"することである。 
siamese networkの崩壊を防ぐための一般的な戦略はいくつかある。 
例えばSimCLR[8]でインスタンス化された対照学習[16]は、同じ画像の2つのビュー(正のペア)を引き寄せる一方で、異なる画像（負のペア)を反発させる。 負のペアは、解空間から定数出力を排除する。 
クラスタリング[5]も一定の出力を避ける方法のひとつであり、SwAV[7]はオンライン・クラスタリングをsiamese networkに組み込んでいる。
対比学習やクラスタリング以外にも、BYOL [15]は正例のペアにのみ依存しているが、momentum encoderを使用した場合には破綻しない。  
本論文では、単純なsiamese networkが、崩壊を防ぐための上記の戦略のどれを用いなくても、驚くほどうまく機能することを報告する。
我々のモデルは、不例ペアもmomentum encoderも使わず、1つの画像の2つのビューの類似度を直接最大化する。 
一般的なバッチサイズで動作し、大規模なバッチ訓練に依存しない。 
この"SimSiam"手法をFigure 1に示す。  
概念が単純なため、SimSiamはいくつかの既存の手法を関連付けるハブの役割を果たすことができる。 
一言で言えば、我々の手法は「momentum encoderのないBYOL」と考えることができる。 
BYOLとは異なるが、SimCLRやSwAVのように、我々の手法は2つのブランチ間で重みを直接共有するため、「不例ペアのないSimCLR」、「オンラインクラスタリングのないSwAV」とも考えることができる。
興味深いことに、SimSiamは各手法の核となるコンポーネントを1つ取り除くことで、各手法と関連している。 
それでも、SimSiamは崩壊を起こさず、競争力のある結果を出すことができる。 
崩壊解が存在することを経験的に示すが、そのような解を防ぐには、stop-gradient(勾配停止)操作(Figure 1)が重要である。 
stop-gradientの重要性は、解こうとしている最適化問題の根底に別の問題があることを示唆している。 
我々は、暗黙のうちに2つの変数セットが存在し、SimSiamはそれぞれのセットを交互に最適化するような振る舞いをするという仮説を立てた。 
この仮説を検証するための概念実証実験を行う。
我々の単純なベースラインは、siamese型アーキテクチャが、関連する手法に共通する成功の本質的な理由になり得ることを示唆している。 
siamese networkは、"不変性"をモデル化するための帰納的バイアスを自然に導入することができる。 
translation-invarianceをモデル化するための重み共有による帰納的バイアスに成功した畳み込み[25]に類似して、重み共有siamese networkは、より複雑な変換(例えばaugmentation)に対する不変性をモデル化することができる。 
我々の研究が、教師なし表現学習におけるsiameseアーキテクチャの基本的な役割を再考する動機付けになれば幸いである。

## 2. Related Work
**Siamese networks** 
シャムネットワーク[4]は、実体を比較するための一般的なモデルである。 
その応用例としては、署名[4]や顔[34]の検証、追跡[3]、ワンショット学習[23]などがある。 
従来の使用例では、シャムネットワークへの入力は異なる画像からであり、比較可能性は監視によって決定される。

**Contrastive learning** 
対照学習[16]の核となる考え方は、正のサンプル・ペアを引き付け、負のサンプル・ペアを斥けることである。 
この方法論は近年、教師なし/自己教師ありの表現学習[36, 30, 20, 37, 21, 2, 35, 17, 29, 8, 9]のために広まっている。 
siameseを用いた、単純で効果的な対比学習法が開発されている[37, 2, 17, 8, 9]。 
実際には、対照学習法は多数の不例サンプルから利益を得ることができる[36, 35, 17, 8]。 
これらのサンプルはメモリーバンクに保持することができる[36]。 
siamese networkでは、MoCo [17]は負サンプルのキューを維持し、キューの一貫性を向上させるために、1つのブランチをmomentum encoderに変える。
SimCLR[8]は、現在のバッチに共存する負サンプルを直接使用する。そして、大きなバッチサイズで有効に動作する。

**Clustering** 
教師なし表現学習のもう一つのカテゴリーは、クラスタリングに基づくものである[5, 6, 1, 7]。  
これらは表現をクラスタリングすることと、クラスタ割り当てを予測する学習を交互に行う。 
SwAV [7]はクラスタリングをsiamese netwworkに組み込み、あるビューから割り当てを計算し、別のビューからそれを予測する。 
SwAVは各バッチに対して、Sinkhorn-Knopp変換[10]によって解かれる均衡分割制約の下でオンラインクラスタリングを実行する。 
クラスタリングベースの手法は不例サンプル抽出器を定義しないが、クラスタ中心は不例の原型として機能することができる。 
対比学習と同様に、クラスタリングに基づく手法は、クラスタリングに十分なサンプルを提供するために、memory bank[5, 6, 1]、大規模バッチ[7]、またはキュー[7]のいずれかを必要とする。

**BYOL** 
BYOL[15]は、あるビューの出力を別のビューから直接予測する。 
これは、siamese networkの片方のブランチがmomentum encoder(*1)になっているものである。
[15]では、BYOLが破綻しないためにはmomentum encoderが重要であるという仮説が立てられており、momentum encoderを取り除くと破綻するという結果が報告されている(0.3%の精度、[15]の表5)
(*2) 我々の実証研究では、破綻を防ぐためのmomentum encoderの必要性に挑戦している。 
我々は、stop-gradient操作が重要であることを発見した。
この発見は、stop-gradient(パラメータの勾配によって更新されないため)を常に伴う運動量エンコーダの使用によって不明瞭になる可能性があります。 
移動平均のふるまいは、適切なmomentum係数があれば精度を向上させる可能性があるが、我々の実験によれば、それは崩壊の防止とは直接関係がない。

(*1) MoCo[17]とBYOL[15]は、理論的にはmomentum encoderは学習可能エンコーダと同じ状態に収束するはずであるが、2つのブランチ間で直接重みを共有していない。 
我々はこれらのモデルを「間接的な」重み共有のsiamese networkとみなす。

(*2)BYOLのarXiv v3アップデートでは、momentum encoderを除去し、予測器の学習率を10倍増加させた場合、300エポックの事前学習で66.9%の精度を報告している。 
我々の研究は、このarXivの更新と同時に行われた。我々の研究は、異なる観点からこのトピックを研究し、より良い結果を達成している。

## 3. Method(手法)
我々のアーキテクチャ(Figure 1)は、画像$x$から2つのランダムなビュー $x_1, x_2$ を拡張子入力とする。

![Figure1](images/Figure1.png)
Figure 1. **SimSiam architecture** 
1つの画像を拡張した2つのビューが、同一のencoder network $f$(バックボーン＋射影MLP)によって処理される。
次に、一方に予測MLP $h$が適用され、もう一方にstop-gradient 演算が適用される。
本モデルは両者の類似度を最大化します。 
negative pairもmomentum encoderも使わない。

この2つのビューは、バックボーン(例えばResNet[19])と射影MLPヘッド[8]から構成されるエンコーダネットワーク$f$によって処理される。
エンコーダ$f$は2つのビュー間で重みを共有する。 
*h*と表記される予測MLPヘッド[15]は、一方のビューの出力を変換し、もう一方のビューにマッチさせる。
2つの出力ベクトルを $p_1 \triangleq h(f(x_1))$ 、 $z_2 \triangleq f(x_2)$ と定義すると、それらのcos類似度の-1倍を最小化する：

$$
\mathcal{D}(p_1, z_2) = -\frac{p_1}{\|p_1\|_2}\cdot \frac{z_2}{\|z_2\|_2} \tag{1}
$$

ここで $\|\cdot\|_2$ は $\mathcal{l}_2$-normである。 
これは$l_2$正規化されたベクトルのMSE[15]に等しく、最大で2のスケールまでとなる。
参考文献[15]に従い, 対称lossを次のように定義する:

$$
\mathcal{L} = \frac{1}{2}\mathcal{D}(p_1, z_2) + \frac{1}{2}\mathcal{D}(p_2, z_1) \tag{2}
$$

これは画像ごとに定義され、全損失は全画像の平均となる。 最小値は-1である。
この方法が機能するための重要な要素は、stop-gradient($stopgrad$)演算である(Figure 1)。
(1)を次のように修正して実装する：

$$
\mathcal{D}(p_1, stapgrad(z_2)) \tag{3}
$$

これは、$z_2$を定数項として扱うという意味である。
同様に、(2)式をこの方式で次の様に実装する:

$$
\mathcal{L} = \frac{1}{2}\mathcal{D}(p_1, stopgrad(z_2)) + \frac{1}{2}\mathcal{D}(p_2, stopgrad(z_1)) \tag{4}
$$

ここで、$x_2$ のエンコーダは、第1項では$z_2$から勾配を受け取らないが、第2項では$p_2$から勾配を受け取る（$x_1$はその逆）。
SimSiamの擬似コードはAlgorithm1にある。

![Algorithm1](images/Algorithm1.png)

**Baseline settings** 
特に指定しない限り、我々の探索では教師なし事前学習に以下の設定を使用する：

- *Optimizer(最適化器)* 
事前学習にはSGD(確率的勾配降下法)を利用する。
提案手法ではLARS[38]のような大きなバッチサイズの最適化器を必要としない([8, 15, 7]とはことなる)。
学習率には $lr \times BatchSize / 256$を使用する(線形スケーリング[14])。基本となる$lr$は$lr= 0.05$とする。
学習率にはcos減衰スケージュール[27, 8]を使用する。
係数の減衰は0.0001でSGDのモメンタムは0.9とする。
バッチサイズは基本的には512を使用する、これは典型的な8-GPUでの実装に合っている。
ほかのバッチサイズでも同様に機能する(Sec. 4.3). 
デバイス間で同期させたbatch normalization(BN)を使用する[22]([8, 15, 7]のように)。

- *Projection MLP(射影MLP)* 
($f$中の)射影MLPは、出力fcを含む全結合層(fc)にBNを適用している。
出力fcにはReLUがない．隠れfcは2048次元とする。 このMLPは3層である。

- *Prediction MLP(推論MLP)* 
推論MLP($h$)の隠れ層のfc層はBNを有する。
出力の全結合層にはBN(Sec 4.4のablationにて)やReLUは無い。
このMLPは2層である。
$h$ の入力と出力($z$と$p$)の次元は $d = 2048$ であり、 $h$ の隠れ層の次元は $512$ であるため、 $h$ はボトルネック構造である(付録のアブレーション)。 
本論文ではデフォルトのバックボーンとしてResNet-50[19]を用いる。
その他の実装の詳細は補足にある。 
アブレーション実験では $100$ epochの事前学習を行う。

**Experimental setup** 
ImageNet-1K[11]に対して、ラベルを使用せずに教師なし事前学習を行う。 
事前学習された表現の品質は、学習セットに対して固定された表現に対して教師あり線形分類器を訓練し、Validationセットでそれをテストすることで評価する。 
線形分類の実装の詳細は補足にある。

## 4. Empirical Study
このセクションでは、SimSiamの動作を実証的に研究する。 
特に、このモデルの崩壊しない解の原因となるものに注目する。

### 4.1. Stop-gradient
Figure 2 は"stop-gradient の有無"の比較を示している。

![Figure2](images/Figure2.png)
Figure 2. SimSiam の stop-gradient の有無の比較。
左: training loss. stop-gradient が無いとすぐに縮退する。
中央: $l_2$正規化した出力のチャネルごとの標準偏差。全チャネルの標準偏差の平均値としてプロットしている。
右: kNN-classifier[36] によるvalidationでの精度
それぞれの進捗となっている。
Table: ImageNet 線形評価(ネットワークを固定し1層の線形層を出力につなぎ、それだけを学習させた場合の精度)("w/ stop-grad" は5回の志向の平均とmean±標準偏差)。

アーキテクチャとすべてのハイパーパラメーターに変更はなく、ストップグラディエントだけが異なる。
Figure 2の(左)は学習lossを示している。
stop-gradientを用いない場合、オプティマイザはすぐに縮退した解を見つけ、可能な限り最小の損失である-1に達する。 
縮退が崩壊によるものであることを示すために、$l_2$正規化した出力の標準偏差 $z/|z|_2$ を調べる。
出力結果が定数ベクトルに崩壊している場合、全てのサンプルの標準偏差は各チャネルで0となる。
これは、Figure 2(中央)の赤い曲線から見ることができる。
比較として、出力 $z$ がゼロ平均等方ガウス分布の場合、$z/|z|_2$ の標準偏差は $\frac{1}{\sqrt{d}}$であることを示せる(*3)。

(*3) 非公式な導出： $z/|z|_2$を$z'$とすると、$i$番目のチャネルに対して、$z'_i = z_i/(\sum^d_{j=1}  z^2_j)^{\frac{1}{2}}$ となる。 
$z_j$ がi.i.d(独立同一分布)ガウス分布： $z_j \approx \mathcal{N}(0,1), \forall j$ に従うとすると、 $z'_i \approx z_i/d^{ \frac{1}{2}}$ そして $std[z'_i]\approx 1/d^{\frac{1}{2}}$ となる。

Figure 2(中)の青い曲線は、stop-gradientの場合、標準偏差が $1\sqrt{d}$ 付近にあることを示している。 
これは出力が崩壊せず、単位超球面上に散らばっていることを示している。
Figure 2(右)は、k-NN分類器[36]の検証精度をプロットしたものです。 
このkNN分類器は、進捗のモニターとして役立ちます。 
stop-gradienありの場合、kNNで見ることで着実に精度が向上していることがわかる。 
線形評価結果はFigure 2の表にある。  
SimSiamは、67.7%の非自明な精度を達成する。 
この結果は、5回の試行の標準偏差が示すように、適度に安定している。  
stop-gradientを取り除くだけで、精度は0.1%になり、これはImageNetにおける偶然レベルの推測値である。

**Discussion**
我々の実験は、崩壊解が存在することを示している。 
崩壊は可能な限り最小の損失と一定の出力によって観察することができる(*4)。

(*4) 偶然レベルの精度(0.1%)では崩壊を示すには不十分であることに注意されたい。 
別の失敗のパターンである損失が発散するモデルも、偶然レベルの精度を示すことがある。

崩壊解が存在するということは、我々の手法がアーキテクチャ設計(例えば、予測子、BN、$l_2$-norm)のみで崩壊を防ぐには不十分であることを意味する。 
stop-gradientの導入は、根本的に別の最適化問題が存在することを意味する。 
Sec. 5 で仮説を提案する。

### 4.2. Predictor
Table 1 で予測器のMLPの効果を研究している。
$h$ を取り除くとモデルは機能しない(Table 1a)、つまり$h$は同一性写像である。
実は、この観測はsymmetric loss(4)を使えば推定できる。 
このとき、損失は $\frac{1}{2}\mathcal{D}(z_1,stopgrad(z_2)) + \frac{1}{2}\mathcal{D}(z_2,stopgrad(z_1))$ となる。
その勾配は $\mathcal{D}(z1,z2)$ の勾配と同じ方向で、大きさは1/2にスケーリングされる。 
この場合、$stopgrad$ を使うことは、stop-gradientを取り除き、損失を1/2にスケーリングすることと等価である。 崩壊が観測される。
勾配方向に関するこの導出は、対称化された損失に対してのみ有効であることに注意。
しかし、非対称変形(3)も$h$を取り除くと失敗し、$h$を残すとうまくいくことが観測されている(Sec 4.6)。 
これらの実験は、$h$ が我々のモデルにとって有用であることを示唆している。
$h$ をランダムな初期値で固定した場合も、我々のモデルはちゃんと働かない(Table 1b)。
しかし、この失敗は崩壊ではない。 
学習は収束せず、損失は高いままである。 
予測器 $h$ は学習により表現に適応する。
定数$lr$(減衰なし)で学習した$h$もうまく機能し、ベースラインよりさらに良い結果を生むことが分かる(Table 1c)。
考えられる説明としては、$h$ は最新の表現に適応すべきなので、表現が十分に訓練される前に(lrを小さくして)強制的に収束させる必要はない、ということである。 
我々のモデルの多くのバリエーションにおいて、$h$を定数 $lr$ で学習することで、わずかに良い結果が得られることが観察されている。 
以下の節ではこの形式を用いる。

![Table1](images/Table1.png)
Table 1. **MLPによる推論の効果** (100epochの事前学習からのImageNetに対するlinear evaluation の精度). 
すべての場合において、 encoder $f$ の学習すジュールは同一とした($lr$ をcos減衰).

### 4.3. Batch Size
Table 2は、バッチサイズが64から4096の場合の結果である。 
バッチサイズを変える場合に、ベース$lr = 0.05$に大して同じ線形スケーリングルール($lr \times BatchSize/256$)[14]を使用する。 
$BatchSize \ge 1024$では、10epochのウォームアップ[14]を使用する。  
なお、検討したすべてのバッチサイズについて、(LARS[38]ではなく)同じSGDオプティマイザを使い続けている。
我々の手法は、この広いバッチサイズ範囲にわたってそれなりにうまく機能する。 
バッチサイズが128や64の場合でも、精度が0.8%または2.0%低下する程度で、まずまずの性能を発揮する。 
バッチサイズが256から2048の場合、結果は同様に良好であり、その差はランダムな変動のレベルである。 
SimSiamのこの挙動は、SimCLR [8]やSwAV [7]とは明らかに異なる。
3手法とも、直接重み共有を行うsiamese networkであるが、SimCLRとSwAVは共に大きなバッチ(例えば4096)を必要とする。 
また、標準的なSGDオプティマイザは、バッチが大きすぎると(教師あり学習[14, 38]でさえ)うまく働かないことに注意する。我々の結果はbatch size $4096$ より小さい。
この場合、特化したオプティマイザ(例えばLARS[38])が役立つと期待される。 
しかし、我々の結果は、特化したオプティマイザは崩壊を防ぐのに必要ないことを示している。

![Table2](images/Table2.png)
Table 2. **バッチサイズの影響** (100epochの事前学習からのImageNetに対するlinear evaluation の精度)

### 4.4. Batch Normalization
Table 3はMLPヘッド上のBNの構成を比較したものである。
Table 3aでは、MLPヘッドのBN層をすべて削除している(10-epoch warmup [14]はこのエントリのために特別に使用されています)。
この変形では、精度は低い(34.6%)ものの、崩壊は起こらない。
精度が低いのは最適化が難しいからだと思われる。
隠れ層にBNを追加すると(Table 3b)、精度は67.4%に向上する。
さらに投影MLPの出力(つまり$f$の出力)にBNを加えると、精度は68.1%に向上する（Table 3c）。
このエントリでは、$f$ の出力BNにおける学習可能なアフィン変換(スケールとオフセット[22])は不要であり、これを無効にすると68.2%の同程度の精度になることもわかります。
予測MLP $h$の出力にBNを追加してもうまくいかない（Table 3d）。
これは崩壊の問題ではないことがわかります。
学習が不安定で、lossが振動している。
まとめると、BNは適切に用いれば最適化に役立つということであり、これは他の教師あり学習シナリオにおけるBNの挙動と同様である。
しかし、BN が崩壊の抑止の一助になっているという証拠は示せていない:
実際、Sec 4.1 での比較(Figure 2)では、両エントリとも全く同じBN構成になっているが、stop-gradientを使用しないとモデルは崩壊している。

![Table3](images/Table3.png)
Table 3. **MLP heads へのbatch normalizationの影響** (100epochの事前学習からのImageNetに対するlinear evaluation の精度).

### 4.5. Similarity Function
cos類似度関数(1)の他に、提案手法はcroll-entorpy類似度でも動作する。 
我々は、$\mathcal{D}$ を次のように修正する： 
$\mathcal{D}(p1,z2) = -softmax(z2) \cdot \log softmax(p1)$. 
ここで、softmax関数はチャンネル次元に沿ったものである。 
softmaxの出力は、$d$ 個の擬似カテゴリーに属する確率と考えることができる。 
余弦類似度を単純にクロスエントロピー類似度に置き換え、(4)を用いて対称化する。 
すべてのハイパーパラメータとアーキテクチャーに変更はないが、この変形では最適ではないかもしれない。
以下はその比較です: 

![t4_5_(1)](images/t4_5_(1).png) 

交差エントロピーの変種は、崩壊することなく妥当な結果に収束する。 
このことは、崩壊防止動作はcos類似度だけではないことを示唆していいる。  
この変形はSec 6.2で議論するSwAV [7]との関連を設定するのに役立つ。

### 4.6. Symmetrization
これまでの実験は、対称化された損失(4)に基づいている。 
我々は、SimSiamの崩壊を防ぐ動作が対称化に依存しないことを観察した。 
非対称の変形(3)と以下のように比較する：

![t4_6_(1)](images/t4_6_(1).png)

非対称バリアントは妥当な結果を達成している。 
対称化は精度を上げるのに役立つが、破綻防止とは関係ない。 
対称化は各画像に対して予測を1つ増やすので、非対称バージョンでは各画像に対して2つのペアをサンプリングすることでこれをおおよそ補うことができる("2×")。 
これによってギャップが小さくなる。

### 4.7. Summary
我々は、SimSiamが様々な設定において、崩壊することなく意味のある結果を出すことができることを経験的に示しました。 
オプティマイザ(バッチサイズ)、バッチ正規化、類似関数、対称化は精度に影響を与えるかもしれませんが、それらが崩壊防止に関係しているという証拠は見当たりません。 
本質的な役割を果たすのは、主にstop-gradient操作です。

## 5. Hypothesis
SimSiamによって暗黙のうちに最適化されるものについての仮説を、概念実証実験を交えて議論する。

### 5.1. Formulation
我々の仮説では、SimSiamは期待値最大化(EM)のようなアルゴリズムの実装である。 
SimSiamは、暗黙のうちに2つの変数セットを含み、2つの基本的な部分問題を解く。 
停止勾配の存在は、余分な変数セットを導入した結果である。 
以下の式で損失関数を考える：

$$
\mathcal{L}(\theta, \eta) = E_{x, \tau}\left[ \|\mathcal{F}_\theta(\Tau(x)) - \eth_x\|^2_2\right] \tau{5}
$$

$F$ はパラメータ $\theta$ を持つネットワーク。 
$\Tau$ はaugmentationである。 $x$ は画像。
期待値 $E[\cdot]$ は 画像とaugmentationの分布上の期待値である。
分析を簡単にするため、ここでは平均二乗誤差 $\|\cdot\|^2_2$ を用いるが、これはベクトルが $l^2$ 正規化されていれば余弦類似度と等価である。 
(5)では、$\eta$ と呼ぶ別の変数集合を導入した。 $\eta$ の大きさは画像数に比例する。 
直観的に、$\eta_x$ は画像$x$の表現であり、添え字 $x$ は、画像のインデックスを使って $\eta$ のサブベクトルにアクセスすることを意味する。 
$\eta$ は必ずしもネットワークの出力ではなく、最適化問題の引数である。 
この定式化で、解くことを考える：

$$
\min_{\theta, \eta}\mathcal{L}(\theta, \eta) \tag{6}
$$

ここでは、$\theta$と $\eta$ の両方が問題である。
この定式化はk-meansクラスタリング[28]に類似している。
変数 $\theta$ は、クラスタリング中心と似ている：エンコーダの学習可能なパラメータである。
変数 $\eta_x$ は、サンプル $x$ の代入ベクトル(k-meansの1-hotベクトル)に類似している：それは $x$ の表現である。 
また、k-meansに類似して、(6)の問題は交互アルゴリズムで解くことができる：変数の1セットを固定し、他のセットについて解く。 
形式的には、これら2つの部分問題を交互に解くことができる：

$$
\begin{align}
\theta^t &\leftarrow& arg \min_\theta \mathcal{L}(\theta, \eta^{t-1}) \tag{7}\\
\eta^t &\leftarrow& arg \min_\eta \mathcal{L}(\theta, \eta) \tag{8}
\end{align}
$$

ここで、$t$ は交替のインデックスであり、" $\leftarrow$ "は代入を意味する。

**Solving for $\theta$** 
SGDを用いて部分問題(7)を解くことができる。 
勾配が $\eta^{t-1}$ に逆伝播しないので、勾配停止操作は自然な帰結である。

**Solving for $\eta$** 
この部分問題(8)は、各 $\eta_x$ について独立に解くことができる。
この問題は次式の最小化である : 各画像 $x$ に対して、 $E_\Tau\left[ \|F_{\theta^t}(\Tau(x)) − \eta_x‖^2_2 \right]$ 期待値は augmentation $\tau$ の分布上のものである。
平均二乗誤差(*5)により、簡単に解くことができる：

$$
\eta^t_x \leftarrow E_\Tau\left[ \mathcal{F}_{\theta^t}(\Tau(x))\right] \tag{9}
$$

(*5) 余弦類似度を使えば、$l_2$- $\mathcal{F}$の出力と$\eta_x$を正規化することで近似的に解くことができる。

これは $\eta_x$ は $x$ のaugmentation分布上の平均表現を割り当てている。

**One-step alternation** 
SimSiamは、(7)と(8)を1ステップずつ交互に繰り返すことで近似できる。
まず、$ET[\cdot]$ を無視し、、*1回だけ* 増強量をサンプリングして( $\Tau'$ と表記) (9)を近似する。

$$
\eta^t_x \leftarrow \mathcal{F}_{\theta^t}(\Tau'(x)) \tag{10}
$$

これを(7)の部分問題に挿入すると、次のようになる：

$$
\theta^{t+1} \leftarrow \min_\theta E_{x, \Tau}\left[ \|\mathcal{F}_\theta(\Tau(x))\| - \mathcal{F}_{\theta^t}(\Tau'(x))\|^2_2\right] \tag{11}
$$

ここで、 $\theta^t$ はこの部分問題の定数であり、 $Tau′$ はそのランダムな性質から別の見方を意味する。 
この定式化はsiamese構造を示す。 
次に、(11)を1つのSGDステップで損失を減らすように実装すると、SimSiamアルゴリズムに近づくことができる。
:stop-gradientが適用されたsiamese ネットワーク。

**Predictor** 
上記の分析は予測変数 $h$ を含まない。
さらに、(10)による近似のため、$h$は本手法において有用であると仮定する。
定義により、予測変数 $h$ は次のように最小化することが期待される： $E_z\left[Γ|h(z_1) -z_2Γ|^2_2\right]$.
$h$ の最適解は次を満たすべきである：任意の画像 $x$ に対して $h(z_1)=E_z[z_2]=E_\Tau[f( \Tau(x))]$.
この項は(9)の項と似ている。
(10)の近似では、期待値 $E_\Tau[\cdot]$ は無視される。
$h$ の使い方はこのギャップを埋めるかもしれない。 
実際には、期待値 $E_\Tau$ を実際に計算するのは非現実的である。
しかし、ニューラルネットワーク(例えばプレディター $h$ )が、 $\Tau$ のサンプリングが暗黙のうちに複数のエポックに分散されている間に、期待値を予測することを学習することは可能かもしれない。

**Symmetrization** 
我々の仮説は対称化を伴わない。 
対称化とは、(11)の $\Tau$ をより密にサンプリングするようなものである。
実際には、SGDオプティマイザは、画像のバッチと1組のaugmentation $( \Tau_1, \Tau_2)$ をサンプリングして、 $E_{x,\Tau}[\cdot]$ の経験的期待値を計算します。
原理的には、サンプリングが密であればあるほど、経験的な期待値はより正確になるはずである。
対称化により、余分なペア $(\Tau_2, \Tau_1)$ ができる。
このことは、我々がSection 4.6で観察したように、対称化は我々の方法にとって必要ではなく、しかし精度を向上させることができることを説明している。

### 5.2. Proof of concept
私たちは、私たちの仮説に由来する一連の概念実証実験をデザインする。 
これらはSimSiamとは異なる手法であり、我々の仮説を検証するためのものである。

**Multi-step alternation**  
我々は、SimSiamアルゴリズムは、SGDの更新が1ステップの間隔で、(7)と(8)を交互に繰り返すようなものであるという仮説を立てた。 
この仮説の下では、SGDが複数ステップ更新される区間であれば、我々の定式化が機能する可能性が高い。
この変形では、(7)と(8)の $t$ を外側ループのインデックスとして扱い、(7)の部分問題を $k$ のSGDステップの内側ループによって更新する。 
各交代において、全ての$k$ SGDステップに必要な$teta_x$を(10)を用いて事前計算し、メモリにキャッシュする。
そして、$k$ SGDステップを実行して $\theta$ を更新する。
SimSiamと同じアーキテクチャとハイパーパラメータを使用する。
The comparison is as follows:

![t5_2_(1)](images/t5_2_(1).png)
ここで、"1-step "はSimSiamと等価であり、"1-epoch "は1エポックに必要な$k$ステップを表す。
すべての多段階変法はうまくいく。 
10ステップ/100ステップのバリエーションは、余分な事前計算の代償としてではあるが、SimSiamよりも良い結果さえ出している。
この実験は、交互最適化が有効な定式化であり、SimSiamがその特殊なケースであることを示唆している。

**Expectation over augmentations** 
The usage of the predictor $h$ is presumably because the expectation $E_\Tau[\cdot]$ in (9) is ignored. 
We consider another way to approximate this expectation, in which we find $h$ is not needed.
In this variant, we do not update $\eta_x$ directly by the assignment (10); instead, we maintain a moving-average: $\eta^t_x \leftarrow m * \eta^{t−1}_x + (1 −m) * F_{\theta^t}(\Tau'(x))$, where m is a momentum coefficient (0.8 here). 
This computation is similar to maintaining the memory bank as in [36]. 
This moving-average provides an approximated expectation of multiple views. 
This variant has 55.0% accuracy without the predictor $h$. 
As a comparison, it fails completely if we remove $h$ but do not maintain the moving average (as shown in Table 1a). 
This proof-of-concept experiment supports that the usage of predictor $h$ is related to approximating $E_\Tau[\cdot]$.

### 5.3. Discussion
Our hypothesis is about what the optimization problem can be. 
It does not explain why collapsing is prevented.  
We point out that SimSiam and its variants’ non-collapsing behavior still remains as an empirical observation.
Here we briefly discuss our understanding on this open question. 
The alternating optimization provides a different trajectory, and the trajectory depends on the initialization.
It is unlikely that the initialized $\eta$, which is the output of a randomly initialized network, would be a constant. 
Starting from this initialization, it may be difficult for the alternating optimizer to approach a constant $\eta_x$ for all $x$, because the method does not compute the gradients w.r.t. $\eta$ jointly for all $x$. 
The optimizer seeks another trajectory (Figure 2 left), in which the outputs are scattered (Figure 2 middle).

## 6. Comparisons
### 6.1. Result Comparisons
**ImageNet**. 
We compare with the state-of-the-art frameworks in Table 4 on ImageNet linear evaluation. 
For fair comparisons, all competitors are based on our reproduction, and "+" denotes improved reproduction vs. the original papers (see supplement). 
For each individual method, we follow the hyper-parameter and augmentation recipes in its original paper.(*6) 
All entries are based on a standard ResNet-50, with two 224×224 views used during pre-training.
Table 4 shows the results and the main properties of the methods. 
SimSiam is trained with a batch size of 256, using neither negative samples nor a momentum encoder. 
Despite it simplicity, SimSiam achieves competitive results. 
It has the highest accuracy among all methods under 100-epoch pre-training, though its gain of training longer is smaller. 
It has better results than SimCLR in all cases.

(*6) In our BYOL reproduction, the 100, 200(400), 800-epoch recipes follow the 100, 300, 1000-epoch recipes in [15]: $lr$ is {0.45, 0.3, 0.2}, $wd$ is
{1e-6, 1e-6, 1.5e-6}, and momentum coefficient is {0.99, 0.99, 0.996}.

![Table4](images/Table4.png)
Table 4. **Comparisons on ImageNet linear classification.** 
All are based on **ResNet-50*** pre-trained with **two 224×224 views**. 
Evaluation is on a single crop. 
All competitors are from our reproduction, and "+" denotes improved reproduction vs. original papers (see supplement).

**Transfer Learning**. 
In Table 5 we compare the representation quality by transferring them to other tasks, including VOC [12] object detection and COCO [26] object detection and instance segmentation. 
We fine-tune the pre-trained models end-to-end in the target datasets. 
We use the public codebase from MoCo [17] for all entries, and search the fine-tuning learning rate for each individual method. 
All methods are based on 200-epoch pre-training in ImageNet
using our reproduction.
Table 5 shows that SimSiam’s representations are transferable beyond the ImageNet task. 
It is competitive among these leading methods. The “base” SimSiam in Table 5 uses the baseline pre-training recipe as in our ImageNet experiments. 
We find that another recipe of $lr=0.5$ and $wd=1e-5$ (with similar ImageNet accuracy) can produce better results in all tasks (Table 5, "SimSiam, optimal").
We emphasize that all these methods are highly successful for transfer learning—in Table 5, they can surpass or be on par with the ImageNet supervised pre-training counterparts in all tasks. 
Despite many design differences, a common structure of these methods is the Siamese network.
This comparison suggests that the Siamese structure is a core factor for their general success.

![Table5](images/Table5.png)
Table 5. **Transfer Learning**. All unsupervised methods are based on 200-epoch pre-training in ImageNet. VOC 07 detection: Faster R-CNN [32] fine-tuned in VOC 2007 trainval, evaluated in VOC 2007 test; VOC 07+12 detection: Faster R-CNN fine-tuned in VOC 2007 trainval + 2012 train, evaluated in VOC 2007 test; COCO detection and COCO instance segmentation: Mask R-CNN [18] (1×schedule) fine-tuned in COCO 2017 train, evaluated in COCO 2017 val. 
All Faster/Mask R-CNN models are with the C4-backbone [13]. 
All VOC results are the average over 5 trials. **Bold entries** are within 0.5 below the best.

## 6.2. Methodology Comparisons
Beyond accuracy, we also compare the methodologies of these Siamese architectures. 
Our method plays as a hub to connect these methods. 
Figure 3 abstracts these methods.  
The “encoder” subsumes all layers that can be shared between both branches (e.g., backbone, projection MLP [8], prototypes [7]). The components in red are those missing in SimSiam. 
We discuss the relations next.
**Relation to SimCLR [8]**. SimCLR relies on negative samples (“dissimilarity”) to prevent collapsing. 
SimSiam can be thought of as “SimCLR without negatives”.
To have a more thorough comparison, we append the prediction MLP $h$ and stop-gradient to SimCLR.(*7) Here is the ablation on our SimCLR reproduction:
![t6_2_(1)](images/t6_2_(1).png)
Neither the stop-gradient nor the extra predictor is necessary or helpful for SimCLR. 
As we have analyzed in Sec. 5, the introduction of the stop-gradient and extra predictor is presumably a consequence of another underlying optimization problem. 
It is different from the contrastive learning problem, so these extra components may not be helpful.

(*7) We append the extra predictor to one branch and stop-gradient to the
other branch, and symmetrize this by swapping.

**Relation to SwAV [7]**. 
SimSiam is conceptually analogous to “SwAV without online clustering”. We build up this connection by recasting a few components in SwAV. 
(i) The shared prototype layer in SwAV can be absorbed into the Siamese encoder. 
(ii) The prototypes were weight-normalized outside of gradient propagation in [7];
we instead implement by full gradient computation [33].(*8)
(iii) The similarity function in SwAV is cross-entropy. 
With these abstractions, a highly simplified SwAV illustration is shown in Figure 3.

(*8) This modification produces similar results as original SwAV, but it can enable end-to-end propagation in our ablation.

![Figure3](images/Figure3.png)
Figure 3. **Comparison on Siamese architectures**. 
The encoder includes all layers that can be shared between both branches.
The dash lines indicate the gradient propagation flow. 
In BYOL, SwAV, and SimSiam, the lack of a dash line implies stop-gradient, and their symmetrization is not illustrated for simplicity. 
The components in red are those missing in SimSiam.

SwAV applies the Sinkhorn-Knopp (SK) transform [10] on the target branch (which is also symmetrized [7]). 
The SK transform is derived from online clustering [7]: it is the outcome of clustering the current batch subject to a balanced partition constraint. 
The balanced partition can avoid collapsing. Our method does not involve this transform.
We study the effect of the prediction MLP h and stop-gradient on SwAV. 
Note that SwAV applies stop-gradient on the SK transform, so we ablate by removing it. 
Here is the comparison on our SwAV reproduction:
![t6_2_(2)](images/t6_2_(2).png)
Adding the predictor does not help either. Removing stop-gradient (so the model is trained end-to-end) leads to divergence. 
As a clustering-based method, SwAV is inherently an alternating formulation [7]. 
This may explain why stop-gradient should not be removed from SwAV.

**Relation to BYOL [15]**. 
Our method can be thought of as "BYOL without the momentum encoder", subject to many implementation differences. 
The momentum encoder may be beneficial for accuracy (Table 4), but it is not necessary for preventing collapsing. Given our hypothesis in Sec. 5, the $\eta$ sub-problem (8) can be solved by other optimizers, e.g., a gradient-based one. 
This may lead to a temporally smoother update on η. Although not directly related, the momentum encoder also produces a smoother version of $\eta$. 
We believe that other optimizers for solving (8) are also plausible, which can be a future research problem.

##  Conclusion
我々はシンプルなデザインでシャム型ネットワークを探索した。 
我々の最小限の手法の競争力は、最近の手法のsiamese型がその有効性の核心的な理由になり得ることを示唆している。 
Siameseは、表現学習の焦点である不変性をモデル化するための自然で効果的なツールである。 
我々の研究が、表現学習におけるsiamese型ネットワークの基本的な役割に関心を集めることを期待している。

## A. Implementation Details
**Unsupervised pre-training** 
我々の実装は、既存の研究 [36, 17, 8, 9, 15]の実践に従っている。

*Data augmentation*. 
We describe data augmentation using the PyTorch [31] notations. 
Geometric augmentation is $RandomResizedCrop$ with scale in [0.2,1.0] [36] and $RandomHorizontalFlip$. 
Color augmentation is $ColorJitter$ with {brightness, contrast, saturation, hue}strength of {0.4, 0.4, 0.4, 0.1}with an applying probability of 0.8, and $RandomGrayscale$ with an applying probability of 0.2. 
Blurring augmentation [8] has a Gaussian kernel with std in [0.1,2.0].

*Initialization*. 
The convolution and fc layers follow the default PyTorch initializers. Note that by default PyTorch initializes fc layers' weight and bias by a uniform distribution $\mu(−\sqrt{k}, \sqrt{k})$ where $k= \frac{1*}{in_ channels}$. 
Models with substantially different fc initializers (e.g., a fixed std of 0.01) may not converge. 
Moreover, similar to the implementation of [8], we initialize the scale parameters as 0 [14] in the last BN layer for every residual block.

*Weight decay*. 
We use a weight decay of 0.0001 for all parameter layers, including the BN scales and biases, in the SGD optimizer. 
This is in contrast to the implementation of [8, 15] that excludes BN scales and biases from weight decay in their LARS optimizer.

**Linear evaluation**. 
Given the pre-trained network, we train a supervised linear classifier on frozen features, which are from ResNet’s global average pooling layer (pool5).
The linear classifier training uses base $lr = 0.02$ with a cosine decay schedule for 90 epochs, weight decay = 0, momentum=0.9, batch size=4096 with a LARS optimizer [38]. 
We have also tried the SGD optimizer following [17] with base $lr = 30.0$, weight decay = 0, momentum = 0.9, and batch size=256, which gives ∼1% lower accuracy. 
After training the linear classifier, we evaluate it on the center 224×224 crop in the validation set.

## B. Additional Ablations on ImageNet
The following table reports the SimSiam results vs. the output dimension $d$:
![tA_(1)](images/tA_(1).png)
It benefits from a larger $d$ and gets saturated at $d = 2048$.
This is unlike existing methods [36, 17, 8, 15] whose accuracy is saturated when d is 256 or 512.
In this table, the prediction MLP’s hidden layer dimension is always 1/4 of the output dimension. 
We find that this bottleneck structure is more robust. 
If we set the hidden dimension to be equal to the output dimension, the training can be less stable or fail in some variants of our exploration. 
We hypothesize that this bottleneck structure, which behaves like an auto-encoder, can force the predictor to digest the information. 
We recommend to use this bottleneck structure for our method.

## C. Reproducing Related Methods
Our comparison in Table 4 is based on our reproduction of the related methods. 
We re-implement the related methods as faithfully as possible following each individual paper.
In addition, we are able to improve SimCLR, MoCo v2, and SwAV by small and straightforward modifications: specifically, we use 3 layers in the projection MLP in SimCLR and SwAV (vs. originally 2), and use symmetrized loss for MoCo v2 (vs. originally asymmetric). 
Table C.1 compares our reproduction of these methods with the original papers' results (if available). 
Our reproduction has better results for SimCLR, MoCo v2, and SwAV (denoted as "+" in Table 4), and has at least comparable results for BYOL.

![TableC1](images/TableC1.png)
Table C.1. **Our reproduction vs. original papers’ results**. 
All are based on ResNet-50 pre-trained with two 224×224 crops.

## D. CIFAR Experiments
We have observed similar behaviors of SimSiam in the CIFAR-10 dataset [24]. 
The implementation is similar to that in ImageNet. 
We use SGD with base $lr = 0.03$ and a cosine decay schedule for 800 epochs, weight decay = 0.0005, momentum = 0.9, and batch size = 512. 
The input image size is 32×32. We do not use blur augmentation. 
The backbone is the CIFAR variant of ResNet-18 [19], followed by a 2-layer projection MLP. 
The outputs are 2048-d.  
Figure D.1 shows the kNN classification accuracy (left) and the linear evaluation (right). 
Similar to the ImageNet observations, SimSiam achieves a reasonable result and does not collapse. 
We compare with SimCLR [8] trained with the same setting. 
Interestingly, the training curves are similar between SimSiam and SimCLR. SimSiam is slightly better by 0.7% under this setting.

![FigureD1](images/FigureD1.png)
Figure D.1. CIFAR-10の実験。
左：事前学習中のモニターとしてのkNN分類の検証精度。右：線形評価精度。バックボーンはResNet-18。
