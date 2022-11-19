# Zero-Shot Learning for Reflection Removal of Single 360-Degree Image

# Abstract
従来の反射除去の手法は主にぼやけた弱い反射アーチファクトを除去することに重点を置いており、深刻で強い反射アーチファクトに対処できないことがよくある。  
しかし多くの場合、実際の反射現象は、人間でさえ透過シーンと反射シーンを完全に区別できないほど、鋭く激しい。  
本論文では360度画像を用いて、このような困難な反射アーチファクト除去を試みる。  
教師あり学習のためのペアデータ収集の負担や、異なるデータセット間のdomeain gapを回避するため、zero-shot learning方式を採用する。  
まず、反射の形状に基づいて360度画像から反射シーンの参照画像を探索し、それを用いて反射画像の忠実な色を復元するようにネットワークを誘導する。  
我々は、困難な反射アーチファクトが現れている30枚の360度テスト画像を収集し、提案手法が360度画像において既存の最先端手法を凌駕することを実証する。

# 1 Introduction
美術館やギャラリーのガラスショーケースを撮影するなど、ガラス越しの撮影はよくあります。  
ガラス越しに撮影された画像には、反射したシーンの望ましくないアーチファクトが現れます。  
このような映り込みは、ガラス越しの透過シーンの視認性を低下させ、多様なコンピュータビジョン技術の性能を低下させます。  
数十年前から、効率的な反射除去方法を開発する試みがなされてきた。  
既存の反射除去手法の多くは、制約のある環境下で撮影された複数のガラス画像を用いていたが、近年の学習ベースの手法は、入力された単一のガラス画像を透過画像と反射画像に分離する深層学習特徴量を利用することで優れた性能を達成している。
既存の手法はガラス前面のピンぼけシーンに関連するぼやけた反射アーチファクトを想定していますが、実際の反射アーチファクトは想定よりも多様な特性を示し、しばしば集約的でシャープになります。  
ゆえに、satate-of-the-artな学習ベースの手法であっても学習と実践の間のgapに苦しんでいる。  
特にVR用途に広く用いられている360度カメラは、特定の物体に焦点を合わせないため、Fig 1（a）に示すように、通常、ガラス領域上にくっきりした映り込みがある画像が生成されます。  
Fig 1（b）、（c）は、それぞれガラス領域と実際の反射シーンの参照領域を切り出した画像で、ガラス画像に反射シーンがはっきりと浮かび上がっていることが分かります。  
Fig 1 (d), (e) に示すように、360度画像の反射特性は通常の画像とは異なるため、既存の学習ベースの手法[16, 30]ではガラス画像からこのような映り込みを除去することができません。  
この場合、ガラス画像内のどのシーンが透過・反射しているかを人間でさえ区別することはとても困難になっています。  
しかし、360度画像内の反射シーンの視覚情報を参考にすることで、反射除去を効果的に誘導することができます。  

![Fig1](images/Fig1.png)  

360度画像に対する既存の唯一の反射除去法[9]は、教師あり学習のためのガラス画像を合成するアルゴリズムを用いるため、理論的には学習データとテストデータセットの間の領域ギャップに悩まされることになる。  
加えて、360度画像に対する反射除去の2つのタスクである画像復元と参照画像照合との連携に関係する部分はほとんどない。  
本論文では360度画像の反射除去のために、学習データセットを収集する負担の回避および、異なるデータセット間の領域ギャップを回避するzero-shot learningの枠組みを適用する。  
また、提案手法は、与えられたテスト画像に対する結果を交互に更新することで、画像復元と参照画像マッチングの両方の最適解を反復的に推定する。
まず、ガラス面の前に垂直に立てたカメラで360度画像を撮影し、360度画像の中心領域をガラス領域と見なすとする。  
そして、360度画像中の復元された反射画像との参照情報のマッチングを調べ、マッチングした参照情報に基づいてネットワークパラメータを更新し、透過画像と反射画像を復元します。  
その結果提案手法は、Fig 1(f)に示すように、360度画像における映り込みを除去する優れた性能を発揮します。  
本研究の主な貢献は、以下のようにまとめられます。

1. 我々の知る限り本論文は単一の360度画像に対する反射除去問題に対してゼロショット学習の枠組みを適用した最初の研究であり、既存の教師あり学習手法でみられる異なるデータセット間の領域ギャップを回避している。
2. 提案手法は、360度画像上の反射形状を利用して参照マッチングを洗練し、洗練された参照を手掛かりに透過画像と反射画像を適応的に復元する。 
3. 実験のために30枚のテスト用の実360度画像を収集し、提案手法が最先端の反射除去技術を凌駕することを実証する。

# 2 Related Works
本節では，既存の反射除去手法について簡単にまとめる．  
既存の手法を教師なしアプローチと教師ありアプローチに分類する。  
教師なしアプローチには、計算による反射除去法と、学習のためのペアデータセットを必要としない最新のゼロショット学習ベースの画像分解法が含まれる。  
一方、教師ありアプローチには学習ベースの単一画像反射除去法が含まれる。

**Unsupervised approach:**  
このような反射光アーチファクトは，特定の撮影環境下で撮影された複数の画像に現れる．  
[5, 13, 21]は，偏光角によって強度が変化する反射光の特性を利用し，複数の偏光画像の反射アーチファクトを除去している．  
また[20]では，焦点距離の異なる複数のガラス画像を2枚の画像に分離し，ボケ度を明確にしている．  
また，[7, 8, 17, 24, 29]は，カメラ位置を変えて撮影した複数のガラス画像について，透過シーンと反射シーンの異なる挙動を解析している．  
さらに、[19]は映像中の透過場面の繰り返し移動を検出している。  
また，[23]はブラックボックス映像から自動車のフロントガラスに映る静止画を抽出した．  
一方1枚のガラス画像から映り込みを除去することは、透過画像と反射画像を区別するための特性がないため困難である。  
[14]はユーザ支援により，ガラス画像上で除去すべき反射エッジを選択した．  
[18]は反射画像は透過画像よりも不鮮明であるという強い仮定のもと、入力画像を鮮明な層と不鮮明な層に分離し、透過画像と反射画像を取得した。  
また[15]では，ガラス画像は異なる2つのエッジ間の交差点が多くなると仮定し，交差点の総数が最小になるようにガラス画像を2つのレイヤーに分離している．  
[22]はガラス窓の表裏で反射した光がゴースト効果をもたらすため、空間的に繰り返される視覚構造を除去している。  
[6]は、透過シーンと反射シーンがそれぞれ動的と静的であるという制約のある環境下で撮影された複数のガラス画像を分解するネットワークを学習させる一般的なフレームワークを提案した。  
既存の手法では、複数のガラス画像を必要としたり、反射物の明確な特徴を想定したりしますが、提案手法では、単一の360度画像から参照情報を検出することにより、透過画像と同様の特徴を示す難しい反射物の除去を行います。  

**Supervised approach:**  
近年ディープラーニングを用いた反射除去法が提案されています。  
これらは，ガラス画像と透過画像のペアデータセットを用いてディープネットワークを学習させ，反射物特有の特性を強く仮定した計算機上での手法よりも信頼性の高い結果を提供する．  
[4]は反射除去に初めてCNNを適用し，2つのネットワークを直列に接続して，透過画像のグラデーションと色をそれぞれ復元する枠組みを提案した。  
[25]は透過画像の色と階調を同時に予測するフレームワークを修正した．  
また、[30]は、過去の結果を用いて再帰的に透過画像と反射画像を予測する新たなフレームワークを提案した。  
同様に[16]はガラス画像から透過画像と反射画像を予測するために、透過と反射の復元結果を事前にフィードバックして繰り返す完全カスケード型のフレームワークを採用した。  
[2]は映り込みが支配的な局所領域を示す確率マップを予測することで、局所的に集中する反射アーチファクトに対処した。  
一方，教師あり学習の学習データの問題に取り組んだ手法もある．  
[27]は、既存の学習データセットで頻繁に観測される、入力ガラス画像とその真正透過画像との間の不整合に関係なく、ネットワークパラメータを学習するための新しい損失項を定義している。  
実画像においてガラス画像と透過画像のペアデータがないため、[32]はガラス面への光線入射角度に依存する光吸収効果を含むガラス画像を合成するための高度な画像定式化をモデル化した。  
また，[12]は，グラフィカルシミュレータを用いて物理的に反射を模倣し，合成ガラス画像を生成している。  
また，[28]では，ガラス画像合成と同時に反射除去のためのディープネットワークを利用し，よりリアルなガラス画像を作成し，学習用に利用した．  
最近、[9]はパノラマ画像のガラス窓と反対方向に撮影した参照画像を用いて、反射アーチファクトを除去しています。
しかし、教師あり学習ベースの手法はすべてドメインギャップに悩まされている。  
[1]では既存手法の反射除去性能は、その学習データセットに含まれる反射アーチファクトの種類によって決定されることを示した。  
一方、提案手法はゼロショット学習の枠組みに基づき、与えられた入力画像に対して適応的に動作し、教師あり学習の枠組みにおける学習データセットの収集の負担を軽減することができる。

![Fig2](images/Fig2.png)  
Fig 2：オプティカルフロー推定器を用いた画像アライメント  
(a) 360度画像上の正面ガラス画像と参照画像のペア  
フローマップはと歪んだ参照画像はそれぞれ(b)DICL[26]よって、(c)FlowNet[3]によって得られたものである。

# 3 Methodology
360度画像はカメラの周囲の環境シーン全体を含むため、ガラスのシーンと関連する反射シーンが一緒に撮影されます。  
提案手法では、反射シーンを含む参照領域から関連する情報を持ってくることで、360度画像中のガラス領域に関連する透過画像と反射画像を復元する。  
本章では、まず360度画像中の反射形状に基づく参照画像の探索方法について説明します。  
次に、提案するゼロショット学習の枠組みを、既存手法であるDDIP[6]と簡単な比較を交えて紹介する。  
最後に提案手法のテスト時における詳細な学習過程を説明する。

## 3.1 Estimation of Reference Image
我々は、反射画像と関連する参照画像との関係を調査している。  
[9]で紹介したように、反射画像は測光歪みと幾何歪みに悩まされ、参照画像を用いてもガラス画像上の反射シーンと透過シーンを区別することが困難である。  
幾何学的な歪みは、ガラスの厚み、光の入射角、光の波長などの外的要因によって発生することがある。  
また、カメラに内蔵された画像信号処理（ISP）も、測光歪みの内部要因となっている。  
反射像と参照像の間の幾何学的歪みは、カメラからガラスや物体までの距離に依存する視差が主な原因である。  
最近のオプティカルフロー推定技術[3, 26]では、図2に示すように、反射画像と透過画像の混合による測光歪みのため、ガラス画像と参照画像の正しい対応関係を推定することができない。

![Fig3](images/Fig3.png)  
Fig 3：反射を利用した360度画像取得の構成。  
丸は360度映像のレンダリングを行う単位球の表面を表す。  

幾何学的歪みと写真的歪みの低減は鶏と卵の問題と考えることができる。  
反射画像とよく一致する参照画像は、反射されたシーンの内容を復元するための忠実な色を提供します。  
一方、反射画像の復元がうまくいけば、参照画像と反射画像の位置関係を合わせるための確かな視覚的特徴を得ることができます。  
提案手法は反射形状に基づき、ガラス画像領域内の各画素に対して信頼性の高い参照領域を求める。  
360度画像は3次元空間の物体から単位球の表面に投影される光線によって撮影される。  
特にガラス領域では、ガラスに反射した光線が追加で発生するため、そのような場合は距離情報がないため、3次元空間における正確な物体位置を推定することができない。  
Fig 3に示すように、360度画像のガラス領域で$x_i$に物体が観測された場合、ガラスが存在しなければ(仮に実態であれば)$\hat{x}$に物体が観測されることになる。  
反射形状に従い、ガラス面の向きで定義されるハウスホルダー行列[10]を用いて仮想点$\hat{x}_i$と$\hat{o}$の座標を算出する。  
物体は$d_i = \hat{x}_i -\hat{o}$の方向に沿って位置すると仮定し、仮想原点$\hat{o}$から物体までの距離を$d_i$に沿って変化させ、$x_i$に対する$c^k_i$の位置候補を検討する。  
そして、候補位置$c^{k'}_i$をそれぞれ単位面に投影してマッチング候補$x^{k'}_i$を集める。  
本研究ではガラスから遠い背景を扱うために、$d_i$の方向に沿ってサンプリングされた50 個の$c^k_i$の候補位置を含む探索空間を定義する。  
そして、探索空間の$x^{k'}_i$の中から、$x_i$との特徴量の差が最小となる$x_i$への最適なマッチング点$m_i$を求める。  
近傍画素を考慮し、$x_i$と$x^k_i$のパッチワイズ特徴差を次のように計算する。

$$
\Omega(x_i, x_i^k) = \frac{1}{|\mathcal{N}_i| + 1}\sum_{p_j\in \mathcal{N_i} \cup \{p_i\}} \parallel F_G(p_j) - F_R(p_j^k)\parallel_1 \tag{1}
$$

ここで、$F_G$と$F_R$はそれぞれ360度画像中のガラス領域と参照領域の任意の平行化特徴マップを表し、$p$は平行化画像領域上の$x$に対応する画素位置を表し、$\mathcal{N}_i$は$p_i$の近傍集合を表している。  
本研究では$\mathcal{N}_iの$大きさを$24$とする。  
具体的には、与えられた$x_i$に対して、色と勾配の特徴量の観点からそれぞれ最適な2つのマッチング点$m^c_i$と$m^g_iを$探索する。  
なお、学習反復回数$t$の表記は簡略化のため省略する。  
色に基づくマッチング点 $m^c_i$ を探索するために、$F_G$ を再構成された反射画像 $\hat{R}$ とし、$F_R$ を平行化された参照画像 $I_{ref}$ とする。  
$\hat{R}$と$m^c_i$は学習のために繰り返し更新され、より忠実な$\hat{R}$はより信頼性の高い$m^c_i$を提供し、逆もまた然りであることに注意されたい。  
勾配ベースのマッチング点$m^g_i$は、$F_G$と$F_R$に対して、それぞれ平行化されたガラス画像$I_G$と$I_{ref}$の勾配を用いることで得られる。  
$m^c_i$ と$m^g_i$ は反射像回復のために部分的に補完的な情報を提供することに注意されたい。  
$m^c_i$は、復元された反射画像が参照画像と見慣れない色を持つことを防ぐ一方で、$m^g_i$は、復元された反射画像がガラス画像の構造を保持するようにする。

![Fig4](images/Fig4.png)  
Fig 4：提案するネットワークの全体構成。  

## 3.2 Network Architecture
提案手法は、Fig 4に示すように、encoder、decoder、2つのgeneratorの4つのサブネットワークから構成される。  
透過画像と反射画像の復元でそれぞれネットワークパラメータ$\theta, \phi$を共有している。  
encoderは360度画像から平行画像を取得し、decoderが入力画像を再構成できるdeep特徴量を抽出する。  
ガラス画像のdeep特徴量は透過と反射の両方の特徴を持つため、generatorはガラス画像のdeep特徴量を透過特徴$h_T$と反射特徴$h_R$に分離するためのマスクマップを提供しそれぞれ次式で与えられる。

$$
\begin{align}
h_T = f_{\theta}(I_G)\cdot f_{\psi_T}(z_T), \tag{2}\\
h_R = f_{\theta}(I_G)\cdot f_{\psi_R}(z_R), \tag{3}
\end{align}
$$

ここで、$z_T$ と $z_R$ は異なるガウスランダムノイズを表し、$(\cdot)$ は要素ごとの乗算を表す。  
しかし、ガラス画像の光歪みにより反射画像の元の色を復元するための反射特徴が不完全なため、提案手法では反射特徴に適応的インスタンス正規化(AdaIN)［11］を適用し、不完全な情報を補う。  
反射特徴量$h_R$は、参照特徴量$h_{ref}$によって以下のように変換される。  

$$
\hat{h}_R = \sigma(h_{ref})(\frac{h_R - \mu(h_R)}{\sigma(h_R)}) + \mu(h_{ref}), \tag{4}
$$

ここで、$h_{ref} = f_{\theta}(I_{ref})$であり、$\mu$ と $\sigma$ は空間次元間の平均と標準偏差を計算するための演算を表す。  
提案手法は最終的に反射画像を$\hat{R} = f_{\phi}(\hat{h}_R)$として復号する。  
AdaINは空間位置をまたいだ統計量に従って特徴量を転送するため、反射画像と参照画像の幾何学的な差異を緩和することができる。  
反射回復とは異なり、透過の歪みは無視できると仮定し、$\hat{T} = f_{\phi}(h_T)$ を介して透過画像を予測する。  
DDIP[6]は、複数のガラス画像を透過画像と反射画像に分離できる一般的なフレームワークを導入しています。  
これは、ガラス画像合成のための線形定式化の下でネットワークを学習させるものである。  
しかし、最近の研究[1, 9, 28]では、このような素朴な定式化では実際のガラス画像をモデル化するには不十分であることが指摘されている。  
一方提案手法は、ガラス画像をdeep特徴量空間に分解し、得られた透過・反射カラーマップを単純に足し合わせるのではなく、透過・反射画像のdeep特徴量を統合してガラス画像を合成するものである。  
また、DDIPが単に複数のガラス画像を要求して透過シーンと反射シーンの特徴を区別するのに対し、提案手法は与えられた360度画像から反射画像と透過画像を区別するための参照情報をもたらす新しい枝を付加している点にも注目されます。  
ネットワークアーキテクチャの詳細については、補足資料をご参照ください。

# 3.3 Training Strategy
he proposed method trains the network parameters in a test time for a given instance. Particularly, each network of the proposed framework is trained respectively according to different training losses.  
For each iteration, the (θ, φ), ψR, and ψT are trained by using three individual Adam optimizers.  
We update the network parameters during 600 iterations for each test image.  
Encoder and decoder: The parameters of the encoder θ and the decoder φ are trained to reconstruct the input image itself according to the reconstruction loss Lrecon between a source map X and a target map Y defined as

$$
\mathcal{L}_{recon}(X, Y) = \mathcal{L}_{mse}(X, Y) + \omega_1 \mathcal{L}_mse(\nabla X, \nabla Y) \tag{5}
$$

where Lmse denotes the mean squared error and w1 denotes the weight to determine the contribution of the gradient difference for training.  
We utilize the rectified images IG and Iref of the glass region and the reference region as training images.  
The encoder extracts the deep features from IG and Iref and the decoder outputs the images ˆIG and ˆIref that minimize the auto-encoder loss LA defined as

$$
\mathcal{L}_A(\theta, \phi)=\mahtcal{L}_{recon}(\hat{I}_G, I_G) + \mathcal{L}_{recon}(\hat{I}_{ref}, I_{ref}) \tag{6}
$$

In addition, it is helpful to reduce the training time to initialize θ and φ by using any photos.  
For all the following experiments, we used θ and φ pre-trained on the natural images in [31] for one epoch.  
Mask generator for transmission recovery: Though the network parameters θ, φ, and ψT are associated with the transmission recovery, ψT is only updated by the transmission loss.  
The gradient prior that the transmission and reflection images rarely have intensive gradients at the same pixel location has been successfully used in reflection removal.  
We enhance this prior for the two images not to have intensive gradients at similar locations.  
The gradient prior loss Lgrad is defined as

$$
\mathcal{\hat{T}\hat{R}} = \frac{1}{N}\displaystyle \sum_{p_j} |\nabla \hat{T}||\nabla \hat{R}^*(p_i)|, \tag{7}
$$

where N represents the total number of pixels and ∇ˆR∗(pi) denotes the gradient having the maximum magnitude around pi, i.e. ∇ˆR∗(pi) = maxpj∈Wi |∇ˆR(pj )| where Wi denotes the set of pixels within a local window centered at pi.  
We empirically set the window size to 5. We also evaluate the additional reconstruction loss for the glass image by synthesizing a glass image using the recovered transmission and reflection images.  
For glass image synthesis, the existing methds [31, 30, 16] manually modify the reflection image to imitate the photometric distortion of reflection and combine them according to the hand-crafted image formation models.  
However, we obtain the distorted reflection image  ̄R by deactivating AdaIN of the proposed framework as  ̄R = fφ(fθ(IG) ·fψR(zR)) and synthesize the glass image by using the encoder and decoder as  ̃IG = fφ(fθ( ˆT) + fθ(  ̄R)). The transmission loss LT is defined as


$$
\mathcal{L}_T(\psi_T) = \mathcal{L}_{recon}(\tilde{I}_G, I_G) + \omega_2 \mahtcal{L}_{grad}(\hat{T}, \hat{R}). \tag{8}
$$

**Mask generator for reflection recovery:**  
While the transmission image is hypothetically estimated by applying the gradient prior, the reflection image has a reference color map R and a reference gradient map M obtained by the reference matching process, such that R(pi) = I(mci ) and M(pi) = ∇I(mg i ) where pi denotes the pixel location corresponding to xi in the rectified image.  
The total reflection loss LR is given by

$$
\mathcal{L}_R(\psi_R) = \mathcal{L}_{recon}(\tilde{I}_G, I_G) + \omega_3\mahtcal{L}_{mse}(\hat{R}, R) + \omega_4\mathcal{L}_{mse}(\nabla \hat{R}, \mathcal{M}). \tag{9}
$$

# 4 Experimental Results
This section provides the experimental results on ten 360-degree images to discuss the effectiveness of each part of the proposed method and compare the proposed method with the state-of-the-art methods qualitatively and quantitatively.  
In this work, we set the weight of w1 for LA to 1 and the weights of w1,w2,w3, and w4 for LT and LR to 10, 3, 5, and 50, respectively.  
Please see the supplementary results for more experimental results.

![Fig5](images/Fig5.png)  
Fig. 5: Effect of feature matching for reference searching. (a) Glass images and (b) reference images rectified from 360-degree images.  
The reflection recovery results are obtained by the proposed methods using the (c) color-based matching, (d)
gradient-based matching, and (e) both of them.

## 4.1 Ablation Study
**Feature matching for reference searching:**  
The proposed method utilizes the color of the recovered reflection image and the gradient of the glass images to determine the matching points to bring the information to recover the reflection image.  
We tested the comparative methods that utilize either of the color-based matching points or the gradient-based matching points to search for the reference images. 
Fig. 5 shows the glass and reference images in the 360-degree images captured in front of the fish tanks of an aquarium. As shown in Figs. 5c and 5d, the method using only the color-based matching destroys the reflected scene structures, and the method using only the gradient-based matching fails to recover the original color of the reflection image faithfully.  
However, when using both of the matching together, the proposed method recovers realistic colors while preserving the reflected scene structures.  
Note that the rectified reference image and the recovered reflection image are misaligned due to the geometric distortion.

**Glass synthesis loss:**  
Although the gradient prior provides a good insight for image decomposition, it may result in a homogeneous image where all pixels have small gradients.  
We can alleviate this problem by using the glass synthesis loss Lrecon(  ̃IG,IG). Fig. 6 shows the effect of the glass synthesis loss.  
The proposed method without Lrecon(  ̃IG,IG) provides the significantly blurred transmission images as shown in Fig. 6b where the mannequins behind the glass are disappeared from the recovered transmission image and the synthesized glass image.

![Fig6](images/Fig6.png)  
Fig. 6: Effect of the glass synthesis loss Lrecon(  ̃IG,IG). 
(a) Glass and reference images rectified from 360-degree images. The triplets of the recovered transmission, reflection, and synthesized glass images obtained by the proposed method (b) without Lrecon(  ̃IG,IG) and (c) with Lrecon(  ̃IG,IG).

![Fig7](images/Fig7.png)  
Fig. 7: Effect of the gradient prior loss Lgrad( ˆT, ˆR). (a) Glass and reference images rectified from 360-degree images.  
The pairs of the recovered transmission and reflection images obtained by the proposed method (b) without Lgrad( ˆT, ˆR) and (c) with Lgrad( ˆT, ˆR).

In contrary, the proposed method using Lrecon(  ̃IG,IG) enforces the synthesized glass images to have the image context not detected in the reflection image, which preserves the context of the transmitted scene.

**Gradient prior loss:**  
The ablation study for the gradient prior loss Lgrad shows how it affects the resulting transmission images.  
As shown in Fig. 7, whereas the method without the gradient prior loss often remains the sharp edges of the intensive reflection artifacts in the transmission images, the proposed method trained with Lgrad successfully suppresses such reflection edges.

![Fig8](images/Fig8.png)  
Fig. 8: Qualitative comparison of the reflection removal performance. (a) Pairs of the glass and reference images in 360-degree images.  
The results of the recovered transmission and reflection images obtained by (b) RS [18], (c) PRR [31], (d) BDN [30], (e) IBCLN [16], (f) PBTI [12], and (g) the proposed method.

## 4.2 Qualitative Comparison
Since there are no existing methods of unsupervised reflection removal for a single 360-degree image, we compared the proposed method with the representative unsupervised method [18] and the state-of-the-art supervised methods [12, 16, 30, 31] that remove the reflection artifacts from a single glass image.  
The rectified images of the glass regions in 360-degree images are given as input images for the existing methods.  
Most of the reflection removal methods restore not only the transmission image but also the reflection image, and thus we evaluate the quality of the recovered transmission and reflection images together.  
Fig. 8 shows the reflection removal results for three challenging glass images that make it hard for even humans to distinguish between the transmission and reflection images.  
Due to the absence of the ground truth images, the rectified images of the misaligned reference regions in the 360-degree images are inferred to display the reflected scenes.  
The unsupervised method RS [18] targets to remove blurred reflection artifacts and therefore rarely removed the reflection artifacts on the test glass images.  
Also, the existing learning-based methods failed to detect the reflection artifacts because they are mainly trained by the synthesized glass images where the reflection images are manually blurred and attenuated except PBTI [12].  
PBTI generates realistic glass images by using a graphic simulator, and suppressed the grey and homogeneous reflection artifacts from the sky as shown in the first image in Fig. 8, however, it failed to remove the colorful and structural reflection artifacts in the other glass images.  
On the other hand, the proposed method successfully estimated the reflection images and suppressed the challenging reflection artifacts with the guidance of the reference regions estimated in the 360-degree images.

## 4.3 Quantitative Comparison
We simply synthesize the glass images in 360-degree images without reflection artifacts.  
In practice, we set the center area of a 360-degree image as the glass region and suppose an arbitrary depth of the region opposite to the glass region as a reflected scene.  
Then we compose the transmission image in the glass region according to the conventional linear glass image formulation. 
Table 1 quantitatively compare the performance of the reflection removal methods using 12 synthetic 360-degree images, where ‘-T’ and ‘-R’ denote the comparison for the transmission and reflection images, respectively.  
We see that the proposed method ranks the first among the compared methods in terms of all the metrics except SSIM-T.  
However, note that the input glass image itself, without any processing, yields the SSIM-T score of 0.666, even higher than that of the most methods.  
It means that the quantitative measures are not sufficient to reflect the actual performance of the reflection removal, and the qualitative comparison on real test datasets is much more informative.

![Table1](images/Table1.png)  
Table 1: Comparison of the quantitative performance of reflection removal.  

![Fig9](images/Fig9.png)  
Fig. 9: Layer separation results accroding to different angles of the glass plane orientation.

## 4.4 Limitations
The angular deviation of the glass plane orientation may cause large displacement of the matching candidates in 3D space, and thus degrade the performance of the proposed method.  
Fig. 9 shows this limitation where the recovered transmission images remain lots of the reflection artifacts in the glass regions as the angular deviation of the glass plane orientation increases.  
Moreover, since the proposed method highly depends on the quality of the reference image captured by the camera, it fails to remove the reflected camera contents itself and it often fails to recover the transmission and/or reflection images when the reference image is overexposed due to intense ambient light.

# 5 Conclusion
This paper proposes a novel reflection removal method for 360-degree images by applying the zero-shot learning scheme. Based on reflection geometry, the proposed method searches for reliable references from outside the grass region in the 360-degree image.  
And then, it adaptively restores the truthful colors for the transmission and reflection images according to the searched references.  
Experimental results demonstrate that the proposed method provides outstanding reflection removal results compared to the existing state-of-the-art methods for 360-degree images.  
**Acknowledgements**  
This work was supported by the National Research Foundation of Korea within the Ministry of Science and ICT(MSIT) under Grant 2020R1A2B5B01002725, and by Institute of Information & communications Technology Planning & Evaluation(IITP) grant funded by the Korea government(MSIT(NO.2021-0-02068, Artificial Intelligence Innovation Hub) and o.2020-0-01336, Artificial Intelligence Graduate School Program(UNIST)).
