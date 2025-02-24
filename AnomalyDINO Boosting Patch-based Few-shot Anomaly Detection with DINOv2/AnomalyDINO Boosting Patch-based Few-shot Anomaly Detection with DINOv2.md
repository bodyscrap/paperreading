# AnomalyDINO Boosting Patch-based Few-shot Anomaly Detection with DINOv2
[paper](https://arxiv.org/abs/2405.14529)

AnomalyDINO: Boosting Patch-based Few-shot Anomaly Detection with DINOv2

## Abstract

最近のマルチモーダル基盤モデルの進歩は、few-shot異常検出における新たな基準を打ち立てた。  
本稿では、高品質の視覚的特徴だけで、既存の最先端の視覚言語モデルに匹敵するのに十分かどうかを検討する。  
DINOv2をone-shotおよびfew-shotの異常検知に適応させことで、産業アプリケーションに焦点を当て、これを肯定する。  
我々は、本アプローチが既存の手法に匹敵するだけでなく、多くの設定においてそれらを凌駕することさえできることを示す。  
提案する視覚特徴量のみのアプローチであるAnomaly-DINOは、パッチの類似性に基づき、画像レベルの異常予測とピクセルレベルの異常セグメンテーションの両方を可能にする。  
このアプローチは方法論的にシンプルで訓練不要であるため、fine-tuningやメタ学習のための追加データを必要としない。  
AnomalyDINOは、そのシンプルさにもかかわらず、1ショットおよび数ショットの異常検知において最先端の結果を達成している(例えば、MVTec-ADのone-shot検知のパフォーマンスを93.1%のAUROCから96.6%に押し上げた)。  
AnomalyDINOは、オーバーヘッドの削減と、その卓越した数ショットのパフォーマンスにより、産業用コンテキストなどでの迅速な展開のための強力な候補となる。

## 1 Introduction

機械学習における異常検出(AD)は、公称データ分布 $p_{norm}(x)$ から大きく逸脱するインスタンスを識別しようとするものである。  
そのため、異常は「別のメカニズムで発生した」疑いが生じ[15]、  多くの場合、重大であったり、まれであったり、または予期せぬ出来事を示す。  
正常なサンプルと異常を確実に区別する能力は、セキュリティ[38]、ヘルスケア[13, 39]、工業検査など、様々な領域で高い価値を持つ。  
本論文では後者に焦点を当てる。  
完全に自動化されたシステムでは、不良品や部品の欠落を検出して下流製品の誤作動を防いだり、潜在的な危険に対して警告を発したり、あるいはそれらを分析して生産ラインを最適化したりする能力が必要となる。  
この文脈における異常サンプルについては、Figure 1の右側を参照のこと。  

![Figure1](images/Figure1.png)
Figure 1: 単一の名目上の参照サンプルに基づくAnomalyDINOによる異常検出（ここではMVTec-ADのカテゴリ「Screw」）。
数少ないパッチ表現を(潜在的に拡張された)参照サンプルから、メモリバンク $\Mu$ に集める。
テスト時には、(該当する場合は)マスキングによって関連するパッチ表現を選択する。  
$\Mu$ 中の名目上の表現との距離から、アノマリーマップと、集約統計量 $q$(←アノマリーマップの値の合計とかだと思われる) を用いた対応するアノマリースコア $s(x_{test})$ が得られる。  
マスキングと特徴抽出の両方にDINOv2を利用した。他のカテゴリの例は右図(およびAppendix. AのFigure.4とFigure.5)に示す。

産業用画像のADは、ここ2、3年で大きな関心を集めている。  
ベンチマークデータにおける最適に近い結果は、異常検出の問題が本質的に解決されたかのように思わせる。  
例えば、Mousakhanら[27]は、一般的なベンチマークであるMVTec-AD [2]とVisA [48]において、それぞれ99.8%と98.9%のAUROCを報告している。  
最も一般的なAD技術は、訓練データを使用しｔ異常分類器[36]を訓練するか、再構成ベース[40, 25, 27]、または尤度ベース[34, 9]の異常スコアリングと組み合わせた生成モデルを使用する。  
しかし、これらのアプローチはfull-shotな設定で動作するため、十分な量のトレーニングデータへのアクセスに依存している。  
データセット取得に関連する課題や、高速で展開が容易な手法の魅力、公称データ分布の共変量シフトに迅速に適応する必要性[21]を考慮すると、few-shotやzero-shotの異常検出への関心が高まっている。  
しかし、few-shotのテクニックは、意味のある特徴に大きく依存している。つまり、[32]が言うように、「異常検出にはより良い表現が必要」なのである。  
このようなより良い表現は、教師なし/自己教師ありの方法で膨大なデータセットに対して学習された大規模なモデル、すなわち基盤モデルの利用可能性と能力の向上により、現在利用可能になっている[30, 5, 29]。  
few-shot異常検知技術の性能は、基盤モデルの使用によって、主に言語と視覚を組み込んだマルチモーダルアプローチによって、すでに向上している[18, 4, 47, 20]。  
ここでは、そのようなマルチモーダル技術とは対照的に、視覚のみのアプローチに焦点を当てることを提案する。  
この視点は、few-shot異常検出は、人間の注釈者が視覚的特徴のみに基づいて実行可能であり、与えられたオブジェクトや予想される異常の種類(一般的に先験的に知られていない)の追加のテキスト記述を必要としないという観察によって動機づけられている。  

AnomalyDINOと呼ぶ我々のアプローチは、メモリバンクベースであり、バックボーンとしてDINOv2 [29]を活用している(ただし、パッチレベルの特徴抽出が強力な他のアーキテクチャにも適応可能である)。  
DINOv2のセグメンテーション能力(別のモデルの追加オーバーヘッドを軽減する)を使用して、one-shot シナリオに適した前処理パイプラインを慎重に設計する。  
テスト時には、パッチ表現と公称メモリバンク内の最も近い対応物との間の距離が大きいことに基づいて、異常なサンプルが検出される。  

AnomalyDINOはシンプルであるため、[7]や[20]のような複雑なアプローチとは対照的に、非常に簡単に産業界に導入することができる。  

しかし、提案手法は、MVTec-AD [2]において、few-shot領域での異常検出において最新の性能を達成し、VisA [48]では、1つを除き競合するすべての手法を凌駕している。  

論文の構成は以下の通り：  
Section 2では、関連する先行研究をレビューし、zero-shotとfew-shot、およびバッチ化zero-shot技法が扱う設定の違いを明確にする。  

Section 3では、提案手法であるAnomalyDINOを紹介する。  
この手法のバッチ化zero-shotシナリオへの拡張はAppendix Dで詳述する。

Section 4では実験結果を示す。  
その他の結果とablation studyは、それぞれAppendix AとCに記載されている。  

Appendix Bでは、AnomalyDINOで確認された失敗事例を取り上げる。
実験を再現するコードはhttps://github.com/dammsi/AnomalyDINO にて公開されている。  

**Contributions(貢献)**  

- AnomalyDINOを提案する。AnomalyDINOは、視覚的異常検出のための、シンプルで訓練不要でありながら非常に効果的なパッチベースの手法である。提案手法はDINOv2によって抽出された高品質な特徴表現に基づいている。

- 広範な分析により、提案アプローチの効率性と有効性が実証され、性能と推論速度の点で他のマルチモーダルfew-shot技術を凌駕している。具体的には、AnomalyDINOはMVTec-ADにおけるfew-shotの異常検知において最先端の結果を達成し、例えば、one-shot検知のAUROCを93.1%から96.6%に押し上げた(これにより、few-shotとfull-shot設定間のギャップを半減させた)。さらに、Visaにおけるの結果は、他のfew-shot手法に匹敵するだけでなく、全ての訓練不要few-shot異常検出に対して新たな最先端を確立した。また,全ての手法にを見ても最高のローカライズ性能を達成している。  

## 2 Related Work

**Foundation Models for Vision**  
マルチモーダル基盤モデルは、さまざまなタスクのための強力なツールとして浮上してきた。例えば[3, 22, 6, 16, 30, 24, 28]を参照して欲しい。  
画像に夜ADに最も関連するのは、CLIP [30] や最近のLLM [28] に基づくマルチモーダルアプローチだが、DINO [5, 29] のような視覚特徴のみのアプローチもある。  
CLIP[30]は、テキスト注釈と対になった画像のデータセットで学習することで、自然言語記述から視覚的概念を学習する。  
このモデルは、画像エンコーダとテキストエンコーダからの埋め込みを整列させ、対応する画像とテキストのペア間の類似度を最適化する対照的学習目的を使用する。  
視覚と言語に共通するこの特徴空間は、クラス固有のプロンプト集合との類似性を評価することで、zero-shot画像の分類など、いくつかの下流タスクに利用することができる。  
dino[5,29]は、vision transformers[12]に基づく自己教師型stutend-teacherフレームワークを活用している。  
マルチビュー戦略を採用し、ソフト化(soft-labelingの事だと思う)された教師出力を予測するモデルを学習することで、下流タスクのためのロバストで高品質な特徴を学習する。  
DINOv2[29]はDINOのアイデアとパッチレベルの再構成技術[46]を組み合わせたもので、より大規模なアーキテクチャとデータセットに対応する。  
DINOによって抽出された特徴量は、ローカル情報とグローバル情報の両方を取り込み、複数のビューやクロップに対してロバストであり、大規模な事前学習が有効であるため、異常検出に適している。  
GroundingDINO [24]は、DINOフレームワークを基礎とし、テキスト情報と視覚情報の整合を改善することに重点を置き、詳細なオブジェクトの位置特定とマルチモーダル理解を必要とするタスクにおけるモデルのパフォーマンスを向上させる。  

**Anomaly Detection**  
Given a predefined notion of normality, the anomaly detection task is to detect test samples that deviate from this concept [43, 35].  
In this work, we focus on low-level sensory anomalies of industrial image data, i.e., we do not target the detection of semantic anomalies but of low-level features such as scratches of images of industrial products (see e.g., Figure 1).  
Several works tackled this task by either training an anomaly classifier [36] or a generative model, which allows for reconstruction-based or likelihood-based AD [40, 25, 45, 9, 27].
However, this work focuses on few-shot AD, which tackles anomaly detection under strong restrictions on the number of available training samples. Various recent methods recognized the effectiveness of pre-trained vision models for few-shot AD.  
The underlying idea in these approaches is to leverage the feature representations originating from pre-trained ResNets [17], Wide ResNets [44], and Vision Transformers (ViT) [11].  
These models can extract representations on a patch and pixel level, which can be compared against a memory bank, which aggregates patch and pixel features provided by the training set.  
The comparison with the memory bank is typically conducted using a nearest neighbor approach [1, 8, 10, 33, 41, 23]. Loosely speaking, patches and pixels corresponding to anomalies are expected to have a high distance to their nearest neighbor in the memory bank.  
Another line of work builds upon the success of pre-trained language-vision models in zero-shot classification.  
The underlying idea consists of two steps.  
First, these approaches define sets of prompts describing nominal samples and anomalies.  
Second, the corresponding textual embeddings are compared against the image embeddings [18, 7, 20, 47].  
Images whose visual embedding is close to the textual embedding of a prompt associated with an anomaly are classified as anomalous.  
However, these methods either require significant prompt engineering (e.g., [7] use a total of 35×7 different prompts for describing normal samples) or fine-tuning of the prompt(-embeddings).  
Lastly, another type of few-shot anomaly detection builds upon the success of multimodal chatbots.  
These methods require more elaborate prompting and techniques for interpreting textual outputs [42].  
Since these methods do not require a memory bank, they are capable of performing zero-shot anomaly detection.  

**Categorization of Few-/Zero-Shot Anomaly Detectors**  
Previous works consider different AD setups, which complicates their evaluation and comparison.  
To remedy this, we provide a taxonomy of recent few- and zero-shot AD based on the particular ‘shot’-setting, the training requirements, and the modes covered by the underlying models.
We categorize three ‘shot’-settings: zero-shot, few-shot, and batched zero-shot.  
Zero- and few-shot settings are characterized by the number of nominal training samples a method can process, before making predictions on the test samples.  
In batched zero-shot, inference is not performed sample-wise but based on a whole batch of test sam-ples, usually the full test set.  
For instance, the method proposed in [23] benefits from the fact that a significant majority of pixels correspond to normal pixels, which motivates the strategy of matching patches across a batch of images.  
Another work that considers this setting [21], deploys a parameter-free anomaly detector based on the effect of batch normalization. We split the training requirements into the categories ‘Training-Free’, ‘Fine-Tuning’, and ‘Meta- Training’.  
‘Training-Free’ approaches do not require any training, while ‘Fine-Tuning’ methods use the few accessible samples to modify the underlying model.  
In contrast, ‘Meta-Training’ is associated with training the model on a dataset related to the test data.  
For example, [21] train their model on MVTec-AD containing all classes except the class they test against. [47] and [7] train their model on VisA when evaluating the test performance on MVTec-AD and vice versa.  
Finally, we differentiate the leveraged models, which are either vision models (such as pre-trained ViT) or language-vision models (such as CLIP).  
We provide a detailed summary in Table 1.

Table 1: Taxonomy of recent few- and zero-shot anomaly
detection methods. The † indicates approaches that were
introduced as full-shot detectors but then considered as few-
shot detectors in later works, see e.g., [33].  

![Table1](images/Table1.png)  

