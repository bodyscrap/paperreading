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

Our approach, termed AnomalyDINO, is memory bank-based and leverages DINOv2 [29] as a backbone (but the approach could be adapted to other architectures with strong patch-level feature extraction).  
We carefully design a suitable preprocessing pipeline for the one-shot scenario, using the segmentation abilities of DINOv2 (which alleviates the additional overhead of another model).  
At test time, anomalous samples are detected based on the high distances between their patch representations and the closest counterparts in the nominal memory bank.  
Due to its simplicity, AnomalyDINO can be deployed in industrial contexts very easily—in strong contrast to more complex approaches such as those proposed by [7] or [20].  
Yet, the proposed method achieves new state-of-the-art performance on anomaly detection in the few-shot regime on MVTec-AD [2] and outperforms all but one competing method on VisA [48].  
The structure of the paper is as follows: Section 2 reviews relevant prior studies and clarifies the distinctions between the settings addressed by zero- and few-shot, and batched zero-shot techniques.  
Section 3 introduces our proposed method, AnomalyDINO.  
An extension of this method to the batched zero-shot scenario is detailed in Appendix D.  
Section 4 presents the experimental outcomes.  
Additional results and an ablation study are provided in Appendices A and C, respectively.  
We address identified failure cases of AnomalyDINO in Appendix B.  
The code to reproduce the experiments is available at https://github.com/dammsi/AnomalyDINO.