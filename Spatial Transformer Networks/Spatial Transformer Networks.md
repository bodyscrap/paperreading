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