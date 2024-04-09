# RMT : Retentive Networks Meet Vision Transformers
[paper](https://arxiv.org/abs/2309.11523v5)
[github](https://github.com/qhfan/RMT)

## Abstract
Vision Transformer (ViT)は、近年コンピュータ・ビジョンのコミュニティで注目を集めている。
しかし、ViTの核となる要素である「Self-Attention」は、明示的な空間的事前分布を持たず、計算量も2次関数的であるため、ViTの適用性に制約がある。
このような問題を解決するために、我々は自然言語処理分野のRetentive Network(RetNet)から着想を得て、汎用の明示的な空間事前分布を持つ強力な視覚バックボーンであるRMTを提案する。 
具体的には、RetNetの時間的減衰メカニズムを空間領域に拡張し、マンハッタン距離に基づく空間的減衰行列を提案し、Self-Attentionに明示的な空間的事前分布を導入する。 
さらに、空間的減衰行列を崩すことなく、大域的な情報をモデル化する計算負荷を軽減することを目的として、明示的な空間的先行に巧みに適応するAttention分解形式を提案する。
空間減衰行列とAttention分解形式に基づき、線形の複雑度で視覚バックボーンに明示的な空間事前分布を柔軟に統合することができる。 
広範な実験により、RMTは様々な視覚タスクにおいて卓越した性能を示すことが実証された。 
具体的には、余分な学習データなしで、RMTはImageNet-1kにおいて27M/4.5GFLOPsのモデルで84.8%、96M/18.2GFLOPsのモデルで86.1%のtop-1 accを達成した。 
下流のタスクでは、RMTはCOCO検出タスクで54.5 box APと47.2 mask APを達成し、ADE20K sematic segmentationタスクで52.8mIoUを達成した。

コードはこちら: //github.com/qhfan/RMT


## 1. Introduction
Vision Transformer (ViT) [12] is an excellent visual architecture highly favored by researchers. 
However, as the core module of ViT, Self-Attention’s inherent structure lacking explicit spatial priors. Besides, the quadratic complexity of Self-Attention leads to significant computational costs when modeling global information. 
These issues limit the application of ViT.
Many works have previously attempted to alleviate these issues [13, 16, 30, 35, 50, 57, 61]. For example, in Swin Transformer [35], the authors partition the tokens used for self-attention by applying windowing operations. 
This operation not only reduces the computational cost of self-attention but also introduces spatial priors to the model through the use of windows and relative position encoding.
In addition to it, NAT [19] changes the receptive field of Self-Attention to match the shape of convolution, reducing computational costs while also enabling the model to perceive spatial priors through the shape of its receptive field.
Different from previous methods, we draw inspiration from the recently successful Retentive Network (Ret- Net) [46] in the field of NLP. 
RetNet utilizes a distance-dependent temporal decay matrix to provide explicit temporal prior for one-dimensional and unidirectional text data.
ALiBi [41], prior to RetNet, also applied a similar approach and succeeded in NLP tasks. 
We extend this temporal decay matrix to the spatial domain, developing a two-dimensional bidirectional spatial decay matrix based on the Manhattan distance among tokens. 
In our space decay matrix, for a target token, the farther the surrounding tokens are, the greater the degree of decay in their attention scores. 
This property allows the target token to perceive global information while simultaneously assigning different levels of attention to tokens at varying distances. 
We introduce explicit spatial prior to the vision backbone using this spatial decay matrix.
We name this Self-Attention mechanism, which is inspired by RetNet and incorporates the Manhattan distance as the explicit spatial prior, as Manhattan Self-Attention (MaSA).
Besides explicit spatial priors, another issue caused by global modeling with Self-Attention is the enormous computational burden. 
Previous sparse attention mechanisms [11, 35, 53, 63, 75] and the way retention is decomposed in RetNet [46] mostly disrupt the spatial decay matrix, making them unsuitable for MaSA. In order to sparsely model global information without compromising the spatial decay matrix, we propose a method to decompose Self-Attention along both axes of the image. 
This decomposition method decomposes Self-Attention and the spatial decay matrix without any loss of prior information. 
The decomposed MaSA models global information with linear complexity and has the same receptive field shape as the original MaSA. 
We compare MaSA with other Self-Attention mechanisms in Fig. 2. 
It can be seen that our MaSA introduces richer spatial priors to the model than its counterparts.
Based on MaSA, we construct a powerful vision backbone called RMT. 
We demonstrate the effectiveness of the proposed method through extensive experiments. 
As shown in Fig. 1, our RMT outperforms the state-of-the-art (SOTA) models on image classification tasks. 
Additionally, our model exhibits more prominent advantages compared to other models in tasks such as object detection, instance segmentation, and semantic segmentation. 
Our contributions can be summarized as follows:

* We propose a spatial decay matrix based on Manhattan distance to augment Self-Attention, creating the Manhattan Self-Attention (MaSA) with an explicit spatial prior.
* We propose a decomposition form for MaSA, enabling linear complexity for global information modeling without disrupting the spatial decay matrix.
* Leveraging MaSA, we construct RMT, a powerful vision backbone for general purposes. RMT attains high top-1 accuracy on ImageNet-1k in image classification without extra training data, and excels in tasks like object detection, instance segmentation, and semantic segmentation.

![Figure1](images/Figure1.png)
Figure 1. FLOPs v.s. Top-1 accuracy on ImageNet-1K with 224 ×224 resolution. “*” indicates the model trained with token labeling [27].

![Figure2](images/Figure2.png)
Figure 2. Comparison among different Self-Attention mechanisms. 
In MaSA, darker colors represent smaller spatial decay rates, while lighter colors represent larger ones. 
The spatial decay rates that change with distance provide the model with rich spatial priors.