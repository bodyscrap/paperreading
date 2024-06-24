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

## 2. Related Work
**Transformer**. 
Transformer architecture was firstly proposed in [52] to address the training limitation of recurrent model and then achieve massive success in many NLP tasks. 
By splitting the image into small, non-overlapped patches sequence, Vision Transformer (ViTs) [12] also have attracted great attention and become widely used on vision tasks [5, 14, 18, 39, 58, 66]. 
Unlike in the past, where RNNs and CNNs have respectively dominated the NLP and CV fields, the transformer architecture has shined through in various modalities and fields [26, 37, 42, 60]. In the computer vision community, many studies are attempting to introduce spatial priors into ViT to reduce the data requirements for training [6, 19, 49]. 
At the same time, various sparse attention mechanisms have been proposed to reduce the computational cost of Self-Attention [13, 53, 54, 57].

**Prior Knowledge in Transformer**. 
Numerous attempts have been made to incorporate prior knowledge into the Transformer model to enhance its performance. 
The original Transformers [12, 52] use trigonometric position en-coding to provide positional information for each token. 
In vision tasks, [35] proposes the use of relative positional encoding as a replacement for the original absolute positional encoding. 
[6] points out that zero padding in convolutional layers could also provide positional awareness for the ViT, and this position encoding method is highly efficient. 
In many studies, Convolution in FFN [13, 16, 54] has been employed for vision models to further enrich the positional information in the ViT. 
For NLP tasks, in the recent Retentive Network [46], the temporal decay matrix has been introduced to provide the model with prior knowledge based on distance changes. 
Before RetNet, ALiBi [41] also uses a similar temporal decay matrix.

## 3. Methodology
### 3.1. Preliminary
**Temporal decay in RetNet**. 
Retentive Network (RetNet) is a powerful architecture for language models. 
This work proposes the retention mechanism for sequence modeling.  
Retention brings the temporal decay to the language model, which Transformers do not have. 
Retention firstly considers a sequence modeling problem in a recurrent manner. 
It can be written as Eq. 1:

$$
o_n = \sum_{m=1}^n \gamma^{n-m}\left(Q_ne^{in0}\right)\left(K_me^{im0}\right)^\dagger v_m \tag{1}
$$

For a parallel training process, Eq. 1 is expressed as:

$$
Q=(XW_Q) \odot \Theta, K=(XW_K) \odot \bar{\Theta}, V=XW_V \\
\Theta_n = e^{in0}, D_{nm} =\begin{cases} \gamma^{n-m} & n \geq m \\ 0 & n < m \end{cases} \\
Retention(X) = (QK^T\odot D)V \tag{2}
$$
where $\bar{\Theta}$ is the complex conjugate of $\Theta$, and $D \in \mathbb{R}^{|x|\times|x|}$ contains both causal masking and exponential decay, which symbolizes the relative distance in one-dimensional sequence and brings the explicit temporal prior to text data.

### 3.2. Manhattan Self-Attention
Starting from the retention in RetNet, we evolve it into Manhattan Self-Attention (MaSA). 
Within MaSA, we transform the unidirectional and one-dimensional temporal decay observed in retention into bidirectional and two-dimensional spatial decay. 
This spatial decay introduces an explicit spatial prior linked to Manhattan distance into the vision backbone. 
Additionally, we devise a straightforward approach to concurrently decompose the Self-Attention and spatial decay matrix along the two axes of the image.

**From Unidirectional to Bidirectional Decay:**  
In RetNet, retention is unidirectional due to the causal nature of text data, allowing each token to attend only to preceding tokens and not those following it. 
This characteristic is ill-suited for tasks lacking causal properties, such as image recognition. 
Hence, we initially broaden the retention to a bidirectional form, expressed as Eq. 3:

$$
\begin{aligned}
BiRetention(X) &= (QK^T\odot D^{Bi}) V \\
D_nm^{Bi} &= \gamma^{|n-m|} \tag{3}
\end{aligned}
$$

where $BiRetention$ signifies bidirectional modeling.

**From One-dimensional to Two-dimensional Decay:**  
While retention now supports bi-directional modeling, this capability remains confined to a one-dimensional level and is inadequate for two-dimensional images. 
To address this limitation, we extend the one-dimensional retention to encompass two dimensions.  
In the context of images, each token is uniquely positioned with a two-dimensional coordinate within the plane, denoted as $(x_n,y_n)$ for the n-th token. 
To adapt to this, we adjust each element in the matrix D to represent the Manhattan distance between the respective token pairs based on their 2D coordinates. 
The matrix $D$ is redefined as follows:

$$
D_{nm}^{2d} = \gamma^{|x_n-x_m|+|y_n-y_m|} \tag{4}
$$

In the retention, the Softmax is abandoned and replaced with a gating function. This variation gives RetNet multiple flexible computation forms, enabling it to adapt to parallel training and recurrent inference processes. 
Despite this flexibility, when exclusively utilizing RetNet’s parallel computation form in our experiments, the necessity of retaining the gating function becomes debatable. 
Our findings indicate that this modification does not improve results for vision models; instead, it introduces extra parameters and computational complexity. 
Consequently, we continue to employ Softmax to introduce nonlinearity to our model.
Combining the aforementioned steps, our Manhattan Self-Attention is expressed as

$$
\begin{aligned}
MaSA(X) &= (Softmax(QK^T)\odot D^{2d}) V \\
D_{nm}^{2d} &= \gamma^{|x_n-x_m|+|y_n-y_m|} \tag{5}
\end{aligned}
$$

**Decomposed Manhattan Self-Attention.** 
In the early stages of the vision backbone, an abundance of tokens leads to high computational costs for Self-Attention when attempting to model global information. 
Our MaSA encounters this challenge as well. 
Utilizing existing sparse attention mechanisms [11, 19, 35, 53, 63], or the original RetNet’s recurrent/chunk-wise recurrent form directly, disrupts the spatial decay matrix based on Manhattan distance, resulting in the loss of explicit spatial prior. 
To address this, we introduce a simple decomposition method that not only decomposes Self-Attention but also decomposes the spatial decay matrix. 
The decomposed MaSA is represented in Eq. 6. Specifically, we calculate attention scores separately for the horizontal and vertical directions in the image. 
Subsequently, we apply the one-dimensional bidirectional decay matrix to these attention weights. 
The one-dimensional decay matrix signifies the horizontal and vertical distances between tokens $(D^H_{nm} = \gamma^{|y_n−y_m|}, D^W_{nm} = \gamma^{|x_n−x_m|})$:

![Figure3](images/Figure3.png)
Figure 3. Overall architecture of RMT.

![Figure4](images/Figure4.png)
Figure 4. Spacial decay matrix in the decomposed MaSA.

$$
\begin{aligned}
Attn_H = Softmax(Q_H K_H^\tau)\odot D^H, \\
Attn_W = Softmax(Q_WK_W^\tau)\odot D^W ,\\
MaSA(X) = Attn_H (Attn_W V)\tau \tag{6}
\end{aligned}
$$

Based on the decomposition of $MaSA$, the shape of the receptive field of each token is shown in Fig. 4, which is identical to the shape of the complete MaSA’s receptive field. 
Fig. 4 indicates that our decomposition method fully preserves the explicit spatial prior.
To further enhance the local expression capability of MaSA, following [75], we introduce a Local Context Enhancement module using DWConv:

$$
X_{out} = MaSA(X) + LCE(V) \tag{7}
$$

### 3.3. Overall Architecture
We construct the RMT based on MaSA, and its architecture is illustrated in Fig. 3. 
Similar to previous general vision backbones [35, 53, 54, 71], RMT is divided into four stages. 
The first three stages utilize the decomposed MaSA, while the last uses the original MaSA. 
Like many previous backbones [16, 30, 72, 75], we incorporate CPE [6] into our model.

## 4. Experiments
We conducted extensive experiments on multiple vision tasks, such as image classification on ImageNet-1K [9],
object detection and instance segmentation on COCO 2017 [33], and semantic segmentation on ADE20K [74].
We also make ablation studies to validate the importance of each component in RMT. More details can be found in Appendix.
### 4.1. Image Classification
Settings. We train our models on ImageNet-1K [9] from scratch. We follow the same training strategy in [49], with the only supervision being classification loss for a fair comparison. 
The maximum rates of increasing stochastic depth [24] are set to 0.1/0.15/0.4/0.5 for RMT-T/S/B/L [24], respectively. 
We use the AdamW optimizer with a cosine decay learning rate scheduler to train the models. 
We set the initial learning rate, weight decay, and batch size to 0.001, 0.05, and 1024, respectively. 
We adopt the strong data augmentation and regularization used in [35]. 
Our settings are RandAugment [8] (randm9-mstd0.5-inc1), Mixup [70] (prob=0.8), CutMix [69] (prob=1.0), Random Erasing [73] (prob=0.25). 
In addition to the conventional training meth- ods, similar to LV-ViT [27] and VOLO [68], we train a model that utilizes token labeling to provide supplementary supervision.

![Table1](images/Table1.png)
Table 1. Comparison with the state-of-the-art on ImageNet-1K classification. “*” indicates the model trained with token labeling [27].

**Results.** 
We compare RMT against many state-of-the-art models in Tab. 1. 
Results in the table demonstrate that RMT consistently outperforms previous models across all settings. 
Specifically, RMT-S achieves 84.1% Top1-accuracy with only 4.5 GFLOPs. RMT-B also surpasses iFormer [45] by 0.4% with similar FLOPs. 
Furthermore, our RMT-L model surpasses MaxViT-B [51] in top1-accuracy by 0.6% while using fewer FLOPs. 
Our RMT-T has also outperformed many lightweight models. 
As for the model trained using token labeling, our RMT-S outperforms the current state-of-the-art BiFormer-S by 0.5%.

### 4.2. Object Detection and Instance Segmentation
**Settings.** 
We adopt MMDetection [4] to implement RetinaNet [32], Mask-RCNN [22] and Cascade Mask R-CNN [2]. 
We use the commonly used “1×” (12 training epochs) setting for the RetinaNet and Mask R-CNN. 
Besides, we use “3 ×+MS” for Mask R-CNN and Cascade Mask R-CNN. 
Following [35], during training, images are resized to the shorter side of 800 pixels while the longer side is within 1333 pixels. 
We adopt the AdamW optimizer with a learning rate of 0.0001 and batch size of 16 to optimize the model. 
For the “1×” schedule, the learning rate declines with the decay rate of 0.1 at the epoch 8 and 11.
While for the “3 ×+MS” schedule, the learning rate declines with the decay rate of 0.1 at the epoch 27 and 33.

![Table2](images/Table2.png)
Table 2. Comparison to other backbones using RetinaNet and Mask R-CNN on COCO val2017 object detection and instance segmentation.

![Table3](images/Table3.png)
Table 3. Comparison to other backbones using Mask R-CNN with ”3 ×+MS” schedule.

![Table4](images/Table4.png)
Table 4. Comparison to other backbones using Cascade Mask R-CNN with ”3 ×+MS” schedule.

**Results.** 
Tab. 2, Tab. 3 and Tab. 4 show the results with different detection frameworks. The results demonstrate that our RMT performs best in all comparisons. 
For the RetinaNet framework, our RMT-T outperforms MPViT-XS by +1.3 AP, while S/B/L also perform better than other methods. 
As for the Mask R-CNN with “1×” schedule, RMT-L outperforms the recent InternImage-B by +2.8 box AP and +1.9 mask AP. 
For “3 × +MS” schedule, RMT-S outperforms InternImage-T for +1.6 box AP and +1.2 mask AP. 
Besides, regarding the Cascade Mask R-CNN, our RMT still performs much better than other backbones. 
All the above results tell that RMT outperforms its counterparts by evident margins.

![Table5](images/Table5.png)
Table 5. Comparison with the state-of-the-art on ADE20K.

## 4.3. Semantic Segmentation
**Settings.** 
We adopt the Semantic FPN [28] and UperNet [59] based on MMSegmentation [7], apply RMTs which are pretrained on ImageNet-1K as backbone. 
We use the same setting of PVT [53] to train the Semantic FPN, and we train the model for 80k iterations. 
All models are trained with the input resolution of 512 ×512. 
When testing the model, we resize the shorter side of the image to 512 pixels. 
As for UperNet, we follow the default settings in Swin [35]. 
We take AdamW with a weight decay of 0.01 as the optimizer to train the models for 160K iterations. 
The learning rate is set to 6×10−5 with 1500 iterations warmup.

**Results.** 
The results of semantic segmentation can be found in Tab. 5. 
All the FLOPs are measured with the resolution of 512 ×2048, except the group of RMT-T, which are measured with the resolution of 512 × 512. 
All our models achieve the best performance in all comparisons.  
Specifically, our RMT-S exceeds Shunted-S for +1.2 mIoU with Semantic FPN. 
Moreover, our RMT-B outperforms the recent InternImage-S for +1.8 mIoU. 
All the above results demonstrate our model’s superiority in dense prediction.

## 4.4. Ablation Study
**Strict comparison with previous works.** 
In order to make a strict comparison with previous methods, we align RMT’s hyperparameters (such as whether to use hierarchical structure, the number of channels in the four stages of the hierarchical model, whether to use positional encoding and convolution stem, etc.) of the overall architecture with DeiT [49] and Swin [35], and only replace the Self-Attention/Window Self-Attention with our MaSA. 
The comparison results are shown in Tab. 6, where RMT significantly outperforms DeiT-S, Swin-T, and Swin-S.  MaSA. 
We verify the impact of Manhattan Self-Attention on the model, as shown in the Tab. 6. 
MaSA improves the model’s performance in image classification and downstream tasks by a large margin. 
Specifically, the classification accuracy of MaSA is 0.8% higher than that of vanilla attention.
**Softmax.** 
In RetNet, Softmax is replaced with a non-linear gating function to accommodate its various computational forms [46]. 
We replace the Softmax in MaSA with this gating function. However, the model utilizing the gating function cannot undergo stable training. 
It is worth noting that this does not mean the gating function is inferior to Softmax. 
The gating function may just not be compatible with our decomposed form or spatial decay.

**LCE.** 
Local Context Enhancement also plays a role in the excellent performance of our model. 
LCE improves the classification accuracy of RMT by 0.3% and enhances the model’s performance in downstream tasks.
**CPE.** 
Just like previous methods, CPE provides our model with flexible position encoding and more positional information, contributing to the improvement in the model’s performance in image classification and downstream tasks.

![Table6](images/Table6.png)
Table 6. Ablation study. We make a strict comparison among RMT, DeiT, and Swin-Transformer.  

![Table7](images/Table7.png)
Table 7. Comparison between decomposed MaSA (MaSA-d) and original MaSA.

![Table8](images/Table8.png)
Table 8. Comparison between MaSA and retention in RMT-S’s architecture.

![Table9](images/Table9.png)
Table 9. Comparison of inference speed among SOTA models.  

**Convolutional Stem.** 
The initial convolutional stem of the model provides better local information, thereby further enhancing the model’s performance on various tasks.
**Decomposed MaSA.** 
In RMT-S, we substitute the decomposed MaSA (MaSA-d) in the third stage with the original MaSA to validate the effectiveness of our decomposition method, as illustrated in Tab. 7. 
In terms of image classification, MaSA-d and MaSA achieve comparable accuracy.
However, for semantic segmentation, employing MaSA-d
significantly reduces computational burden while yielding
similar result.
**MaSA v.s. Retention.**  
As shown in Tab. 8, we replace MaSA with the original retention in the architecture of RMT-S. 
We partition the tokens into chunks using the method employed in Swin-Transformer [35] for chunk-wise retention. 
Due to the limitation of retention in modeling one-dimensional causal data, the performance of the vision backbone based on it falls behind RMT. Moreover, the chunk-wise and recurrent forms of retention disrupt the parallelism of the vision backbone, resulting in lower inference speed.
**Inference Speed.**
 We compare the RMT’s inference speed with the recent best performing vision backbones in Tab. 9. Our RMT demonstrates the optimal trade-off be-
tween speed and accuracy.

## 5. Conclusion
In this work, we propose RMT, a vision backbone with explicit spatial prior. 
RMT extends the temporal decay used for causal modeling in NLP to the spatial level and introduces a spatial decay matrix based on the Manhattan distance. 
The matrix incorporates explicit spatial prior into the Self-Attention. Additionally, RMT utilizes a Self-Attention decomposition form that can sparsely model global information without disrupting the spatial decay matrix. 
The combination of spatial decay matrix and attention decomposition form enables RMT to possess explicit spatial prior and linear complexity. 
Extensive experiments in image classification, object detection, instance segmentation, and semantic segmentation validate the superiority of RMT.

## A. Architecture Details
Our architectures are illustrated in the Tab. 10. For convolution stem, we apply five 3 ×3 convolutions to embed the image into 56 ×56 tokens. 
GELU and batch normalization are used after each convolution except the last one, which is only followed by batch normalization. 
3 ×3 convolutions with stride 2 are used between stages to reduce the feature map’s resolution. 
3 ×3 depth-wise convolutions are adopted in CPE. Moreover, 5 ×5 depth-wise convolutions are adopted in LCE. RMT-DeiT-S, RMT-Swin-T, and RMT-Swin-S are models that we used in our ablation experiments. 
Their structures closely align with the structure of DeiT [49] and Swin-Transformer [35] without using techniques like convolution stem, CPE, and others.

## B. Experimental Settings
**ImageNet Image Classification.** 
We adopt the same training strategy with DeiT [49] with the only supervision is the classification loss. 
In particular, our models are trained from scratch for 300 epochs. 
We use the AdamW optimizer with a cosine decay learning rate scheduler and 5 epochs of linear warm-up. 
The initial learning rate, weight decay, and batch size are set to 0.001, 0.05, and 1024, respectively. 
Our augmentation settings are RandAugment [8] (randm9-mstd0.5-inc1), Mixup [70] (prob=0.8), CutMix [69] (probe=1.0), Random Erasing [73] (prob=0.25) and Exponential Moving Average (EMA) [40].
The maximum rates of increasing stochastic depth [24] are set to 0.1/0.15/0.4/0.5 for RMT-T/S/B/L, respectively. 
For a more comprehensive comparison, we train two versions of the model. 
The first version uses only classification loss as the supervision, while the second version, in addition to the classification loss, incorporates token labeling introduced by [27] for additional supervision. 
Models using token labeling are marked with“*”.

**COCO Object Detection and Instance Segmentation.**
We apply RetinaNet [32], Mask-RCNN [22] and Cascaded Mask-CNN [2] as the detection frameworks to conduct experiments. We implement them based on the MMDetection [4]. 
All models are trained under two common settings:“1×” (12 epochs for training) and“3×+MS” (36 epochs with multi-scale augmentation for training). 
For the “1×” setting, images are resized to the shorter side of 800 pixels. 
For the “3×+MS”, we use the multi-scale training strategy and randomly resize the shorter side between 480 to 800 pixels. We apply AdamW optimizer with the initial learning rate of 1e-4. 
For RetinaNet, we use the weight decay of 1e-4 for RetinaNet while we set it to 5e-2 for Mask-RCNN and Cascaded Mask-RCNN. For all settings, we use the batch size of 16, which follows the previous works [35, 63, 64].

**ADE20K Semantic Segmentation.** 
Based on MMSegmentation [7], we implement UperNet [59] and Seman- ticFPN [28] to validate our models. 
For UperNet, we follow the previous setting of Swin-Transformer [35] and train the model for 160k iterations with the input size of 512 ×512. 
For SemanticFPN, we also use the input resolution of 512 ×512 but train the models for 80k iterations.

## C. Efficiency Comparison
We compare the inference speed of RMT with other backbones, as shown in Tab. 11. 
Our models achieve the best trade-off between speed and accuracy among many competitors.

## D. Details of Explicit Decay
We use different $\gamma$ for each head of the multi-head ReSA to control the receptive field of each head, enabling the ReSA to perceive multi-scale information. 
We keep all the $\gamma$ of ReSA’s heads within a certain range. Assuming the given receptive field control interval of a specific ReSA module is [a,b], where both a and b are positive real numbers. 
And the total number of the ReSA module’s heads is N. 
The $\gamma$ for its ith head can be written as Eq. 8:

$$
\gamma_i = 1 - 2^{-a - \frac{(b-a)i}{N}}\tag{8}
$$

For different stages of different backbones, we use different values of a and b, with the details shown in Tab. 12.

![Table10](images/Table10.png)
Table 10. Detailed Architectures of our models.

![Table11](images/Table11.png)
Table 11. Comparison of inference speed.

![Table12](images/Table12.png)
Table 12. Details about the $\gamma$ decay.


