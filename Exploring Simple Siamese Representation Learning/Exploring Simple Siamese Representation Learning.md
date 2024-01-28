# Exploring Simple Siamese Representation Learning
[paper](https://arxiv.org/abs/2011.10566)
[code](https://github.com/facebookresearch/simsiam)

## Abstract
Siamese networks have become a common structure in various recent models for unsupervised visual representation learning. 
These models maximize the similarity between two augmentations of one image, subject to certain conditions for avoiding collapsing solutions. 
In this paper, we report surprising empirical results that simple Siamese networks can learn meaningful representations even using none of the following: (i) negative sample pairs, (ii) large batches, (iii) momentum encoders. 
Our experiments show that collapsing solutions do exist for the loss and structure, but a stop-gradient operation plays an essential role in preventing collapsing. 
We provide a hypothesis on the impli- cation of stop-gradient, and further show proof-of-concept experiments verifying it. Our “SimSiam” method achieves competitive results on ImageNet and downstream tasks. 
We hope this simple baseline will motivate people to rethink the roles of Siamese architectures for unsupervised representation learning. 
Code will be made available.

## 1. Introduction
Recently there has been steady progress in un-/self-supervised representation learning, with encouraging results on multiple visual tasks (e.g., [2, 17, 8, 15, 7]). 
Despite various original motivations, these methods generally involve certain forms of Siamese networks [4]. 
Siamese networks are weight-sharing neural networks applied on two or more inputs. 
They are natural tools for comparing (including but not limited to “contrasting”) entities. 
Recent methods define the inputs as two augmentations of one image, and maximize the similarity subject to different conditions.
An undesired trivial solution to Siamese networks is all outputs “collapsing” to a constant. 
There have been several general strategies for preventing Siamese networks from collapsing. 
Contrastive learning [16], e.g., instantiated in SimCLR [8], repulses different images (negative pairs) while attracting the same image’s two views (positive pairs).
The negative pairs preclude constant outputs from the solution space. 
Clustering [5] is another way of avoiding constant output, and SwAV [7] incorporates online clustering into Siamese networks. Beyond contrastive learning and clustering, BYOL [15] relies only on positive pairs but it does not collapse in case a momentum encoder is used.  
In this paper, we report that simple Siamese networks can work surprisingly well with none of the above strategies for preventing collapsing. Our model directly maximizes the similarity of one image’s two views, using neither negative pairs nor a momentum encoder. 
It works with typical batch sizes and does not rely on large-batch training. 
We illustrate this “SimSiam” method in Figure 1.  
Thanks to the conceptual simplicity, SimSiam can serve as a hub that relates several existing methods. 
In a nutshell, our method can be thought of as “BYOL without the momentum encoder”. 
Unlike BYOL but like SimCLR and SwAV, our method directly shares the weights between the two branches, so it can also be thought of as “SimCLR without negative pairs”, and “SwAV without online clustering”. 
Interestingly, SimSiam is related to each method by removing one of its core components. 
Even so, SimSiam does not cause collapsing and can perform competitively.
We empirically show that collapsing solutions do exist, but a stop-gradient operation (Figure 1) is critical to prevent such solutions. 
The importance of stop-gradient suggests that there should be a different underlying optimization problem that is being solved. 
We hypothesize that there are implicitly two sets of variables, and SimSiam behaves like alternating between optimizing each set. 
We provide proof-of-concept experiments to verify this hypothesis.

Our simple baseline suggests that the Siamese architectures can be an essential reason for the common success of the related methods. 
Siamese networks can naturally introduce inductive biases for modeling invariance, as by definition “invariance” means that two observations of the same concept should produce the same outputs. 
Analogous to convolutions [25], which is a successful inductive bias via weight-sharing for modeling translation-invariance, the weight-sharing Siamese networks can model invariance w.r.t. more complicated transformations (e.g., augmentations). 
We hope our exploration will motivate people to rethink the fundamental roles of Siamese architectures for unsupervised representation learning.

## 2. Related Work
**Siamese networks** 
Siamese networks [4] are general models for comparing entities. 
Their applications include signature [4] and face [34] verification, tracking [3], one-shot learning [23], and others. 
In conventional use cases, the inputs to Siamese networks are from different images, and the comparability is determined by supervision.
**Contrastive learning** 
The core idea of contrastive learning [16] is to attract the positive sample pairs and repulse the negative sample pairs. 
This methodology has been recently popularized for un-/self-supervised representation learning [36, 30, 20, 37, 21, 2, 35, 17, 29, 8, 9]. 
Simple and effective instantiations of contrastive learning have been developed using Siamese networks [37, 2, 17, 8, 9].
In practice, contrastive learning methods benefit from a large number of negative samples [36, 35, 17, 8]. 
These samples can be maintained in a memory bank [36]. 
In a Siamese network, MoCo [17] maintains a queue of negative samples and turns one branch into a momentum encoder to improve consistency of the queue. SimCLR [8] directly uses negative samples coexisting in the current batch, and it
requires a large batch size to work well.
**Clustering** 
Another category of methods for unsupervised representation learning are based on clustering [5, 6, 1, 7].  
They alternate between clustering the representations and learning to predict the cluster assignment. 
SwAV [7] incorporates clustering into a Siamese network, by computing the assignment from one view and predicting it from another view. 
SwAV performs online clustering under a balanced partition constraint for each batch, which is solved by the Sinkhorn-Knopp transform [10].
While clustering-based methods do not define negative exemplars, the cluster centers can play as negative protopes. 
Like contrastive learning, clustering-based methods require either a memory bank [5, 6, 1], large batches [7], or a queue [7] to provide enough samples for clustering.
**BYOL** 
BYOL [15] directly predicts the output of one view from another view. 
It is a Siamese network in which one branch is a momentum encoder.(*1) 
It is hypothesized in [15] that the momentum encoder is important for BYOL to avoid collapsing, and it reports failure results if removing the mo- mentum encoder (0.3% accuracy, Table 5 in [15]).(*2) 
Our empirical study challenges the necessity of the momentum encoder for preventing collapsing. 
We discover that the stop-gradient operation is critical. This discovery can be obscured with the usage of a momentum encoder, which is always accompanied with stop-gradient (as it is not updated by its parameters’ gradients). 
While the moving-average behavior may improve accuracy with an appropriate momentum coefficient, our experiments show that it is not directly related to preventing collapsing.

(*1) MoCo [17] and BYOL [15] do not directly share the weights between the two branches, though in theory the momentum encoder should converge to the same status as the trainable encoder. 
We view these models as Siamese networks with “indirect” weight-sharing.

(*2) n BYOL’s arXiv v3 update, it reports 66.9% accuracy with 300-epoch pre-training when removing the momentum encoder and increasing the predictor’s learning rate by 10×. 
Our work was done concurrently with this arXiv update. Our work studies this topic from different perspectives, with better results achieved.

## 3. Method
Our architecture (Figure 1) takes as input two randomly
augmented views x1 and x2 from an image x. 
The two views are processed by an encoder network f consisting of a backbone (e.g., ResNet [19]) and a projection MLP head [8]. 
The encoder f shares weights between the two views.  
A prediction MLP head [15], denoted as h, transforms the output of one view and matches it to the other view. 
Denoting the two output vectors as p1 ,h(f(x1)) and z2 ,f(x2), we minimize their negative cosine similarity:

![Figure1](images/Figure1.png)
Figure 1. **SimSiam architecture** 
Two augmented views of one image are processed by the same encoder network $f$ (a backbone plus a projection MLP). 
Then a prediction MLP h is applied on one side, and a stop-gradient operation is applied on the other side. 
The model maximizes the similarity between both sides. 
It uses neither negative pairs nor a momentum encoder.

$$
\mathcal{D}(p_1, z_2) = -\frac{p_1}{\|p_1\|_2}\cdot \frac{z_2}{\|z_2\|_2} \tag{1}
$$
where $\|\cdot\|_2$ is $\mathcal{l}_2$-norm. 
This is equivalent to the mean squared error of `2-normalized vectors [15], up to a scale of 2. 
Following [15], we define a symmetrized loss as:

$$
\mathcal{L} = \frac{1}{2}\mathcal{D}(p_1, z_2) + \frac{1}{2}\mathcal{D}(p_2, z_1) \tag{2}
$$

This is defined for each image, and the total loss is averaged over all images. 
Its minimum possible value is −1.
An important component for our method to work is a stop-gradient ($stopgrad$) operation (Figure 1). 
We implement it by modifying (1) as:

$$
\mathcal{D}(p_1, stapgrad(z_2)) \tag{3}
$$

This means that z2 is treated as a constant in this term. 
Similarly, the form in (2) is implemented as:

$$
\mathcal{L} = \frac{1}{2}\mathcal{D}(p_1, stopgrad(z_2)) + \frac{1}{2}\mathcal{D}(p_2, stopgrad(z_1)) \tag{4}
$$

Here the encoder on $x2$ receives no gradient from $z2$ in the first term, but it receives gradients from $p2$ in the second term (and vice versa for $x1$).
The pseudo-code of SimSiam is in Algorithm 1.

![Algorithm1](images/Algorithm1.png)

**Baseline settings** 
Unless specified, our explorations use the following settings for unsupervised pre-training:

- *Optimizer* 
We use SGD for pre-training. Our method does not require a large-batch optimizer such as LARS [38] (unlike [8, 15, 7]). 
We use a learning rate of lr×BatchSize/256 (linear scaling [14]), with a base $lr= 0.05$. 
The learning rate has a cosine decay schedule [27, 8]. 
The weight decay is 0.0001 and the SGD momentum is 0.9.
The batch size is 512 by default, which is friendly to typical 8-GPU implementations. 
Other batch sizes also work well (Sec. 4.3). 
We use batch normalization (BN) [22] synchronized across devices, following [8, 15, 7].

- *Projection MLP* 
The projection MLP (in $f$) has BN applied to each fully-connected (fc) layer, including its output fc. 
Its output fc has no ReLU. The hidden fc is 2048-d.
This MLP has 3 layers.

- *Prediction MLP* 
The prediction MLP ($h$) has BN applied to its hidden fc layers. 
Its output fc does not have BN (ablation in Sec. 4.4) or ReLU. This MLP has 2 layers.
The dimension of $h$’s input and output ($z$ and $p$) is $d = 2048$, and $h$’s hidden layer’s dimension is 512, making h a bottleneck structure (ablation in supplement).
We use ResNet-50 [19] as the default backbone. Other im-
plementation details are in supplement. We perform 100-
epoch pre-training in ablation experiments.

Experimental setup. We do unsupervised pre-training on the 1000-class ImageNet training set [11] without using labels. 
The quality of the pre-trained representations is evaluated by training a supervised linear classifier on frozen representations in the training set, and then testing it in the validation set, which is a common protocol. 
The implementaion details of linear classification are in supplement.

## 4. Empirical Study
In this section we empirically study the SimSiam behaviors. 
We pay special attention to what may contribute to the model’s non-collapsing solutions.

### 4.1. Stop-gradient
Figure 2 presents a comparison on “with vs. without stop-gradient”. 
The architectures and all hyper-parameters are kept unchanged, and stop-gradient is the only difference.
Figure 2 (left) shows the training loss. Without stop-gradient, the optimizer quickly finds a degenerated solution and reaches the minimum possible loss of −1. 
To show that the degeneration is caused by collapsing, we study the standard deviation (std) of the $l_2$-normalized output $z/\|z\|_2$ . 
If the outputs collapse to a constant vector, their std over all samples should be zero for each channel. 
This can be observed from the red curve in Figure 2 (middle).  
As a comparison, if the output $z$ has a zero-mean isotropic Gaussian distribution, we can show that the std of $z/\|z\|_2$ is $\frac{1}{\sqrt{d}}$ (*3).

(*3)  Here is an informal derivation: denote $z/\|z\|_2$ as $z'$, that is, $z'_i = z_i/(\sum^d_{j=1} z^2_j)^{\frac{1}{2}}$ for the $i$-th channel. 
If $z_j$ is subject to an i.i.d Gaussian distribution: $z_j \approx \mathcal{N}(0,1), \forall j$, then $z'_i \approx z_i/d^{\frac{1}{2}}$ and $std[z'_i]\approx 1/d^{\frac{1}{2}}$.

![Figure2](images/Figure2.png)
Figure 2. SimSiam with vs. without stop-gradient. Left plot: training loss. Without stop-gradient it degenerates immediately. Middle
plot: the per-channel std of the `2-normalized output, plotted as the averaged std over all channels. Right plot: validation accuracy of a
kNN classifier [36] as a monitor of progress. Table: ImageNet linear evaluation (“w/ stop-grad” is mean±std over 5 trials).

The blue curve in Figure 2 (middle) shows that with stop-gradient, the std value is near 1√d. This indicates that the outputs do not collapse, and they are scattered on the unit hypersphere.
Figure 2 (right) plots the validation accuracy of a k-nearest-neighbor (kNN) classifier [36]. 
This kNN classifier can serve as a monitor of the progress. 
With stop-gradient, the kNN monitor shows a steadily improving accuracy.
The linear evaluation result is in the table in Figure 2.  
SimSiam achieves a nontrivial accuracy of 67.7%. 
This result is reasonably stable as shown by the std of 5 trials.  
Solely removing stop-gradient, the accuracy becomes 0.1%, which is the chance-level guess in ImageNet.

**Discussion**
Our experiments show that there exist collapsing solutions. 
The collapse can be observed by the minimum possible loss and the constant outputs.(*4) 

(*4) We note that a chance-level accuracy (0.1%) is not sufficient to indicate collapsing. 
A model with a diverging loss, which is another pattern of failure, may also exhibit a chance-level accuracy.

The existence of the collapsing solutions implies that it is insufficient for
our method to prevent collapsing solely by the architecture designs (e.g., predictor, BN, `2-norm). 
In our comparison, all these architecture designs are kept unchanged, but they do not prevent collapsing if stop-gradient is removed.
The introduction of stop-gradient implies that there should be another optimization problem that is being solved underlying. 
We propose a hypothesis in Sec. 5.

### 4.2. Predictor
In Table 1 we study the predictor MLP’s effect.
The model does not work if removing $h$ (Table 1a), i.e., $h$ is the identity mapping. 
Actually, this observation can be expected if the symmetric loss (4) is used. 
Now the loss is $\frac{1}{2}\mathcal{D}(z_1,stopgrad(z_2)) + \frac{1}{2}\mathcal{D}(z_2,stopgrad(z_1))$. 
Its gradient has the same direction as the gradient of $\mathcal{D}(z1,z2)$, with the magnitude scaled by 1/2. 
In this case, using stopgradient is equivalent to removing stop-gradient and scaling the loss by 1/2. 
Collapsing is observed (Table 1a).
We note that this derivation on the gradient direction is valid only for the symmetrized loss. 
But we have observed that the asymmetric variant (3) also fails if removing $h$, while it can work if $h$ is kept (Sec. 4.6). 
These experiments suggest that h is helpful for our model.
If $h$ is fixed as random initialization, our model does not work either (Table 1b). 
However, this failure is not about collapsing. 
The training does not converge, and the loss remains high. 
The predictor $h$ should be trained to adapt to the representations.
We also find that $h$ with a constant $lr$ (without decay) can work well and produce even better results than the baseline (Table 1c). 
A possible explanation is that h should adapt to the latest representations, so it is not necessary to force it converge (by reducing lr) before the representations are sufficiently trained. 
In many variants of our model, we have observed that $h$ with a constant $lr$ provides slightly better results. 
We use this form in the following subsections.

![Table1](images/Table1.png)
Table 1. **Effect of prediction MLP** (ImageNet linear evaluation accuracy with 100-epoch pre-training). 
In all these variants, we use the same schedule for the encoder $f$ (lr with cosine decay).

### 4.3. Batch Size
Table 2 reports the results with a batch size from 64 to 4096. 
When the batch size changes, we use the same linear scaling rule (lr×BatchSize/256) [14] with base $lr = 0.05$.
We use 10 epochs of warm-up [14] for batch sizes $\ge 1024$.  
Note that we keep using the same SGD optimizer (rather than LARS [38]) for all batch sizes studied.
Our method works reasonably well over this wide range of batch sizes. 
Even a batch size of 128 or 64 performs decently, with a drop of 0.8% or 2.0% in accuracy. 
The results are similarly good when the batch size is from 256 to 2048, and the differences are at the level of random variations.
This behavior of SimSiam is noticeably different from SimCLR [8] and SwAV [7]. All three methods are Siamese networks with direct weight-sharing, but SimCLR and SwAV both require a large batch (e.g., 4096) to work well.
We also note that the standard SGD optimizer does not work well when the batch is too large (even in supervised learning [14, 38]), and our result is lower with a 4096 batch.
We expect a specialized optimizer (e.g., LARS [38]) will help in this case. 
However, our results show that a specialized optimizer is not necessary for preventing collapsing.

![Table2](images/Table2.png)
Table 2. **Effect of batch sizes** (ImageNet linear evaluation accuracy with 100-epoch pre-training).

### 4.4. Batch Normalization
Table 3 compares the configurations of BN on the MLP heads. 
In Table 3a we remove all BN layers in the MLP heads (10-epoch warmup [14] is used specifically for this entry). 
This variant does not cause collapse, although the accuracy is low (34.6%). 
The low accuracy is likely because of optimization difficulty. Adding BN to the hidden layers (Table 3b) increases accuracy to 67.4%.
Further adding BN to the output of the projection MLP (i.e., the output of $f$) boosts accuracy to 68.1% (Table 3c), which is our default configuration. 
In this entry, we also find that the learnable affine transformation (scale and offset [22]) in f’s output BN is not necessary, and disabling it leads to a comparable accuracy of 68.2%.
Adding BN to the output of the prediction MLP $h$ does not work well (Table 3d). 
We find that this is not about collapsing. 
The training is unstable and the loss oscillates.
In summary, we observe that BN is helpful for optimization when used appropriately, which is similar to BN’s behavior in other supervised learning scenarios. 
But we have seen no evidence that BN helps to prevent collapsing: actually, the comparison in Sec. 4.1 (Figure 2) has exactly the same BN configuration for both entries, but the model collapses if stop-gradient is not used.

![Table3](images/Table3.png)
Table 3. **Effect of batch normalization on MLP heads** (ImageNet linear evaluation accuracy with 100-epoch pre-training).

### 4.5. Similarity Function
Besides the cosine similarity function (1), our method also works with cross-entropy similarity. 
$We modify $\mathcal{D} as:
$\mathcal{D}(p1,z2) = −softmax(z2)\cdot log softmax(p1)$. 
Here the softmax function is along the channel dimension. 
The output of softmax can be thought of as the probabilities of belonging to each of d pseudo-categories.
We simply replace the cosine similarity with the cross-entropy similarity, and symmetrize it using (4). 
All hyper-parameters and architectures are unchanged, though they
may be suboptimal for this variant. Here is the comparison:
![t4_5_(1)](images/t4_5_(1).png)
The cross-entropy variant can converge to a reasonable result without collapsing. 
This suggests that the collapsing prevention behavior is not just about the cosine similarity.  
This variant helps to set up a connection to SwAV [7], which we discuss in Sec. 6.2.

### 4.6. Symmetrization
Thus far our experiments have been based on the symmetrized loss (4). 
We observe that SimSiam’s behavior of preventing collapsing does not depend on symmetrization.
We compare with the asymmetric variant (3) as follows:
![t4_6_(1)](images/t4_6_(1).png)
The asymmetric variant achieves reasonable results. 
Symmetrization is helpful for boosting accuracy, but it is not related to collapse prevention. 
Symmetrization makes one more prediction for each image, and we may roughly compensate for this by sampling two pairs for each image in the asymmetric version ("2×"). 
It makes the gap smaller.

### 4.7. Summary
We have empirically shown that in a variety of settings, SimSiam can produce meaningful results without collapsing. 
The optimizer (batch size), batch normalization, similarity function, and symmetrization may affect accuracy, but we have seen no evidence that they are related to collapse prevention. 
It is mainly the stop-gradient operation that plays an essential role.

## 5. Hypothesis
We discuss a hypothesis on what is implicitly optimized by SimSiam, with proof-of-concept experiments provided.
### 5.1. Formulation
Our hypothesis is that SimSiam is an implementation of an Expectation-Maximization (EM) like algorithm. 
It implicitly involves two sets of variables, and solves two underlying sub-problems. 
The presence of stop-gradient is the consequence of introducing the extra set of variables.
We consider a loss function of the following form:

$$
\mathcal{L}(\theta, \eta) = E_{x, \tau}\left[ \|\mathcal{F}_\theta(\Tau(x)) - \eth_x\|^2_2\right] \tau{5}
$$

$F$ is a network parameterized by $\theta$. 
$\Tau$ is the augmentation.
$x$ is an image. 
The expectation $E[\cdot]$ is over the distribution
of images and augmentations. 
For the ease of analysis, here we use the mean squared error ‖·‖22, which is equivalent to the cosine similarity if the vectors are $l^2$-normalized. 
We do not consider the predictor yet and will discuss it later.
In (5), we have introduced another set of variables which we denote as $\eta$. 
The size of $\eta$ is proportional to the number of images. 
Intuitively, $\eta_x$ is the representation of the image $x$, and the subscript $x$ means using the image index to access a sub-vector of $\eta$. 
$\eta$ is not necessarily the output of a network; it is the argument of an optimization problem.
With this formulation, we consider solving:

$$
\min_{\theta, \eta}\mathcal{L}(\theta, \eta) \tag{6}
$$

Here the problem is w.r.t. both $\theta$ and $\eta$. 
This formulation is analogous to k-means clustering [28]. 
The variable $\theta$ is analogous to the clustering centers: it is the learnable parameters of an encoder. 
The variable $\eta_x$ is analogous to the assignment vector of the sample $x$ (a one-hot vector in k-means): it is the representation of $x$.
Also analogous to k-means, the problem in (6) can be solved by an alternating algorithm, fixing one set of variables and solving for the other set. 
Formally, we can alternate between solving these two subproblems:

$$
\begin{align}
\theta^t &\leftarrow& arg \min_\theta \mathcal{L}(\theta, \eta^{t-1}) \tag{7}\\
\eta^t &\leftarrow& arg \min_\eta \mathcal{L}(\theta, \eta) \tag{8}
\end{align}
$$
Here t is the index of alternation and “←” means assigning.