# Unsupervised convolutional neural networks for motion estimation
[paper](https://arxiv.org/abs/1601.06087)

## ABSTRACT
従来の動き推定法は、一対の画像間の動き場$F$を、設計済みのコスト関数を最小化するものとして推定する。
本論文では、直接的な方法を提案する、またテスト時に一対の画像を入力として与えると、その出力層で密なmotion field場$F$を生成する畳み込みニューラルネットワーク(CNN)を学習する。
古典的な教師あり学習を可能にするような、motionのground truth を有する大規模なデータセットがない場合、我々は教師なし学習でネットワークを学習することを提案する。
学習中に最適化される提案コスト関数は、古典的なoptical flow制約に基づいている。
後者は運動場に対して微分可能であるため、誤差をネットワーク前方の層にbackpropagationすることができる。
我々の手法は、合成画像と実画像の両方でテストされ、state-of-the-artな手法と同様の性能を示した。

検索語— Motion Estimation, Convolutional Neural Network, Unsupervised Training

## 1. INTRODUCTION
Motion fields, that is fields that describe how pixels move from a reference to a target frame, are rich source of information for the analysis of image sequences and beneficial for several applications such as video coding [1, 2], medical image processing [3], segmentation [4] and human action recognition [5, 6]. 
Traditionally, motion fields are estimated using the variational model proposed by Horn and Schunck [7] and its variants such as [8, 9]. 
Very recently, inspired by the great success of Deep Neural Networks in several Computer Vision problems [10], a CNN has been proposed Fischer et al. in [11] for motion estimation. 
The method showed performance that was close to the state-of-the-art in a number of synthetically generated image sequences.
A major problem with the method proposed in [11] is that the proposed CNN needed to be trained in a supervised manner, that is, it required for training synthetic image sequences where ground truth motion fields were available. 
Furthermore, in order to generalize well to an unseen dataset, it needed fine tuning, also requiring ground truth data on samples from that dataset. 
Ground truth motion estimation are not easily available though. 
For this reason, the method proposed in [11] was applied only on synthetic image sequences.  
In this paper, we propose training a CNN for motion estimation in an unsupervised manner. 
We do so by designing a cost function that is differentiable with respect to the unknown motion field and, therefore, allows the backpropagation of the error and the end to end training of the CNN. 
The cost function builds on the widely used optical flow constraint - our major difference to Horn-Schunk based methods is that the cost function is used only during training and without regularization. 
Once trained, given a pair of frames as input the CNN gives at its output layer an estimate of the motion field.
In order to deal with motions large in magnitude, we embed the proposed network in a classical iterative scheme, in which at the end of each iteration the reference image is warped towards the target image and in a classical coarse-to-fine multi-scale framework. 
We train our CNN using randomly chosen pairs of consecutive frames from UCF101 dataset and test it on both the UCF101 where it performs similarly to the state- of-the-art methods and on the synthetic MPI-Sintel dataset where it outperforms them.
The remainder of the paper is organized as follows. In Section 2 we describe our method. In Section 3 we present our experimental results. Finally, in Section 4 we give some conclusion.