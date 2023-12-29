# DiffBIR: Towards Blind Image Restoration with Generative Diffusion Prior

论文地址：[arxiv.org/pdf/2308.15070.pdf](https://arxiv.org/pdf/2308.15070.pdf)

项目地址：[GitHub - XPixelGroup/DiffBIR: Official codes of DiffBIR: Towards Blind Image Restoration with Generative Diffusion Prior](https://github.com/XPixelGroup/DiffBIR)

## 论文解决的问题

利用文生图的扩散先验解决盲图像恢复问题（超分和人脸重建）

## 背景

经典的图像恢复（去噪、超分、去模糊）都是在一定假设条件下进行的，退化的过程通常很单一并且是已知的（如高斯噪声和双三次下采样），而真实任务中模糊核未知的超分任务叫做盲超分。这些方法的泛化能力并不好，因此需要用blind image restoration（BIR）来实现对真实图片的重建。

x和y表示HR和LR图片，k表示模糊核，$\otimes$表示卷积操作，$$\downarrow_k$$表示下采样，n表示附加噪声，退化模型为：
$$
y=(x\otimes k)\downarrow_s+n
$$
一般的退化分为三种**blur、resize、noise**。

**blur：**isotropic Gaussian kernel（各向同性高斯核） 和 anisotropic Gaussian kernel（各向异性高斯核）。

**resize：**区域调整大小、双线性插值和双三次插值。

**noise：**加性高斯噪声、泊松噪声、JPEG压缩噪声。

high-order degradation高阶退化：使用经典退化模型blur-resize-noise重复n次

<u>*盲超分退化模型参考资料：[底层任务超详细解读 (十)：为图像盲超分学习通用的退化模型 - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/482261337)*</u>

目前BIR主要分为三类： blind image super-resolution (BSR)，zero-shot image restoration (ZIR)，blind face restoration (BFR)。

BSR：BSRGAN和Real-ESRGAN，他们可以移除大部分退化，但无法生成真实的细节。此外超分设置在4倍或8倍，这对BIR问题来说是不完整的。（无法生成现实世界的细节）

ZIR：新产生的方向DDRM、DDNM、GDP，将扩散模型作为附加先验，因此比基于GAN的网络的生成能力更好。在退化假设恰当时，他们在经典任务中表现得很好。但ZIR与BIR的问题设定不同，ZIR只能解决退化明确定义的图片重建问题，而不能推广到未知的退化定义上去，也就是说，ZIR只能在一般的图片上进行真实的重建，而不能在一般退化的图片上进行。（无法适用于一般退化）

BFR：CodeFormer和VQFR，与BSR有类似的pipeline，但在退化模型和生成网络不同。BFR是BIR的一个子域，BFR通常设定一个固定尺寸的输入和和一个有限的图像空间，因此不能应用于一般的图像。（无法使用于一般图像）

![image-20231214184732984](C:\Users\86187\AppData\Roaming\Typora\typora-user-images\image-20231214184732984.png)

上图为BSR和本文的DiffBIR的效果对比。对于复杂的纹理，BSR效果并不好（row1）。对于语义的有关细节，BSR会过渡平滑。对于过小的纹理，BSR会进行擦除，而DiffBIR会进行增强。

## 所做工作

这项工作提出了DiffBIR来解决上述的问题，DiffBIR 将上述的优点整合为一个统一的框架：

1. 采用一个扩展的退化模型来生成各种退化。
2. 用一个训练好的SD作为先验来提高生成能力。
3. 引入一个两阶段的pipeline来保证真实性和保真度。

总的来说就是一个两阶段的pipeline，第一阶段是用来处理退化的回复模型，第二阶段是来生成细节的生成模型。

![image-20231215125805808](C:\Users\86187\AppData\Roaming\Typora\typora-user-images\image-20231215125805808.png)

### 1. Pretraining for Degradation Removal（stage 1）

为了模拟更全面的退化，作者采用了同时考虑退化多样性和高阶退化的综合模型。

首先输入高质量图片经过两阶退化得到低质量图片。然后通过restoration model（SwinIR）得到$I_{reg}$，其中HQ Image Reconstruction采取最近邻插值（3次），每一次插值采用一个卷积层和一个Leaky ReLU激活层。通过L2损失来优化恢复模块的参数。

#### a. nearest interpolation（补充知识）

最近邻法插值：插值非整数处的像素时，此时小数位 ≤ 0.5，舍去小数位，若 ＞ 0.5，小数位进位。

#### b. SwinIR（补充知识）

<img src="C:\Users\86187\AppData\Roaming\Typora\typora-user-images\image-20231215190145436.png" alt="image-20231215190145436" style="zoom:67%;" />

先通过卷积操作提取浅层特征，然后通过多个RSTB提取深层特征，将提取的深层特征通过卷积映射回原始图像空间。然后通过上采样和卷积的操作对图像进行重建。

通常重建的过程：

1. 经典超分 (卷积 + pixelshuffle 上采样 + 卷积)。
2. 轻量超分 (卷积 + pixelshuffle 上采样)。
3. 真实图像超分 (卷积 + 卷积插值上采样 + 卷积插值上采样 + 卷积)。
4. 去噪和 JPEG 压缩去伪影 （卷积 + 引入残差）。

### 2. Leverage Generative Prior for Image Reconstruction（stage 2）

虽然经过stage 1的图像可以去除大部分的退化，但通常获得的图像会过于平滑，确实很多细节，离高质量图片的分布依旧很远，所以采用stage 2进行图像细节的生成。

经过预训练的encoder后，得到$I_{reg}$在应空间的表示，再与每个阶段的随机噪声进行concat操作，由于经过concat操作后通道数会增加，所以每一个平行模块的第一个卷积层的权重设为0，其余所有的参数都用预训练的。

在平行模块得到的结果加到denoiser的decoder上之前，经过一个$1\times1$的卷积。

在微调的过程中，对这些并行模块和$1\times1$的卷积同时进行优化，其中文本提示设为空。

相较于controlNet，本文提出的方法（LAControlNet）对图片的重构更有效。controlNet是从头开始训练网络的附加条件，而LAControlNet则直接利用训练好的encoder将条件映射到与图片相同的隐空间。

### 3. Latent Image Guidance for Fidelity-Realness Trade-off

传统的classifier guidance是针对图像进行引导的，本文中所有操作都是在隐空间完成的，所以对算法做了调整，使其在隐空间进行引导。作者通过将$\varepsilon(I_{reg})$作为guidance

上述guidance强制空间对齐和颜色一致性，相当于在x和$I_{reg}$的隐空间上计算L2距离（x为估计生成的图像，$I_{reg}$为原始图像-参考图像）。
$$
\mathcal D_{latent}(x,I_{reg})=\mathcal L(\tilde z_0,\varepsilon(I_{reg}))=\sum_j\frac1 {C_jH_jW_j}||\tilde z_0-\varepsilon(I_{reg})||^2_2
$$
![image-20231217175446508](C:\Users\86187\AppData\Roaming\Typora\typora-user-images\image-20231217175446508.png)

#### classifier guidance（补充知识）

DDPM的优化目标（预测每一步所添加的噪声）实际上可以理解为学习一个当前输入对目标数据分布的最优梯度方向。这实际上非常符合直觉：即我们对输入所添加的噪声实际上使得输入远离了其原本的数据分布，而学习一个数据空间上最优的梯度方向实际上便等同于往去除噪声的方向去行走。

基础的扩散模型优化的是常规的无条件生成的梯度，而加上分类器就相当于额外添加一个基于噪声输入的分类器，常规的无条件生成的梯度$\nabla\log p(x_t)$，添加分类器的生成的梯度$\nabla \log p(x_t|y)$。

<img src="C:\Users\86187\AppData\Roaming\Typora\typora-user-images\image-20231219145707508.png" alt="image-20231219145707508" style="zoom:67%;" />

通常情况下，作为guidance的图像是目标图片，通过修改采样时的均值来指导图像按类别生成，计算$\log p_\phi(y|x_t)$的梯度，来使生成的x沿着梯度的方向进行。伪代码如下：

![image-20231218194312655](C:\Users\86187\AppData\Roaming\Typora\typora-user-images\image-20231218194312655.png)

## 效果

### BSR的评价

Real47：作者从网络收集的47张图片包括各种场景情况。

![image-20231217161534078](C:\Users\86187\AppData\Roaming\Typora\typora-user-images\image-20231217161534078.png)

![image-20231217161620929](C:\Users\86187\AppData\Roaming\Typora\typora-user-images\image-20231217161620929.png)

主观评价：

<img src="C:\Users\86187\AppData\Roaming\Typora\typora-user-images\image-20231217162122107.png" alt="image-20231217162122107" style="zoom:67%;" />

### BFR的评价

![image-20231217162554566](C:\Users\86187\AppData\Roaming\Typora\typora-user-images\image-20231217162554566.png)

![image-20231217162506779](C:\Users\86187\AppData\Roaming\Typora\typora-user-images\image-20231217162506779.png)

![image-20231217162528431](C:\Users\86187\AppData\Roaming\Typora\typora-user-images\image-20231217162528431.png)

### 消融实验

![image-20231217172721580](C:\Users\86187\AppData\Roaming\Typora\typora-user-images\image-20231217172721580.png)

#### Restoration Module（是否存在该模块）

根据a中row1，RM模块错误地将退化当成是语义信息，这表明RM模块有助于保真。

根据a中row2，没有RM模块，只对SD进行微调会出现噪点和伪影，这表明RM在降噪时的必要性。

![image-20231217173448484](C:\Users\86187\AppData\Roaming\Typora\typora-user-images\image-20231217173448484.png)

#### Finetuning Stable Diffusion（是否微调）

根据b中row1，不经过微调的SD生成的鸟少了一条腿，这表明直接使用图像空间中的guidance无法有效推广到隐空间，因此需要微调。

#### LAControlNet（ControlNet可否替代）

根据c，可知ControlNet生成的颜色有偏移，因为训练过程中没有对颜色进行明确的正则化。

#### Controllable Module（scale的影响）

![image-20231217181442923](C:\Users\86187\AppData\Roaming\Typora\typora-user-images\image-20231217181442923.png)

## 问题

1. 文本驱动的图像恢复待探索。
2. 本文恢复一个低质量图片需要采样50次，因此相较于其他的图像恢复模型，需要更多计算成本和推理时间。
