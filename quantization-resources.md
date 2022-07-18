
<p align="center">
  <a href="https://discord.gg/RbeQMu886J">Join the community</a> •
  <a href="#contribute">Contribute to the library</a>
</p>

<img height="25" width="100%" src="https://user-images.githubusercontent.com/83510798/171454644-d4b980bc-15ab-4a31-847c-75c36c5bd96b.png">


# Resources

Awesome resource collection on pruning techniques.

- Literature reviews and papers
- Courses, webinars, and blogs
- Open-source libraries

And check [overview page on quantization](https://github.com/emilecourthoud/learning-AI-optimization/blob/main/quantization-resources) f an overview of what quantization, as well as a mapping of quantization techniques, 
And check out [this page](https://github.com/emilecourthoud/learning-AI-optimization/blob/main/quantization-overview) for an overview of quantization process and a concept map of quantization techniques.


IMAGE

## Literature reviews and papers
Legenda: 
- ✏️  More than 20 citations (20+, 50+, 100+)
- ⭐ With GitHub code and more than 50 stars (100+, 300+, 1k+, 3k+)
Sorting: alphabetic order
<br>

### List of literature reviews

2017-2022
- ...
- ...

### List of papers
2022
- FQ-ViT: Post-Training Quantization for Fully Quantized Vision Transformer [[IJCAI](https://arxiv.org/abs/2111.13824)][[github](https://github.com/megvii-research/FQ-ViT)]

2021
- MQBench: Towards Reproducible and Deployable Model Quantization Benchmark [[paper](https://arxiv.org/abs/2111.03759)][[github](http://mqbench.tech/)]

2020
- APQ: Joint Search for Network Architecture, Pruning and Quantization Policy [[✏️CVPR](https://openaccess.thecvf.com/content_CVPR_2020/html/Wang_APQ_Joint_Search_for_Network_Architecture_Pruning_and_Quantization_Policy_CVPR_2020_paper.html)][⭐[github](https://github.com/mit-han-lab/apq)]
- Compression of Deep Learning Models for Text: A Survey [[✏️paper](https://arxiv.org/pdf/2008.05221.pdf)]
- Forward and Backward Information Retention for Accurate Binary Neural Networks [[✏️CVPR](https://openaccess.thecvf.com/content_CVPR_2020/html/Qin_Forward_and_Backward_Information_Retention_for_Accurate_Binary_Neural_Networks_CVPR_2020_paper.html)][⭐[github](https://github.com/htqin/IR-Net)]
- GhostNet: More Features from Cheap Operations [[✏️✏️CVPR](https://openaccess.thecvf.com/content_CVPR_2020/html/Han_GhostNet_More_Features_From_Cheap_Operations_CVPR_2020_paper.html)][⭐⭐[github](https://github.com/huawei-noah/Efficient-AI-Backbones)]
- Learned Step Size Quantization [[✏️ICLR](https://arxiv.org/abs/1902.08153)]
- MeliusNet: Can Binary Neural Networks Achieve MobileNet-level Accuracy? [[✏️paper](https://arxiv.org/abs/2001.05936)][⭐[github](https://github.com/hpi-xnor/BMXNet-v2)]
- Mixed Precision DNNs: All you need is a good parametrization [[✏️ICLR](https://arxiv.org/abs/1905.11452)][⭐[github](https://github.com/sony/ai-research-code)]
- Q-BERT: Hessian Based Ultra Low Precision Quantization of BERT [[✏️AAAI](https://ojs.aaai.org/index.php/AAAI/article/view/6409)]
- ReActNet: Towards Precise Binary Neural Network with Generalized Activation Functions [[✏️ECCV](https://link.springer.com/chapter/10.1007/978-3-030-58568-6_9)]
- Soft Weight-Sharing for Neural Network Compression [[✏️ICLR](https://arxiv.org/abs/1702.04008)]

<details> <summary> Other papers </summary>
- PROFIT: A Novel Training Method for sub-4-bit MobileNet Models [[ECCV](https://link.springer.com/chapter/10.1007/978-3-030-58539-6_26)]
- Linear Symmetric Quantization of Neural Networks for Low-precision Integer Hardware [[ICML](https://openreview.net/pdf?id=H1lBj2VFPS)]
- AutoQ: Automated Kernel-Wise Neural Network Quantization [[paper](https://arxiv.org/abs/1902.05690)]
- Gradient ℓ1 Regularization for Quantization Robustness [[ICLR](https://arxiv.org/abs/2002.07520)]
- BinaryDuo: Reducing Gradient Mismatch in Binary Activation Network by Coupling Binary Activations [[paper](https://arxiv.org/abs/2002.06517)]
- Rotation Consistent Margin Loss for Efficient Low-Bit Face Recognition [[CVPR](https://openaccess.thecvf.com/content_CVPR_2020/html/Wu_Rotation_Consistent_Margin_Loss_for_Efficient_Low-Bit_Face_Recognition_CVPR_2020_paper.html)]
- BiDet: An Efficient Binarized Object Detector [[CVPR](https://openaccess.thecvf.com/content_CVPR_2020/html/Wang_BiDet_An_Efficient_Binarized_Object_Detector_CVPR_2020_paper.html)][⭐[github](https://github.com/ZiweiWangTHU/BiDet)]
- Differentiable Joint Pruning and Quantization for Hardware Efficiency [[ECCV](https://link.springer.com/chapter/10.1007/978-3-030-58526-6_16)]
- Riptide: Fast End-to-End Binarized Neural Networks [[MLSys](https://proceedings.mlsys.org/paper/2020/hash/2a79ea27c279e471f4d180b08d62b00a-Abstract.html)][⭐[github](https://github.com/jwfromm/Riptide)]
- Balanced Binary Neural Networks with Gated Residual [[IEEE](https://ieeexplore.ieee.org/abstract/document/9054599)]
</details>
<br>





2019
- Additive Powers-of-Two Quantization: A Non-uniform Discretization for Neural Networks [[✏️paper](https://arxiv.org/abs/1909.13144)][⭐[github](https://github.com/yhhhli/APoT_Quantization)]
- Data-Free Quantization through Weight Equalization and Bias Correction [[✏️ICCV](https://openaccess.thecvf.com/content_ICCV_2019/html/Nagel_Data-Free_Quantization_Through_Weight_Equalization_and_Bias_Correction_ICCV_2019_paper.html)]
- Defensive Quantization: When Efficiency Meets Robustness [[✏️paper](https://arxiv.org/abs/1904.08444)]
- Differentiable Soft Quantization: Bridging Full-Precision and Low-Bit Neural Networks [[✏️ICCV](https://openaccess.thecvf.com/content_ICCV_2019/html/Gong_Differentiable_Soft_Quantization_Bridging_Full-Precision_and_Low-Bit_Neural_Networks_ICCV_2019_paper.html)]
- Eyeriss v2: A Flexible Accelerator for Emerging Deep Neural Networks on Mobile Devices [[✏️✏️IEEE](https://ieeexplore.ieee.org/abstract/document/8686088)]
- HAQ: Hardware-Aware Automated Quantization With Mixed Precision [[✏️✏️CVPR](https://openaccess.thecvf.com/content_CVPR_2019/html/Wang_HAQ_Hardware-Aware_Automated_Quantization_With_Mixed_Precision_CVPR_2019_paper.html)]
- Improving Neural Network Quantization without Retraining using Outlier Channel Splitting [[✏️PMLR](http://proceedings.mlr.press/v97/zhao19c.html)]
- Learning to Quantize Deep Networks by Optimizing Quantization Intervals With Task Loss [[✏️CVPR](https://openaccess.thecvf.com/content_CVPR_2019/html/Jung_Learning_to_Quantize_Deep_Networks_by_Optimizing_Quantization_Intervals_With_CVPR_2019_paper.html)]
- Qsparse-local-SGD: Distributed SGD with Quantization, Sparsification, and Local Computations [[✏️NeurIPS](https://proceedings.neurips.cc/paper/2019/hash/d202ed5bcfa858c15a9f383c3e386ab2-Abstract.html)]
- Quantization Networks [[✏️CVPR](https://openaccess.thecvf.com/content_CVPR_2019/html/Yang_Quantization_Networks_CVPR_2019_paper.html)][⭐[github](https://github.com/aliyun/alibabacloud-quantization-networks)]

<details> <summary> Other papers </summary>
- Binary Ensemble Neural Network: More Bits per Network or More Networks per Bit? [[✏️CVPR](https://openaccess.thecvf.com/content_CVPR_2019/html/Zhu_Binary_Ensemble_Neural_Network_More_Bits_per_Network_or_More_CVPR_2019_paper.html)]
- Fully Quantized Network for Object Detection [[✏️CVPR](https://openaccess.thecvf.com/content_CVPR_2019/html/Li_Fully_Quantized_Network_for_Object_Detection_CVPR_2019_paper.html)]
- Regularizing Activation Distribution for Training Binarized Deep Networks [[✏️CVPR](https://openaccess.thecvf.com/content_CVPR_2019/html/Ding_Regularizing_Activation_Distribution_for_Training_Binarized_Deep_Networks_CVPR_2019_paper.html)]
- Latent Weights Do Not Exist: Rethinking Binarized Neural Network Optimization [[✏️NeurIPS](https://proceedings.neurips.cc/paper/2019/hash/9ca8c9b0996bbf05ae7753d34667a6fd-Abstract.html)]
- Learning Channel-Wise Interactions for Binary Convolutional Neural Networks [[✏️CVPR](https://openaccess.thecvf.com/content_CVPR_2019/html/Wang_Learning_Channel-Wise_Interactions_for_Binary_Convolutional_Neural_Networks_CVPR_2019_paper.html)]
- An Empirical study of Binary Neural Networks' Optimisation [[✏️ICLR](https://openreview.net/forum?id=rJfUCoR5KX)][[github](https://github.com/mil-ad/studying-binary-neural-networks)]
- Same, Same But Different: Recovering Neural Network Quantization Error Through Weight Factorization [[✏️PMLR](https://proceedings.mlr.press/v97/meller19a.html)]
- Simultaneously Optimizing Weight and Quantizer of Ternary Neural Network Using Truncated Gaussian Approximation [[✏️CVPR](https://openaccess.thecvf.com/content_CVPR_2019/html/He_Simultaneously_Optimizing_Weight_and_Quantizer_of_Ternary_Neural_Network_Using_CVPR_2019_paper.html)]
- Circulant Binary Convolutional Networks: Enhancing the Performance of 1-Bit DCNNs With Circulant Back Propagation [[✏️CVPR](https://openaccess.thecvf.com/content_CVPR_2019/html/Liu_Circulant_Binary_Convolutional_Networks_Enhancing_the_Performance_of_1-Bit_DCNNs_CVPR_2019_paper.html)]
- ARM: Augment-REINFORCE-Merge Gradient for Stochastic Binary Networks [[✏️ICLR](https://arxiv.org/abs/1807.11143)]
- SeerNet: Predicting Convolutional Neural Network Feature-Map Sparsity Through Low-Bit Quantization [[CVPR](https://openaccess.thecvf.com/content_CVPR_2019/html/Cao_SeerNet_Predicting_Convolutional_Neural_Network_Feature-Map_Sparsity_Through_Low-Bit_Quantization_CVPR_2019_paper.html)]
- Per-Tensor Fixed-Point Quantization of the Back-Propagation Algorithm [[ICLR](https://arxiv.org/abs/1812.11732)]
- ECC: Platform-Independent Energy-Constrained Deep Neural Network Compression via a Bilinear Regression Model [[CVPR](https://openaccess.thecvf.com/content_CVPR_2019/html/Yang_ECC_Platform-Independent_Energy-Constrained_Deep_Neural_Network_Compression_via_a_Bilinear_CVPR_2019_paper.html)]
- Fully Quantized Transformer for Machine Translation [[paper](V)]
- Energy-Constrained Compression for Deep Neural Networks via Weighted Sparse Projection and Layer Input Masking [[paper](https://arxiv.org/abs/1806.04321)]
- MetaQuant: Learning to Quantize by Learning to Penetrate Non-differentiable Quantization [[NeurIPS](https://proceedings.neurips.cc/paper/2019/hash/f8e59f4b2fe7c5705bf878bbd494ccdf-Abstract.html)][[github](https://github.com/csyhhu/MetaQuant)]
- Integer Networks for Data Compression with Latent-Variable Models [[ICLR](https://openreview.net/forum?id=S1zz2i0cY7)][⭐[github](https://github.com/1adrianb/binary-human-pose-estimation)]
- A Main/Subsidiary Network Framework for Simplifying Binary Neural Networks [[CVPR](https://openaccess.thecvf.com/content_CVPR_2019/html/Xu_A_MainSubsidiary_Network_Framework_for_Simplifying_Binary_Neural_Networks_CVPR_2019_paper.html)]
- Self-Binarizing Networks [[paper](https://arxiv.org/abs/1902.00730)]
- Proximal Mean-Field for Neural Network Quantization [[ICCV](https://openaccess.thecvf.com/content_ICCV_2019/html/Ajanthan_Proximal_Mean-Field_for_Neural_Network_Quantization_ICCV_2019_paper.html)]
- Double Viterbi: Weight Encoding for High Compression Ratio and Fast On-Chip Reconstruction for Deep Neural Network [[ICLR](https://openreview.net/forum?id=HkfYOoCcYX)]![image](https://user-images.githubusercontent.com/83510798/179468440-9633acb4-5fd6-493e-bb1a-0e3b8e2af038.png)
</details>
<br>

2018
- Bi-Real Net: Enhancing the Performance of 1-bit CNNs With Improved Representational Capability and Advanced Training Algorithm [[✏️ECCV](https://openaccess.thecvf.com/content_ECCV_2018/html/zechun_liu_Bi-Real_Net_Enhancing_ECCV_2018_paper.html)][⭐[github](https://github.com/liuzechun/Bi-Real-net)]
- DoReFa-Net: Training Low Bitwidth Convolutional Neural Networks with Low Bitwidth Gradients [[✏️✏️paper](https://arxiv.org/abs/1606.06160)]
- FP-BNN: Binarized neural network on FPGA [[✏️paper](https://www.sciencedirect.com/science/article/abs/pii/S0925231217315655)]
- LQ-Nets: Learned Quantization for Highly Accurate and Compact Deep Neural Networks [[✏️ECCV](https://openaccess.thecvf.com/content_ECCV_2018/html/Dongqing_Zhang_Optimized_Quantization_for_ECCV_2018_paper.html)]
- Model compression via distillation and quantization [[✏️ICLR](https://arxiv.org/abs/1802.05668)][⭐⭐[github](https://github.com/antspy/quantized_distillation)]
- PACT: Parameterized Clipping Activation for Quantized Neural Networks [[✏️✏️paper](https://arxiv.org/abs/1805.06085)]
- Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference [[✏️✏️CVPR](https://openaccess.thecvf.com/content_cvpr_2018/html/Jacob_Quantization_and_Training_CVPR_2018_paper.html)]
- Quantizing deep convolutional networks for efficient inference: A whitepaper [[✏️✏️paper](https://arxiv.org/abs/1806.08342)]
- Scalable Methods for 8-bit Training of Neural Networks [[✏️NeurIPS](https://proceedings.neurips.cc/paper/2018/hash/e82c4b19b8151ddc25d4d93baf7b908f-Abstract.html)]
- SIGNSGD: compressed optimisation for non-convex problems [[✏️✏️PMLR](https://proceedings.mlr.press/v80/bernstein18a.html)]

<details> <summary> Other papers </summary>
- FINN-R: An End-to-End Deep-Learning Framework for Fast Exploration of Quantized Neural Networks [[✏️paper](https://dl.acm.org/doi/abs/10.1145/3242897)]
- CLIP-Q: Deep Network Compression Learning by In-Parallel Pruning-Quantization [[✏️CVPR](Q_Deep_Network_CVPR_2018_paper)]
- Relaxed Quantization for Discretized Neural Networks [[✏️paper](https://arxiv.org/abs/1810.01875)]
- Loss-aware Weight Quantization of Deep Networks [[✏️paper](https://arxiv.org/abs/1802.08635)]
- Learning Discrete Weights Using the Local Reparameterization Trick [[✏️ICLR](https://arxiv.org/abs/1710.07739)]
- Explicit Loss-Error-Aware Quantization for Low-Bit Deep Neural Networks [[✏️CVPR](https://openaccess.thecvf.com/content_cvpr_2018/html/Zhou_Explicit_Loss-Error-Aware_Quantization_CVPR_2018_paper.html)]
- ProxQuant: Quantized Neural Networks via Proximal Operators [[✏️paper](https://arxiv.org/abs/1810.00861)]
- BinaryRelax: A Relaxation Approach For Training Deep Neural Networks With Quantized Weights [[✏️SIAM](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=BinaryRelax%3A+A+Relaxation+Approach+For+Training+Deep+Neural+Networks+With+Quantized+Weights&btnG=)]
- HitNet: Hybrid Ternary Recurrent Neural Network [[NeurIPS](https://proceedings.neurips.cc/paper/2018/hash/82cec96096d4281b7c95cd7e74623496-Abstract.html)]
- Adaptive Quantization of Neural Networks [[ICLR](https://openreview.net/forum?id=SyOK1Sg0W)]
- Combinatorial Attacks on Binarized Neural Networks [[paper](https://arxiv.org/abs/1810.03538)]
- Heterogeneous Bitwidth Binarization in Convolutional Neural Networks [[NeurIPS](https://proceedings.neurips.cc/paper/2018/hash/1b36ea1c9b7a1c3ad668b8bb5df7963f-Abstract.html)]
- LSQ++: Lower running time and higher recall in multi-codebook quantization [[ECCV](https://openaccess.thecvf.com/content_ECCV_2018/html/Julieta_Martinez_LSQ_lower_runtime_ECCV_2018_paper.html)]
- Efficient end-to-end learning for quantizable representations [[PMLR](https://proceedings.mlr.press/v80/jeong18a.html)]
</details>
  

2017
- Apprentice: Using Knowledge Distillation Techniques To Improve Low-Precision Network Accuracy [[✏️paper](https://arxiv.org/abs/1711.05852)]
- Binarized Convolutional Landmark Localizers for Human Pose Estimation and Face Alignment with Limited Resources [[✏️ICCV](https://openaccess.thecvf.com/content_iccv_2017/html/Bulat_Binarized_Convolutional_Landmark_ICCV_2017_paper.html)]
- Deep Learning with Low Precision by Half-wave Gaussian Quantization [[✏️CVPR](https://openaccess.thecvf.com/content_cvpr_2017/html/Cai_Deep_Learning_With_CVPR_2017_paper.html)]
- Extremely Low Bit Neural Network: Squeeze the Last Bit Out with ADMM [[✏️paper](https://arxiv.org/abs/1707.09870)]
- FINN: A Framework for Fast, Scalable Binarized Neural Network Inference [[✏️✏️paper](https://dl.acm.org/doi/abs/10.1145/3020078.3021744)]
- Incremental Network Quantization: Towards Lossless CNNs with Low-Precision Weights [[✏️✏️ICLR](https://arxiv.org/abs/1702.03044)][⭐[github](https://github.com/AojunZhou/Incremental-Network-Quantization)]
- Local Binary Convolutional Neural Networks [[✏️CVPR](https://openaccess.thecvf.com/content_cvpr_2017/html/Juefei-Xu_Local_Binary_Convolutional_CVPR_2017_paper.html)][[github](https://github.com/juefeix/lbcnn.torch)]
- Network Sketching: Exploiting Binary Structure in Deep CNNs [[✏️CVPR](https://openaccess.thecvf.com/content_cvpr_2017/html/Guo_Network_Sketching_Exploiting_CVPR_2017_paper.html)]
- QSGD: Communication-Efficient SGD via Gradient Quantization and Encoding [[✏️✏️NIPS](QSGD: Communication-efficient SGD via gradient quantization and encoding)]
- Quantized Neural Networks Quantized Neural Networks: Training Neural Networks with Low Precision Weights and Activations [[✏️✏️paper](https://www.jmlr.org/papers/volume18/16-456/16-456.pdf)]

Before 2017
- Binarized Neural Networks: Training Deep Neural Networks with Weights and Activations Constrained to +1 or -1 [[✏️✏️paper](https://arxiv.org/abs/1602.02830)][⭐⭐[github](https://github.com/itayhubara/BinaryNet)]
- Bitwise Neural Networks [[✏️ICML](https://arxiv.org/abs/1601.06071)]
- Loss-aware Binarization of Deep Networks [[✏️paper](https://arxiv.org/abs/1611.01600)]
- Overcoming Challenges in Fixed Point Training of Deep Convolutional Networks [[ICML](https://arxiv.org/abs/1607.02241)]
- Quantized Convolutional Neural Networks for Mobile Devices [[✏️✏️CVPR](https://openaccess.thecvf.com/content_cvpr_2016/html/Wu_Quantized_Convolutional_Neural_CVPR_2016_paper.html)]
- BinaryConnect: Training Deep Neural Networks with binary weights during propagations [[✏️✏️NIPS](https://arxiv.org/abs/1511.00363)][⭐⭐[github](https://github.com/MatthieuCourbariaux/BinaryConnect)]
- Expectation Backpropagation: Parameter-Free Training of Multilayer Neural Networks with Continuous or Discrete Weights [[✏️NIPS](https://proceedings.neurips.cc/paper/2014/hash/076a0c97d09cf1a0ec3e19c7f2529f2b-Abstract.html)]
- Expectation Backpropagation: Parameter-Free Training of Multilayer Neural Networks with Continuous or Discrete Weights [[✏️NIPS](https://proceedings.neurips.cc/paper/2014/hash/076a0c97d09cf1a0ec3e19c7f2529f2b-Abstract.html)]
- Back to Simplicity: How to Train Accurate BNNs from Scratch? [[paper](https://arxiv.org/abs/1906.08637)][⭐[github](https://github.com/hpi-xnor/BMXNet-v2)]




## Courses, webinars, blogs

Legenda: 🥇🥈🥉 Combined project-quality score

2017-2022

- …
- …
- …

Before 2017

- …
- …

### Libraries and resources

Legenda

- 🐣 New project *(less than 6 months old)*
- 💤 Inactive project *(6 months no activity)*

Sorting: alphabetic order

List of libraries and resources
- 🐣 Nebulgym G stars 151 - Accelerate AI training in a few lines of code without changing the training setup. Apache-2
- OpenVINO™ G stars 3.4k - Open-source toolkit for optimizing and deploying AI inference. Apache-2




<img height="25" width="100%" src="https://user-images.githubusercontent.com/83510798/171454644-d4b980bc-15ab-4a31-847c-75c36c5bd96b.png">

<p align="center">
  <a href="https://discord.gg/RbeQMu886J">Join the community</a> •
  <a href="#contribute">Contribute to the library</a>
</p>
