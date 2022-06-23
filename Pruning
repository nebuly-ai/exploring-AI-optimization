
<img height="25" width="100%" src="https://user-images.githubusercontent.com/83510798/171454644-d4b980bc-15ab-4a31-847c-75c36c5bd96b.png">

# Pruning
The pruning algorithm usually can be divided in three main parts: 
* The selection of the parameters to prune. 
* The methodology to perfom the pruning. 
* The fine-tuning of the remaining parameters. 

## Pruning a way to Find the "Lottery Ticket"
Given a network, the lottery ticket hypothesis suggests that there is a subnetwork that is at least as accurate as the original network. Pruning, which is a technique of removing weights from a network without affecting its accuracy, aims to find this lottery ticket. 
The main advantages of a pruned network are:
* the model is smaller: it has fewer weights and therefore occupies a smaller portion of memory. 
* The model is faster: the smaller weights can reduce the amount of FLOPS of the model. 
* The training of the model is faster. 

These "winning tickets" or subnetworks are found to have special properties:
 * They are independent of the optimizer. 
 * They are transferable between similar tasks. 
 * The winning tickets of large-scale tasks are more transferable.


## Pruning Types
Various pruning technique have been proposed in the licterature, here an attempt to cathegorized the common traits is proposed. 

### Scale: Structured Vs Unstructured 
Unstructured pruning occurs when the weights to be pruned are individually targeted without taking into account the layer structure. This means that the selection of weights to prune is easy once the principle for doing so is defined.  Since the layer structure is not taken into account, this type of pruning may not improve the performance of the model. The typical case is that as the number of pruned weights increases, the matrix becomes more and more sparse; sparsity requires ad hoc computational techniques that may produce even worse results if the tradeoff between representation overhead and amount of computation performed is not balanced. For this reason, the performance increase with this type of pruning is usually only observable for a high pruning ratio. 
Structured Pruning, instead of focusing on individual weights, attempts to prune an entire structure by producing a more ordered sparsity that is computationally easier to handle, sacrificing the simplicity of the pruning algorithm. 

### Data Dependency: Data Dependent Vs Data Indipendent
Distinction between pruning techniques that use only weight as information for the pruning algorithm (Data Indipendet) and techniques that perform further analysis that require some input data to be provided (Data Dependent). Usually data-dependent techniques, because they require performing more calculations, are more costly in terms of time and resources. 

### Granularity: One-shoot Vs Iterative
One-shot techniques establish a criterion, such as the amount of pruning weight or model compression, and perform pruning in a single pass. Iterative techniques, on the other hand, adapt their learning and pruning ratio through several training epochs, usually producing much better results in terms of both compression achieved and accuracy degradation. 

### Initialization: Random, Later Iteration, Fine-Tuning
Especially when performing iterative pruning, with different pruning ratios, there are several possibilities to set the initial weight between pruning steps, which can be set randomly each time, or maintained from the previous epoch by fine-tuning the remaining weights to balance those being pruned. Another technique is to take a later iteration as a starting point, using a trade-off between random and maintaining the same from the previous epoch; in this way the model has more freedom to adapt to the pruned weight and generally adapts better to change. 

### Reversibility: Masked Vs Unmasked
One of the problems with pruning is that some of the weights that are removed in the first iterations may actually be critical, and their saliency may be more pronounced as pruning increases. Some techniques therefore, instead of removing the weights completely, adopt a masking technique that is able to maintain and restore the value of the weights if in a later iteration they start to become relevant. 

### Element Type: Neuron Vs Connection
Simply the different types of pruned element, that could be a connection between two neurons or directly the entire neuron that is pruned. 

### Timing: Dynamic Vs Static
Dynamic pruning is performed at runtime introducing some overhead but can be adaptivly perfomed per computation, Static pruning is performed offline before deploying the model

```mermaid
flowchart LR
A["Pruning \n Types"] --> B["Scale"]
B --> BA["Structured"]
B --> BB["Unstructured"]
A --> C["Data \n Dependency"]
C --> CA["Data \n Dependent"]
C --> CB["Data \n Independent"]
A --> D["Granularity"]
D --> DA["One-Shot"]
D --> DB["Iterative"]
A --> E["Initialization"]
E --> EA["Random"]
E --> EB["Later Iteration"]
E --> EC["Fine-Tuning"]
A --> F["Reversibility"]
F --> FA["Masked"]
F --> FB["Unmasked"]
A --> G["Element Type"]
G --> GA["Neuron"]
G --> GB["Connection"]
A --> H["Timings"]
H --> HA["Static"]
H --> HB["Dynamic"]
```

## Pruning Techniques

### Magnitude Based: Simple, Regularized.
Under the hypothesis that smaller weight have a minor impact on the model accuracy the weight that are smaller of a given threshold are pruned. To enforce the weight to be pruned some regularization can be applied. Usualy $L_1$ norm is better right after pruning while $L_2$ works better if the weight of the pruned network are fine-tuned. 

### Inboud Pruning [[1]](#1)
The input pruning method targets the number of channels on which each filter operates. The amount of information each channel brings is measured by the variance of the activation output of the specific channel.

$$ \sigma_{ts} = var(|| W_{ts} * X_{s} ||_F) $$

Where $t$ is the filter, $s$ the activation channel.  
Pruning of the entire network is performed sequentially on the layers of the network: from lower to higher layers, pruning is followed by fine-tuning, which is followed by pruning of the next layer. Speeding up the input pruning scheme is directly achieved by reducing the amount of computation that each filter performs.

### Reduce and Reuse [[1]](#1)
To find candidates for pruning, the variance of each filter output on a sample of the training set is calculated. Then, all filters whose score is less than the percentile $\mu$ are eliminated.

$$ \sigma_{t} = var(|| W_{t} * X ||_F) $$

The pruning performed requires adaptation, since the next layer expects to receive inputs of the size prior to pruning and with similar activation patterns. Therefore, the remaining output channels are reused to reconstruct each pruned channel, through the use of linear combinations.

$$ A = min_{A} \left(\sum_i || Y_i - A Y'_i) ||_2^2 \right)$$

Where $Y_i$ is the output before pruning, $Y'_i$ is the output after pruning, and $i$ are the iterations on the input samples, each input sample producing different activations.  
The operation of A is added to the network by introducing a convolution layer with filters of size 1 × 1, which contains the elements of A.

### Entropy Based Pruning [[2]](#2)
Entropy-based metric to evaluate the weakness of each channel: a larger entropy value means the system contains more information. First is used global average pooling to convert the output of layer $i$, which is a $c \times h \times w$ tensor, into a $1 \times c$ vector. In order to calculate the entropy, more output values need to be collected, which can be obtained using an evaluation set.
Finally, we get a matrix $M \in R^{ n \times c}$, where $n$ is the number of images in the evaluation set, and $c$ is the channel number. For each channel $j$, we would pay attention to the distribution of $M[:,j]$. To compute the entropy value of this channel, we first divide it into $m$ different bins, and calculate the probability of each bin. 
The entropy is then computed as

$$ H_j = - \sum_i^m p_i log(p_i) $$

Where, $p_i$ is the probability of bin $i$, $H_j$ is the entropy of channel $j$. Whenever one layer is pruned, the whole network is fine-tuned with one or two epochs to recover its performance slightly. Only after the final layer has been pruned, the network is fine-tuned carefully with many epochs.

### APoZ [[3]](#3)
It is defined Average Percentage of Zeros (APoZ) to measure the percentage of zero activations of a neuron after the ReLU mapping. $APoZ_c^{(i)}$ of the $c-th$ neuron in $i-th$ layer is defined as:

$$ APoZ_c^{(i)} =  \frac{\sum_k^N \sum_j^M f(O_{c,j}^{(i)}(k))}{N \times M}$$

Where, $O_c^{(i)}$ denotes the output of the $c-th$ neuron in $i-th$ layer, $M$ denotes the dimension of output feature map of $O_c^{(i)}$ , and $N$ denotes the total number of validation examples. While $f(\cdot)$ is a function that is equal to one only if $O_c^{(i)} = 0$ and zero otherwise.  
The higher mean APoZ also indicates more redundancy in a layer. Since a neural network has a multiplication-addition-activation computation process, a neuron which has its outputs mostly zeros will have very little contribution to the output of subsequent layers, as well as to the final results. Thus, we can remove those neurons without harming too much to the overall accuracy of the network.  
Empirically, is found that starting to trim from a few layers with high mean APoZ, and then progressively trim its neighboring layers can rapidly reduce the number of neurons while maintaining the performance of the original network. Pruning the neurons whose APoZ is larger than one standard derivation from the mean APoZ of the target trimming layer would produce good retraining results.

### Filter Weight Summing [[4]](#4)
The relative importance of a filter in each layer is measured by calculating the sum of its absolute weights.

$$ s_j = \sum |F_{i,j}|$$

This value gives an expectation of the magnitude of the output feature map. Then each filter is sorted based on $s_j$, and $m$ are pruned togheter with the kernels in next layer corresponding to the prenued features. After a layer is pruned, the network is fine-tuned, and pruning is continued layer by layer.

### Geometric Median [[5]](#5)
The geometric median is used to get the common information of all the filters within the single ith layer:

$$ x^{GM} = \argmin_x \sum_{j'} || x - F_{i,j'} ||_2$$
$$ x \in R ^ {N_i \times K \times K} $$
$$ j'\in [1, N_{i+1}] $$

Where $N_i$ and $N_{i+1}$, to represent the number of input chan-nels and the output channels for the $i-th$ convolution layer, respectively. $F_{i,j}$ represents the $j-th$ filter of the $i-th$ layer, then the dimension of filter $F_{i,j}$ is $N_i \times K \times K$, where $K$ is the kernel size of the network.

Then is found the filters that are closer to geometric mean of the filters and pruned since it can be represented by the other filters.

Such computations is rather complex than instead is possible to find which filter minimizes the summation of the distance with other filters. 

$$ F_{i,x^*} = \argmin_x \sum_{j'} || x - F_{i,j'} ||_2$$
$$ x \in [F_{i,1} \dots  F_{i,N_{i+1}}] $$
$$ j'\in [1, N_{i+1}] $$

Since the selected filter(s), $F_{i,x^*}$ , and left ones share the most common information. This indicates the information of the filter(s) $F_{i,x^*}$ could be replaced by others.


```mermaid
flowchart LR
A["Pruning \n Techniques"] --> B["Magnitude \n Based"]
A --> C["Filter/ \n Feature Map \n Selection"]
C --> CA["Variance Selection"]
C --> CB["Entropy Based"]
C --> CC["Zeros Counting \n (APoZ)"]
C --> CD["Filter Weight \n Summing"]
C --> CE["Geometric Mean \n Distance"]
A --> D["Optimization \n Based"]
D --> DA["Greedy Search"]
D --> DB["LASSO"]
A --> E["Sensitivity \n Based"]
E --> EA["First Order \n Taylor Expansion"]
E --> EB["Hessian \n Approximation"]
```

## References

<a id="1">[1]</a> 
Polyak, Adam and Wolf, Lior, ["Channel-level acceleration of deep face representations"](https://ieeexplore.ieee.org/document/7303876), in IEEE Access (2015). 

<a id="2">[2]</a>
Luo, Jian-Hao, and Jianxin Wu, ["An Entropy-based Pruning Method for CNN Compression"](https://arxiv.org/abs/1706.05791), in arXiv preprint arXiv:1706.05791 (2017).


<a id="3">[3]</a>
Hu, H., Peng, R., Tai, Y. W., & Tang, C. K., ["Network Trimming: A Data-Driven Neuron Pruning Approach towards Efficient Deep Architectures"](https://arxiv.org/abs/1607.03250) in arXiv preprint arXiv:1607.03250 (2016).

<a id ="4">[4]</a>
Li, H., Kadav, A., Durdanovic, I., Samet, H., & Graf, H. P. ["Pruning filters for efficient convnets"](https://arxiv.org/abs/1608.08710). arXiv preprint arXiv:1608.08710 (2016).

<a id ="5">[5]</a>
He, Y., Liu, P., Wang, Z., Hu, Z., & Yang, Y. . ["Filter pruning via geometric median for deep convolutional neural networks acceleration"](https://arxiv.org/abs/1811.00250). In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition (2019).


<img height="25" width="100%" src="https://user-images.githubusercontent.com/83510798/171454644-d4b980bc-15ab-4a31-847c-75c36c5bd96b.png">


<p align="center">
  <a href="https://discord.gg/RbeQMu886J">Join the community</a> •
  <a href="#contribute">Contribute to the library</a>
</p>
