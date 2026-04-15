## 01-Tokenizer

分词器（tokenizer）是将字符串和整数序列（token）转换的工具，这里的整数序列将字符串切分成段，每个段对应着一个整数

这个课程使用的是**Byte-Pair Encoding(BPE) tokenizer**

当前还有tokennizer-free的方法，不对字符串进行分词处理直接使用原始的字节，但是这种方法还没应用落地

对tokenizer的讨论要从原始的Tranformer架构开始：

自从tranformer出现以来出现了很多新的变化：

- 激活函数的变化：ReLU，SwiGLU
- 位置编码的变化：正余弦编码，RoPE旋转位置编码
- 正则化：LayerNorm，RMSNorm
- 正则化的位置：pre-norm&post-norm
- MLP：dense MLP，混合专家
- 注意力Attention：full Attention、sliding window Attention、linear Attention
- 低维度的注意力：GQA、MLA
- 状态空间模型：Hyena

### kernels、parallelism、inference

GPU上有若干处理浮点运算的单元适合大型运算，但是数据的搬运是问题，主要的成本在数据移动上

当存在多个GPU时通过MVlink和CPU节点相连，在GPU之间的数据移动更加缓慢

inference分为两个阶段，分别为：prefill和decode

prefill时给模型输入一个句子，后面的decode是模型根据句子不断推理出新的token，这种方式使GPU的利用率不高，因为GPU在这种方式下没有并行的进行推理

### Scaling Laws

在很小的规模下进行实验，预测大规模的超参数/loss

在训练时的训练的token计算为模型参数的20倍，例如在1.4B参数模型上需要训练28B的tokens，这里没有算上推理部分所需的tokens