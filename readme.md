# 结果
---
>tinymind运行链接[https://www.tinymind.com/executions/n084nlgz](https://www.tinymind.com/executions/n084nlgz)  

下图所示为word2vec的词嵌入结果图
![](https://i.imgur.com/u040XTQ.png)

下图所示为RNN_LSTM在运行完30个epoch共计13小时后的输出log
![](https://i.imgur.com/BmpXk5i.png)

# 总结
---
### word2vec
感觉这个词嵌入其实没啥好说的，代码的分析可以见word2vec basic这个notebook，主要通过以下几个过程实现embedding  

1. 读入数据  
2. 计算文本总长度，不同词的数量以及单个词出现的频率  
3. 根据词出现的频率排序，设置索引，再根据索引建立相应的dictionary和reserve dictionary，将稀有数据用unk表示，但是我觉得本次的num words 取得相对较少，unk相对较多，导致最后的预测过程中出现好多unk  
4. 生成训练用的batch 主要用到一个skip window这样的东西
5. 创建模型，主要的一个是embedding矩阵，等于vocabularySize * embeddingSize（即向量维度）
6. 开始训练，loss用nce loss ，待loss收敛后，进行降维可以使其可视化，最后将embedding和dictionary和reverse dictionary 保存下来

感觉其实就是根据某个单词或者字的上下文如果和其他词相似，那么它们两个就挨的近一些，然后降维到一个较低的空间维度，对于上面这幅图来说，它就是字的向量在降维的表示，靠在一起的说明有相似的上下文，从而判定他们有某种特征相似

---

### LSTM RNN
详细的代码分析还是直接看代码里面吧，这里说下模型的设计流程
感觉打字好累... 画了下面这幅图，感觉更直观形象吧
![](https://i.imgur.com/u9pbDGM.png)

1. 读入数据，并且根据dictionary将文字用index表示
2. 根据batch size 将vocabulary等比例分割，再根据num steps 再分割一次
3. 读入前面word2vec训练出来的embedding矩阵，通过TensorFlow的embedding lookup方法迅速找到各个index对应的词向量
4. 设计lstm cell基础单元，这里比较重要的是一个state size ，根据RNN前向运算可知，最后的outputs是input*state size 这里为了后面做计算方便，另state size = embedding dim 即词的维度
5. 设计dropout 控制输出的dropout 防止过拟合
6. 设计mutiRNNcell 将lstmcell 基础单元进行堆叠，增加模型深度，具有更好的表达能力
7. 全零初始化一个 init state 表示cell的初始state
8. 将前面的muticell 送入 dymanic Rnn 进行循环，其输出有两个，一个是最终的outputs，另一是最后一个state，state可以作为下一个batch的初始值，outputs经过压平处理后变成一个二维矩阵
9. 前面之所以令statesize=embedding dim 的维度，就是在这里用到，跟一个W矩阵相乘，将最后一维变为num words的个数，（其实我觉得还是没必要一定让state size = embedding dim吧，只要让W的raw=state size就行了，保证最后出来的是 num words的长度就行）
10. 最后就是进行softmax激活做loss，这里我觉得loss是不是不用交叉熵好一些，因为我看训练时候的loss基本在后面没什么变化了，这里能用前面的nce loss 吗 》。。。。。 