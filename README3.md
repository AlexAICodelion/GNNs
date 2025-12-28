Related to gcn_explain.ipynb in code




1、定义并训练 GCN

第一层：GCNConv(1433 -> hidden_channels)

ReLU + Dropout

第二层：GCNConv(hidden_channels -> 7) 输出 7 类的 logits

这里的关键点：
GCNConv 的输出不仅依赖节点自身特征，还依赖邻居节点特征（邻域聚合）

























训练一个 GCN 做 Cora 节点分类

在测试集里挑两个节点：一个预测对（node_ok）、一个预测错（node_bad）

用解释器给这两个节点生成解释（mask）：

node_mask：哪些特征维度更重要

edge_mask：哪些边更重要

把解释“翻译成人能读懂的东西”：Top 特征、Top 邻接边、关键邻居类别混杂程度

做一类最基础的“解释质量检验”（你做的是 fidelity 类指标）：

删除最重要的边，看置信度掉多少（deletion fidelity）

只保留最重要的边，看能不能维持原预测（insertion fidelity）

再做一个更稳的版本：只在 2-hop 子图里做 insertion




























