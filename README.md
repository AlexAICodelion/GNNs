
Why GNNs


MLP&amp;GCN

<img width="882" height="297" alt="image" src="https://github.com/user-attachments/assets/572ec35d-179c-40c6-97cc-cfa349989f08" />
PyG(pytorch_geometric) address：https://github.com/pyg-team/pytorch_geometric?tab=readme-ov-file#installation

第一个数据集：https://pytorch-geometric.readthedocs.io/en/latest/modules/datasets.html
核心是 Zachary’s Karate Club 的 PyG 入门：
写可视化函数（NetworkX 绘图、embedding 绘图）
定义一个小 GCN（多层 GCNConv + tanh）
训练并每 10 个 epoch 可视化一次节点 embedding 的变化

第二个数据集：
核心是 Cora 数据集上的对比实验：
数据集加载与统计信息打印
t-SNE 可视化 logits/embedding
先做 MLP（只用节点特征，不用图结构）
再做 GCN（用邻域聚合利用图结构）
训练、测试、可视化对比

PyG文档（GCNconv包）：https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.conv.GCNConv.html#torch_geometric.nn.conv.GCNConv



核心是 在 Cora 上训练好 GCN 后做节点级可解释性：
选一个预测对的测试节点 + 一个预测错的测试节点
用 PyG 的 Explainer + GNNExplainer 产出解释 mask：node_mask / edge_mask
读 top 特征维度、top 边
做 fidelity（deletion / insertion）来检验解释是否“真的影响模型决策”
