
Why GNNs

contents：

    加载图数据（KarateClub）
    
    理解图结构（edge_index）
    
    可视化原始图（NetworkX）
    
    定义一个小  GCN（3 层卷积 + 分类头）
    
    训练（semi-supervised mask 上的交叉熵）
    
    把“训练过程”用 embedding 动态图展示出来，每 10 个 epoch 可视化一次节点 embedding 的变化

<img width="882" height="297" alt="image" src="https://github.com/user-attachments/assets/572ec35d-179c-40c6-97cc-cfa349989f08" />

PyG(pytorch_geometric) address：https://github.com/pyg-team/pytorch_geometric?tab=readme-ov-file#installation


一、PyG.ipynb（KarateClub）

第一个数据集KarateClub address：https://pytorch-geometric.readthedocs.io/en/latest/modules/datasets.html

1、加载 KarateClub 数据集 + 查看数据集基本信息

<img width="300" height="150" alt="image" src="https://github.com/user-attachments/assets/54269491-de98-4ced-b507-270d83d63bf5" />

KarateClub() 载入 Zachary’s Karate Club 经典小图数据。

打印：

len(dataset)：这个数据集里有多少张图（KarateClub 通常只有 1 张图）

dataset.num_features：每个节点的输入特征维度（KarateClub 数据集默认给每个节点配置的特征表示的维度）

dataset.num_classes：标签类别数


2、取出唯一那张图，查看 Data 对象

<img width="738" height="57" alt="image" src="https://github.com/user-attachments/assets/cbbc39ce-13ee-4148-9afd-6605acbda378" />

print(data) 会打印 PyG 的 Data(...) 概览，通常包含：

x：节点特征矩阵（形状大致是 [num_nodes, num_features]）

edge_index：边列表（形状 [2, num_edges]）

y：节点标签（形状 [num_nodes]）

还有 train_mask/test_mask（看数据集是否自带划分）


3、把边列表打印出来看清楚“谁连谁”，也就是说谁和谁有关系

<img width="255" height="864" alt="image" src="https://github.com/user-attachments/assets/6a29196f-a1f2-4db1-b749-fd75f6231625" />


4、把 PyG 图转成 NetworkX，并画出图结构，也就是原始图结构 + 原始标签的可视化

把 PyG 图转成无向图（KarateClub 本质上常当无向图看）。

visualize_graph：把节点画出来，并按真实标签 data.y 上色。

<img width="843" height="831" alt="image" src="https://github.com/user-attachments/assets/b61aa2cc-cd06-4e02-bcef-37ab0a9805a8" />


5、定义 3 层 GCN + 分类器，并打印模型结构，把“图数据 → 节点表示 → 节点分类”写成一个可训练的模型

<img width="642" height="639" alt="image" src="https://github.com/user-attachments/assets/17da5e61-dd8d-4bd8-aa10-e7b8ccf107d1" />

三层 GCNConv 的含义（严格对应你的实现）：

第 1 层：把每个节点从 num_features 维映射到 4 维，并融合 1-hop 邻域信息

第 2 层：4 → 4，再融合一次邻域信息（感受野扩大）

第 3 层：4 → 2，输出 2 维嵌入（方便直接二维散点图可视化）

classifier = Linear(2, num_classes)：把 2 维嵌入映射成类别 logits（每个节点输出 num_classes 维得分）。

打印 model ：是核对结构（层数、维度）

<img width="761" height="188" alt="image" src="https://github.com/user-attachments/assets/ac8bda45-8c91-4ff9-9e41-fb743cacf40a" />


6、不训练，直接前向一次，并把 2D embedding 画出来，结果由随机初始化权重 + 图结构 + 节点特征共同决定

model(...)：对整张图前向传播一次，得到：

    out：分类 logits（你这里丢掉不用）
    
    h：2 维节点嵌入（你用来画图）

visualize_embedding(...)：按真实标签上色，看看随机初始化的嵌入分布长什么样。

<img width="876" height="909" alt="image" src="https://github.com/user-attachments/assets/2d389dd3-3975-45ce-bf70-350e14e4394c" />


7、训练 GCN，并每 10 个 epoch 画一次 embedding

定义损失函数：交叉熵 CrossEntropyLoss（用于多分类）。

定义优化器：Adam，更新模型参数。

train(data) 做一轮训练：

  1）清梯度
  
  2）前向得到 out（logits）和 h（embedding）
  
  3）用 data.train_mask 选出“训练节点”子集，计算这些节点的交叉熵损失
  
  4）反向传播 + 参数更新
  
训练 401 轮，每 10 轮画一次当前 h 的二维嵌入，并在 x 轴标签上写 epoch 与 loss。

其中：

  loss 只在 train_mask 对应的那些节点上计算：
  
    训练节点的真实标签 data.y[train_mask]
    
    训练节点的预测 logits out[train_mask]
    
模型之所以能利用“未标注节点的信息”，来自 GCNConv 的邻域聚合：即便某些节点不在 mask 里，它们仍会通过图结构影响邻居的信息传播。

<img width="831" height="863" alt="image" src="https://github.com/user-attachments/assets/213fabc6-8fd9-43e9-af51-5012cca8605f" />

<img width="828" height="866" alt="image" src="https://github.com/user-attachments/assets/84aa485c-28c2-477a-8acd-7af4770eb1d5" />

<img width="834" height="867" alt="image" src="https://github.com/user-attachments/assets/badf4060-f884-46cc-93b0-8e76a469b7ad" />

<img width="831" height="858" alt="image" src="https://github.com/user-attachments/assets/985f06ea-ff0e-427c-9fea-a52d94c28a53" />

<img width="831" height="864" alt="image" src="https://github.com/user-attachments/assets/e0a2521f-061a-43bd-adb7-d26fe1700ca2" />




