Related to Cora_MLP&GCN.ipynb（Cora）in code

contents：


PyG文档（GCNconv包）：https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.conv.GCNConv.html#torch_geometric.nn.conv.GCNConv


1、下载 / 读取 Cora + 打印图数据统计

加载 Planetoid 系列数据集中的 Cora，GitHub地址：https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.datasets.Planetoid.html#torch_geometric.datasets.Planetoid

但是Cora的raw data如果不能自动下载，那么可以在这里面去下载：https://github.com/kimiyoung/planetoid/tree/master/data

<img width="255" height="270" alt="image" src="https://github.com/user-attachments/assets/3acca709-605f-4414-87ea-2818edc5ba6a" />

打印的这些结果解释：（简单的说就是2708篇论文，1433向量，10556条边，即中等规模图、特征很高维（1433）、边数量约一万级）

<img width="1248" height="465" alt="image" src="https://github.com/user-attachments/assets/bb8e4750-e74d-4d27-95e5-c8324f6b5f29" />


len(dataset)：图的数量（Cora 是 1 张图，所以一般是 1）

dataset.num_features：每个节点的特征维度（Cora 是 1433）

dataset.num_classes：类别数（Cora 是 7）

data.num_nodes / data.num_edges：节点数 / 边数

data.train_mask.sum()：训练集中有标签节点数（Cora 常见是 140，即每类 20 个）

train_mask / num_nodes：训练标签占比（很小，所以叫半监督）

has_isolated_nodes / has_self_loops / is_undirected：图结构性质检查


2、t-SNE 可视化函数 visualize

t-SNE 本身的性质，t-SNE 是非线性降维，重点保留“局部邻近关系”

3、对比实验——MLP vs GCN

      MLP：只用节点特征 x，不看图结构 edge_index
      
      GCN：同时用 x 和 edge_index（邻域聚合）

4、传统的全连接层(Multi-layer Perception Network)-（两层全连接）

<img width="765" height="120" alt="image" src="https://github.com/user-attachments/assets/75dbf447-89d2-4ecb-be9a-cf94292d2fbb" />

lin1: 1433 -> hidden_channels

lin2: hidden_channels -> 7

中间：ReLU + Dropout


4、训练 、测试MLP（只用 data.x）

<img width="1209" height="585" alt="image" src="https://github.com/user-attachments/assets/229f49d8-dac1-42a1-898e-96afd68500e9" />

1）训练 MLP

model.train()：进入训练模式（dropout 生效）

前向：对所有节点算 logits：out.shape = [num_nodes, num_classes]

但 loss 只在训练 mask 的节点上计算（半监督关键点）

反向传播 + Adam 更新参数

循环 200 个 epoch 打印 loss

<img width="318" height="861" alt="image" src="https://github.com/user-attachments/assets/4bfaed1c-ec23-4878-87f5-83144d9cfc04" />

2）测试 MLP（只用 data.x）

model.eval()：评估模式（dropout 关闭）

pred = out.argmax(dim=1)：取最大得分类别

test_mask 上算 accuracy

<img width="549" height="192" alt="image" src="https://github.com/user-attachments/assets/0dd91cf0-03b6-4576-8f3b-d81edc9a5432" />


5、可视化 MLP 的输出

MLP 的输出空间结构（不是图结构）看到“大家混在一起”，也就是说MLP 学到的表示对类别区分不够强（至少在二维投影里看起来不明显）

<img width="1179" height="1175" alt="image" src="https://github.com/user-attachments/assets/0fc5b7f2-b823-4a00-bfe8-085cf1ea80b0" />


6、对比试验

Graph Neural Network (GNN)-GCN 的每层都会把邻居的特征/表示融合进当前节点表示

将全连接层替换成GCN层（两层 GCNConv）

<img width="348" height="141" alt="image" src="https://github.com/user-attachments/assets/e8b91764-6455-4b14-8b1c-a7e31c9561b9" />

MLP 的线性层是：XW

GCNConv是在每层把“邻居信息”混进来（邻域聚合）：AXW

7、未训练的 GCN 直接可视化（baseline）

不会形成稳定的 7 类簇

<img width="1182" height="1179" alt="image" src="https://github.com/user-attachments/assets/2c1ad3ac-0530-4e2b-9456-cbba7e5e3cb2" />


8、测试 GCN 并打印 Test Accuracy

<img width="270" height="855" alt="image" src="https://github.com/user-attachments/assets/e255e522-fff7-4909-ab8b-e0e9cfb81bb8" />


<img width="525" height="192" alt="image" src="https://github.com/user-attachments/assets/8f1ad6f0-fd42-4eea-8a87-76a5d3f8a737" />

其实可以对比两个结果，前者为MLP，后者为GCN

<img width="549" height="192" alt="image" src="https://github.com/user-attachments/assets/0dd91cf0-03b6-4576-8f3b-d81edc9a5432" />  <img width="525" height="192" alt="image" src="https://github.com/user-attachments/assets/8f1ad6f0-fd42-4eea-8a87-76a5d3f8a737" />


9、训练后的 GCN 输出可视化

<img width="1182" height="1170" alt="image" src="https://github.com/user-attachments/assets/8f858c63-be5a-43af-b55e-2b7ba062bffe" />

用训练好的 GCN 得到每个节点的 7 类 logits

t-SNE 压到二维

用真实标签上色

直观看：不同类别形成更明显的区域/簇，比 MLP 更可分








































第二个数据集：
核心是 Cora 数据集上的对比实验：
数据集加载与统计信息打印
t-SNE 可视化 logits/embedding
先做 MLP（只用节点特征，不用图结构）
再做 GCN（用邻域聚合利用图结构）
训练、测试、可视化对比

