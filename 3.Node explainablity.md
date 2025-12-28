Related to gcn_explain.ipynb in code

contents：

                    一、训练一个 GCN 做 Cora 节点分类
                    
                    二、在测试集里挑两个节点：一个预测对（node_ok）、一个预测错（node_bad）
                    
                    三、用解释器给这两个节点生成解释（mask）：
                    
                              node_mask：哪些特征维度更重要
                              
                              edge_mask：哪些边更重要
                    
                    四、把解释转成可读的：Top 特征、Top 邻接边、关键邻居类别混杂程度
                    
                    五、做一类最基础的“解释质量检验”（你做的是 fidelity 类指标）：
                    
                              删除最重要的边，看置信度掉多少（deletion fidelity）
                              
                              只保留最重要的边，看能不能维持原预测（insertion fidelity）
                              
                              再做一个更稳的版本：只在 2-hop 子图里做 insertion


一、定义并训练 GCN

<img width="359" height="129" alt="image" src="https://github.com/user-attachments/assets/ecdd29dc-69e2-4e20-a1c8-735fc109e616" />

第一层：GCNConv(1433 -> hidden_channels)

ReLU + Dropout

第二层：GCNConv(hidden_channels -> 7) 输出 7 类的 logits

这里的关键点：

GCNConv 的输出不仅依赖节点自身特征，还依赖邻居节点特征（邻域聚合）


二、训练与测试函数 + 训练 200 epoch，Test Accuracy

<img width="279" height="864" alt="image" src="https://github.com/user-attachments/assets/d7a5760a-4b83-4b04-90ad-82139179370c" />


三、可视化（t-SNE）

<img width="1179" height="1170" alt="image" src="https://github.com/user-attachments/assets/e6806347-61f9-405f-8d8d-e72fa0caa12b" />

模型学到的表示在某种程度上对类别有可分性。

但它不是严格意义的“可解释性”，因为它没有回答：

为什么某个节点被判成某一类？

哪些特征/哪些邻居支持了这个判断？

如果去掉某些边/特征，预测会不会变？

所以解释器 + fidelity来处理


四、节点可解释性

可解释性（Explainer/GNNExplainer + fidelity drop）

节点级可解释性：对某一个具体节点，模型为什么会给出当前的预测？它主要依赖了哪些邻居边、哪些输入特征？

          步骤：
          
          1、选节点（解释对象）：从测试集中挑选若干节点作为解释案例
          
          2、生成解释（解释器）：用GNNExplainer等方法输出
          
                  重要边/重要邻居（结构解释）
                  
                  重要特征维度（特征解释）
          
          3、解释可视化与阅读：画出解释子图、列出top特征
          
          4、解释质量验证（fidelity）：把解释器认为关键的边/特征移除，看预测置信度是否显著下降（证明“解释确实影响模型决策”）



1.1、选节点（解释对象）：从测试集中挑选若干节点作为解释案例

          选“预测正确 + 预测错误”的节点各一个
          
          A-解释“模型为什么成功”（正确样本）
          
          选一个预测正确的测试节点来解释，目的在于：
          
                看模型在“正常工作”的情况下，确实在依赖哪些邻居/哪些特征
            
                验证解释是否符合直觉：例如解释子图主要集中在该节点的局部邻域（1~2 hop），而不是随机散边
            
                这类解释更容易讲清楚：模型靠什么信息做对了，说明模型的决策路径在理想情形下是什么样的
          
          B-解释“模型为什么失败”（错误样本）
          
          选一个预测错误的测试节点来解释，目的在于：
          
                定位误判的原因：
            
                      是否因为邻居跨类别连接太多（结构混杂）
            
                      是否因为节点特征与其他类别更相似（特征模糊）
                  
                      是否因为局部子图本身就不典型/不纯净（类间边界点）
                
          t-SNE 看到“分簇但交叉多”，解释能给出“交叉从何而来”的机制性证据
          
          说明模型在什么情况下会做错，以及错在哪里

<img width="447" height="108" alt="image" src="https://github.com/user-attachments/assets/1f58cbd5-f4eb-4a1b-8a0d-b1696b81a602" />

          node_ok = 1709：模型把它预测为 2 类，真实也是 2 类，这是一个“预测正确”的测试节点，用来解释“模型为什么做对”。
          
          node_bad = 1708：模型把它预测为 1 类，真实是 3 类，这是一个“预测错误”的测试节点，用来解释“模型为什么会错/哪里混淆”。
          
          测试集 1000 个节点里错了 186 个，说明模型整体测试准确率大约是1−186/1000=0.814
          
          与Test Accuracy: 0.8140 一致，说明model.eval()推理与测试函数是对齐的

1.2、生成可解释性mask（GNNExplainer）

对节点 1709 和 1708，各生成一份解释，输出两类 mask：

          edge_mask：每条边的重要性分数（哪些边/邻居最影响该节点预测）
          
          node_mask（feature mask）：每个输入特征维度的重要性分数（哪些特征维度最关键）

<img width="507" height="144" alt="image" src="https://github.com/user-attachments/assets/a9e5333d-0a1a-4509-a476-907c1286251d" />

          edge_index 一共有 10556 条边（有向边计数）。
          
          edge_mask 长度 10556，表示：每一条边都有一个重要性分数（通常 0~1 之间，越大越重要）。
          
          主要是需要对于节点 1709 的预测，哪些边/邻居连接是“关键证据”
          
          图一共有 2708 个节点，每个节点有 1433 维特征。
          
          node_mask 给的是一个 (2708, 1433) 的矩阵：
          
          解释器为“每个节点的每一维特征”都分配了重要性分数

1.3、把 mask 输出成“Top 重要特征 + Top 重要边”

<img width="3210" height="438" alt="image" src="https://github.com/user-attachments/assets/dd15d273-8bcc-4b6d-9bc2-35389e190a71" />

          Top-10 feature dims：403、931…是 Cora 的 1433 维输入特征的“维度编号”
          
          Top-20 edge indices：7002, 6996, ...不是“节点编号”，而是 edge_index 这张表里的“第几列”
          
          node_ok 的 top 特征
          
          node_bad 的 top 特征
          
          两个节点的 top 边（先打印 top-20 的边“索引位置”）


1.4、把重要边映射成具体边，并画解释子图

<img width="684" height="725" alt="image" src="https://github.com/user-attachments/assets/b3658c5d-9423-4003-9a29-dd45ed2cdceb" />

<img width="828" height="852" alt="image" src="https://github.com/user-attachments/assets/1e4c0eab-6b24-43a7-8bf0-b685a6827b9f" />

周围小点大多是同一种颜色（同类邻居占主导）

<img width="828" height="861" alt="image" src="https://github.com/user-attachments/assets/815c69e4-607a-4ce5-87ad-a0ee97e32f1d" />

周围小点颜色明显更多、更杂

            === node 1709 top incident edges ===
            
            00. eid= 6852  (1739 -> 1709)  w=0.8849   y[s]=2, y[t]=2
             
            01. eid= 6851  (1738 -> 1709)  w=0.8822   y[s]=2, y[t]=2
               
            02. eid= 6854  (2365 -> 1709)  w=0.8581   y[s]=2, y[t]=2
               
            03. eid= 6850  (1358 -> 1709)  w=0.7778   y[s]=2, y[t]=2
               
            说明：解释器认为最关键的证据基本都来自“同类别（2类）邻居”
            
            === node 1708 top incident edges ===
            
            00. eid= 1925  (1708 -> 467)  w=0.9186   y[s]=3, y[t]=0
               
            02. eid= 9509  (1708 -> 2313)  w=0.9169   y[s]=3, y[t]=2
               
            04. eid= 3471  (1708 -> 873)  w=0.8972   y[s]=3, y[t]=0
               
            说明：1708 的邻域不是“纯 3 类社区”，而是 0 / 2 / 3 混在一起，并且“跨类边”的权重很高
            
            node_ok：邻域证据比较干净 → 模型更稳
            
            node_bad：邻域证据更混杂 → 容易被邻居“带偏”


1.5、解释质量检验（fidelity）：删除最重要边（把解释器说最重要的边删掉，模型对这个节点的预测会不会明显受影响？）

      删除 top_eids 中前 k 条边（最重要的 k 条），观察预测变化。
       
      返回：原预测类别/置信度、删边后原预测类别置信度、删边后新预测类别/置信度。

打印了 deletion 的结果（k=4/8/12）

<img width="3225" height="333" alt="image" src="https://github.com/user-attachments/assets/9f17ceec-93b1-4095-b93d-832d9cd2e1f1" />

            node_ok=1709 的解释：高度 faithful（删关键边→预测崩）
            
            原始：
            
                真值 y=2，预测 pred_before=2
                
                对类 2 的置信度 conf_before=0.8195
                
            删掉 top-4 关键边后：
            
                原预测类 2 的置信度掉到 0.1649 conf_after_on_pred_before
                
                下降 drop = 0.6546
                
                预测类别直接从 2 变成 3（pred_after=3）
                
            删到更多（k=8）后：
            
                原预测类 2 置信度进一步掉到 0.1053 conf_after_on_pred_before
                
                下降 drop =0.7142
                
                仍然预测成 3
            
            对节点 1709 来说，解释器选出的那几条“关键邻居连接”几乎是模型判断为 2 类的主要证据；一旦删掉这些边，模型马上失去关键结构信息，预测从 2 翻到 3，说明这份解释对模型决策是高度忠实的。


            node_bad=1708 的解释
            
            原始：
            
                真值 y=3，但预测 pred_before=1
                
                预测为 1 的置信度只有 conf_before=0.2080
                
            k=4 的现象：删边后反而更像 1（drop 负数）
            
                conf_after_on_pred_before=0.2528
                
                drop=-0.0448（置信度反而上升）
                
                这通常说明：你删掉的那 4 条边不一定是“支持预测为 1”的证据，可能删掉的是对 1 有抑制作用、或对其他类有支持作用的边，导致模型更偏向 1。
                
            k=8：轻微下降，但仍预测 1
            
                conf_after_on_pred_before=0.2079
                
                drop=0.0185，仍然 pred_after=1
                
                说明删 8 条还不足以改变决策。
                
            k=12：预测翻转到正确类 3（最关键）
            
                原预测类 1 的置信度下降到 0.1329（drop 0.0751）
                
                pred_after 从 1 变成 3
                
                新预测（类 3）的置信度 conf_after=0.3000
            
            对节点 1708，模型本身对“预测为 1”并不自信（0.208）。当移除更多解释器认为重要的邻域连接（到 k≈12）后，预测从 1 翻转为真实类别 3，说明这些关键边里包含了会把节点表示“拉离真实 3 类”的结构因素。也就是说，1708 的误判与其关键邻域中混杂/跨类连接有关，而不是单一特征造成。


1.6、对照组1：解释质量检验（fidelity）：只保留 top-k 边

<img width="3216" height="711" alt="image" src="https://github.com/user-attachments/assets/da34a42b-5852-4092-916e-803491d45257" />

                    node_ok=1709：解释非常强
                    
                    Deletion：删 top-4 关键边就让预测从 2 翻到 3，置信度大幅下降（drop≈0.65+）
                    
                    Insertion：只保留 top-2 / top-4 边，预测仍是 2，而且置信度甚至接近 1
                    
                    
                    对 1709 来说，这些关键边既是“必要的”（删掉就不行），也是“近乎充分的”（只保留也行），解释高度 faithful。
                    
                    验证了之前论文中的Plausibility(合理性)和Fidelity(保真度)
                    
                    
                    
                    node_bad=1708：局部证据很不稳定（这正是错分节点的典型特征）
                    
                    Deletion：删到 k=12 才把预测从 1 翻到真实类 3
                    
                    Insertion：只保留少量边时，预测会在 3、0、5 之间跳（k=2/4 是 3，k=6 是 0，k=8 是 5…）
                    
                    
                    1708 本身处在“结构/特征混杂”的区域，模型决策边界不稳，所以“只靠少量边”时预测更容易漂移。
                    
                    你的 insertion 实验因为“全图只剩几条边”，导致归一化与消息传递环境剧烈改变，所以预测更容易异常跳变。


1.7、对照组2：解释质量检验（fidelity）：只在“目标节点的 k-hop 局部子图”里做保留

先抽出 node 的 2-hop 子图（GCN 两层主要依赖 2-hop 邻域），再在这个子图里“只保留 top-k 解释边”。

这样不会把全图删成几条边（比如对照组1），结果更稳、更可信，因为每个节点的新表示 = 自己 + 邻居的加权平均（权重与度数有关）

                    更稳的 insertion：
                    
                    2-hop 子图 和 top-k 保留边的组合方法旨在提升解释的稳定性和可信度。
                    
                    2-hop 子图：
                    
                        2-hop是通过当前节点的邻居节点再进一步访问其邻居。也就是说，我们考虑一个节点的两跳邻居，而不是直接只依赖一跳邻居。
                        
                        2-hop邻居能提供更全面的信息，因为它不仅包括直接连接的节点（1-hop邻居），还包括这些邻居的邻居。这样做可以更准确地反映节点在图中的“局部结构”。
                        
                        不只依赖直接邻居：如果只考虑1-hop邻居，可能会导致过于局部化的解释，忽略了图中更广泛的结构信息。2-hop邻居有助于增加信息的传递范围，避免模型过度依赖单一节点的特征。
                    
                    top-k 保留边：
                    
                        top-k是，我们保留与目标节点最相关的k条边，这些边在节点的分类决策中起到关键作用。
                        
                        通过保留top-k边，我们避免了将整个图的边都删掉，只保留对分类决策最有影响的那部分。这能使得解释结果更加稳定，因为不至于在删除大量边的情况下导致结果剧烈波动。
                        
                        减少不必要的噪音：如果我们不限制边的数量，可能会导致模型在删去边时，决策变得非常不稳定。通过只保留top-k边，我们确保解释集中在最有影响的部分，避免了噪音干扰。

<img width="3234" height="456" alt="image" src="https://github.com/user-attachments/assets/7d0f1e85-3c50-45b8-8ace-3dda867af5fb" />

<img width="447" height="429" alt="image" src="https://github.com/user-attachments/assets/fb212f4c-4e06-4dd4-9293-2a51911a717b" />

                    1709：subgraph_edges_available: 868
                    
                    1708：subgraph_edges_available: 692
                    
                    目标节点 2-hop 邻域内，其实原本有几百条边，这才是两层 GCN 真正能“接触到”的结构环境。
                    
                    node_ok=1709：
                    
                    原图：预测 2，conf_before=0.8195
                    
                    只保留 top-2：仍预测 2，conf_after=0.9738
                    
                    只保留 top-4：仍预测 2，conf_after=0.9996
                    
                    只保留 top-8：仍预测 2，conf_after=0.9632
                    
                    只保留 top-10：仍预测 2，conf_after=0.9286
                    
                    对 1709 来说，解释器找出的关键边不仅“重要”，甚至对模型而言接近“充分证据”：只要这些边存在，模型就能非常自信地给出类别 2。
                    
                    1709 的解释边同时满足“必要性（deletion）”与“近似充分性（insertion）


                    node_bad=1708：解释显示“证据混杂”，并且少量关键边可能是误判来源
                    
                    原图：预测 1（错），conf_before=0.208
                    
                    只保留 top-2 / top-4：预测变成 3（对），conf_after=0.3000，在最关键的少数连接上，模型其实能倾向真实类 3
                    
                    只保留 top-8：预测变成 5（错）
                    
                    只保留 top-12：预测变成 0（错）
                    
                    对 1708，解释器挑出来的“重要边”并不是同一种作用：
                    
                    有的边把它推向真实类 3，有的边把它推向其他类（0/5/1）。
                    
                    “证据冲突 / 邻域混杂”：节点周围存在跨类连接，模型在不同边子集下会被不同社群拉扯。
                    
                    1708 的误分类与其关键邻域存在跨类连接/混杂证据有关；少量核心边可支持正确类 3，但随着更多高权重边加入，表示被不同类别邻域拉扯，导致预测不稳定并偏离真实类别。






















