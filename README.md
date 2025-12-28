GNN Explainability Research and Implementation

This project focuses on exploring the explainability of Graph Neural Networks (GNNs), specifically focusing on node-level explainability. By applying node classification tasks to classical graph datasets (such as Cora), we aim to uncover the decision-making process of GNN models and further improve their transparency and interpretability.

The project covers the following components:

      Data Loading and Preprocessing
      
      GNN Model Construction and Training (MLP and GCN)
      
      Node-Level Explainability Analysis
      
      Optimizing Explainability with Masking and Top-k Edges
      
      Model Evaluation and Results Analysis


Project Structure:（see README1）

1. Data Loading and Preprocessing:

Using the torch-geometric library, we loaded the Cora dataset, a classic graph dataset containing 2708 nodes and 10556 edges, and performed preprocessing steps like feature normalization.

2. Model Construction and Training:（see README2）

We implemented two models for the node classification task:

      MLP (Multilayer Perceptron):
      
      MLP is a simple baseline model that performs classification using fully connected layers without considering the graph structure.
      
      GCN (Graph Convolutional Network):
      
      GCN model uses graph convolutional layers (GCNConv) to aggregate information from neighboring nodes, and is suitable for graph-based data.

3. Node-Level Explainability Analysis:（see README3）

We performed the following steps to analyze node-level explainability:

      Selecting Test Nodes: Chose correctly and incorrectly predicted nodes for feature and neighbor analysis.
      
      Feature Importance Analysis: Used t-SNE to visualize the importance of features for node classification.
      
      Neighbor Node Influence Analysis: We observed how the removal of certain neighbors affects model predictions, understanding the role of neighbors in decision-making.
      
      Error Prediction Node Analysis: Compared correct and incorrect predictions to identify misleading features or neighbors.

4. Optimizing Explainability:（see README3）

      Stable Insertion: This approach involves first extracting a 2-hop subgraph (nodes up to 2 hops away), and then retaining the most influential top-k edges. This helps to provide more stable and reliable explanations by reducing the noise that might come from removing too many edges.
      
          2-hop Subgraph: By considering the 2-hop neighbors, we capture broader neighborhood information, improving the explanation's reliability.
      
          Top-k Edges: By limiting to the top-k edges, we focus only on the most influential connections, enhancing the explanation's stability and reducing the possibility of noisy interpretations.



