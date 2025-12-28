# **GNN Explainability on Cora (PyTorch Geometric)**

This repository documents a hands-on study of **Graph Neural Network (GNN) explainability** on the Cora node classification benchmark using **PyTorch Geometric (PyG)**.

The workflow starts from training **MLP vs. GCN** as a baseline comparison, and then applies **post-hoc**,**mask-based explanations (PyG Explainer)** to interpret **node predictions** through both:

            Feature attributions (node_mask)
            
            Edge/neighbor attributions (edge_mask)

In addition, the explanations are validated with **fidelity tests** (deletion/insertion), and compared on a **correctly predicted node** and a **misclassified node**.

## **Dataset**

            Cora (Planetoid)
            
            Nodes:2708
            
            Edges: 10556 (PyG stores edges as directed pairs, so undirected links appear twice)
            
            Node features: 1433
            
            Classes: 7
            
            Masks: train_mask, val_mask, test_mask

## **Project Structure (Notebooks)**

**1）PyG.ipynb — Setup & Dataset Familiarization**

            Loads Cora with Planetoid(...) and feature normalization (NormalizeFeatures()).
            
            Inspects data.x, data.edge_index, data.y, and mask splits.
            
            Confirms the environment and that a basic GCN pipeline runs correctly.

**2）Cora_MLP&GCN.ipynb — Baseline vs GNN (MLP vs GCN)**

            Implements and trains:
            
            MLP baseline: two fully connected layers (no graph structure).
            
            GCN model: two GCNConv layers with neighbor aggregation (uses graph structure).
            
            Typical training setup:

                        Loss: CrossEntropyLoss
                        
                        Optimizer: Adam (lr=0.01, weight_decay=5e-4)
                        
                        Training on train_mask, evaluation on test_mask
                        
                        Example result: Test Accuracy ≈ 0.814 (GCN)

Also includes a 2D visualization (e.g., t-SNE/UMAP) of model outputs/embeddings to inspect class separability in representation space.

**3) gcn_explain.ipynb — Node-Level Explainability (Feature + Edge Masks)**
   
Uses PyG’s Explainer to explain a specific node prediction and outputs:

            node_mask: importance over (node × feature dimension), shape (2708, 1433)
            
            edge_mask: importance over edges, shape (10556,)

A pairwise case study is performed:

            node_ok = 1709 (predicted correctly)
            
            node_bad = 1708 (predicted incorrectly)

For each node, the notebook extracts:

            Top feature dimensions (from node_mask) contributing to the node’s decision
            
            Top incident edges (from edge_mask) contributing to the node’s decision

**4) Edge_Explainability.ipynb — Edge/Neighbor Explanation + Fidelity Validation**

Focuses on **edge-level explanations for node predictions**, including:

**(a) Top incident edges for a node**

            Lists the node’s incident edges ranked by edge_mask.
            
            Converts edge IDs into readable (source → target) neighbor relations.

            Reads neighbor labels to analyze evidence consistency (homophily vs mixing).

**(b) Neighbor label distribution & homophily**

            Aggregates edge weights by neighbor labels.
            
            Computes a simple homophily indicator:
            
                  node_ok=1709: high homophily (neighbors mostly same class)
                  
                  node_bad=1708: low homophily (neighbors mixed across classes)

**(c) Fidelity tests (deletion / insertion)**

            Deletion: remove top-k important neighbors/edges and measure:
            
                  confidence drop on the original predicted class
                  
                  whether the predicted class flips
            
            Insertion: keep only top-k neighbors/edges and measure:

                  whether the prediction can be reproduced with a small explanatory subgraph

Observed patterns (from the recorded outputs):

            For node 1709 (correct):
            
                  removing a few key neighbors can significantly reduce confidence and may flip the class
                  
                  keeping top neighbors often preserves the original prediction with high confidence
            
            For node 1708 (incorrect):
            
                  important neighbors include multiple different labels (mixed evidence)
                  
                  removing several influential neighbors can change the predicted class (sometimes correcting it)


## **What Has Been Completed (Current Status)**

            Modeling
            
                  Trained MLP and GCN on Cora for node classification.
                  
                  Achieved a reasonable GCN performance (e.g., ~0.81 test accuracy).
            
            Explainability (Post-hoc)
            
                  Generated feature masks (node_mask) and extracted top feature dimensions for target nodes.
                  
                  Generated edge masks (edge_mask) and extracted top incident edges / key neighbors for target nodes.
                  
                  Conducted a case study on one correct and one incorrect node (1709 vs 1708).
            
            Explanation Validation
            
                  Implemented and reported deletion fidelity and insertion fidelity on the explained node’s neighborhood.

## **Environment**

            Example environment (as recorded):
            
            Python 3.10
            
            PyTorch 2.8.0 (CPU)
            
            PyG 2.7.0

## **How to Run**

            Install PyTorch and PyTorch Geometric following official instructions for your platform.
            
            Run notebooks in order:
            
                  PyG.ipynb
                  
                  Cora_MLP&GCN.ipynb
                  
                  gcn_explain.ipynb
                  
                  Edge_Explainability.ipynb


## **Next Steps (Possible Extensions)**

**1.The next extensions can be organized into three directions:**

      A. Explainer dimension (same model, different explainers)
      
            GNNExplainer (already used in this project)
      
            PGExplainer (learns a parameterized explainer; may be more stable but requires training)
      
            Integrated Gradients / Gradient-based attribution (more common for node-feature explanations)
      
            GraphLIME (local linear surrogate, LIME-style explanations)
      
      B. Model architecture dimension (same explainer, different GNN)
      
            GCN → GraphSAGE → GAT (attention provides an explainability signal, but note: attention ≠ explanation)
            
            Deeper GNNs + residual connections / JKNet (explanations may become more complex)
      
      C. Evaluation dimension (evaluation beyond fidelity)
      
            Comprehensiveness / Sufficiency (common concepts in explainability literature)
            
            Stability (whether masks remain stable across random seeds / perturbations)
            
            Plausibility (whether explanations align with domain priors; on Cora, homophily can serve as a weak prior)

**2.LLMs × GNN Explainability**

In current workflow, the explainer already outputs structured evidence:

      node_mask: which feature dimensions matter
      
      edge_mask: which incident edges / neighbors matter

      deletion/insertion: whether the explanation is causal for the prediction

However, the outputs are still **hard to read and hard to compare** across:

      different nodes (correct vs incorrect),
      
      different explainers,
      
      different models.

LLMs can be added as a **post-processing, analysis, and reporting layer** to make explanations:

      more readable,
      
      more comparable,
      
      more systematic,
            without claiming new causal evidence beyond fidelity tests.

**A. LLM as an “Explanation Summarizer” (reporting layer)**

      Input to LLM (structured, no free-form):
      
            node id, true label, predicted label, confidence
            
            top-N features (from node_mask): feature ids + scores
            
            top-N neighbors/edges (from edge_mask): (u→v), neighbor labels + scores
            
            deletion/insertion results (k, confidence drop, whether class flips)

      LLM outputs:

            a concise narrative:
      
                  “The model predicts class X mainly because…”
      
            an evidence table summary:
      
                  “Top supporting neighbors are mostly class …”
      
            a contrastive explanation:
      
                  “Compared with node 1709, node 1708 shows mixed neighbor evidence…”
      
      This directly upgrades your current outputs into a presentation-ready explanation.

**B. LLM as an “Explanation Comparator” (across nodes / explainers / models)**

      Once you run multiple explainers or models later, LLM can do alignment:
      
            Consensus features/edges: appear in top-k across methods
            
            Disagreement: edges/features only supported by one explainer
            
            Stability narrative: if top edges vary across random seeds
      
      Deliverable: a “comparison report”:
      
            “Which evidence is robust vs method-dependent?”

**C. LLM as an “Evaluation Auditor” (beyond fidelity)**

      You already compute fidelity. LLM can help interpret why fidelity behaves differently:
      
            If deletion causes a flip: summarize which neighbor removal caused it
            
            If insertion reproduces prediction: describe the minimal evidence subgraph
            
            Plausibility checks:
      
                  on Cora, homophily can be a weak prior:
      
                        “Explanation relies on same-label neighbors” (plausible)
                        
                        “Explanation relies heavily on different-label neighbors” (potential conflict)
      
      Important: the LLM does not decide correctness; it flags and organizes.

## **a clear and defensible contribution:**

## **LLM improves interpretability for humans, not model accuracy.**






















