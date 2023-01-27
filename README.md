# GAT-DL-DIY
Final project for the course DL-DIY. Implementation of graph attention network (GAT). 

We implemented, using Pytorch, the graph attention layer proposed by [Velickovic et al.](https://arxiv.org/pdf/1710.10903.pdf). Our implementation is largely based in [this repo](https://github.com/gordicaleksa/pytorch-GAT). 

We tested the model in transductive learning settings, in the citation datasets Cora, Pubmed and Citeseer, reaching accuracies around 75 %. 

We implemented the Nueral tree representation with the objective of applying the GAT network to this more structured representation, but we faced some issues in the manipulation of this structure by GAT. 
