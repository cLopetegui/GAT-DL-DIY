#!/usr/bin/env python
# coding: utf-8

# In[5]:


import torch
import torch.nn as nn
import sys
sys.path.append('../')
from utils.layers import GAT_layer


# In[7]:


class GAT(nn.Module):
    def __init__(self,num_layers,num_nodes,num_features_per_layer,num_heads_per_layer,dropout=0.6):
        super().__init__()
        
        layers=[]
        
        for i in range(num_layers):
            if i==num_layers-1:
                self.concat=False
                self.activation=nn.Softmax(dim=-1)
            else: 
                self.concat=True
                self.activation=nn.ELU()
            if i==0:
                layers.append(GAT_layer(num_nodes,num_features_per_layer[i],num_features_per_layer[i+1],num_heads_per_layer[i],activation=self.activation,concat=self.concat,dropout_prob=dropout))
            else: 
                layers.append(GAT_layer(num_nodes,num_features_per_layer[i]*num_heads_per_layer[i-1],num_features_per_layer[i+1],num_heads_per_layer[i],activation=self.activation,concat=self.concat,dropout_prob=dropout))
        self.gat_network=nn.Sequential(*layers)
    def forward(self,data):  #data consists of (input_features, edge_index)
        return self.gat_network(data)
    
        


# In[ ]:




