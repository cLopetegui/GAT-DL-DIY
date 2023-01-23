#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn


# In[59]:


class GAT_layer(nn.Module):
    """
    Implementation of the Graph attentional layer (the only layer used througout the architecture):
    
    Input: h={h1,...,hN}, nodes features, where hi \in R^F, with F the number of features and N the number of nodes
    
    Output: h'={h'1,...,h'N}, new set of node features, where h'i\in R^F', with F' the new number of features. 
    
    Structure of the layer: 
    1- Shared Linear layer: weights W\in R^F'xF | acts parallel on each node
    2- Self-attention mechanism: 
        - e_ij=a(W hi,W hj)=LeakyReLu(a^T.(W hi||W hj)) only for j in the neighborhood of i (masked attention)
        - alpha_ij=softmax(e_ij)
    3- Compute the new features: h'i=sigmoid(\sum_{j\in Neighbor(i)}alpha_ij W h_j)
    
    We may allow for multi-headed attention, where K independent attention mechanisms run each time, giving an output 
    which is the concatenation of all of them and then lives in R^{K F'}
    
    """
    def __init__(self,num_nodes,num_in_features,num_out_features,num_heads,activation,concat=True,dropout_prob=0.6):
        super().__init__()
        self.num_in_features=num_in_features
        self.num_out_features=num_out_features
        self.num_nodes=num_nodes
        self.num_heads=num_heads
        self.concat=concat # if True: concatenate the output of each attention head, if False (classification step): average the output of each attention head
        
        
        self.linear=nn.Linear(self.num_in_features,self.num_heads*self.num_out_features)
        #Initialize the weights of the linear projection
        nn.init.xavier_uniform_(self.linear.weight)
        
        
        self.leakyReLu=nn.LeakyReLU(0.2)
        self.softmax=nn.Softmax(dim=-1)
        
        self.dropout=nn.Dropout(p=dropout_prob)
        
        self.activation=activation
        
        #for the attention mechanism, we divide the attention weight in two parts, one for the source(i) and one for the target (j)
        
        
        self.attention_weight_source=nn.Parameter(torch.Tensor(1,num_heads,num_out_features))
        self.attention_weight_target=nn.Parameter(torch.Tensor(1,num_heads,num_out_features))
        
        nn.init.xavier_uniform_(self.attention_weight_source)
        nn.init.xavier_uniform_(self.attention_weight_target)
    
    def forward(self,data):
        
        h_in,edge_index=data
        
        # dim(h_in) = (N,F)
        # shape(h_proj) = (N,num_heads,num_out_features)
        h_in=self.dropout(h_in)
        h_proj=self.linear(h_in).view(h_in.shape[0],self.num_heads,self.num_out_features)
        
        
        # Attention mechanism:
        # the goal is to get for each i,j a number e_ij (for each head)
        # we do first h.a[1:F_out]: (N,N_heads,F')x (1,N_heads,F')->(N,N_heads,1) 
        att_coefficients_contrib_source=torch.sum(h_proj*self.attention_weight_source,dim=-1)
        att_coefficients_contrib_target=torch.sum(h_proj*self.attention_weight_target,dim=-1)
        
        #The idea now is the following: 
        #From the data we have edge_index which has a shape (2,E), E number of edges so that edge_index[0] stores the index of
        #the source nodes
        #and edge_index[1] stores the index of the target nodes. So, now, what we do is to lift h_proj,att_coeff...,att_coeff...
        #from (N,N_heads,...)->(E,N_heads,...) with ...=1 for the att weights and =F' for the h_proj_lift
        
        h_proj_lift=h_proj.index_select(0,edge_index[0])
        
        att_coefficients_cotrib_source_lift=att_coefficients_contrib_source.index_select(0,edge_index[0])
        att_coefficients_contrib_target_lift=att_coefficients_contrib_target.index_select(0,edge_index[1])
        
        #Now we apply the leaky relu to the sum of both contributions 
        
        att_weights_per_edge=self.leakyReLu(att_coefficients_contrib_target_lift+att_coefficients_cotrib_source_lift)
        
        
        
        #Now we need to apply a soft max that normalizes over all neighbors of node i. 
        att_weights_per_edge_soft_maxed=self.soft_max_att_weights_over_neighbors(att_weights_per_edge,edge_index)

        att_weights_per_edge_soft_maxed=self.dropout(att_weights_per_edge_soft_maxed)
        #Now we can add the contribution of neighbors to compute the new values of the features
        
        h_new=self.get_weighted_contirbutions_new_features(h_proj_lift,att_weights_per_edge_soft_maxed,edge_index)
        
        return (h_new,edge_index)
        
    def soft_max_att_weights_over_neighbors(self,att_weights,edge_index):
        exp_att_weights=att_weights.exp()
        soft_max_weights=torch.clone(exp_att_weights)
        already_computed=torch.zeros(edge_index.shape[0])
        for i in range(edge_index.shape[0]):
            equal_index=[i]
            if already_computed[i]==1:
                continue
            already_computed[i]=1
            for j in range(i,edge_index.shape[0]):
                if edge_index[0][i]==edge_index[0][j]:
                    equal_index.append(j)
                    already_computed[j]=1
            neighborhood_weights_denominator=exp_att_weights[equal_index].sum(dim=0).unsqueeze(dim=0)
            soft_max_weights[equal_index]=soft_max_weights[equal_index]/(neighborhood_weights_denominator+1e-16)
        return soft_max_weights.unsqueeze(dim=-1)
    
    def get_weighted_contirbutions_new_features(self,h_proj_lift,att_weights,edge_index):
        # Here we just multiply the lifted features (shape (E,N_heads,F')) with the att weights already passed through the softmax
        # which have shape (E,N_heads,1) -> the output should have shape (E,N_heads,F')
        
        mult_features_weights=att_weights*h_proj_lift
        
        # Now we need to get the new features by adding over the neighbors of each node
        
        new_features=torch.zeros((self.num_nodes,self.num_heads,self.num_out_features),dtype=mult_features_weights.dtype)
        
        for i in range(self.num_nodes):
            equal_index=[]
            for j in range(edge_index[0].shape[0]):
                if edge_index[0][j]==i:
                    equal_index.append(j)
            #Now compute the new features for node i 
            new_features[i]=mult_features_weights[equal_index].sum(dim=0).unsqueeze(dim=0) # input: (E,Nheads,F')-> intermediate (E',Nheads,F') -> output (1,Nheads,F')
        if self.concat: 
            new_features_sigm=self.activation(new_features)
            new_features_output=new_features.view(self.num_nodes,self.num_heads*self.num_out_features)
            return new_features_output

        new_features_average=new_features.sum(dim=1)/self.num_heads # (N,N_heads,F') -> (N,F')
        new_features_output=self.activation(new_features_average)
        
        return new_features_output
    

        
        
        

