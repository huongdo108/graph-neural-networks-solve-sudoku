import time
import torch
import numpy as np
import matplotlib.pyplot as plt

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

import tools
import tests

class GNN(nn.Module):
    def __init__(self,device, n_iters=7, n_node_features=10, n_node_inputs=9, n_edge_features=11, n_node_outputs=9):
        """
        Args:
          n_iters: Number of graph iterations.
          n_node_features: Number of features in the states of each node.
          n_node_inputs: Number of inputs to each graph node (on each graph iteration).
          n_edge_features: Number of features in the messages sent along the edges of the graph (produced
              by the message network).
          n_node_outputs: Number of outputs produced by at each node of the graph.
        """
        super(GNN, self).__init__()
        self.device = device
        self.n_iters = n_iters
        self.n_node_features = n_node_features
        self.n_node_inputs = n_node_inputs
        self.n_edge_features = n_edge_features
        self.n_node_outputs = n_node_outputs
        self.msg_net = nn.Sequential(
            nn.Linear(2*n_node_features, 96),
            nn.ReLU(),
            nn.Linear(96, 96),
            nn.ReLU(),
            nn.Linear(96, n_edge_features),
        )
        
        self.gru = nn.GRU(input_size=n_node_inputs+n_edge_features, hidden_size=n_node_features)
        self.out = nn.Linear(n_node_features, n_node_outputs)
    def forward(self, node_inputs, src_ids, dst_ids): #
        """
        Args:
          node_inputs of shape (n_nodes, n_node_inputs): Tensor of inputs to every node of the graph.
          src_ids of shape (n_edges): Indices of source nodes of every edge.
          dst_ids of shape (n_edges): Indices of destination nodes of every edge.
          
        Returns:
          outputs of shape (n_iters, n_nodes, n_node_outputs): Outputs of all the nodes at every iteration of the
              graph neural network. 7 , 81, 9
        """
        self = self.to(self.device)
        node_inputs=node_inputs.to(self.device)
        src_ids = src_ids.to(self.device)
        dst_ids = dst_ids.to(self.device)
        n_nodes = node_inputs.shape[0]
        n_edges = len(src_ids)
        hidden =  torch.zeros(n_nodes,self.n_node_features).unsqueeze(dim=0)
        hidden = hidden.to(self.device)
        outputs = torch.zeros(self.n_iters,n_nodes,self.n_node_outputs)
        
        for i in range(self.n_iters):

            src_message = hidden.squeeze(dim=0)[src_ids]
            dst_message = hidden.squeeze(dim=0)[dst_ids]
            input_msg_net = torch.cat((src_message,dst_message),dim=1)
            input_msg_net = input_msg_net.to(self.device)
            message = self.msg_net(input_msg_net)
            aggregated_message = torch.zeros(n_nodes,self.n_edge_features)
            aggregated_message = aggregated_message.to(self.device)
            message = message.to(self.device)      
            aggregated_message = aggregated_message.index_add_(0, dst_ids, message)
            concat_message = torch.cat((node_inputs,aggregated_message),dim=1).unsqueeze(dim=0) 
            output,hidden = self.gru(concat_message,hidden)
            out = self.out(output.squeeze(dim=0))
            outputs[i] = out

        return outputs    