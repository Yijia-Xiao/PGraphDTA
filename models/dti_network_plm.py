import torch
import torch.nn as nn
from .gat import GATEmbedding
import pudb

class DTINetworkPLM(nn.Module):
    """
    Custom Class for DTINetwork based on GraphDTA with Protein Language Models (PLM) [SeqVec/DistilProtBERT/ProtBERT] for protein sequence. For more information about GraphDTA, please refer to the following paper:
    GraphDTA: predicting drug–target binding affinity with graph neural networks.
    """
    def __init__(self,
                 prot_model,
                 prot_dim=1024,
                 in_feats=74,
                 graph_hidden_feats=[74, 128], # GraphDTA: [74, 128], Original: [32, 32]
                 graph_num_heads=[10, 1], # GraphDTA: [10, 1], Original: [4, 4]
                 dense_hidden_feats=[1024, 256], # GraphDTA: [1024, 256], Original: 64
                 dropout=0.2, # GraphDTA: 0.2
                 verbose=False):

        """
        Args:
            prot_model (string): Protein language model.
            prot_dim (int): Protein embedding dimension.
            in_feats (int): Input feature size for molecules.
            graph_hidden_feats (list): Hidden feature sizes for GAT layers.
            graph_num_heads (list): Number of attention heads for GAT layers.
            dense_hidden_feats (list): Hidden feature sizes for dense layers.
            dropout (float): Dropout rate.
            verbose (bool): Whether to print out information.
        """
        super(DTINetworkPLM, self).__init__()
        self.verbose = verbose
        self.prot_model = prot_model
        self.prot_dim = prot_dim

        self.mol_dim = 64

        self.mol_model = GATEmbedding(in_feats=in_feats, 
                                      hidden_feats=graph_hidden_feats,
                                      num_heads=graph_num_heads,
                                      dropouts=[dropout]*len(graph_hidden_feats))
        self.dense = nn.ModuleList()

        self.dist_fc = nn.Linear(self.mol_dim * self.mol_dim, 2*self.mol_model.gnn_out_feats)
        self.prot_fc = nn.Linear(self.prot_dim, 2*self.mol_model.gnn_out_feats)
        self.dense.append(nn.Linear(2*3*self.mol_model.gnn_out_feats, dense_hidden_feats[0]))
        for i in range(1, len(dense_hidden_feats)):
            self.dense.append(nn.Linear(dense_hidden_feats[i-1], dense_hidden_feats[i]))
        self.output = nn.Linear(dense_hidden_feats[-1], 1)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
           
    def forward(self, mol_graphs, target, dist_dicts):
        """
        Args:
            mol_graphs (DGLGraph): A batch of DGLGraphs for molecules.
            target (torch.Tensor): A batch of protein sequences.
        Returns:
            torch.Tensor: A batch of predicted DTI binding affinity.
        """
        x_mol = self.mol_model(mol_graphs, in_feats=mol_graphs.ndata['h'], readout=True)
        if self.verbose: print("Molecule tensor shape:", x_mol.shape)
        x_prot = self.prot_fc(target)
        # pu.db
        # # print("Protein tensor shape:", x_prot.shape)
        # print("Distance tensor shape:", dist_dicts.shape)
        # # print dtype of dist_dicts
        # print("Distance tensor dtype:", dist_dicts.dtype)
        # # print dist_fc nn layer
        # print("Distance fc layer shape:", self.dist_fc)
        # # print dtype of dist_fc nn layer
        # print("Distance fc layer dtype:", self.dist_fc.weight.dtype)
        x_dist = self.dist_fc(dist_dicts)

        x = torch.cat((x_prot, x_mol, x_dist), axis=1)
        for layer in self.dense:
            x = self.dropout(self.activation(layer(x)))
        return self.output(x)