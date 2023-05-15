import torch
import torch.nn as nn
from .gat import GATEmbedding

class DTINetworkCNN(nn.Module):
    """
    Custom Class for DTINetwork based on GraphDTA with CNN for protein sequence. For more information about GraphDTA, please refer to the following paper:
    GraphDTA: predicting drug–target binding affinity with graph neural networks.
    """
    def __init__(self,
                 prot_dim=1024,
                 in_feats=74,
                 graph_hidden_feats=[74, 128], # GraphDTA: [74, 128], Original: [32, 32]
                 graph_num_heads=[10, 1], # GraphDTA: [10, 1], Original: [4, 4]
                 dense_hidden_feats=[1024, 256], # GraphDTA: [1024, 256], Original: 64
                 dropout=0.2, # GraphDTA: 0.2
                 verbose=False, 
                 num_features_xt=25,
                 embed_dim=128,
                 n_filters=32):

        """
        Args:
            prot_dim (int): Protein embedding dimension.
            in_feats (int): Input feature size for molecules.
            graph_hidden_feats (list): Hidden feature sizes for GAT layers.
            graph_num_heads (list): Number of attention heads for GAT layers.
            dense_hidden_feats (list): Hidden feature sizes for dense layers.
            dropout (float): Dropout rate.
            verbose (bool): Whether to print out information.
            num_features_xt (int): Number of features for protein sequence.
            embed_dim (int): Embedding dimension for protein sequence.
            n_filters (int): Number of filters for 1D CNN of protein sequence.
        """
        super(DTINetworkCNN, self).__init__()
        self.verbose = verbose
        self.prot_dim = prot_dim

        self.mol_model = GATEmbedding(in_feats=in_feats, 
                                      hidden_feats=graph_hidden_feats,
                                      num_heads=graph_num_heads,
                                      dropouts=[dropout]*len(graph_hidden_feats))
        self.dense = nn.ModuleList()

        # CNN Module for protein sequence
        self.embedding_xt = nn.Embedding(num_features_xt + 1, embed_dim)
        self.conv_xt1 = nn.Conv1d(in_channels=1024, out_channels=n_filters, kernel_size=8)
        self.fc_xt1 = nn.Linear(32*121, 2*self.mol_model.gnn_out_feats)
        self.prot_fc = nn.Linear(self.prot_dim, 2*self.mol_model.gnn_out_feats)
        self.dense.append(nn.Linear(2*2*self.mol_model.gnn_out_feats, dense_hidden_feats[0]))
        for i in range(1, len(dense_hidden_feats)):
            self.dense.append(nn.Linear(dense_hidden_feats[i-1], dense_hidden_feats[i]))
        self.output = nn.Linear(dense_hidden_feats[-1], 1)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
           
    def forward(self, mol_graphs, target):
        """
        Args:
            mol_graphs (DGLGraph): A batch of DGLGraphs for molecules.
            target (torch.Tensor): A batch of protein sequences.
        Returns:
            torch.Tensor: A batch of predicted DTI binding affinity.
        """
        x_mol = self.mol_model(mol_graphs, in_feats=mol_graphs.ndata['h'], readout=True)
        if self.verbose: print("Molecule tensor shape:", x_mol.shape)
        embedded_xt = self.embedding_xt(target)
        conv_xt = self.conv_xt1(embedded_xt)
        conv_xt = self.activation(conv_xt)

        # flatten
        xt = conv_xt.view(-1, 32 * 121)
        x_prot = self.fc_xt1(xt)

        x = torch.cat((x_prot, x_mol), axis=1)
        for layer in self.dense:
            x = self.dropout(self.activation(layer(x)))
        return self.output(x)