import torch
import torch.nn as nn
from dgllife.model import GAT
from dgllife.model.readout.weighted_sum_and_max import WeightedSumAndMax
from torch.nn.utils.rnn import pad_sequence

class GATEmbedding(nn.Module):
    """
    GAT Network Class based on GraphDTA. For more information about GraphDTA, please refer to the following paper:
    GraphDTA: predicting drug–target binding affinity with graph neural networks.
    """
    def __init__(self,
                 in_feats, 
                 hidden_feats,
                 num_heads,
                 dropouts):
        """
        Args:
            in_feats (int): Input feature size for molecules.
            hidden_feats (list): Hidden feature sizes for GAT layers.
            num_heads (list): Number of attention heads for GAT layers.
            dropouts (list): Dropout rate for GAT layers.
        """
        super(GATEmbedding, self).__init__()

        self.gnn = GAT(in_feats, 
                       hidden_feats=hidden_feats, 
                       num_heads=num_heads,
                       feat_drops=dropouts,
                       attn_drops=dropouts)
        if self.gnn.agg_modes[-1] == 'flatten':
            gnn_out_feats = self.gnn.hidden_feats[-1] * self.gnn.num_heads[-1]
        else:
            gnn_out_feats = self.gnn.hidden_feats[-1]
        self.gnn_out_feats = gnn_out_feats
        self.readout = WeightedSumAndMax(gnn_out_feats)

    def forward(self, g, in_feats, readout=False):
        """
        Args:
            g (DGLGraph): A DGLGraph for a molecule.
            in_feats (torch.Tensor): Input node features.
            readout (bool): Whether to perform readout.
        Returns:
            torch.Tensor: The output of GAT network.
        """
        node_feats = self.gnn(g, in_feats)
        if readout:
            return self.readout(g, node_feats)
        else:
            batch_num_nodes = g.batch_num_nodes().tolist()
            return pad_sequence(torch.split(node_feats, batch_num_nodes, dim=0), batch_first=True)