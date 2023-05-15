import dgl
from torch.utils.data import Dataset
from dgllife.utils import CanonicalAtomFeaturizer, CanonicalBondFeaturizer, smiles_to_bigraph
import numpy as np

class DTIDatasetCNN(Dataset):
    """
    Custom Dataset class for DTI data to the trained using CNN.
    """
    def __init__(self, drugs, targets, affinities, inference=False):
        """
        Args:
            drugs (list): List of SMILES strings
            targets (list): List of protein sequences
            affinities (list): List of binding affinities
        """
        self.drugs = drugs
        self.targets = targets
        self.affinities = affinities
        self.inference = inference
        self.atom_featurizer = CanonicalAtomFeaturizer()
        self.bond_featurizer = CanonicalBondFeaturizer()

        self.seq_voc = "ABCDEFGHIKLMNOPQRSTUVWXYZ"
        self.seq_dict = {v:(i+1) for i,v in enumerate(self.seq_voc)}
        self.seq_dict_len = len(self.seq_dict)
        self.max_seq_len = 1024

    def __len__(self):
        return len(self.affinities)

    def __getitem__(self, idx):
        """
        Args:
            idx (int): Index
        Returns:
            graph (dgl.DGLGraph): Graph
            protein_sequence (np.ndarray): Protein sequence
            label (torch.Tensor): Label
        """
        smiles = self.drugs[idx]
        mol_graph = smiles_to_bigraph(smiles, 
                                      node_featurizer=self.atom_featurizer, 
                                      edge_featurizer=self.bond_featurizer,
                                      )
        mol_graph = dgl.add_self_loop(mol_graph)
        sequence = self.targets[idx]
        prot_seq_embed = np.zeros(self.max_seq_len)
        for i, ch in enumerate(sequence[:self.max_seq_len]): 
            prot_seq_embed[i] = self.seq_dict[ch]
        label = self.affinities[idx]
        if self.inference:
            return mol_graph, prot_seq_embed, sequence, label
        return mol_graph, prot_seq_embed, label
