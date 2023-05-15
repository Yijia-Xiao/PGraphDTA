import dgl
from torch.utils.data import Dataset
from dgllife.utils import CanonicalAtomFeaturizer, CanonicalBondFeaturizer, smiles_to_bigraph
import numpy as np

class DTIDatasetPLM(Dataset):
    def __init__(self, drugs, prot_embeds, affinities, dist_dict):
        """
        Args:
            drugs (list): List of SMILES strings
            prot_embeds (list): List of protein embeddings
            affinities (list): List of binding affinities
        """
        self.drugs = drugs
        self.prot_embeds = prot_embeds
        self.affinities = affinities
        self.atom_featurizer = CanonicalAtomFeaturizer()
        self.bond_featurizer = CanonicalBondFeaturizer()
        self.dist_dict = dist_dict

    def __len__(self):
        return len(self.affinities)

    def __getitem__(self, idx):
        """
        Args:
            idx (int): Index
        Returns:
            graph (dgl.DGLGraph): Graph
            prot_embedding (torch.Tensor): Protein embedding
            label (torch.Tensor): Label
        """
        smiles = self.drugs[idx]
        mol_graph = smiles_to_bigraph(smiles, 
                                      node_featurizer=self.atom_featurizer, 
                                      edge_featurizer=self.bond_featurizer,
                                      )
        mol_graph = dgl.add_self_loop(mol_graph)
        prot_embedding = self.prot_embeds[idx]
        dist = self.dist_dict[idx]
        label = self.affinities[idx]
        return mol_graph, prot_embedding, dist, label


class DTIDatasetPLMInference(Dataset):
    def __init__(self, drugs, prot_embeds, targets_seq, affinities, dist_dict):
        """
        Args:
            drugs (list): List of SMILES strings
            prot_embeds (list): List of protein embeddings
            affinities (list): List of binding affinities
        """
        self.drugs = drugs
        self.prot_embeds = prot_embeds
        self.targets_seq = targets_seq
        self.affinities = affinities
        self.atom_featurizer = CanonicalAtomFeaturizer()
        self.bond_featurizer = CanonicalBondFeaturizer()
        self.dist_dict = dist_dict

    def __len__(self):
        return len(self.affinities)

    def __getitem__(self, idx):
        """
        Args:
            idx (int): Index
        Returns:
            graph (dgl.DGLGraph): Graph
            prot_embedding (torch.Tensor): Protein embedding
            targets_sequence (str): Target sequence
            label (torch.Tensor): Label
        """
        smiles = self.drugs[idx]
        mol_graph = smiles_to_bigraph(smiles, 
                                      node_featurizer=self.atom_featurizer, 
                                      edge_featurizer=self.bond_featurizer,
                                      )
        mol_graph = dgl.add_self_loop(mol_graph)
        prot_embedding = self.prot_embeds[idx]
        targets_sequence = self.targets_seq[idx]
        dist = self.dist_dict[idx]
        label = self.affinities[idx]
        return mol_graph, prot_embedding, targets_sequence, dist, label