
from cgi import test
import dgl
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from .dti_dataset_plm import DTIDatasetPLM
from .dti_dataset_plm import DTIDatasetPLMInference
from torch.utils.data._utils.collate import default_collate
import numpy as np
import os

def get_dataloaders(dataset, dist_dict, dataset_name, prot_model, target_scaler, seed, batch_size, world_size, rank, inference=False, pin_memory=False, num_workers=0):

    """
    Returns dataloaders for the training, validation, and test sets by loading and processing the data from the dataset object. Also loads the proteinb embeddings from the protein model.
    Args:
        dataset (TDC Dataset): TDC dataset object
        target_scaler (sklearn Scaler): Scaler object for scaling the target values
        seed (int): Random seed
        batch_size (int): Batch size
        world_size (int): Number of processes in the distributed training
        rank (int): Rank of the current process
        inference (bool): Whether to return dataloaders for inference
        pin_memory (bool): Whether to pin memory
        num_workers (int): Number of workers for the dataloader
    Returns:
        train_dataloader (DataLoader): Dataloader for the training set
        valid_dataloader (DataLoader): Dataloader for the validation set
        test_dataloader (DataLoader): Dataloader for the test set
        target_scaler (sklearn Scaler): Scaler object for scaling the target values
    """

    split = dataset.get_split()
    train_data = split["train"]
    valid_data = split["valid"]
    test_data = split["test"]

    train_dist_dict = dist_dict["train"]
    valid_dist_dict = dist_dict["valid"]
    test_dist_dict = dist_dict["test"]
    
    # Load protein embeddings
    PATH = './embeddings/protein_emb_train_{}_{}.npy'.format(prot_model, dataset_name)
    if not os.path.exists(PATH):
        print("Pre-computed Protein Embedding File does not exist. Please run the protein embedding script for dataset: {} and model: {} in data_processing first.".format(dataset_name, prot_model))
        print(PATH)
        return None, None, None, None
    protein_emb_train = np.load(PATH)

    PATH = './embeddings/protein_emb_valid_{}_{}.npy'.format(prot_model, dataset_name)
    if not os.path.exists(PATH):
        print("Pre-computed Protein Embedding File does not exist. Please run the protein embedding script for dataset: {} and model: {} in data_processing first.".format(dataset_name, prot_model))
        return None, None, None, None
    protein_emb_valid = np.load(PATH)

    PATH = './embeddings/protein_emb_test_{}_{}.npy'.format(prot_model, dataset_name)
    if not os.path.exists(PATH):
        print("Pre-computed Protein Embedding File does not exist. Please run the protein embedding script for dataset: {} and model: {} in data_processing first.".format(dataset_name, prot_model))
        return None, None, None, None
    protein_emb_test = np.load(PATH)

    train_data.Y = target_scaler.fit_transform(train_data.Y.values.reshape(-1, 1))
    
    # Create dataset and dataloader from PyTDC 
    train_drugs = train_data.Drug.tolist()
    train_targets = protein_emb_train
    train_affinities = train_data.Y.tolist()
    if inference:
        train_targets_seq = train_data.Target.tolist()
        train_dataset = DTIDatasetPLMInference(train_drugs, train_targets, train_targets_seq, train_affinities, train_dist_dict)
    else:
        train_dataset = DTIDatasetPLM(train_drugs, train_targets, train_affinities, train_dist_dict)
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True, drop_last=False)
    if inference:
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, pin_memory=pin_memory, num_workers=num_workers, sampler=train_sampler, collate_fn=collate_fn_inference)
    else:
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, pin_memory=pin_memory, num_workers=num_workers, sampler=train_sampler, collate_fn=collate_fn)
    
    # Create dataset and dataloader from PyTDC 
    valid_drugs = valid_data.Drug.tolist()
    valid_targets = protein_emb_valid
    valid_affinities = valid_data.Y.tolist()
    if inference:
        valid_targets_seq = valid_data.Target.tolist()
        valid_dataset = DTIDatasetPLMInference(valid_drugs, valid_targets, valid_targets_seq, valid_affinities, valid_dist_dict)
    else:
        valid_dataset = DTIDatasetPLM(valid_drugs, valid_targets, valid_affinities, valid_dist_dict)
    valid_sampler = DistributedSampler(valid_dataset, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False)

    if inference:
        valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, pin_memory=pin_memory, num_workers=num_workers, sampler=valid_sampler, collate_fn=collate_fn_inference)
    else:
        valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, pin_memory=pin_memory, num_workers=num_workers, sampler=valid_sampler, collate_fn=collate_fn)
    
    # Create dataset and dataloader from PyTDC 
    test_drugs = test_data.Drug.tolist()
    test_targets = protein_emb_test
    test_affinities = test_data.Y.tolist()
    if inference:
        test_targets_seq = test_data.Target.tolist()
        test_dataset = DTIDatasetPLMInference(test_drugs, test_targets, test_targets_seq, test_affinities, test_dist_dict)
    else:
        test_dataset = DTIDatasetPLM(test_drugs, test_targets, test_affinities, test_dist_dict)
    test_sampler = DistributedSampler(test_dataset, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False)

    if inference:
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, pin_memory=pin_memory, num_workers=num_workers, sampler=test_sampler, collate_fn=collate_fn_inference)
    else:
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, pin_memory=pin_memory, num_workers=num_workers, sampler=test_sampler, collate_fn=collate_fn)

    return train_dataloader, valid_dataloader, test_dataloader, target_scaler

def collate_fn(batch):
    """
    Args:
        batch (list): List of tuples of (graph, label)
    Returns:
        batch_graphs (dgl.DGLGraph): Batched graph
        protein_sequences (torch.Tensor): Protein Sequences
        labels (torch.Tensor): Labels
    """
    mol_graphs, prot_embeddings, dist_dict, labels = tuple(zip(*batch))
    return dgl.batch(mol_graphs), default_collate(prot_embeddings), default_collate(dist_dict), default_collate(labels)


def collate_fn_inference(batch):
    """
    Args:
        batch (list): List of tuples of (graph, label)
    Returns:
        batch_graphs (dgl.DGLGraph): Batched graph
        protein_embeddings (torch.Tensor): Protein Embeddings
        targets_sequences (str): Protein Sequences
        labels (torch.Tensor): Labels
    """
    mol_graphs, prot_embeddings, targets_sequences, dist_dict, labels = tuple(zip(*batch))
    return dgl.batch(mol_graphs), default_collate(prot_embeddings), default_collate(targets_sequences), default_collate(dist_dict), default_collate(labels) 