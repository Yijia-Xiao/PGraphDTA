import os
import time
import pickle
import numpy as np
from tqdm import tqdm
import argparse
import pudb

# PyTorch
import torch
import torch.nn as nn
import torch.optim as optim

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from sklearn.preprocessing import StandardScaler, FunctionTransformer
from scipy.stats import pearsonr

from data_processing.dti_dataloader_cnn import get_dataloaders as get_dataloaders_cnn
from data_processing.dti_dataloader_plm import get_dataloaders as get_dataloaders_plm
from models.dti_network_cnn import DTINetworkCNN
from models.dti_network_plm import DTINetworkPLM

from rdkit import Chem
from rdkit.Geometry import Point3D


def init_process_group(world_size, rank):
    """
    Args:
        world_size (int): Number of processes participating in the job.
        rank (int): Rank of the current process.
    Initializes the distributed backend which will take care of synchronizing GPUs.
    """
    dist.init_process_group(
        # backend='gloo',     # change to 'nccl' for multiple GPUs
        backend='nccl',
        init_method='tcp://127.0.0.1:12345',
        world_size=world_size,
        rank=rank)

###############################################################################
# To ensure same initial model parameters across processes, we need to set the
# same random seed before model initialization. Once we construct a model
# instance, we wrap it with :func:`~torch.nn.parallel.DistributedDataParallel`.
#

def init_model(seed, device, prot_model, **kwargs):
    """
    Args:
        seed (int): Random seed used to initialize model weights.
        device (torch.device): Device on which the model will be allocated.
        prot_model (str): Protein model choice
        **kwargs: Keyword arguments to pass to the model constructor.
    Returns:
        model (torch.nn.Module): DistributedDataParallel model instance.
    """
    torch.manual_seed(seed)
    
    if prot_model.lower() == 'cnn':
        model = DTINetworkCNN(**kwargs)
    else:
        model = DTINetworkPLM(prot_model, **kwargs)
    model.to(device)
    
    if device.type == 'cpu':
        model = DistributedDataParallel(model,
                                        find_unused_parameters=True)
    else:
        model = DistributedDataParallel(model, 
                                        device_ids=[device], 
                                        output_device=device, 
                                        find_unused_parameters=True)

    return model

def evaluate(model, prot_model, dataloader, target_scaler, device):
    """
    Args:
        model (torch.nn.Module): DistributedDataParallel model instance.
        prot_model (str): Protein model choice
        dataloader (torch.utils.data.DataLoader): Validation dataloader.
        target_scaler (sklearn.preprocessing.StandardScaler): Scaler for the target values.
        device (torch.device): Device on which the model will be allocated.
    Returns:
        mse (float): Mean squared error.
        pcc (float): Pearson correlation coefficient.
    """
    model.eval()
    y_true_list = []
    y_pred_list = []
    total_samples = 0
    for i, batch in enumerate(dataloader):
        mol_graphs, protein_embeds, dist_dicts, labels = batch
        y_true = labels.unsqueeze(1).to(device)
        total_samples += y_true.shape[0]

        mol_graphs = mol_graphs.to(device)
        # Convert dist_dicts to tensors
        # dist_dicts = torch.stack([torch.FloatTensor(dist_dict) for dist_dict in dist_dicts])
        dist_dicts = torch.Tensor(dist_dicts).float()
        dist_dicts = dist_dicts.to(device)
        if prot_model.lower() == 'cnn':
            protein_embeds = protein_embeds.long().to(device)
        else:
            protein_embeds = torch.FloatTensor(protein_embeds).to(device)
        y_pred = model(mol_graphs, protein_embeds, dist_dicts)
        y_true_list.append(y_true.squeeze(1).detach().cpu().numpy())
        y_pred_list.append(y_pred.squeeze(1).detach().cpu().numpy())

    y_true_final = np.concatenate(y_true_list)
    y_pred_final = target_scaler.inverse_transform(np.concatenate(y_pred_list).reshape(-1, 1)).flatten()
    mse = (np.square(y_pred_final - y_true_final)).mean()
    pcc = pearsonr(y_true_final, y_pred_final)[0]
    return mse, pcc


def train(model, prot_model, optimizer, loss_fn, train_loader, val_loader, target_scaler, 
        epochs=20, max_batches_per_epoch=1000, save_dir="", device="cpu"):

    """
    Args:
        model (torch.nn.Module): DistributedDataParallel model instance.
        prot_model (str): Protein model choice
        optimizer (torch.optim.Optimizer): Optimizer instance.
        loss_fn (torch.nn.Module): Loss function.
        train_loader (torch.utils.data.DataLoader): Training dataloader.
        val_loader (torch.utils.data.DataLoader): Validation dataloader.
        target_scaler (sklearn.preprocessing.StandardScaler): Scaler for the target values.
        epochs (int): Number of training epochs.
        max_batches_per_epoch (int): Maximum number of batches to process per epoch.
        save_dir (str): Directory to save model checkpoints.
        device (torch.device): Device on which the model will be allocated.
    
    Trains the model and saves the last model checkpoint.
    """
    train_losses = []
    val_losses = []
    val_pccs = []
    for epoch in range(1, epochs+1):
        model.train()
        training_loss = 0.0
        train_samples = 0
        train_batches = 0
        min_train_loss = float('inf')
        for i, batch in enumerate(train_loader):
            optimizer.zero_grad()
            mol_graphs, protein_embeds, dist_dicts, labels = batch
            y_true = labels.unsqueeze(1).float().to(device)
            train_samples += y_true.shape[0]
            train_batches += 1

            mol_graphs = mol_graphs.to(device)
            # Convert dist_dicts to tensors
            # dist_dicts = torch.stack([torch.FloatTensor(dist_dict) for dist_dict in dist_dicts])
            dist_dicts = torch.Tensor(dist_dicts).float()
            dist_dicts = dist_dicts.to(device)

            if prot_model.lower() == 'cnn':
                protein_embeds = protein_embeds.long().to(device)
            else:
                protein_embeds = torch.FloatTensor(protein_embeds).to(device)

            y_pred = model(mol_graphs, protein_embeds, dist_dicts).float()
            loss = loss_fn(y_pred, y_true)
            loss.backward()
            optimizer.step()
            training_loss += loss.cpu().item()
            if (i+1) >= max_batches_per_epoch:
                break
            elif (i+1)%100 == 0:
                print('Device: {}, Batch: {}, MSE: {:.4f}'.format(device, i+1, training_loss/train_batches))
            else:
                continue
        training_loss /= train_batches
        val_mse, val_pcc = evaluate(model, prot_model, val_loader, target_scaler, device)

        train_losses.append(training_loss)
        val_losses.append(val_mse)
        val_pccs.append(val_pcc)

        print('Device: {}, Epoch: {}, Training MSE: {:.4f}, Validation MSE: {:.4f}, Validation PCC: {:.4f}'.format(device, epoch, training_loss, val_mse, val_pcc))
        # Save the model for best epoch
        if training_loss < min_train_loss:
            min_train_loss = training_loss
            PATH = '{}{}_{}.pt'.format(save_dir, epoch, training_loss)
            torch.save(model.state_dict(), PATH)

    # Save the model
    PATH = '{}{}_{}.pt'.format(save_dir, device, val_pcc)
    torch.save(model.state_dict(), PATH)

    # Save the losses to a file
    loses = {'train_losses': train_losses, 'val_losses': val_losses, 'val_pccs': val_pccs}
    PATH = '{}{}_{}.pkl'.format(save_dir, device, val_pcc)
    with open(PATH, 'wb') as f:
        pickle.dump(loses, f)

# CLI
def parse_args():
    # Argument Parser for command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=512, help='Batch size')
    parser.add_argument('--epochs' ,type=int, default=1500, help='Number of epochs')
    parser.add_argument('--max_epochs_per_batch' ,type=int, default=500, help='Number of epochs per batch')
    parser.add_argument('--lr', type=float, default=0.0005, help='Learning rate')
    parser.add_argument('--num_workers', type=int, default=0, help='Number of workers')
    parser.add_argument('--num_gpus', type=int, default=1, help='Number of GPUs')
    parser.add_argument('--max_prot_len', type=int, default=1024, help='Maximum protein length')
    parser.add_argument('--scale_target', type=bool, default=True, help='Scale target')
    parser.add_argument('--dataset_choice', type=int, default=0, help='Dataset choice')
    parser.add_argument('--prot_lm_model_choice', type=int, default=0, help='Protein Language Model choice')
    parser.add_argument('--save_dir', type=str, default="./models/", help='Directory to save models')

    opt = parser.parse_args()
    return opt

###############################################################################
# Define the main function for each process.
#

def main(rank, world_size, args, dataset, dist_dict, seed=0):
    """
    Args:
        rank (int): Rank of the process.
        world_size (int): Number of processes.
        args (argparse.Namespace): Arguments.
        dataset (torch.utils.data.Dataset): Dataset.
        seed (int): Random seed.
    
    Main Function for each process which trains the model and returns all necesssary metrics.
    """
    init_process_group(world_size, rank)
    if torch.cuda.is_available():
        device = torch.device('cuda:{:d}'.format(rank))
        torch.cuda.set_device(device)
    else:
        device = torch.device('cpu')
    print("Spawning process on device", 'cuda:{:d}'.format(rank))

    torch.cuda.empty_cache()
    model = init_model(seed, 
                       device,
                       prot_model=args.prot_model,
                       prot_dim=args.max_prot_len,
                       graph_hidden_feats=[74, 128], # GraphDTA: [74, 128], Original: [32, 32]
                       graph_num_heads=[10, 1], # GraphDTA: [10, 1], Original: [4, 4]
                       dense_hidden_feats=[1024, 256], # GraphDTA: [1024, 256], Original: 64
                       dropout=0.2, # GraphDTA: 0.2
                      )
    
    lr = args.lr
    criterion = nn.MSELoss() # Loss function
    optimizer = optim.Adam(model.parameters(), lr=lr) # Optimizer
    
    scale_target = args.scale_target
    if scale_target:
        target_scaler = StandardScaler() # Standard scaling of target
    else:
        target_scaler = FunctionTransformer(lambda x: x) # Do not scale target

    batch_size = args.batch_size # Per-GPU batch size for training
    if args.prot_model.lower() == 'cnn':
        train_loader, val_loader, test_loader, target_scaler = get_dataloaders_cnn(dataset,
                                                                           target_scaler,
                                                                           seed,
                                                                           batch_size=batch_size,
                                                                           world_size=world_size,
                                                                           rank=rank)
    else:
        train_loader, val_loader, test_loader, target_scaler = get_dataloaders_plm(dataset,
                                                                           dist_dict,
                                                                           args.dataset_name,
                                                                           args.prot_model,
                                                                           target_scaler,
                                                                           seed,
                                                                           batch_size=batch_size,
                                                                           world_size=world_size,
                                                                           rank=rank)

    test_mse, test_pcc = evaluate(model, args.prot_model, test_loader, target_scaler, device)
    print('Device: {:}, Test MSE: {:.4f}, Test PCC: {:.4f}'.format(device, test_mse, test_pcc))
    
    epochs = args.epochs
    max_batches_per_epoch = args.max_epochs_per_batch # Maximum number of batches per GPU per epoch during training
    start = time.time()
    train(model=model, 
          prot_model=args.prot_model,
          loss_fn=criterion, 
          optimizer=optimizer,
          train_loader=train_loader,
          val_loader=val_loader,
          target_scaler=target_scaler,
          epochs=epochs,
          max_batches_per_epoch=max_batches_per_epoch,
          save_dir=args.save_dir,
          device=device)
    end = time.time()
    test_mse, test_pcc = evaluate(model, args.prot_model, test_loader, target_scaler, device)
    print('Device: {:}, Test MSE: {:.4f}, Test PCC: {:.4f}'.format(device, test_mse, test_pcc))
    print('Device: {:}, Time: {:.4f}s'.format(device, end-start))
    dist.destroy_process_group()

###############################################################################
# Finally we load the dataset and launch the processes.
#
# .. code:: python
#
if __name__ == '__main__':
    import torch.multiprocessing as mp
    from tdc.multi_pred import DTI
    import pickle
    import pandas as pd

    args = parse_args()

    datasets = ["DAVIS", "KIBA"]
    args.dataset_name = datasets[args.dataset_choice]

    num_gpus = args.num_gpus
    procs = []

    dataset = DTI(name = args.dataset_name)
    if args.dataset_name == 'DAVIS':
        dataset.convert_to_log(form = 'binding')

    models = ["SeqVec", "DistilProtBERT", "ProtBERT", "CNN"]
    args.prot_model = models[args.prot_lm_model_choice]

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    
    save_dir = args.save_dir + str(args.prot_model.lower()) + "_" + str(args.epochs) + "_" + str(args.dataset_name.lower())

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    args.save_dir = save_dir + "/"
    
    # Print all the arguments
    print(args)

    with open('data/af_davis_reverse_map.pickle', 'rb') as handle:
        af_davis_reverse_map = pickle.load(handle)
    
    af_davis_reverse_map_train = af_davis_reverse_map['train']
    af_davis_reverse_map_test = af_davis_reverse_map['test']
    # af_davis_reverse_map_valid = af_davis_reverse_map['valid']

    # open alphafold_davis2.csv with pandas
    df = pd.read_csv('data/alphafold_davis2.csv')

    distances_path = "/local2/rakesh/DiffDock/results/alpahfold_davis_without_afpdbs/"

    split = dataset.get_split()
    train = split['train']
    val = split['valid']
    test = split['test']

    MAX_MOLECULE_LEN = 64

    def extract_complex_distances(PATH):
        # Load the SDF file
        suppl = Chem.SDMolSupplier(PATH)

        # Distances are 1D np array of size 100 * 100 each value being 0 initially
        distances = np.zeros(MAX_MOLECULE_LEN * MAX_MOLECULE_LEN)

        # Iterate over each molecule in the file
        for mol in suppl:
            if mol is not None:
                # Get the coordinates of each atom
                conf = mol.GetConformer()
                coords = [(atom.GetAtomMapNum(), conf.GetAtomPosition(atom.GetIdx())) for atom in mol.GetAtoms()]

                for i in range(len(coords)):
                    dist_i = []
                    for j in range(len(coords)):
                        distance = (coords[i][1] - coords[j][1]).Length()
                        index = i * MAX_MOLECULE_LEN + j
                        if index < MAX_MOLECULE_LEN * MAX_MOLECULE_LEN and distance < 10:
                            distances[index] = 1
                    # distances.append(dist_i)
                    # distances.append((coords[i][0], coords[j][0], distance))
        return distances

    # train_dist = []
    # for index, row in tqdm(train.iterrows(), total=len(train)):
    #     distances = np.zeros(MAX_MOLECULE_LEN * MAX_MOLECULE_LEN)
    #     if index in af_davis_reverse_map_train:
    #         complex_index = af_davis_reverse_map_train[index]
    #         # Add the complex index to the dstances path
    #         complex_path = distances_path + "complex_" + str(complex_index) + "/"
    #         # Find the number of files in the complex path
    #         num_files = len([name for name in os.listdir(complex_path) if os.path.isfile(os.path.join(complex_path, name))])
    #         if num_files != 0:
    #             # Load the distances
    #             distances = extract_complex_distances(complex_path + "rank1.sdf")
        
    #     # Print shape of distances
    #     # if distances.shape != MAX_MOLECULE_LEN * MAX_MOLECULE_LEN:
    #     #     print("ERROR: ", index, distances.shape)

    #     train_dist.append(distances)

    # # Save the train distances
    # with open('data/train_distances.pickle', 'wb') as handle:
    #     pickle.dump(train_dist, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # Load the train distances
    with open('data/train_distances.pickle', 'rb') as handle:
        train_dist = pickle.load(handle)

    valid_dist = [np.zeros(MAX_MOLECULE_LEN * MAX_MOLECULE_LEN)] * len(val)
    test_dist = [np.zeros(MAX_MOLECULE_LEN * MAX_MOLECULE_LEN)] * len(test)

    # test_dist = []
    # for index, row in tqdm(test.iterrows(), total=len(test)):
    #     distances = []
    #     if index in af_davis_reverse_map_test:
    #         complex_index = af_davis_reverse_map_test[index]
    #         # Add the complex index to the dstances path
    #         complex_path = distances_path + str(complex_index) + "/"
    #         # Find the number of files in the complex path
    #         num_files = len([name for name in os.listdir(complex_path) if os.path.isfile(os.path.join(complex_path, name))])
    #         if num_files == 0:
    #             distances = []
    #         else:
    #             # Load the distances
    #             distances = extract_complex_distances(complex_path + "rank1.sdf")
            
    #     test_dist.append(distances)

    dist_dict = {"train": train_dist, "valid": valid_dist, "test": test_dist}

    mp.spawn(main, args=(num_gpus, args, dataset, dist_dict), nprocs=num_gpus)
